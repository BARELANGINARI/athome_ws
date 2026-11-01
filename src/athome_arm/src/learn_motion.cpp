#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include <filesystem>
#include <map>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <dynamixel_sdk.h>
#include "calibration_data.h"

namespace fs = std::filesystem;

// ---------------- REGISTER AX-18A (PROTOCOL 1.0) ----------------
#define ADDR_MODEL_NUMBER       0
#define ADDR_FIRMWARE           2
#define ADDR_CW_ANGLE_LIMIT     6
#define ADDR_CCW_ANGLE_LIMIT    8
#define ADDR_TORQUE_ENABLE      24
#define ADDR_LED                25
#define ADDR_CW_SLOPE           28
#define ADDR_CCW_SLOPE          29
#define ADDR_GOAL_POSITION      30
#define ADDR_MOVING_SPEED       32
#define ADDR_TORQUE_LIMIT       34
#define ADDR_PRESENT_POS        36
#define ADDR_PRESENT_SPEED      38
#define ADDR_PRESENT_LOAD       40
#define ADDR_PRESENT_VOLT       42
#define ADDR_PRESENT_TEMP       43
#define ADDR_MOVING             46

#define LEN_GOAL_POSITION       2
#define PROTOCOL_VERSION        1.0
#define BAUDRATE                1000000
#define DEVICENAME              "/dev/ttyUSB1"

#define MAX_ANGLE               300.0
#define MAX_TICKS               1023
#define WORK_RANGE_DEG          260.0
#define UPDATE_RATE_HZ          50
#define MAX_SPEED_DPS           60.0
#define MIN_DURATION_S          0.4
#define COMPLIANCE_SLOPE        64         //32
#define DEFAULT_TORQUE_LIM      1023     // ~78% max torque to hold weight

// Deadzone & Haptic Wall Protection Parameters
#define DEADZONE_LOW_TICK       15       // Lower threshold near deadzone 0
#define DEADZONE_HIGH_TICK      1008     // Upper threshold near deadzone 1023
#define DEADZONE_MARGIN_TICKS   35       // Detection margin before deadzone
#define MAX_TICK_JUMP           120      // Jump > ~35 deg per cycle (20ms) = deadzone wrap-around
#define HAPTIC_WALL_MARGIN      8        // Enable holding torque within 8 ticks from limit

// Adaptive Grasping Parameters (ID 6)
#define GRIPPER_ID              6
#define DEFAULT_GRASP_LOAD      170     //180 // Detection load threshold (~17.6% torque)
#define GRASP_STEP_TICKS        4       // Step per iteration (~1.17 deg)
#define GRASP_STEP_MS           40      // 25 Hz sampling period
#define GRASP_STALL_TICKS       15      // Lag (Pos - Goal) indicating object stall
#define GRASP_SQUEEZE_TICKS     20      // Pre-load squeeze after contact
#define GRASP_CONFIRM_COUNT     3       // Contact confirmation cycles
#define GRASP_COMPLIANCE_SLOPE  50      // Soft compliance during grasp
#define GRASP_CLOSING_SPEED     60      // Closing speed (~6.8 RPM)
#define MAX_OPEN_DEG            80.0    // Full open angle (~780 ticks, covers physical opening of 753 ticks)
#define DEFAULT_RELEASE_DEG     65.0    // Default open angle

const std::string MOTION_DIR = "motions";
const uint8_t ARM_CORE_IDS[] = {4, 5, 1};
const uint8_t ALL_IDS[]      = {4, 5, 1, 2};

struct Servo { uint8_t id; bool inverted; };
const Servo JOINT1[] = { {4, false}, {5, true} };
const Servo JOINT2[] = { {1, false} };
const Servo JOINT3[] = { {2, false} };

enum class FrameAction {
    MOVE,           // Normal waypoint / motion sample
    ADAPTIVE_GRASP, // Adaptive grasping step on ID 6
    RELEASE         // Explicit gripper open on ID 6
};

struct Frame {
    double j1;          // Joint 1 (ID 4 & 5)
    double j2;          // Joint 2 (ID 1)
    double grip;        // Gripper (ID 6)
    double duration;    // Transition duration (sec); <= 0 auto via MAX_SPEED_DPS
    FrameAction action = FrameAction::MOVE;
    int grasp_load     = DEFAULT_GRASP_LOAD;
};

enum class MotionMode { WAYPOINT, STREAM };

struct MotionData {
    MotionMode mode = MotionMode::WAYPOINT;
    std::vector<Frame> frames;
};

// ---------------- NON-BLOCKING KEYBOARD INPUT (LINUX TERMIOS) ----------------
class RawTerminalGuard {
    struct termios orig_termios_{};
    bool active_ = false;
public:
    RawTerminalGuard() {
        if (tcgetattr(STDIN_FILENO, &orig_termios_) == 0) {
            struct termios raw = orig_termios_;
            raw.c_lflag &= ~(ICANON | ECHO); // Disable canonical mode & echo
            raw.c_cc[VMIN]  = 0;
            raw.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &raw);
            active_ = true;
        }
    }
    ~RawTerminalGuard() {
        if (active_) {
            tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios_);
        }
    }
};

// Read 1 character from keyboard without blocking 50Hz loop (-1 if no key)
int readKeyNonBlocking() {
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);
    struct timeval tv = {0, 0};
    if (select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, &tv) > 0) {
        unsigned char c = 0;
        if (read(STDIN_FILENO, &c, 1) == 1) return c;
    }
    return -1;
}

dynamixel::PortHandler   *portHandler   = nullptr;
dynamixel::PacketHandler *packetHandler = nullptr;
bool g_gripper_online = false;

// Anti-jump deadzone filter state & background monitor for free mode
static std::map<uint8_t, uint16_t> g_last_valid_raw;
static std::map<uint8_t, bool>     g_in_deadzone;
static std::map<uint8_t, bool>     g_servo_is_free;
static std::atomic<bool>           g_free_monitor_active{false};
static std::thread                 g_free_monitor_thread;

// ---------------- KALIBRASI & KONVERSI ----------------
const CalibratedServoConfig *getCalib(uint8_t id) {
    for (const auto &c : ARM_CALIBRATION) {
        if (c.id == id) return &c;
    }
    return nullptr;
}

double ticksToDeg(int ticks) {
    return ticks * (MAX_ANGLE / MAX_TICKS);
}

int degToTicks(double deg) {
    return static_cast<int>(std::round(deg / MAX_ANGLE * MAX_TICKS));
}

double rawToRelAngle(uint8_t id, uint16_t raw_pos, bool inverted) {
    const auto *c = getCalib(id);
    int center = c ? c->center_tick : 512;
    int delta_ticks = static_cast<int>(raw_pos) - center;
    if (inverted) delta_ticks = -delta_ticks;
    return ticksToDeg(delta_ticks);
}

uint16_t relAngleToPosition(uint8_t id, double rel_angle, bool inverted) {
    rel_angle = std::clamp(rel_angle, -WORK_RANGE_DEG, WORK_RANGE_DEG);
    if (id == GRIPPER_ID) {
        rel_angle = std::clamp(rel_angle, 0.0, MAX_OPEN_DEG);
    }

    const auto *c = getCalib(id);
    int center = c ? c->center_tick : 512;
    int dt = degToTicks(rel_angle);
    if (inverted) dt = -dt;

    int goal = std::clamp(center + dt, 0, MAX_TICKS);
    if (c) {
        goal = std::clamp(goal, static_cast<int>(c->cw_limit), static_cast<int>(c->ccw_limit));
    }
    return static_cast<uint16_t>(goal);
}

// ---------------- KOMUNIKASI SERVO & FILTER DEADZONE ----------------
bool readRawPos(uint8_t id, uint16_t &pos) {
    uint8_t err = 0;
    return packetHandler->read2ByteTxRx(portHandler, id, ADDR_PRESENT_POS, &pos, &err) == COMM_SUCCESS;
}

void resetDeadzoneFilter() {
    for (uint8_t id : ALL_IDS) {
        if (id == GRIPPER_ID && !g_gripper_online) continue;
        const auto *c = getCalib(id);
        uint16_t pos = c ? c->center_tick : 512;
        if (readRawPos(id, pos) && c) {
            pos = std::clamp<uint16_t>(pos, c->cw_limit, c->ccw_limit);
        }
        g_last_valid_raw[id] = pos;
        g_in_deadzone[id]    = false;
    }
}

uint16_t readFilteredRawPos(uint8_t id, bool enable_haptic_wall = false) {
    const auto *c = getCalib(id);
    uint16_t min_lim = c ? c->cw_limit  : 10;
    uint16_t max_lim = c ? c->ccw_limit : 1010;

    uint16_t raw = 0;
    if (!readRawPos(id, raw)) {
        return g_last_valid_raw.count(id) ? g_last_valid_raw[id] : (c ? c->center_tick : 512);
    }

    if (g_last_valid_raw.count(id)) {
        int prev = g_last_valid_raw[id];
        int diff = std::abs(static_cast<int>(raw) - prev);

        bool near_high = (prev >= static_cast<int>(max_lim) - DEADZONE_MARGIN_TICKS) ||
                         (prev >= DEADZONE_HIGH_TICK - DEADZONE_MARGIN_TICKS);
        bool near_low  = (prev <= static_cast<int>(min_lim) + DEADZONE_MARGIN_TICKS) ||
                         (prev <= DEADZONE_LOW_TICK  + DEADZONE_MARGIN_TICKS);

        if ((near_high || near_low) && diff > MAX_TICK_JUMP) {
            if (!g_in_deadzone[id]) {
                std::cerr << "\n  [WARN] ID " << (int)id << " exceeded angle limit/deadzone (raw="
                          << raw << ")! Holding reading at limit: " << prev << " ticks.\n";
                g_in_deadzone[id] = true;
            }
            raw = static_cast<uint16_t>(near_high ? max_lim : min_lim);
        } else if (g_in_deadzone[id]) {
            if (std::abs(static_cast<int>(raw) - prev) <= MAX_TICK_JUMP) {
                g_in_deadzone[id] = false;
            } else {
                raw = static_cast<uint16_t>(prev);
            }
        }
    }

    uint16_t clamped = std::clamp<uint16_t>(raw, min_lim, max_lim);
    g_last_valid_raw[id] = clamped;

    // Active Haptic Wall only for servos currently in free / record mode
    if (enable_haptic_wall && g_servo_is_free[id]) {
        uint8_t err = 0;
        if (clamped >= max_lim - HAPTIC_WALL_MARGIN || clamped <= min_lim + HAPTIC_WALL_MARGIN || g_in_deadzone[id]) {
            uint16_t wall_pos = (clamped >= max_lim - HAPTIC_WALL_MARGIN)
                                    ? (max_lim - HAPTIC_WALL_MARGIN)
                                    : (min_lim + HAPTIC_WALL_MARGIN);
            packetHandler->write2ByteTxRx(portHandler, id, ADDR_TORQUE_LIMIT, 500, &err);
            packetHandler->write2ByteTxRx(portHandler, id, ADDR_GOAL_POSITION, wall_pos, &err);
            packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, 1, &err);
            packetHandler->write1ByteTxRx(portHandler, id, ADDR_LED, 1, &err);
        } else {
            packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, 0, &err);
            packetHandler->write1ByteTxRx(portHandler, id, ADDR_LED, 0, &err);
        }
    }

    return clamped;
}

int readLoadMagnitude(uint8_t id, int &signed_load) {
    uint16_t raw = 0;
    uint8_t err = 0;
    if (packetHandler->read2ByteTxRx(portHandler, id, ADDR_PRESENT_LOAD, &raw, &err) != COMM_SUCCESS) {
        signed_load = 0;
        return 0;
    }
    int mag = raw & 0x3FF;
    bool cw = (raw & 0x400) != 0;
    signed_load = cw ? -mag : mag;
    return mag;
}

double readServoRelAngle(uint8_t id, bool inverted, bool enable_haptic_wall = false) {
    uint16_t pos = readFilteredRawPos(id, enable_haptic_wall);
    return rawToRelAngle(id, pos, inverted);
}

void readCurrentPose(double pose[3], bool enable_haptic_wall = false) {
    double a4 = readServoRelAngle(4, false, enable_haptic_wall);
    double a5 = readServoRelAngle(5, true,  enable_haptic_wall);
    pose[0] = 0.5 * (a4 + a5);
    pose[1] = readServoRelAngle(1, false, enable_haptic_wall);
    if (g_gripper_online) {
        pose[2] = std::clamp(readServoRelAngle(GRIPPER_ID, false, false), 0.0, MAX_OPEN_DEG);
    }
}

// ---------------- BACKGROUND MONITOR THREAD (FOR FREE MODE) ----------------
void stopFreeMonitor() {
    if (g_free_monitor_active.load()) {
        g_free_monitor_active.store(false);
        if (g_free_monitor_thread.joinable()) {
            g_free_monitor_thread.join();
        }
    }
}

void startFreeMonitor() {
    stopFreeMonitor();
    g_free_monitor_active.store(true);
    g_free_monitor_thread = std::thread([]() {
        const auto period = std::chrono::milliseconds(1000 / UPDATE_RATE_HZ);
        while (g_free_monitor_active.load()) {
            double dummy[3];
            readCurrentPose(dummy, true);
            std::this_thread::sleep_for(period);
        }
    });
}

template <size_t N>
void addJointParam(dynamixel::GroupSyncWrite &gsw, const Servo (&joint)[N], double rel_angle) {
    for (const Servo &s : joint) {
        if (s.id == GRIPPER_ID && !g_gripper_online) continue;
        uint16_t pos = relAngleToPosition(s.id, rel_angle, s.inverted);
        uint8_t param[2] = { DXL_LOBYTE(pos), DXL_HIBYTE(pos) };
        gsw.addParam(s.id, param);
    }
}

void sendJoints(dynamixel::GroupSyncWrite &gsw, const double a[3]) {
    gsw.clearParam();
    addJointParam(gsw, JOINT1, a[0]);
    addJointParam(gsw, JOINT2, a[1]);
    addJointParam(gsw, JOINT3, a[2]);
    gsw.txPacket();
}

void setTorqueAll(bool enable) {
    stopFreeMonitor();
    uint8_t err = 0;
    if (!enable) resetDeadzoneFilter();
    for (uint8_t id : ALL_IDS) {
        if (id == GRIPPER_ID && !g_gripper_online) continue;
        g_servo_is_free[id] = !enable;
        packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, enable ? 1 : 0, &err);
        if (!enable) packetHandler->write1ByteTxRx(portHandler, id, ADDR_LED, 0, &err);
    }
}

// Set torque for main arm J1 & J2 (ID 4, 5, 1) WITHOUT altering Gripper (ID 6) status
void setArmOnlyTorque(dynamixel::GroupSyncWrite &gsw, double cur[3], bool enable) {
    uint8_t err = 0;
    if (enable) {
        readCurrentPose(cur, false);
        gsw.clearParam();
        addJointParam(gsw, JOINT1, cur[0]);
        addJointParam(gsw, JOINT2, cur[1]);
        gsw.txPacket();
    } else {
        resetDeadzoneFilter();
    }

    for (uint8_t id : ARM_CORE_IDS) {
        g_servo_is_free[id] = !enable;
        packetHandler->write1ByteTxRx(portHandler, id, ADDR_LED, 0, &err);
        if (enable) {
            packetHandler->write2ByteTxRx(portHandler, id, ADDR_MOVING_SPEED, 0, &err);
            packetHandler->write2ByteTxRx(portHandler, id, ADDR_TORQUE_LIMIT, DEFAULT_TORQUE_LIM, &err);
            packetHandler->write1ByteTxRx(portHandler, id, ADDR_CW_SLOPE,  COMPLIANCE_SLOPE, &err);
            packetHandler->write1ByteTxRx(portHandler, id, ADDR_CCW_SLOPE, COMPLIANCE_SLOPE, &err);
        }
        packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, enable ? 1 : 0, &err);
    }
}

void setJointTorque(const std::string &joint_name, bool enable) {
    stopFreeMonitor();
    uint8_t err = 0;
    if (!enable) resetDeadzoneFilter();
    if (joint_name == "j1") {
        g_servo_is_free[4] = !enable;
        g_servo_is_free[5] = !enable;
        packetHandler->write1ByteTxRx(portHandler, 4, ADDR_TORQUE_ENABLE, enable ? 1 : 0, &err);
        packetHandler->write1ByteTxRx(portHandler, 5, ADDR_TORQUE_ENABLE, enable ? 1 : 0, &err);
    } else if (joint_name == "j2") {
        g_servo_is_free[1] = !enable;
        packetHandler->write1ByteTxRx(portHandler, 1, ADDR_TORQUE_ENABLE, enable ? 1 : 0, &err);
    } else if (joint_name == "arm") {
        for (uint8_t id : ARM_CORE_IDS) {
            g_servo_is_free[id] = !enable;
            packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, enable ? 1 : 0, &err);
        }
    } else if (joint_name == "grip" && g_gripper_online) {
        g_servo_is_free[GRIPPER_ID] = !enable;
        packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, enable ? 1 : 0, &err);
    }
}

void lockCurrentPose(dynamixel::GroupSyncWrite &gsw, double cur[3]) {
    stopFreeMonitor();
    readCurrentPose(cur, false);
    sendJoints(gsw, cur);
    uint8_t err = 0;
    for (uint8_t id : ALL_IDS) {
        if (id == GRIPPER_ID && !g_gripper_online) continue;
        g_servo_is_free[id] = false;
        packetHandler->write1ByteTxRx(portHandler, id, ADDR_LED, 0, &err);
        packetHandler->write2ByteTxRx(portHandler, id, ADDR_MOVING_SPEED, 0, &err);
        packetHandler->write2ByteTxRx(portHandler, id, ADDR_TORQUE_LIMIT, DEFAULT_TORQUE_LIM, &err);
        packetHandler->write1ByteTxRx(portHandler, id, ADDR_CW_SLOPE,  COMPLIANCE_SLOPE, &err);
        packetHandler->write1ByteTxRx(portHandler, id, ADDR_CCW_SLOPE, COMPLIANCE_SLOPE, &err);
        packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, 1, &err);
    }
}

bool initServos() {
    uint8_t err = 0;
    for (uint8_t id : ARM_CORE_IDS) {
        uint16_t model = 0;
        if (packetHandler->ping(portHandler, id, &model, &err) != COMM_SUCCESS) {
            std::cerr << "[ERROR] Main servo ID " << (int)id << " not detected!\n";
            return false;
        }
        g_servo_is_free[id] = false;
        packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, 0, &err);
        if (const auto *c = getCalib(id)) {
            packetHandler->write2ByteTxRx(portHandler, id, ADDR_CW_ANGLE_LIMIT,  c->cw_limit,  &err);
            packetHandler->write2ByteTxRx(portHandler, id, ADDR_CCW_ANGLE_LIMIT, c->ccw_limit, &err);
        }
    }

    uint16_t grip_model = 0;
    if (packetHandler->ping(portHandler, GRIPPER_ID, &grip_model, &err) == COMM_SUCCESS) {
        g_gripper_online = true;
        g_servo_is_free[GRIPPER_ID] = false;
        packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 0, &err);
        if (const auto *c = getCalib(GRIPPER_ID)) {
            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_ANGLE_LIMIT,  c->cw_limit,  &err);
            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_ANGLE_LIMIT, c->ccw_limit, &err);
        }
        std::cout << "[OK] Gripper ID 6 detected (Adaptive Grasping active).\n";
    } else {
        g_gripper_online = false;
        std::cout << "[WARN] Gripper ID 6 not detected on bus.\n";
    }
    resetDeadzoneFilter();
    return true;
}

std::string makeLoadBar(int load_mag, int threshold) {
    const int width = 20;
    int filled = std::clamp(load_mag * width / 512, 0, width);
    int thresh_pos = std::clamp(threshold * width / 512, 0, width - 1);
    std::string bar(width, '.');
    for (int i = 0; i < filled; i++) bar[i] = '#';
    bar[thresh_pos] = '|';
    return "[" + bar + "]";
}

// ---------------- ADAPTIVE GRASPING (ID 6) ----------------
bool adaptiveGrasp(double &cur_grip_deg, int load_threshold = DEFAULT_GRASP_LOAD) {
    stopFreeMonitor();
    uint8_t err = 0;
    uint16_t model = 0;
    if (!g_gripper_online) {
        if (packetHandler->ping(portHandler, GRIPPER_ID, &model, &err) == COMM_SUCCESS) {
            g_gripper_online = true;
        } else {
            std::cout << "  [!] Gripper ID 6 not connected!\n";
            return false;
        }
    }

    load_threshold = std::clamp(load_threshold, 50, 800);
    const auto *c = getCalib(GRIPPER_ID);
    const int closed_tick   = c ? c->center_tick : 510;
    const int max_open_tick = c ? c->ccw_limit    : 750;

    uint16_t start_pos = closed_tick;
    if (!readRawPos(GRIPPER_ID, start_pos)) {
        std::cout << "  [!] Failed to read ID 6 start position.\n";
        return false;
    }

    g_servo_is_free[GRIPPER_ID] = false;
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 1, &err);

    int search_torque = std::clamp(load_threshold + 250, 350, 850);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_SLOPE,  GRASP_COMPLIANCE_SLOPE, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_SLOPE, GRASP_COMPLIANCE_SLOPE, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING_SPEED, GRASP_CLOSING_SPEED, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT, search_torque, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 1, &err);

    if (static_cast<int>(start_pos) <= closed_tick + 10) {
        uint16_t open_tick = relAngleToPosition(GRIPPER_ID, 80.0, false);
        std::cout << "  [PRE-OPEN] Jaws closed, opening to +80.0 deg (tick=" << open_tick << ")...\n";
        packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, open_tick, &err);
        std::this_thread::sleep_for(std::chrono::milliseconds(650));
        readRawPos(GRIPPER_ID, start_pos);
    }

    std::cout << "\n================ ADAPTIVE GRASPING PROCESS (ID 6) ================\n";
    std::cout << "  Closing from tick " << start_pos << " (" << std::fixed << std::setprecision(1)
              << rawToRelAngle(GRIPPER_ID, start_pos, false) << " deg) down to tick " << closed_tick << " (0.0 deg)\n"
              << "  Load Threshold: " << load_threshold << " | Stall Limit: " << GRASP_STALL_TICKS << " ticks\n";
    std::cout << std::left
              << std::setw(6)  << "Step"
              << std::setw(6)  << "Goal"
              << std::setw(6)  << "Pos"
              << std::setw(9)  << "Angle"
              << std::setw(6)  << "Lag"
              << std::setw(12) << "Load(Raw/%)"
              << std::setw(24) << "Load Bar"
              << "State\n";
    std::cout << "-----------------------------------------------------------------------------\n";

    int goal_tick = std::min<int>(max_open_tick, start_pos);
    int contact_count = 0;
    int step = 0;
    uint16_t present_pos = start_pos;
    int signed_load = 0;
    int load_mag = 0;

    while (goal_tick > closed_tick) {
        step++;
        goal_tick = std::max(closed_tick, goal_tick - GRASP_STEP_TICKS);
        packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION,
                                      static_cast<uint16_t>(goal_tick), &err);
        std::this_thread::sleep_for(std::chrono::milliseconds(GRASP_STEP_MS));

        readRawPos(GRIPPER_ID, present_pos);
        load_mag = readLoadMagnitude(GRIPPER_ID, signed_load);
        int lag = static_cast<int>(present_pos) - goal_tick;
        double load_pct = (load_mag / 1023.0) * 100.0;

        bool trig_load  = (load_mag >= load_threshold);
        bool trig_stall = (lag >= GRASP_STALL_TICKS);
        std::string state = "SEARCHING";

        if (step > 2 && (trig_load || trig_stall)) {
            contact_count++;
            state = "CONTACT (" + std::to_string(contact_count) + "/" + std::to_string(GRASP_CONFIRM_COUNT) + ")";
        } else {
            contact_count = 0;
        }

        std::ostringstream load_str;
        load_str << std::setw(3) << load_mag << " (" << std::fixed << std::setprecision(1) << load_pct << "%)";

        std::cout << std::left
                  << std::setw(6)  << step
                  << std::setw(6)  << goal_tick
                  << std::setw(6)  << present_pos
                  << std::setw(9)  << std::fixed << std::setprecision(1) << rawToRelAngle(GRIPPER_ID, present_pos, false)
                  << std::setw(6)  << std::showpos << lag << std::noshowpos
                  << std::setw(12) << load_str.str()
                  << std::setw(24) << makeLoadBar(load_mag, load_threshold)
                  << state << "\n";

        if (contact_count >= GRASP_CONFIRM_COUNT) {
            int hold_tick = std::clamp(static_cast<int>(present_pos) - GRASP_SQUEEZE_TICKS,
                                       closed_tick, max_open_tick);
            // int hold_torque = std::clamp(load_mag + 160, 280, 650);
            int hold_torque = std::clamp(load_mag + 200, 360, 450);

            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT,
                                          static_cast<uint16_t>(hold_torque), &err);
            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION,
                                          static_cast<uint16_t>(hold_tick), &err);

            cur_grip_deg = rawToRelAngle(GRIPPER_ID, static_cast<uint16_t>(hold_tick), false);
            g_last_valid_raw[GRIPPER_ID] = static_cast<uint16_t>(hold_tick);
            std::cout << "-----------------------------------------------------------------------------\n";
            std::cout << "  [HOLDING OBJECT] Object grasped successfully!\n"
                      << "    -> Object Position : tick " << present_pos << " ("
                      << std::fixed << std::setprecision(1) << cur_grip_deg << " deg)\n"
                      << "    -> Lock Target     : tick " << hold_tick << " (squeeze -" << GRASP_SQUEEZE_TICKS << " ticks)\n"
                      << "    -> Contact Load    : " << load_mag << " (" << load_pct << "%)\n"
                      << "    -> Holding Torque  : " << hold_torque << " / 1023 ("
                      << (hold_torque / 1023.0 * 100.0) << "%)\n";
            std::cout << "=============================================================================\n";
            return true;
        }
    }

    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
    cur_grip_deg = 0.0;
    g_last_valid_raw[GRIPPER_ID] = static_cast<uint16_t>(closed_tick);
    std::cout << "-----------------------------------------------------------------------------\n";
    std::cout << "  [CLOSED EMPTY] Jaws reached 0.0 deg (tick=" << closed_tick
              << ") without detecting object.\n";
    std::cout << "=============================================================================\n";
    return false;
}

// ---------------- TRAJECTORY & PLAYBACK ----------------
inline double minJerk(double t) {
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t);
}

void moveSmooth(dynamixel::GroupSyncWrite &gsw, double cur[3], const double target[3], double custom_duration = 0.0) {
    double maxDelta = 0;
    for (int i = 0; i < 3; i++) {
        maxDelta = std::max(maxDelta, std::fabs(target[i] - cur[i]));
    }
    if (maxDelta < 0.05 && custom_duration <= 0.0) return;

    double T = (custom_duration > 0.0)
                   ? std::max(0.1, custom_duration)
                   : std::max(MIN_DURATION_S, maxDelta * 1.875 / MAX_SPEED_DPS);
    int steps = std::max(1, static_cast<int>(std::ceil(T * UPDATE_RATE_HZ)));

    double start[3] = { cur[0], cur[1], cur[2] };
    auto next = std::chrono::steady_clock::now();
    const auto period = std::chrono::microseconds(1000000 / UPDATE_RATE_HZ);

    for (int k = 1; k <= steps; k++) {
        double s = minJerk(static_cast<double>(k) / steps);
        double a[3];
        for (int i = 0; i < 3; i++) {
            a[i] = start[i] + (target[i] - start[i]) * s;
        }
        sendJoints(gsw, a);
        next += period;
        std::this_thread::sleep_until(next);
    }
    for (int i = 0; i < 3; i++) cur[i] = target[i];
    resetDeadzoneFilter();
}

// Open gripper without altering arm J1 & J2 torque state
void openGripperOnly(double &cur_grip_deg, double open_deg = DEFAULT_RELEASE_DEG) {
    open_deg = std::clamp(open_deg, 0.0, MAX_OPEN_DEG);
    if (!g_gripper_online) return;

    uint8_t err = 0;
    g_servo_is_free[GRIPPER_ID] = false;
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_SLOPE,  COMPLIANCE_SLOPE, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_SLOPE, COMPLIANCE_SLOPE, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING_SPEED, 120, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT, DEFAULT_TORQUE_LIM, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 1, &err);

    uint16_t goal_tick = relAngleToPosition(GRIPPER_ID, open_deg, false);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, goal_tick, &err);
    std::this_thread::sleep_for(std::chrono::milliseconds(450));
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING_SPEED, 0, &err);

    cur_grip_deg = open_deg;
    g_last_valid_raw[GRIPPER_ID] = goal_tick;
}

void releaseGripper(dynamixel::GroupSyncWrite &gsw, double cur[3], double open_deg = DEFAULT_RELEASE_DEG) {
    stopFreeMonitor();
    open_deg = std::clamp(open_deg, 0.0, MAX_OPEN_DEG);
    if (g_gripper_online) {
        uint8_t err = 0;
        g_servo_is_free[GRIPPER_ID] = false;
        packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
        packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_SLOPE,  COMPLIANCE_SLOPE, &err);
        packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_SLOPE, COMPLIANCE_SLOPE, &err);
        packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING_SPEED, 0, &err);
        packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT, DEFAULT_TORQUE_LIM, &err);
    }
    double target[3] = { cur[0], cur[1], open_deg };
    std::cout << "  [RELEASE] Opening gripper to +" << open_deg << " deg...\n";
    moveSmooth(gsw, cur, target, 0.5);
}

void executeMotion(dynamixel::GroupSyncWrite &gsw, double cur[3], const MotionData &motion, int loops) {
    stopFreeMonitor();
    if (motion.frames.empty()) {
        std::cout << "  [!] Motion data is empty.\n";
        return;
    }

    lockCurrentPose(gsw, cur);

    for (int l = 1; l <= loops; l++) {
        if (loops > 1) {
            std::cout << "  -> Running cycle " << l << " / " << loops << "...\n";
        }
        bool holding_object = false;

        if (motion.mode == MotionMode::WAYPOINT) {
            for (size_t i = 0; i < motion.frames.size(); i++) {
                const auto &f = motion.frames[i];
                if (f.action == FrameAction::ADAPTIVE_GRASP) {
                    double arm_target[3] = { f.j1, f.j2, cur[2] };
                    moveSmooth(gsw, cur, arm_target, f.duration);
                    std::cout << "     Waypoint #" << (i + 1) << " [ADAPTIVE GRASP]: load_thresh="
                              << f.grasp_load << "\n";
                    holding_object = adaptiveGrasp(cur[2], f.grasp_load);
                } else if (f.action == FrameAction::RELEASE) {
                    double arm_target[3] = { f.j1, f.j2, cur[2] };
                    moveSmooth(gsw, cur, arm_target, f.duration);
                    std::cout << "     Waypoint #" << (i + 1) << " [RELEASE]: +" << f.grip << " deg\n";
                    releaseGripper(gsw, cur, f.grip);
                    holding_object = false;
                } else {
                    double grip_target = holding_object ? cur[2] : f.grip;
                    double target[3] = { f.j1, f.j2, grip_target };
                    std::cout << "     Waypoint #" << (i + 1) << ": J1=" << target[0]
                              << ", J2=" << target[1] << ", Grip=" << target[2] << "\n";
                    moveSmooth(gsw, cur, target, f.duration);
                }
            }
        } else {
            // STREAM Mode: seamless ADAPTIVE_GRASP & RELEASE
            const auto &first = motion.frames.front();
            double start_target[3] = { first.j1, first.j2, first.grip };
            std::cout << "  -> Moving to start position...\n";
            moveSmooth(gsw, cur, start_target, 0.0);

            std::cout << "  -> Replaying stream (" << motion.frames.size() << " samples @ "
                      << UPDATE_RATE_HZ << " Hz)...\n";
            auto next = std::chrono::steady_clock::now();
            const auto period = std::chrono::microseconds(1000000 / UPDATE_RATE_HZ);

            for (const auto &f : motion.frames) {
                if (f.action == FrameAction::ADAPTIVE_GRASP) {
                    double arm_hold[3] = { f.j1, f.j2, cur[2] };
                    sendJoints(gsw, arm_hold);
                    cur[0] = f.j1; cur[1] = f.j2;
                    std::cout << "\n  [STREAM EVENT] Executing Adaptive Grasp (load=" << f.grasp_load << ")...\n";
                    holding_object = adaptiveGrasp(cur[2], f.grasp_load);
                    next = std::chrono::steady_clock::now();
                } else if (f.action == FrameAction::RELEASE) {
                    double arm_hold[3] = { f.j1, f.j2, cur[2] };
                    sendJoints(gsw, arm_hold);
                    cur[0] = f.j1; cur[1] = f.j2;
                    std::cout << "\n  [STREAM EVENT] Opening Gripper to +" << f.grip << " deg...\n";
                    releaseGripper(gsw, cur, f.grip);
                    holding_object = false;
                    next = std::chrono::steady_clock::now();
                } else {
                    double grip_cmd = holding_object ? cur[2] : f.grip;
                    double a[3] = { f.j1, f.j2, grip_cmd };
                    sendJoints(gsw, a);
                    cur[0] = a[0]; cur[1] = a[1]; cur[2] = a[2];
                    next += period;
                    std::this_thread::sleep_until(next);
                }
            }
            resetDeadzoneFilter();
        }
    }
    std::cout << "  [OK] Motion execution completed.\n";
}

// ---------------- INTERACTIVE CONTINUOUS RECORDING (PAUSE-AND-LOCK) ----------------
void recordInteractiveStream(dynamixel::GroupSyncWrite &gsw, double cur[3],
                             MotionData &buffer, double max_secs) {
    stopFreeMonitor();

    // 1. Open gripper to ready position (+65 deg) and keep ID 6 locked,
    //    while J1 & J2 (ID 4, 5, 1) are free.
    readCurrentPose(cur, false);
    if (g_gripper_online) {
        openGripperOnly(cur[2], DEFAULT_RELEASE_DEG);
    }
    setArmOnlyTorque(gsw, cur, false);

    std::cout << "\n================ INTERACTIVE RECORD MODE (50 Hz) ================\n"
              << "  - Arm torque (J1 & J2) DISABLED; Gripper ready at +" << cur[2] << " deg.\n"
              << "  - Real-time Key Controls (no need to press ENTER):\n"
              << "      [g]         -> PAUSE & LOCK Arm + Run Adaptive Grasp\n"
              << "      [r]         -> PAUSE & LOCK Arm + Open Gripper (Release)\n"
              << "      [q] / SPACE -> Finish recording early & lock position\n"
              << "----------------------------------------------------------------\n"
              << "  Place your hand on the robot arm, press ENTER to START...";
    std::string dummy;
    std::getline(std::cin, dummy);

    resetDeadzoneFilter();
    buffer.frames.clear();
    buffer.mode = MotionMode::STREAM;

    int max_steps = static_cast<int>(std::ceil(max_secs * UPDATE_RATE_HZ));
    const auto period = std::chrono::microseconds(1000000 / UPDATE_RATE_HZ);
    bool holding_object = false;

    std::cout << "\n  [REC] Recording motion (Max " << max_secs << "s)... "
              << "Press [g]=Grasp, [r]=Release, [q]=Stop\n";

    {
        RawTerminalGuard raw_term;
        auto next = std::chrono::steady_clock::now();

        for (int k = 0; k < max_steps; k++) {
            double p[3] = {cur[0], cur[1], cur[2]};
            readCurrentPose(p, true);
            if (holding_object) {
                p[2] = cur[2];
            }
            cur[0] = p[0]; cur[1] = p[1]; cur[2] = p[2];

            buffer.frames.push_back(Frame{
                p[0], p[1], p[2], 1.0 / UPDATE_RATE_HZ, FrameAction::MOVE, DEFAULT_GRASP_LOAD
            });

            if (k % 5 == 0) {
                double elapsed = static_cast<double>(k) / UPDATE_RATE_HZ;
                std::cout << "\r  [REC " << std::fixed << std::setprecision(1)
                          << std::setw(4) << elapsed << "s / " << max_secs << "s] "
                          << "J1=" << std::setw(6) << p[0] << " deg | "
                          << "J2=" << std::setw(6) << p[1] << " deg | "
                          << "Grip=" << std::setw(5) << p[2] << " deg "
                          << (holding_object ? "[HOLDING]" : "[OPEN]   ") << std::flush;
            }

            int key = readKeyNonBlocking();
            if (key == 'g' || key == 'G') {
                std::cout << "\n\n  >>> [TRIGGER 'g'] PAUSE & LOCK Arm for Adaptive Grasp...\n";
                setArmOnlyTorque(gsw, cur, true);

                holding_object = adaptiveGrasp(cur[2], DEFAULT_GRASP_LOAD);

                buffer.frames.push_back(Frame{
                    cur[0], cur[1], cur[2], 0.0, FrameAction::ADAPTIVE_GRASP, DEFAULT_GRASP_LOAD
                });

                setArmOnlyTorque(gsw, cur, false);
                std::cout << "  <<< [RESUME REC] Arm J1 & J2 freed (Gripper holding object). "
                          << "Move the object now!\n";
                next = std::chrono::steady_clock::now();
            } else if (key == 'r' || key == 'R') {
                std::cout << "\n\n  >>> [TRIGGER 'r'] PAUSE & LOCK Arm for Gripper Release...\n";
                setArmOnlyTorque(gsw, cur, true);

                openGripperOnly(cur[2], DEFAULT_RELEASE_DEG);
                holding_object = false;

                buffer.frames.push_back(Frame{
                    cur[0], cur[1], cur[2], 0.5, FrameAction::RELEASE, DEFAULT_GRASP_LOAD
                });

                setArmOnlyTorque(gsw, cur, false);
                std::cout << "  <<< [RESUME REC] Object released, arm J1 & J2 freed.\n";
                next = std::chrono::steady_clock::now();
            } else if (key == 'q' || key == 'Q' || key == ' ') {
                std::cout << "\n\n  >>> [STOP] Recording stopped by user.\n";
                break;
            }

            next += period;
            std::this_thread::sleep_until(next);
        }
    }

    lockCurrentPose(gsw, cur);
    std::cout << "\n  [DONE] Continuous recording complete (" << buffer.frames.size()
              << " samples) and arm position locked.\n";
}

// ---------------- SAVE & LOAD MOTION FILES ----------------
std::string getMotionPath(const std::string &name) {
    fs::create_directories(MOTION_DIR);
    return MOTION_DIR + "/" + name + ".motion";
}

bool saveMotionToFile(const std::string &name, const MotionData &motion) {
    if (motion.frames.empty()) {
        std::cout << "  [!] No motion points recorded yet. Use 'add' or 'record' first.\n";
        return false;
    }
    std::string path = getMotionPath(name);
    std::ofstream out(path);
    if (!out) {
        std::cerr << "  [ERROR] Failed to write file: " << path << "\n";
        return false;
    }

    out << (motion.mode == MotionMode::STREAM ? "MODE STREAM\n" : "MODE WAYPOINT\n");
    out << std::fixed << std::setprecision(2);
    for (const auto &f : motion.frames) {
        if (f.action == FrameAction::ADAPTIVE_GRASP) {
            out << "GRASP " << f.j1 << " " << f.j2 << " " << f.grasp_load << " " << f.duration << "\n";
        } else if (f.action == FrameAction::RELEASE) {
            out << "RELEASE " << f.j1 << " " << f.j2 << " " << f.grip << " " << f.duration << "\n";
        } else {
            out << f.j1 << " " << f.j2 << " " << f.grip << " " << f.duration << "\n";
        }
    }
    out.close();
    std::cout << "  [SUCCESS] Motion '" << name << "' (" << motion.frames.size()
              << " points) saved to " << path << "\n";
    return true;
}

bool loadMotionFromFile(const std::string &name, MotionData &motion) {
    std::string path = getMotionPath(name);
    std::ifstream in(path);
    if (!in) {
        std::cout << "  [!] Motion file '" << path << "' not found.\n";
        return false;
    }

    motion.frames.clear();
    motion.mode = MotionMode::WAYPOINT;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.find("MODE") == 0) {
            if (line.find("STREAM") != std::string::npos) {
                motion.mode = MotionMode::STREAM;
            }
            continue;
        }
        std::stringstream ss(line);
        std::string first_token;
        ss >> first_token;
        if (first_token == "GRASP") {
            Frame f{};
            f.action = FrameAction::ADAPTIVE_GRASP;
            if (ss >> f.j1 >> f.j2 >> f.grasp_load) {
                if (!(ss >> f.duration)) f.duration = 0.0;
                motion.frames.push_back(f);
            }
        } else if (first_token == "RELEASE") {
            Frame f{};
            f.action = FrameAction::RELEASE;
            if (ss >> f.j1 >> f.j2 >> f.grip) {
                if (!(ss >> f.duration)) f.duration = 0.5;
                motion.frames.push_back(f);
            }
        } else {
            std::stringstream ls(line);
            Frame f{};
            f.action = FrameAction::MOVE;
            if (ls >> f.j1 >> f.j2 >> f.grip) {
                if (!(ls >> f.duration)) f.duration = 0.0;
                motion.frames.push_back(f);
            }
        }
    }
    return !motion.frames.empty();
}

void listSavedMotions() {
    fs::create_directories(MOTION_DIR);
    std::cout << "  Saved motions in folder '" << MOTION_DIR << "/':\n";
    bool any = false;
    for (const auto &entry : fs::directory_iterator(MOTION_DIR)) {
        if (entry.path().extension() == ".motion") {
            std::cout << "    - " << entry.path().stem().string() << "\n";
            any = true;
        }
    }
    if (!any) {
        std::cout << "    (No motion files saved yet)\n";
    }
}

void printHelp() {
    std::cout << "\n================ TEACH, PLAYBACK & ADAPTIVE GRASPING MENU ================\n"
              << "  1. Torque Control & Manual Pose:\n"
              << "     free                 -> Disable all servo torque (+50Hz Deadzone & Haptic Wall Monitor)\n"
              << "     free arm             -> Disable J1 & J2 torque only (Gripper keeps holding!)\n"
              << "     free <j1|j2|grip>    -> Disable single joint torque (+50Hz Deadzone & Haptic Wall Monitor)\n"
              << "     lock                 -> Lock (Torque ON) arm at current position\n"
              << "     move <j1> <j2> <grip>-> Move arm to specific angles (J1, J2, Grip: 0=closed..+65=open)\n"
              << "     zero                 -> Move all joints to calibrated zero position\n\n"
              << "  2. Adaptive Grasping (Gripper ID 6):\n"
              << "     grasp [load_thresh]  -> Grasp object adaptively (default load=" << DEFAULT_GRASP_LOAD << ")\n"
              << "     release [deg]        -> Open gripper jaws (default +" << DEFAULT_RELEASE_DEG << " deg)\n\n"
              << "  3. Full Motion Recording (Stream & Waypoint):\n"
              << "     record [seconds]     -> Record 50 Hz continuous motion (default 20s) with live keys:\n"
              << "                             [g]=Pause & Grasp, [r]=Pause & Release, [q]/SPACE=Stop\n"
              << "     add [duration_sec]   -> Record current arm position as waypoint\n"
              << "     add grasp [load]     -> Run & record Adaptive Grasping at current position\n"
              << "     add release [deg]    -> Run & record gripper open at current position\n"
              << "     show                 -> View current position & recorded waypoints in memory\n"
              << "     clear                -> Clear temporary recording from memory\n\n"
              << "  4. Save & Play Motion:\n"
              << "     save <name>          -> Save recording to file (e.g.: save pick_item)\n"
              << "     play <name> [loop]   -> Play saved motion (e.g.: play pick_item)\n"
              << "     play                 -> Play recording currently in memory\n"
              << "     list                 -> List all saved motion names\n"
              << "     help                 -> Show this help menu\n"
              << "     quit                 -> Disable torque and exit\n"
              << "==========================================================================\n";
}

// ---------------- MAIN ----------------
int main() {
    portHandler   = dynamixel::PortHandler::getPortHandler(DEVICENAME);
    packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);
    dynamixel::GroupSyncWrite gsw(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION);

    if (!portHandler->openPort() || !portHandler->setBaudRate(BAUDRATE)) {
        std::cerr << "[ERROR] Failed to open port " << DEVICENAME << " or set baudrate!\n";
        return -1;
    }

    if (!initServos()) {
        portHandler->closePort();
        return -1;
    }

    double cur[3] = {0.0, 0.0, 0.0};
    lockCurrentPose(gsw, cur);
    std::cout << "[OK] All servos ready and locked at initial pose: "
              << "J1=" << std::fixed << std::setprecision(1) << cur[0]
              << ", J2=" << cur[1] << ", Grip=" << cur[2] << "\n";

    MotionData buffer;
    printHelp();

    while (true) {
        std::cout << "\nCommand (type 'help' for menu): ";
        std::string line;
        if (!std::getline(std::cin, line)) break;

        std::stringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd.empty()) continue;

        if (cmd == "free") {
            std::string which;
            if (ss >> which) {
                setJointTorque(which, false);
                startFreeMonitor();
                std::cout << "  [FREE] Torque '" << which << "' disabled (50Hz Deadzone & Haptic Wall Protection ACTIVE).\n";
            } else {
                setTorqueAll(false);
                startFreeMonitor();
                std::cout << "  [FREE] All servo torque disabled (50Hz Deadzone & Haptic Wall Protection ACTIVE).\n";
            }
        } else if (cmd == "lock") {
            lockCurrentPose(gsw, cur);
            std::cout << "  [LOCK] Position locked at: J1=" << cur[0]
                      << " deg, J2=" << cur[1] << " deg, Grip=" << cur[2] << " deg\n";
        } else if (cmd == "grasp") {
            bool was_free = g_free_monitor_active.load();
            int load_thresh = DEFAULT_GRASP_LOAD;
            ss >> load_thresh;
            adaptiveGrasp(cur[2], load_thresh);
            if (was_free) startFreeMonitor();
        } else if (cmd == "release") {
            bool was_free = g_free_monitor_active.load();
            double open_deg = DEFAULT_RELEASE_DEG;
            ss >> open_deg;
            openGripperOnly(cur[2], open_deg);
            std::cout << "  [RELEASE] Gripper opened to +" << cur[2] << " deg.\n";
            if (was_free) startFreeMonitor();
        } else if (cmd == "add") {
            bool was_free = g_free_monitor_active.load();
            stopFreeMonitor();

            if (buffer.mode != MotionMode::WAYPOINT) {
                buffer.frames.clear();
                buffer.mode = MotionMode::WAYPOINT;
            }
            readCurrentPose(cur, false);

            std::string sub;
            if (ss >> sub) {
                if (sub == "grasp") {
                    int load_thresh = DEFAULT_GRASP_LOAD;
                    ss >> load_thresh;
                    setArmOnlyTorque(gsw, cur, true);
                    adaptiveGrasp(cur[2], load_thresh);
                    Frame f{cur[0], cur[1], cur[2], 0.0, FrameAction::ADAPTIVE_GRASP, load_thresh};
                    buffer.frames.push_back(f);
                    std::cout << "  [ADD GRASP] Waypoint #" << buffer.frames.size()
                              << " recorded: J1=" << cur[0] << " deg, J2=" << cur[1]
                              << " deg + Adaptive Grasp (load=" << load_thresh << ")\n";
                    if (was_free) {
                        setArmOnlyTorque(gsw, cur, false);
                        startFreeMonitor();
                    }
                    continue;
                } else if (sub == "release") {
                    double open_deg = DEFAULT_RELEASE_DEG;
                    ss >> open_deg;
                    open_deg = std::clamp(open_deg, 0.0, MAX_OPEN_DEG);
                    openGripperOnly(cur[2], open_deg);
                    Frame f{cur[0], cur[1], open_deg, 0.5, FrameAction::RELEASE, DEFAULT_GRASP_LOAD};
                    buffer.frames.push_back(f);
                    std::cout << "  [ADD RELEASE] Waypoint #" << buffer.frames.size()
                              << " recorded: J1=" << cur[0] << " deg, J2=" << cur[1]
                              << " deg, Release Grip=+" << open_deg << " deg\n";
                    if (was_free) startFreeMonitor();
                    continue;
                } else {
                    double dur = std::stod(sub);
                    buffer.frames.push_back(Frame{cur[0], cur[1], cur[2], dur});
                    std::cout << "  [ADD] Waypoint #" << buffer.frames.size() << " recorded: "
                              << "J1=" << cur[0] << " deg, J2=" << cur[1]
                              << " deg, Grip=" << cur[2] << " deg (duration=" << dur << "s)\n";
                    if (was_free) startFreeMonitor();
                    continue;
                }
            }
            buffer.frames.push_back(Frame{cur[0], cur[1], cur[2], 0.0});
            std::cout << "  [ADD] Waypoint #" << buffer.frames.size() << " recorded: "
                      << "J1=" << cur[0] << " deg, J2=" << cur[1] << " deg, Grip=" << cur[2] << " deg\n";
            if (was_free) startFreeMonitor();
        } else if (cmd == "record") {
            double secs = 20.0;
            ss >> secs;
            secs = std::clamp(secs, 1.0, 120.0);
            recordInteractiveStream(gsw, cur, buffer, secs);
        } else if (cmd == "move") {
            double j1, j2, gr;
            if (ss >> j1 >> j2 >> gr) {
                lockCurrentPose(gsw, cur);
                double target[3] = {
                    std::clamp(j1, -WORK_RANGE_DEG, WORK_RANGE_DEG),
                    std::clamp(j2, -WORK_RANGE_DEG, WORK_RANGE_DEG),
                    std::clamp(gr, 0.0, MAX_OPEN_DEG)
                };
                moveSmooth(gsw, cur, target, 0.0);
                std::cout << "  [MOVE] Current position: J1=" << cur[0]
                          << ", J2=" << cur[1] << ", Grip=" << cur[2] << "\n";
            } else {
                std::cout << "  [!] Format: move <j1_deg> <j2_deg> <grip_deg> (e.g.: move -30 -20 30)\n";
            }
        } else if (cmd == "zero") {
            lockCurrentPose(gsw, cur);
            double zero_pose[3] = {0.0, 0.0, 0.0};
            std::cout << "  -> Returning to calibrated zero position...\n";
            moveSmooth(gsw, cur, zero_pose, 0.0);
        } else if (cmd == "show") {
            bool was_free = g_free_monitor_active.load();
            stopFreeMonitor();
            readCurrentPose(cur, false);
            int signed_load = 0;
            int grip_load = g_gripper_online ? readLoadMagnitude(GRIPPER_ID, signed_load) : 0;
            std::cout << "  Current position: J1=" << cur[0] << " deg, J2=" << cur[1]
                      << " deg, Grip=" << cur[2] << " deg (Grip Load=" << grip_load << ")\n";
            std::cout << "  Memory buffer   : " << buffer.frames.size() << " points ("
                      << (buffer.mode == MotionMode::STREAM ? "STREAM 50Hz" : "WAYPOINT") << ")\n";
            if (buffer.mode == MotionMode::WAYPOINT) {
                for (size_t i = 0; i < buffer.frames.size(); i++) {
                    const auto &f = buffer.frames[i];
                    if (f.action == FrameAction::ADAPTIVE_GRASP) {
                        std::cout << "    #" << (i + 1) << " [ADAPTIVE GRASP]: J1=" << f.j1
                                  << ", J2=" << f.j2 << ", load_thresh=" << f.grasp_load << "\n";
                    } else if (f.action == FrameAction::RELEASE) {
                        std::cout << "    #" << (i + 1) << " [RELEASE]: J1=" << f.j1
                                  << ", J2=" << f.j2 << ", Grip=+" << f.grip << " deg\n";
                    } else {
                        std::cout << "    #" << (i + 1) << ": J1=" << f.j1
                                  << ", J2=" << f.j2 << ", Grip=" << f.grip
                                  << " (dur=" << f.duration << "s)\n";
                    }
                }
            }
            if (was_free) startFreeMonitor();
        } else if (cmd == "clear") {
            buffer.frames.clear();
            buffer.mode = MotionMode::WAYPOINT;
            std::cout << "  [CLEAR] Memory recording buffer cleared.\n";
        } else if (cmd == "save") {
            std::string name;
            if (ss >> name) {
                saveMotionToFile(name, buffer);
            } else {
                std::cout << "  [!] Format: save <motion_name> (e.g.: save pick_item)\n";
            }
        } else if (cmd == "play") {
            std::string name;
            int loops = 1;
            if (ss >> name) {
                if (!(ss >> loops) || loops < 1) loops = 1;
                MotionData loaded;
                if (loadMotionFromFile(name, loaded)) {
                    std::cout << "  [PLAY] Executing motion '" << name << "'...\n";
                    executeMotion(gsw, cur, loaded, loops);
                }
            } else {
                std::cout << "  [PLAY] Executing motion from memory...\n";
                executeMotion(gsw, cur, buffer, 1);
            }
        } else if (cmd == "list") {
            listSavedMotions();
        } else if (cmd == "help") {
            printHelp();
        } else if (cmd == "quit" || cmd == "exit" || cmd == "q") {
            break;
        } else {
            std::cout << "  [!] Unknown command. Type 'help' for command list.\n";
        }
    }

    stopFreeMonitor();
    setTorqueAll(false);
    portHandler->closePort();
    std::cout << "Torque disabled and port closed.\n";
    return 0;
}