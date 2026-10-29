#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <dynamixel_sdk.h>
#include "calibration_data.h"

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

#define PROTOCOL_VERSION        1.0
#define BAUDRATE                1000000
#define DEVICENAME              "/dev/ttyUSB0"

#define GRIPPER_ID              6
#define MAX_ANGLE               300.0
#define MAX_TICKS               1023
#define MAX_OPEN_DEG            80.0    // Full open angle (~780 ticks, covers physical opening of 753 ticks)
#define DEFAULT_RELEASE_DEG     65.0    // Default open angle

// Default Adaptive Grasping Parameters
struct GraspConfig {
    int    load_threshold   = 180;   // Object detection load threshold (0-1023, 180 ≈ 17.6%)
    int    step_ticks       = 4;     // Closing step per iteration (~1.17 deg)
    int    step_ms          = 40;    // Sampling interval (40 ms = 25 Hz)
    int    stall_ticks      = 10;    // Difference (Pos - Goal) indicating physical obstruction
    int    squeeze_ticks    = 5;     // Additional squeeze (+ticks toward closing) after contact
    int    confirm_cycles   = 3;     // Consecutive cycles to validate contact
    int    closing_speed    = 60;    // Closing speed (1-1023, 60 ≈ 6.8 RPM)
    uint8_t compliance_slope = 64;   // PID compliance slope
};

struct GripperTelemetry {
    uint16_t goal_pos     = 0;
    uint16_t present_pos  = 0;
    uint16_t torque_limit = 0;
    int      signed_load  = 0;   // Negative = CW (close), Positive = CCW (open)
    int      load_mag     = 0;   // 0 - 1023
    int      speed_signed = 0;
    double   voltage      = 0.0;
    uint8_t  temperature  = 0;
    uint8_t  moving       = 0;
    uint8_t  hw_error     = 0;
    bool     comm_ok      = false;
};

dynamixel::PortHandler   *portHandler   = nullptr;
dynamixel::PacketHandler *packetHandler = nullptr;
GraspConfig g_cfg;
bool g_is_holding = false;
uint16_t g_hold_tick = 510;

// ---------------- CALIBRATION & CONVERSION ----------------
const CalibratedServoConfig *getGripperCalib() {
    for (const auto &c : ARM_CALIBRATION) {
        if (c.id == GRIPPER_ID) return &c;
    }
    return nullptr;
}

int getCenterTick() {
    const auto *c = getGripperCalib();
    return c ? c->center_tick : 510;
}

int getCwLimit() {
    const auto *c = getGripperCalib();
    return c ? c->cw_limit : 490;
}

int getCcwLimit() {
    const auto *c = getGripperCalib();
    return c ? c->ccw_limit : 800;
}

double ticksToDeg(int ticks) {
    return ticks * (MAX_ANGLE / MAX_TICKS);
}

int degToTicks(double deg) {
    return static_cast<int>(std::round(deg / MAX_ANGLE * MAX_TICKS));
}

// 0.0 deg = fully closed (510), +55.0 deg = fully open (~700)
double rawToRelAngle(uint16_t raw_pos) {
    return ticksToDeg(static_cast<int>(raw_pos) - getCenterTick());
}

uint16_t relAngleToTick(double rel_deg) {
    rel_deg = std::clamp(rel_deg, 0.0, MAX_OPEN_DEG);
    int goal = getCenterTick() + degToTicks(rel_deg);
    return static_cast<uint16_t>(std::clamp(goal, getCwLimit(), getCcwLimit()));
}

// ---------------- TELEMETRY READING & DEBUG ----------------
std::string decodeHwError(uint8_t err) {
    if (err == 0) return "OK";
    std::string s;
    if (err & (1 << 0)) s += "[InputVoltage] ";
    if (err & (1 << 1)) s += "[AngleLimit] ";
    if (err & (1 << 2)) s += "[Overheating] ";
    if (err & (1 << 3)) s += "[Range] ";
    if (err & (1 << 4)) s += "[Checksum] ";
    if (err & (1 << 5)) s += "[Overload!] ";
    if (err & (1 << 6)) s += "[Instruction] ";
    return s;
}

GripperTelemetry readTelemetry() {
    GripperTelemetry t{};
    uint8_t err = 0;
    uint16_t raw_load = 0, raw_speed = 0;
    uint8_t raw_volt = 0;

    if (packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_POS, &t.present_pos, &err) != COMM_SUCCESS) {
        t.comm_ok = false;
        return t;
    }
    t.hw_error |= err;
    t.comm_ok = true;

    packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, &t.goal_pos, &err);
    t.hw_error |= err;
    packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT, &t.torque_limit, &err);
    t.hw_error |= err;
    packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_SPEED, &raw_speed, &err);
    t.hw_error |= err;
    packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_LOAD, &raw_load, &err);
    t.hw_error |= err;
    packetHandler->read1ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_VOLT, &raw_volt, &err);
    t.hw_error |= err;
    packetHandler->read1ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_TEMP, &t.temperature, &err);
    t.hw_error |= err;
    packetHandler->read1ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING, &t.moving, &err);
    t.hw_error |= err;

    t.load_mag = raw_load & 0x3FF;
    bool load_cw = (raw_load & 0x400) != 0;
    t.signed_load = load_cw ? -t.load_mag : t.load_mag;

    int spd_mag = raw_speed & 0x3FF;
    bool spd_cw = (raw_speed & 0x400) != 0;
    t.speed_signed = spd_cw ? -spd_mag : spd_mag;

    t.voltage = raw_volt / 10.0;
    return t;
}

std::string makeLoadBar(int load_mag, int threshold) {
    const int width = 20;
    int filled = std::clamp(load_mag * width / 512, 0, width);
    int thresh_pos = std::clamp(threshold * width / 512, 0, width - 1);
    std::string bar(width, '.');
    for (int i = 0; i < filled; i++) bar[i] = '#';
    bar[thresh_pos] = '|'; // grasp threshold marker
    return "[" + bar + "]";
}

void printDetailedStatus() {
    GripperTelemetry t = readTelemetry();
    if (!t.comm_ok) {
        std::cout << "  [ERROR] Failed to read telemetry from Servo ID " << GRIPPER_ID << "!\n";
        return;
    }

    int lag = static_cast<int>(t.present_pos) - static_cast<int>(t.goal_pos);
    double load_pct = (t.load_mag / 1023.0) * 100.0;

    std::cout << "\n---------------- GRIPPER TELEMETRY DEBUG (ID 6) ----------------\n"
              << "  Grip Status    : " << (g_is_holding ? "HOLDING OBJECT" : "IDLE / FREE") << "\n"
              << "  Present Pos    : " << t.present_pos << " ticks ("
              << std::fixed << std::setprecision(2) << rawToRelAngle(t.present_pos)
              << " deg from closed 0 deg)\n"
              << "  Goal Position  : " << t.goal_pos << " ticks ("
              << rawToRelAngle(t.goal_pos) << " deg)\n"
              << "  Position Lag   : " << std::showpos << lag << " ticks ("
              << ticksToDeg(lag) << " deg)" << std::noshowpos << "\n"
              << "  Present Load   : " << t.load_mag << " / 1023 (" << std::setprecision(1) << load_pct
              << "%, dir=" << (t.signed_load < 0 ? "CW/Close" : (t.signed_load > 0 ? "CCW/Open" : "Neutral"))
              << ") " << makeLoadBar(t.load_mag, g_cfg.load_threshold) << "\n"
              << "  Torque Limit   : " << t.torque_limit << " / 1023 ("
              << (t.torque_limit / 1023.0 * 100.0) << "%)\n"
              << "  Voltage & Temp : " << t.voltage << " V | " << (int)t.temperature << " C\n"
              << "  Hardware Alarm : " << decodeHwError(t.hw_error) << "\n"
              << "----------------------------------------------------------------\n";
}

// ---------------- LIVE DEBUG MONITOR ----------------
void runLiveMonitor(double duration_sec) {
    duration_sec = std::clamp(duration_sec, 1.0, 120.0);
    int steps = static_cast<int>(std::ceil(duration_sec * 10.0)); // 10 Hz

    std::cout << "\n================ LIVE DEBUG MONITOR ID 6 (" << duration_sec << "s @ 10Hz) ================\n";
    std::cout << std::left
              << std::setw(7)  << "Time"
              << std::setw(6)  << "Goal"
              << std::setw(6)  << "Pos"
              << std::setw(9)  << "Angle"
              << std::setw(6)  << "Lag"
              << std::setw(12) << "Load(Raw/%)"
              << std::setw(24) << "Load Bar (| = Thresh)"
              << std::setw(7)  << "Volt"
              << std::setw(6)  << "Temp"
              << "Status / Event\n";
    std::cout << "---------------------------------------------------------------------------------------------\n";

    for (int i = 0; i < steps; i++) {
        GripperTelemetry t = readTelemetry();
        if (!t.comm_ok) {
            std::cout << "  [COMM ERROR]\n";
            break;
        }

        int lag = static_cast<int>(t.present_pos) - static_cast<int>(t.goal_pos);
        double load_pct = (t.load_mag / 1023.0) * 100.0;
        std::string event = decodeHwError(t.hw_error);

        if (g_is_holding) {
            if (t.load_mag < 35 && std::abs(static_cast<int>(t.present_pos) - getCenterTick()) <= 5) {
                event = "[SLIP / OBJECT DROPPED!]";
                g_is_holding = false;
            } else if (t.load_mag >= g_cfg.load_threshold) {
                event = "HOLDING (SOLID)";
            } else {
                event = "HOLDING (LIGHT)";
            }
        }

        std::ostringstream load_str;
        load_str << std::setw(3) << t.load_mag << " (" << std::fixed << std::setprecision(1) << load_pct << "%)";

        std::cout << std::left
                  << std::setw(7)  << (std::to_string(i / 10) + "." + std::to_string(i % 10) + "s")
                  << std::setw(6)  << t.goal_pos
                  << std::setw(6)  << t.present_pos
                  << std::setw(9)  << std::fixed << std::setprecision(1) << rawToRelAngle(t.present_pos)
                  << std::setw(6)  << std::showpos << lag << std::noshowpos
                  << std::setw(12) << load_str.str()
                  << std::setw(24) << makeLoadBar(t.load_mag, g_cfg.load_threshold)
                  << std::setw(7)  << std::setprecision(1) << t.voltage
                  << std::setw(6)  << (std::to_string(t.temperature) + "C")
                  << event << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "=============================================================================================\n";
}

// ---------------- ADAPTIVE GRASPING ALGORITHM ----------------
// Closes jaws from open position (+deg, ~700 tick) towards 0.0 deg (510 tick)
bool executeAdaptiveGrasp(int custom_threshold = -1) {
    int thresh = (custom_threshold > 0) ? std::clamp(custom_threshold, 50, 800) : g_cfg.load_threshold;
    uint8_t err = 0;

    const int closed_tick = getCenterTick(); // 510
    const int max_open_tick = relAngleToTick(MAX_OPEN_DEG); // ~700

    GripperTelemetry init_t = readTelemetry();
    if (!init_t.comm_ok) {
        std::cout << "  [ERROR] Servo ID 6 not responding!\n";
        return false;
    }

    // Turn on servo LED during grasping
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 1, &err);

    // Set compliance slope & search torque
    int search_torque = std::clamp(thresh + 220, 320, 750);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_SLOPE,  g_cfg.compliance_slope, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_SLOPE, g_cfg.compliance_slope, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING_SPEED, g_cfg.closing_speed, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT, search_torque, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 1, &err);

    uint16_t start_pos = init_t.present_pos;
    // If jaws are tightly closed (< 10 ticks from closed_tick 510), pre-open to +40 deg
    if (static_cast<int>(start_pos) <= closed_tick + 10) {
        uint16_t open_tick = relAngleToTick(40.0);
        std::cout << "  [PRE-OPEN] Jaws closed, opening to +40.0 deg (tick=" << open_tick << ")...\n";
        packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, open_tick, &err);
        std::this_thread::sleep_for(std::chrono::milliseconds(650));
        start_pos = readTelemetry().present_pos;
    }

    std::cout << "\n================ ADAPTIVE GRASPING PROCESS (ID 6) ================\n";
    std::cout << "  Closing from tick " << start_pos << " (" << std::fixed << std::setprecision(1) << rawToRelAngle(start_pos)
              << " deg) down to tick " << closed_tick << " (0.0 deg)\n"
              << "  Load Threshold: " << thresh << " | Stall Limit: " << g_cfg.stall_ticks << " ticks\n";
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

    // Closing step: goal_tick moves down towards closed_tick (510)
    while (goal_tick > closed_tick) {
        step++;
        goal_tick = std::max(closed_tick, goal_tick - g_cfg.step_ticks);
        packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION,
                                      static_cast<uint16_t>(goal_tick), &err);
        std::this_thread::sleep_for(std::chrono::milliseconds(g_cfg.step_ms));

        GripperTelemetry t = readTelemetry();
        // When closing, if blocked by object above goal_tick, present_pos > goal_tick
        int lag = static_cast<int>(t.present_pos) - goal_tick;
        double load_pct = (t.load_mag / 1023.0) * 100.0;

        bool trig_load  = (t.load_mag >= thresh);
        bool trig_stall = (lag >= g_cfg.stall_ticks);
        std::string state = "SEARCHING";

        // Ignore first 2 steps (motor inrush current)
        if (step > 2 && (trig_load || trig_stall)) {
            contact_count++;
            state = "CONTACT (" + std::to_string(contact_count) + "/" + std::to_string(g_cfg.confirm_cycles) + ")";
        } else {
            contact_count = 0;
        }

        std::ostringstream load_str;
        load_str << std::setw(3) << t.load_mag << " (" << std::fixed << std::setprecision(1) << load_pct << "%)";

        std::cout << std::left
                  << std::setw(6)  << step
                  << std::setw(6)  << goal_tick
                  << std::setw(6)  << t.present_pos
                  << std::setw(9)  << std::fixed << std::setprecision(1) << rawToRelAngle(t.present_pos)
                  << std::setw(6)  << std::showpos << lag << std::noshowpos
                  << std::setw(12) << load_str.str()
                  << std::setw(24) << makeLoadBar(t.load_mag, thresh)
                  << state << "\n";

        if (contact_count >= g_cfg.confirm_cycles) {
            // Lock target at contact position minus squeeze_ticks (slight squeeze)
            int hold_tick = std::clamp(static_cast<int>(t.present_pos) - g_cfg.squeeze_ticks,
                                       closed_tick, max_open_tick);
            int hold_torque = std::clamp(t.load_mag + 160, 280, 650);

            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT,
                                          static_cast<uint16_t>(hold_torque), &err);
            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION,
                                          static_cast<uint16_t>(hold_tick), &err);

            g_is_holding = true;
            g_hold_tick  = static_cast<uint16_t>(hold_tick);

            std::cout << "-----------------------------------------------------------------------------\n";
            std::cout << "  [HOLDING OBJECT] Object grasped successfully!\n"
                      << "    -> Object Position : tick " << t.present_pos << " ("
                      << std::fixed << std::setprecision(1) << rawToRelAngle(t.present_pos) << " deg)\n"
                      << "    -> Lock Target     : tick " << hold_tick << " (squeeze -" << g_cfg.squeeze_ticks << " ticks)\n"
                      << "    -> Contact Load    : " << t.load_mag << " (" << load_pct << "%)\n"
                      << "    -> Holding Torque  : " << hold_torque << " / 1023 ("
                      << (hold_torque / 1023.0 * 100.0) << "%)\n";
            std::cout << "=============================================================================\n";
            return true;
        }
    }

    g_is_holding = false;
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
    std::cout << "-----------------------------------------------------------------------------\n";
    std::cout << "  [CLOSED EMPTY] Jaws reached 0.0 deg (tick=" << closed_tick
              << ") without detecting object.\n";
    std::cout << "=============================================================================\n";
    return false;
}

void moveGripperDeg(double deg, int speed = 80) {
    uint8_t err = 0;
    uint16_t goal = relAngleToTick(deg);
    g_is_holding = false;

    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_SLOPE,  g_cfg.compliance_slope, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_SLOPE, g_cfg.compliance_slope, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING_SPEED, speed, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT, 600, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 1, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, goal, &err);

    std::cout << "  [MOVE] Gripper ID 6 moving to " << std::fixed << std::setprecision(1)
              << std::clamp(deg, 0.0, MAX_OPEN_DEG) << " deg (tick=" << goal << ")\n";
}

void printMenu() {
    std::cout << "\n================ ADAPTIVE GRIPPER & DEBUG MONITOR (ID 6) ================\n"
              << "  Control Commands:\n"
              << "    grasp [load_thresh]   -> Grasp object adaptively (default load=" << g_cfg.load_threshold << ")\n"
              << "    release [deg]         -> Open gripper jaws (default +45.0 deg, e.g.: release or release 50)\n"
              << "    move <deg>            -> Move to angle (0.0=fully closed to +55.0=fully open)\n"
              << "    free                  -> Disable ID 6 torque\n"
              << "    lock                  -> Lock ID 6 at current position\n\n"
              << "  Debug & Monitoring Commands:\n"
              << "    status                -> Show complete ID 6 telemetry report\n"
              << "    monitor [sec]         -> Live debug stream 10 Hz for [sec] (default 10s)\n"
              << "    config <load> <squeeze> <speed> <slope>\n"
              << "                          -> Update adaptive grasp params (current: load=" << g_cfg.load_threshold
              << ", squeeze=" << g_cfg.squeeze_ticks << ", speed=" << g_cfg.closing_speed
              << ", slope=" << (int)g_cfg.compliance_slope << ")\n"
              << "    help                  -> Show this menu\n"
              << "    quit                  -> Disable torque & exit\n"
              << "=========================================================================\n";
}

// ---------------- MAIN ----------------
int main() {
    portHandler   = dynamixel::PortHandler::getPortHandler(DEVICENAME);
    packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    if (!portHandler->openPort() || !portHandler->setBaudRate(BAUDRATE)) {
        std::cerr << "[ERROR] Failed to open port " << DEVICENAME << " @ " << BAUDRATE << " bps!\n";
        return -1;
    }

    uint16_t model = 0;
    uint8_t err = 0;
    if (packetHandler->ping(portHandler, GRIPPER_ID, &model, &err) != COMM_SUCCESS) {
        std::cerr << "[ERROR] Gripper Servo ID " << GRIPPER_ID << " not detected!\n";
        portHandler->closePort();
        return -1;
    }

    // Apply safe CW/CCW limits from calibration_data.h (490 - 750)
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 0, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_ANGLE_LIMIT,  getCwLimit(),  &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_ANGLE_LIMIT, getCcwLimit(), &err);

    std::cout << "[OK] Gripper ID 6 detected (Model: " << model
              << ", Center=" << getCenterTick()
              << ", CW_Lim=" << getCwLimit()
              << ", CCW_Lim=" << getCcwLimit() << ")\n";

    printDetailedStatus();
    printMenu();

    while (true) {
        std::cout << "\nGripper(ID6)> ";
        std::string line;
        if (!std::getline(std::cin, line)) break;

        std::stringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd.empty()) continue;

        if (cmd == "grasp") {
            int thresh = -1;
            ss >> thresh;
            executeAdaptiveGrasp(thresh);
        } else if (cmd == "release") {
            double deg = DEFAULT_RELEASE_DEG;
            ss >> deg;
            moveGripperDeg(deg);
        } else if (cmd == "move") {
            double deg = 0.0;
            if (ss >> deg) {
                moveGripperDeg(deg);
            } else {
                std::cout << "  [!] Format: move <deg> (range 0.0 to +55.0 deg)\n";
            }
        } else if (cmd == "free") {
            g_is_holding = false;
            packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
            packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 0, &err);
            std::cout << "  [FREE] ID 6 torque disabled.\n";
        } else if (cmd == "lock") {
            GripperTelemetry t = readTelemetry();
            if (t.comm_ok) {
                packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, t.present_pos, &err);
                packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 1, &err);
                std::cout << "  [LOCK] ID 6 locked at tick " << t.present_pos
                          << " (" << rawToRelAngle(t.present_pos) << " deg)\n";
            }
        } else if (cmd == "status") {
            printDetailedStatus();
        } else if (cmd == "monitor") {
            double secs = 10.0;
            ss >> secs;
            runLiveMonitor(secs);
        } else if (cmd == "config") {
            int load, squeeze, speed, slope;
            if (ss >> load >> squeeze >> speed >> slope) {
                g_cfg.load_threshold   = std::clamp(load, 50, 800);
                g_cfg.squeeze_ticks    = std::clamp(squeeze, 0, 30);
                g_cfg.closing_speed    = std::clamp(speed, 10, 400);
                g_cfg.compliance_slope = static_cast<uint8_t>(std::clamp(slope, 8, 254));
                std::cout << "  [CONFIG] Updated: load=" << g_cfg.load_threshold
                          << ", squeeze=" << g_cfg.squeeze_ticks
                          << ", speed=" << g_cfg.closing_speed
                          << ", slope=" << (int)g_cfg.compliance_slope << "\n";
            } else {
                std::cout << "  [!] Format: config <load_thresh> <squeeze_ticks> <speed> <slope>\n"
                          << "      Example: config 180 5 60 64\n";
            }
        } else if (cmd == "help") {
            printMenu();
        } else if (cmd == "quit" || cmd == "exit" || cmd == "q") {
            break;
        } else {
            std::cout << "  [!] Unknown command. Type 'help' for assistance.\n";
        }
    }

    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 0, &err);
    portHandler->closePort();
    std::cout << "ID 6 torque disabled and port closed.\n";
    return 0;
}
