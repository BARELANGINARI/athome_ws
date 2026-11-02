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
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

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

#define MAX_ANGLE               300.0
#define MAX_TICKS               1023
#define WORK_RANGE_DEG          260.0
#define UPDATE_RATE_HZ          50
#define MAX_SPEED_DPS           60.0
#define MIN_DURATION_S          0.4
#define COMPLIANCE_SLOPE        64
#define DEFAULT_TORQUE_LIM      1023

#define DEADZONE_LOW_TICK       15
#define DEADZONE_HIGH_TICK      1008
#define DEADZONE_MARGIN_TICKS   35
#define MAX_TICK_JUMP           120
#define HAPTIC_WALL_MARGIN      8

#define GRIPPER_ID              6
#define DEFAULT_GRASP_LOAD      170
#define GRASP_STEP_TICKS        4
#define GRASP_STEP_MS           40
#define GRASP_STALL_TICKS       15
#define GRASP_SQUEEZE_TICKS     20
#define GRASP_CONFIRM_COUNT     3
#define GRASP_COMPLIANCE_SLOPE  50
#define GRASP_CLOSING_SPEED     60
#define MAX_OPEN_DEG            80.0
#define DEFAULT_RELEASE_DEG     80.0

const uint8_t ARM_CORE_IDS[] = {4, 5, 1};
const uint8_t ALL_IDS[]      = {4, 5, 1, 6};

struct Servo { uint8_t id; bool inverted; };
const Servo JOINT1[] = { {4, false}, {5, true} };
const Servo JOINT2[] = { {1, false} };
const Servo JOINT3[] = { {6, false} };

enum class FrameAction {
    MOVE,
    ADAPTIVE_GRASP,
    RELEASE
};

struct Frame {
    double j1;
    double j2;
    double grip;
    double duration;
    FrameAction action = FrameAction::MOVE;
    int grasp_load     = DEFAULT_GRASP_LOAD;
};

enum class MotionMode { WAYPOINT, STREAM };

struct MotionData {
    MotionMode mode = MotionMode::WAYPOINT;
    std::vector<Frame> frames;
};

class ArmMotionServer : public rclcpp::Node {
public:
    ArmMotionServer() : Node("arm_motion_server") {
        this->declare_parameter<std::string>("port", "/dev/ttyUSB0");
        this->declare_parameter<int>("baudrate", 1000000);
        this->declare_parameter<std::string>("motions_dir", "");

        port_name_ = this->get_parameter("port").as_string();
        baudrate_ = this->get_parameter("baudrate").as_int();
        motions_dir_ = this->get_parameter("motions_dir").as_string();

        ambil_tas_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/ambil_tas_simple",
            std::bind(&ArmMotionServer::handleAmbilTas, this, std::placeholders::_1, std::placeholders::_2)
        );

        letakkan_tas_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/letakkan_tas",
            std::bind(&ArmMotionServer::handleLetakkanTas, this, std::placeholders::_1, std::placeholders::_2)
        );

        play_motion_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/play_motion",
            std::bind(&ArmMotionServer::handleAmbilTas, this, std::placeholders::_1, std::placeholders::_2)
        );

        RCLCPP_INFO(this->get_logger(), "Arm Motion Server ready.");
        RCLCPP_INFO(this->get_logger(), "Port: %s, Baudrate: %d", port_name_.c_str(), baudrate_);
        RCLCPP_INFO(this->get_logger(), "Services: /arm/ambil_tas_simple, /arm/letakkan_tas, /arm/play_motion");

        // Initialize hardware connection
        if (initHardware()) {
            RCLCPP_INFO(this->get_logger(), "[Arm Motion Server] Dynamixel hardware connected.");
        } else {
            RCLCPP_WARN(this->get_logger(), "[Arm Motion Server] Failed opening port %s. Will retry on service call.", port_name_.c_str());
        }
    }

    ~ArmMotionServer() override {
        cleanupHardware();
    }

private:
    std::string port_name_;
    int baudrate_;
    std::string motions_dir_;
    std::mutex hardware_mutex_;

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr ambil_tas_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr letakkan_tas_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr play_motion_srv_;

    dynamixel::PortHandler   *portHandler   = nullptr;
    dynamixel::PacketHandler *packetHandler = nullptr;
    bool is_connected_ = false;
    bool g_gripper_online = false;

    std::map<uint8_t, uint16_t> g_last_valid_raw;
    std::map<uint8_t, bool>     g_in_deadzone;
    std::map<uint8_t, bool>     g_servo_is_free;

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

    bool readRawPos(uint8_t id, uint16_t &pos) {
        if (!packetHandler || !portHandler) return false;
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

    uint16_t readFilteredRawPos(uint8_t id) {
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
                raw = static_cast<uint16_t>(near_high ? max_lim : min_lim);
            }
        }

        uint16_t clamped = std::clamp<uint16_t>(raw, min_lim, max_lim);
        g_last_valid_raw[id] = clamped;
        return clamped;
    }

    int readLoadMagnitude(uint8_t id, int &signed_load) {
        if (!packetHandler || !portHandler) return 0;
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

    double readServoRelAngle(uint8_t id, bool inverted) {
        uint16_t pos = readFilteredRawPos(id);
        return rawToRelAngle(id, pos, inverted);
    }

    void readCurrentPose(double pose[3]) {
        double a4 = readServoRelAngle(4, false);
        double a5 = readServoRelAngle(5, true);
        pose[0] = 0.5 * (a4 + a5);
        pose[1] = readServoRelAngle(1, false);
        if (g_gripper_online) {
            pose[2] = std::clamp(readServoRelAngle(GRIPPER_ID, false), 0.0, MAX_OPEN_DEG);
        } else {
            pose[2] = 0.0;
        }
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

    void lockCurrentPose(dynamixel::GroupSyncWrite &gsw, double cur[3]) {
        readCurrentPose(cur);
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

    bool adaptiveGrasp(double &cur_grip_deg, int load_threshold = DEFAULT_GRASP_LOAD) {
        uint8_t err = 0;
        uint16_t model = 0;
        if (!g_gripper_online) {
            if (packetHandler->ping(portHandler, GRIPPER_ID, &model, &err) == COMM_SUCCESS) {
                g_gripper_online = true;
            } else {
                RCLCPP_WARN(this->get_logger(), "[AdaptiveGrasp] Gripper ID 6 not connected!");
                return false;
            }
        }

        load_threshold = std::clamp(load_threshold, 50, 800);
        const auto *c = getCalib(GRIPPER_ID);
        const int closed_tick   = c ? c->center_tick : 510;
        const int max_open_tick = c ? c->ccw_limit    : 750;

        uint16_t start_pos = closed_tick;
        if (!readRawPos(GRIPPER_ID, start_pos)) {
            RCLCPP_WARN(this->get_logger(), "[AdaptiveGrasp] Failed reading initial position of ID 6.");
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
            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, open_tick, &err);
            std::this_thread::sleep_for(std::chrono::milliseconds(650));
            readRawPos(GRIPPER_ID, start_pos);
        }

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

            bool trig_load  = (load_mag >= load_threshold);
            bool trig_stall = (lag >= GRASP_STALL_TICKS);

            if (step > 2 && (trig_load || trig_stall)) {
                contact_count++;
            } else {
                contact_count = 0;
            }

            if (contact_count >= GRASP_CONFIRM_COUNT) {
                int hold_tick = std::clamp(static_cast<int>(present_pos) - GRASP_SQUEEZE_TICKS,
                                           closed_tick, max_open_tick);
                int hold_torque = std::clamp(load_mag + 200, 360, 450);

                packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT,
                                              static_cast<uint16_t>(hold_torque), &err);
                packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION,
                                              static_cast<uint16_t>(hold_tick), &err);

                cur_grip_deg = rawToRelAngle(GRIPPER_ID, static_cast<uint16_t>(hold_tick), false);
                g_last_valid_raw[GRIPPER_ID] = static_cast<uint16_t>(hold_tick);
                RCLCPP_INFO(this->get_logger(), "[AdaptiveGrasp] Object grasped! Position: %.1f deg, Load: %d", cur_grip_deg, load_mag);
                return true;
            }
        }

        packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
        cur_grip_deg = 0.0;
        g_last_valid_raw[GRIPPER_ID] = static_cast<uint16_t>(closed_tick);
        RCLCPP_INFO(this->get_logger(), "[AdaptiveGrasp] Gripper fully closed without load detection.");
        return false;
    }

    void releaseGripper(dynamixel::GroupSyncWrite &gsw, double cur[3], double open_deg = DEFAULT_RELEASE_DEG) {
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
        RCLCPP_INFO(this->get_logger(), "[ReleaseGripper] Opening gripper to +%.1f deg...", open_deg);
        moveSmooth(gsw, cur, target, 0.5);
    }

    void executeMotion(dynamixel::GroupSyncWrite &gsw, double cur[3], const MotionData &motion, int loops) {
        if (motion.frames.empty()) {
            RCLCPP_WARN(this->get_logger(), "[ExecuteMotion] Motion data is empty.");
            return;
        }

        lockCurrentPose(gsw, cur);

        for (int l = 1; l <= loops; l++) {
            bool holding_object = false;

            if (motion.mode == MotionMode::WAYPOINT) {
                for (size_t i = 0; i < motion.frames.size(); i++) {
                    const auto &f = motion.frames[i];
                    if (f.action == FrameAction::ADAPTIVE_GRASP) {
                        double arm_target[3] = { f.j1, f.j2, cur[2] };
                        moveSmooth(gsw, cur, arm_target, f.duration);
                        holding_object = adaptiveGrasp(cur[2], f.grasp_load);
                    } else if (f.action == FrameAction::RELEASE) {
                        double arm_target[3] = { f.j1, f.j2, cur[2] };
                        moveSmooth(gsw, cur, arm_target, f.duration);
                        releaseGripper(gsw, cur, f.grip);
                        holding_object = false;
                    } else {
                        double grip_target = holding_object ? cur[2] : f.grip;
                        double target[3] = { f.j1, f.j2, grip_target };
                        moveSmooth(gsw, cur, target, f.duration);
                    }
                }
            } else {
                // STREAM Mode
                const auto &first = motion.frames.front();
                double start_target[3] = { first.j1, first.j2, first.grip };
                RCLCPP_INFO(this->get_logger(), "[ExecuteMotion] Moving to recorded start pose...");
                moveSmooth(gsw, cur, start_target, 0.0);

                RCLCPP_INFO(this->get_logger(), "[ExecuteMotion] Replaying stream (%zu samples @ %d Hz)...",
                            motion.frames.size(), UPDATE_RATE_HZ);
                auto next = std::chrono::steady_clock::now();
                const auto period = std::chrono::microseconds(1000000 / UPDATE_RATE_HZ);

                for (const auto &f : motion.frames) {
                    if (f.action == FrameAction::ADAPTIVE_GRASP) {
                        double arm_hold[3] = { f.j1, f.j2, cur[2] };
                        sendJoints(gsw, arm_hold);
                        cur[0] = f.j1; cur[1] = f.j2;
                        holding_object = adaptiveGrasp(cur[2], f.grasp_load);
                        next = std::chrono::steady_clock::now();
                    } else if (f.action == FrameAction::RELEASE) {
                        double arm_hold[3] = { f.j1, f.j2, cur[2] };
                        sendJoints(gsw, arm_hold);
                        cur[0] = f.j1; cur[1] = f.j2;
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
        RCLCPP_INFO(this->get_logger(), "[ExecuteMotion] Motion execution completed.");
    }

    std::string resolveMotionPath(const std::string &name) {
        std::vector<std::string> candidate_dirs;
        if (!motions_dir_.empty()) {
            candidate_dirs.push_back(motions_dir_);
        }
        try {
            std::string share_dir = ament_index_cpp::get_package_share_directory("athome_arm");
            candidate_dirs.push_back(share_dir + "/motions");
        } catch (...) {}

        candidate_dirs.push_back("src/athome_arm/motions");
        candidate_dirs.push_back("motions");
        candidate_dirs.push_back(".");

        for (const auto &dir : candidate_dirs) {
            std::vector<std::string> filenames = {
                dir + "/" + name,
                dir + "/" + name + ".motion",
                dir + "/" + name + ".csv"
            };
            for (const auto &fpath : filenames) {
                if (fs::exists(fpath)) {
                    return fpath;
                }
            }
        }
        return "";
    }

    bool loadMotionFromFile(const std::string &name, MotionData &motion) {
        std::string path = resolveMotionPath(name);
        if (path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "[LoadMotion] File '%s' not found in candidate motion dirs.", name.c_str());
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "[LoadMotion] Loading motion file from: %s", path.c_str());
        std::ifstream in(path);
        if (!in) {
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

    bool initHardware() {
        if (is_connected_) return true;

        portHandler = dynamixel::PortHandler::getPortHandler(port_name_.c_str());
        packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

        if (!portHandler->openPort()) {
            return false;
        }
        if (!portHandler->setBaudRate(baudrate_)) {
            portHandler->closePort();
            return false;
        }

        uint8_t err = 0;
        for (uint8_t id : ARM_CORE_IDS) {
            uint16_t model = 0;
            if (packetHandler->ping(portHandler, id, &model, &err) != COMM_SUCCESS) {
                RCLCPP_WARN(this->get_logger(), "Servo ID %d did not respond to ping.", id);
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
            RCLCPP_INFO(this->get_logger(), "Gripper ID 6 detected.");
        } else {
            g_gripper_online = false;
        }

        resetDeadzoneFilter();
        is_connected_ = true;
        return true;
    }

    void cleanupHardware() {
        if (!is_connected_) return;
        uint8_t err = 0;
        for (uint8_t id : ALL_IDS) {
            if (packetHandler && portHandler) {
                packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, 0, &err);
            }
        }
        if (portHandler) {
            portHandler->closePort();
        }
        is_connected_ = false;
    }

    bool runMotionByName(const std::string &name, std::string &error_msg) {
        if (!initHardware()) {
            error_msg = "Failed to connect to Dynamixel hardware on port " + port_name_;
            return false;
        }

        MotionData motion;
        if (!loadMotionFromFile(name, motion)) {
            error_msg = "Failed to load motion file '" + name + "'";
            return false;
        }

        try {
            double cur[3] = {0.0, 0.0, 0.0};
            readCurrentPose(cur);
            dynamixel::GroupSyncWrite gsw(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION);

            executeMotion(gsw, cur, motion, 1);
            return true;
        } catch (const std::exception &e) {
            error_msg = std::string("Motion execution error: ") + e.what();
            return false;
        }
    }

    void handleAmbilTas(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        std::lock_guard<std::mutex> lock(hardware_mutex_);

        RCLCPP_INFO(this->get_logger(), "[Service Call] Received trigger to execute motion 'ambil_tas'...");
        std::string err;
        if (runMotionByName("ambil_tas_simple", err)) {
            response->success = true;
            response->message = "Motion ambil_tas executed successfully.";
            RCLCPP_INFO(this->get_logger(), "[Service Call] %s", response->message.c_str());
        } else {
            response->success = false;
            response->message = err;
            RCLCPP_ERROR(this->get_logger(), "%s", err.c_str());
        }
    }

    void handleLetakkanTas(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        std::lock_guard<std::mutex> lock(hardware_mutex_);

        RCLCPP_INFO(this->get_logger(), "[Service Call] Received trigger to execute motion 'letakkan_tas'...");
        std::string err;
        if (runMotionByName("letakkan_tas", err)) {
            response->success = true;
            response->message = "Motion letakkan_tas executed successfully.";
            RCLCPP_INFO(this->get_logger(), "[Service Call] %s", response->message.c_str());
        } else {
            response->success = false;
            response->message = err;
            RCLCPP_ERROR(this->get_logger(), "%s", err.c_str());
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArmMotionServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

