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
#include <std_msgs/msg/string.hpp>
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

// constructor ArmMotionServer
class ArmMotionServer : public rclcpp::Node {
public:
    ArmMotionServer() : Node("arm_motion_server") {
        this->declare_parameter<std::string>("task", "all");
        task_mode_ = this->get_parameter("task").as_string();
        if (task_mode_ == "all") {
            const char * env_task = std::getenv("ATHOME_TASK");
            if (env_task) task_mode_ = env_task;
        }
        std::transform(task_mode_.begin(), task_mode_.end(), task_mode_.begin(), ::tolower);

        this->declare_parameter<std::string>("port", "/dev/ttyUSB0");
        this->declare_parameter<int>("baudrate", 1000000);
        this->declare_parameter<std::string>("motions_dir", "");

        port_name_ = this->get_parameter("port").as_string();
        baudrate_ = this->get_parameter("baudrate").as_int();
        motions_dir_ = this->get_parameter("motions_dir").as_string();

        this->declare_parameter<bool>("ik.enabled", true);
        this->declare_parameter<double>("ik.l1_m", 0.21);
        this->declare_parameter<double>("ik.l2_m", 0.075);
        this->declare_parameter<double>("ik.shoulder_x_m", 0.13);
        this->declare_parameter<double>("ik.shoulder_z_m", 0.61);
        this->declare_parameter<double>("ik.j1_zero_offset_deg", 90.0);
        this->declare_parameter<double>("ik.j2_zero_offset_deg", 0.0);
        this->declare_parameter<double>("ik.j1_sign", -1.0);
        this->declare_parameter<double>("ik.j2_sign", 1.0);
        this->declare_parameter<bool>("ik.elbow_up", true);
        this->declare_parameter<bool>("ik.clamp_to_workspace", true);
        this->declare_parameter<double>("ik.max_reach_margin_m", 0.08);
        this->declare_parameter<double>("ik.pregrasp_dx_m", -0.04);
        this->declare_parameter<double>("ik.pregrasp_dz_m", 0.02);
        this->declare_parameter<double>("ik.carry_j1_deg", 0.0);
        this->declare_parameter<double>("ik.carry_j2_deg", 0.0);
        this->declare_parameter<double>("ik.target_timeout_sec", 5.0);
        this->declare_parameter<std::string>("ik.fallback_motion", "ambil_tas_simple");

        ik_enabled_            = this->get_parameter("ik.enabled").as_bool();
        ik_l1_m_               = this->get_parameter("ik.l1_m").as_double();
        ik_l2_m_               = this->get_parameter("ik.l2_m").as_double();
        ik_shoulder_x_m_       = this->get_parameter("ik.shoulder_x_m").as_double();
        ik_shoulder_z_m_       = this->get_parameter("ik.shoulder_z_m").as_double();
        ik_j1_zero_offset_deg_ = this->get_parameter("ik.j1_zero_offset_deg").as_double();
        ik_j2_zero_offset_deg_ = this->get_parameter("ik.j2_zero_offset_deg").as_double();
        ik_j1_sign_            = this->get_parameter("ik.j1_sign").as_double();
        ik_j2_sign_            = this->get_parameter("ik.j2_sign").as_double();
        ik_elbow_up_           = this->get_parameter("ik.elbow_up").as_bool();
        ik_clamp_workspace_    = this->get_parameter("ik.clamp_to_workspace").as_bool();
        ik_max_reach_margin_m_ = this->get_parameter("ik.max_reach_margin_m").as_double();
        ik_pregrasp_dx_m_      = this->get_parameter("ik.pregrasp_dx_m").as_double();
        ik_pregrasp_dz_m_      = this->get_parameter("ik.pregrasp_dz_m").as_double();
        ik_carry_j1_deg_       = this->get_parameter("ik.carry_j1_deg").as_double();
        ik_carry_j2_deg_       = this->get_parameter("ik.carry_j2_deg").as_double();
        ik_target_timeout_sec_ = this->get_parameter("ik.target_timeout_sec").as_double();
        ik_fallback_motion_    = this->get_parameter("ik.fallback_motion").as_string();

        target_point_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/arm/target_point", 10,
            std::bind(&ArmMotionServer::onTargetPoint, this, std::placeholders::_1)
        );

        // Menambahkan service untuk motion arm

        ambil_tas_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/motion_task_1",
            std::bind(&ArmMotionServer::handleAmbilTas, this, std::placeholders::_1, std::placeholders::_2)
        );

        ambil_tas_simple_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/ambil_tas_simple",
            std::bind(&ArmMotionServer::handleAmbilTasSimple, this, std::placeholders::_1, std::placeholders::_2)
        );

        ambil_order_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/ambil_order",
            std::bind(&ArmMotionServer::handleAmbilOrder, this, std::placeholders::_1, std::placeholders::_2)
        );

        release_gripper_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/release_gripper",
            std::bind(&ArmMotionServer::handleReleaseGripper, this, std::placeholders::_1, std::placeholders::_2)
        );

        letakkan_tas_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/letakkan_tas",
            std::bind(&ArmMotionServer::handleLetakkanTas, this, std::placeholders::_1, std::placeholders::_2)
        );

        point_pos_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/point_position",
            [this] (const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
                std::shared_ptr<std_srvs::srv::Trigger::Response>res) {
                std::lock_guard<std::mutex> lock(hardware_mutex_);
                std::string err;
                if (runMotionByName("point_position", err)) {
                    res->success = true;
                    res->message = "Motion point_position executed successfully";
                } else {
                    res->success = false;
                    res->message = err;
                }
            }
        );

        point_to_chair_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/point_to_chair",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                std::lock_guard<std::mutex> lock(hardware_mutex_);
                std::string err;
                if (runMotionByName("point_to_chair", err)) {
                    res->success = true;
                    res->message = "Motion point_to_chair executed successfully";
                } else {
                    res->success = false;
                    res->message = err;
                }
            }
        );

        free_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/free",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                std::lock_guard<std::mutex> lock(hardware_mutex_);
                setTorqueAll(false);
                res->success = true;
                res->message = "Arm torque disabled (free mode).";
                RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Free mode (torque off)", getTaskTag().c_str());
            }
        );

        relax_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/arm/relax",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                std::lock_guard<std::mutex> lock(hardware_mutex_);
                setTorqueAll(false);
                res->success = true;
                res->message = "Arm torque disabled (free mode).";
                RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Relax (torque off)", getTaskTag().c_str());
            }
        );

        RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Ready (port: %s, baud: %d)", getTaskTag().c_str(), port_name_.c_str(), baudrate_);

        // Initialize hardware connection
        if (initHardware()) {
            RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Dynamixel connected", getTaskTag().c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "%s\033[1;31m[ARM]\033[0m Failed opening port %s", getTaskTag().c_str(), port_name_.c_str());
        }
    }

    std::string getTaskTag() const {
        if (task_mode_ == "receptionist" || task_mode_ == "rec") {
            return "\033[1;36m[RECEPTIONIST]\033[0m ";
        } else if (task_mode_ == "help_me_carry" || task_mode_ == "hmc") {
            return "\033[1;32m[HELP_ME_CARRY]\033[0m ";
        } else if (task_mode_ == "restaurant" || task_mode_ == "resto" || task_mode_ == "waiter") {
            return "\033[1;34m[RESTAURANT]\033[0m ";
        }
        return "";
    }

    ~ArmMotionServer() override {
        cleanupHardware();
    }

private:
    std::string task_mode_{"all"};
    std::string port_name_;
    int baudrate_;
    std::string motions_dir_;
    std::mutex hardware_mutex_;

    bool   ik_enabled_            = true;
    double ik_l1_m_               = 0.21;
    double ik_l2_m_               = 0.075;
    double ik_shoulder_x_m_       = 0.13;
    double ik_shoulder_z_m_       = 0.61;
    double ik_j1_zero_offset_deg_ = 90.0;
    double ik_j2_zero_offset_deg_ = 0.0;
    double ik_j1_sign_            = -1.0;
    double ik_j2_sign_            = 1.0;
    bool   ik_elbow_up_           = true;
    bool   ik_clamp_workspace_    = true;
    double ik_max_reach_margin_m_ = 0.08;
    double ik_pregrasp_dx_m_      = -0.04;
    double ik_pregrasp_dz_m_      = 0.02;
    double ik_carry_j1_deg_       = 0.0;
    double ik_carry_j2_deg_       = 0.0;
    double ik_target_timeout_sec_ = 5.0;
    std::string ik_fallback_motion_ = "ambil_tas_simple";

    std::mutex target_mutex_;
    double target_x_ = 0.0;
    double target_z_ = 0.0;
    rclcpp::Time target_stamp_{0, 0, RCL_ROS_TIME};
    bool has_target_ = false;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr target_point_sub_;

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr ambil_tas_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr ambil_tas_simple_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr ambil_order_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr release_gripper_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr letakkan_tas_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr point_pos_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr point_to_chair_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr free_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr relax_srv_;

    dynamixel::PortHandler   *portHandler   = nullptr;
    dynamixel::PacketHandler *packetHandler = nullptr;
    bool is_connected_ = false;
    bool g_gripper_online = false;

    std::map<uint8_t, uint16_t> g_last_valid_raw;

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
                RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Grasped: %.1f deg (load: %d)", getTaskTag().c_str(), cur_grip_deg, load_mag);
                return true;
            }
        }

        packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
        cur_grip_deg = 0.0;
        g_last_valid_raw[GRIPPER_ID] = static_cast<uint16_t>(closed_tick);
        RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Gripper closed (no load)", getTaskTag().c_str());
        return false;
    }

    void releaseGripper(dynamixel::GroupSyncWrite &gsw, double cur[3], double open_deg = DEFAULT_RELEASE_DEG) {
        open_deg = std::clamp(open_deg, 0.0, MAX_OPEN_DEG);
        if (g_gripper_online) {
            uint8_t err = 0;
            packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
            packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_SLOPE,  COMPLIANCE_SLOPE, &err);
            packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_SLOPE, COMPLIANCE_SLOPE, &err);
            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING_SPEED, 0, &err);
            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT, DEFAULT_TORQUE_LIM, &err);
        }
        double target[3] = { cur[0], cur[1], open_deg };
        RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Open gripper: +%.1f deg", getTaskTag().c_str(), open_deg);
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
                moveSmooth(gsw, cur, start_target, 0.0);

                RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Replaying stream (%zu samples)", getTaskTag().c_str(), motion.frames.size());
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
        RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Motion completed", getTaskTag().c_str());
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
            packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, 0, &err);
            if (const auto *c = getCalib(id)) {
                packetHandler->write2ByteTxRx(portHandler, id, ADDR_CW_ANGLE_LIMIT,  c->cw_limit,  &err);
                packetHandler->write2ByteTxRx(portHandler, id, ADDR_CCW_ANGLE_LIMIT, c->ccw_limit, &err);
            }
        }

        uint16_t grip_model = 0;
        if (packetHandler->ping(portHandler, GRIPPER_ID, &grip_model, &err) == COMM_SUCCESS) {
            g_gripper_online = true;
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

    void setTorqueAll(bool enable) {
        if (!packetHandler || !portHandler || !is_connected_) return;
        uint8_t err = 0;
        if (!enable) resetDeadzoneFilter();
        for (uint8_t id : ALL_IDS) {
            if (id == GRIPPER_ID && !g_gripper_online) continue;
            packetHandler->write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, enable ? 1 : 0, &err);
            if (!enable) {
                packetHandler->write1ByteTxRx(portHandler, id, ADDR_LED, 0, &err);
            }
        }
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

    void onTargetPoint(const std_msgs::msg::String::SharedPtr msg) {
        if (!msg || msg->data.empty()) return;
        std::string s = msg->data;
        for (char &c : s) {
            if (c == ',' || c == ';') c = ' ';
        }
        std::stringstream ss(s);
        double x = 0.0, y = 0.0, z = 0.0;
        if (ss >> x >> y >> z) {
            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
                std::lock_guard<std::mutex> lock(target_mutex_);
                target_x_ = x;
                target_z_ = z;
                target_stamp_ = this->now();
                has_target_ = true;
                RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Target: %.2f, %.2f, %.2f", getTaskTag().c_str(), x, y, z);
            }
        }
    }

    bool solvePlanarIK(double target_x, double target_z, double &out_j1_deg, double &out_j2_deg) {
        const double dx = target_x - ik_shoulder_x_m_;
        const double dz = target_z - ik_shoulder_z_m_;
        double r = std::hypot(dx, dz);

        const double max_reach = ik_l1_m_ + ik_l2_m_;
        const double min_reach = std::fabs(ik_l1_m_ - ik_l2_m_);

        if (r > max_reach) {
            if (!ik_clamp_workspace_ || (r - max_reach) > ik_max_reach_margin_m_) {
                RCLCPP_WARN(this->get_logger(),
                    "[IK] Target (dx=%.3f, dz=%.3f, r=%.3f m) di luar jangkauan maks %.3f m!",
                    dx, dz, r, max_reach);
                return false;
            }
            r = max_reach - 1e-4;
        } else if (r < min_reach) {
            if (!ik_clamp_workspace_) {
                RCLCPP_WARN(this->get_logger(),
                    "[IK] Target (dx=%.3f, dz=%.3f, r=%.3f m) terlalu dekat (< %.3f m)!",
                    dx, dz, r, min_reach);
                return false;
            }
            r = min_reach + 1e-4;
        }

        const double cos_q2 = std::clamp(
            (r * r - ik_l1_m_ * ik_l1_m_ - ik_l2_m_ * ik_l2_m_) / (2.0 * ik_l1_m_ * ik_l2_m_),
            -1.0, 1.0);
        const double sin_q2 = (ik_elbow_up_ ? -1.0 : 1.0) * std::sqrt(std::max(0.0, 1.0 - cos_q2 * cos_q2));
        const double q2_rad = std::atan2(sin_q2, cos_q2);

        const double phi = std::atan2(dz, dx);
        const double k1  = ik_l1_m_ + ik_l2_m_ * cos_q2;
        const double k2  = ik_l2_m_ * sin_q2;
        const double q1_rad = phi - std::atan2(k2, k1);

        const double q1_deg = q1_rad * 180.0 / M_PI;
        const double q2_deg = q2_rad * 180.0 / M_PI;

        out_j1_deg = (q1_deg - ik_j1_zero_offset_deg_) * ik_j1_sign_;
        out_j2_deg = (q2_deg - ik_j2_zero_offset_deg_) * ik_j2_sign_;

        if (std::fabs(out_j1_deg) > WORK_RANGE_DEG || std::fabs(out_j2_deg) > WORK_RANGE_DEG) {
            RCLCPP_WARN(this->get_logger(), "[IK] Solusi melanggar batas kerja: J1=%.1f, J2=%.1f", out_j1_deg, out_j2_deg);
            return false;
        }
        return true;
    }

    bool executeDynamicGrasp(double tx, double tz, std::string &error_msg) {
        if (!initHardware()) {
            error_msg = "Failed to connect to Dynamixel hardware on port " + port_name_;
            return false;
        }

        double grasp_j1 = 0.0, grasp_j2 = 0.0;
        if (!solvePlanarIK(tx, tz, grasp_j1, grasp_j2)) {
            error_msg = "IK solution out of workspace";
            return false;
        }

        double pre_j1 = grasp_j1, pre_j2 = grasp_j2;
        if (!solvePlanarIK(tx + ik_pregrasp_dx_m_, tz + ik_pregrasp_dz_m_, pre_j1, pre_j2)) {
            pre_j1 = grasp_j1;
            pre_j2 = grasp_j2;
        }

        RCLCPP_INFO(this->get_logger(),
            "[DynamicGrasp] Target (x=%.3f, z=%.3f) -> pre=(%.2f, %.2f) deg, grasp=(%.2f, %.2f) deg",
            tx, tz, pre_j1, pre_j2, grasp_j1, grasp_j2);

        try {
            double cur[3] = {0.0, 0.0, 0.0};
            readCurrentPose(cur);
            dynamixel::GroupSyncWrite gsw(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION);
            lockCurrentPose(gsw, cur);

            releaseGripper(gsw, cur, DEFAULT_RELEASE_DEG);

            double pre_target[3]   = { pre_j1,   pre_j2,   cur[2] };
            double grasp_target[3] = { grasp_j1, grasp_j2, cur[2] };
            moveSmooth(gsw, cur, pre_target, 1.2);
            moveSmooth(gsw, cur, grasp_target, 0.8);

            adaptiveGrasp(cur[2], DEFAULT_GRASP_LOAD);

            double carry_target[3] = { ik_carry_j1_deg_, ik_carry_j2_deg_, cur[2] };
            moveSmooth(gsw, cur, carry_target, 1.5);
            return true;
        } catch (const std::exception &e) {
            error_msg = std::string("Dynamic grasp error: ") + e.what();
            return false;
        }
    }

    void handleAmbilTasSimple(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        std::lock_guard<std::mutex> lock(hardware_mutex_);

        double tx = 0.0, tz = 0.0;
        bool use_dynamic_ik = false;
        if (ik_enabled_) {
            std::lock_guard<std::mutex> tlock(target_mutex_);
            if (has_target_ && (this->now() - target_stamp_).seconds() <= ik_target_timeout_sec_) {
                tx = target_x_;
                tz = target_z_;
                use_dynamic_ik = true;
            }
        }

        std::string err;
        if (use_dynamic_ik) {
            RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Dynamic IK grasp (x=%.2f, z=%.2f)...", getTaskTag().c_str(), tx, tz);
            if (executeDynamicGrasp(tx, tz, err)) {
                response->success = true;
                response->message = "Dynamic IK grasp executed successfully.";
                RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Dynamic IK grasp done", getTaskTag().c_str());
                return;
            }
            RCLCPP_WARN(this->get_logger(), "%s\033[1;33m[ARM]\033[0m IK failed (%s), fallback to '%s'", getTaskTag().c_str(), err.c_str(), ik_fallback_motion_.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Fallback motion '%s'...", getTaskTag().c_str(), ik_fallback_motion_.c_str());
        }

        if (runMotionByName(ik_fallback_motion_, err) || runMotionByName("motion_task_1", err)) {
            response->success = true;
            response->message = "Fallback motion executed successfully.";
            RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Motion done", getTaskTag().c_str());
        } else {
            response->success = false;
            response->message = err;
            RCLCPP_ERROR(this->get_logger(), "%s\033[1;31m[ARM]\033[0m %s", getTaskTag().c_str(), err.c_str());
        }
    }

    void handleAmbilOrder(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        std::lock_guard<std::mutex> lock(hardware_mutex_);

        RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Ambil order start...", getTaskTag().c_str());
        std::string err;
        if (runMotionByName("ambil_order", err)) {
            response->success = true;
            response->message = "Motion ambil_order executed successfully.";
            RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Ambil order done", getTaskTag().c_str());
        } else {
            response->success = false;
            response->message = err;
            RCLCPP_ERROR(this->get_logger(), "%s\033[1;31m[ARM]\033[0m %s", getTaskTag().c_str(), err.c_str());
        }
    }

    void handleReleaseGripper(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        std::lock_guard<std::mutex> lock(hardware_mutex_);

        RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Release gripper start...", getTaskTag().c_str());
        std::string err;
        if (runMotionByName("release_gripper", err)) {
            response->success = true;
            response->message = "Motion release_gripper executed successfully.";
            RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Release gripper done", getTaskTag().c_str());
        } else {
            response->success = false;
            response->message = err;
            RCLCPP_ERROR(this->get_logger(), "%s\033[1;31m[ARM]\033[0m %s", getTaskTag().c_str(), err.c_str());
        }
    }

    void handleAmbilTas(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        std::lock_guard<std::mutex> lock(hardware_mutex_);

        RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Ambil tas start...", getTaskTag().c_str());
        std::string err;
        if (runMotionByName("motion_task_1", err)) {
            response->success = true;
            response->message = "Motion ambil_tas executed successfully.";
            RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Ambil tas done", getTaskTag().c_str());
        } else {
            response->success = false;
            response->message = err;
            RCLCPP_ERROR(this->get_logger(), "%s\033[1;31m[ARM]\033[0m %s", getTaskTag().c_str(), err.c_str());
        }
    }

    void handleLetakkanTas(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        std::lock_guard<std::mutex> lock(hardware_mutex_);

        RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Letakkan tas start...", getTaskTag().c_str());
        std::string err;
        if (runMotionByName("letakkan_tas", err)) {
            response->success = true;
            response->message = "Motion letakkan_tas executed successfully.";
            RCLCPP_INFO(this->get_logger(), "%s\033[1;35m[ARM]\033[0m Letakkan tas done", getTaskTag().c_str());
        } else {
            response->success = false;
            response->message = err;
            RCLCPP_ERROR(this->get_logger(), "%s\033[1;31m[ARM]\033[0m %s", getTaskTag().c_str(), err.c_str());
        }
    }
};

int main(int argc, char **argv) {
    setenv("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}", 1);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArmMotionServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

