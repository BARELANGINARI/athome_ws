#ifndef LUGGAGE_TASK_TASK_HPP
#define LUGGAGE_TASK_TASK_HPP

// Task 1 Carry My Luggage — Navigasi & Pelacakan. Satu header untuk
// seluruh task (pengganti behavior/controller/strategy/world/parse_utils
// yang terpisah): tipe data + deklarasi logika (task_logic.cpp) + node
// ROS (task_node.cpp). Alur state ikut flowchart:
// START -> DETECT_PERSON -> PERSON_POSE -> DETECT_GESTURE -> TARGET_DIR
// -> SEARCH_BAG -> BAG_POSE -> ROTATE_TO_BAG -> APPROACH(0.5m) -> WAIT_TURN
// (operator membelakangi) -> TRACK -> FOLLOW(jaga 0.5m) -> DONE/ABORT.

#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace luggage_task {

// --- Nama topik terpusat (nilai sama seperti literal lama, anti-typo) --------
namespace topics {
constexpr char kPersons[] = "/zed/person_keypoints";
constexpr char kBags[] = "/zed/bag_detections";
constexpr char kYaw[] = "/zed/yaw_deg";
constexpr char kImage[] = "/zed/image";
constexpr char kScan[] = "/scan";
constexpr char kCmdVel[] = "cmd_vel";
constexpr char kStatus[] = "/luggage/status";
constexpr char kTargetDir[] = "/luggage/target_dir";
constexpr char kGoal[] = "/luggage/goal_reached";
constexpr char kHriEvent[] = "/hri/event";
constexpr char kHriTts[] = "/hri/tts_say";
constexpr char kFollowStart[] = "/follow/person";
constexpr char kFollowStop[] = "/follow/stop";
}  // namespace topics

enum class State
{
  START,
  WAIT_WAKEWORD,
  WAIT_FOLLOW,
  DETECT_PERSON,
  PERSON_POSE,
  DETECT_GESTURE,
  TARGET_DIR,
  SEARCH_BAG,
  BAG_POSE,
  ROTATE_TO_BAG,
  APPROACH,
  WAIT_TURN,
  TRACK,
  FOLLOW,
  RECOVERY,
  RETURN_START,
  DONE,
  ABORT
};

const char * stateName(State s);

// --- Parameter tunable (diisi task_node dari ROS param + params.yaml) --------
struct Params
{
  bool use_hri{true};
  double person_timeout{30.0};
  double hand_timeout{30.0};
  double search_timeout{40.0};
  double follow_duration{0.0};  // <= 0.0 = tanpa batas waktu (mengikuti terus sampai arrived)
  double lost_timeout{5.0};
  int reacquire_retries{3};
  double rotate_speed{0.3};
  double rotate_tol{5.0};
  double approach_stop_m{0.5};  // flowchart: berhenti 0.5 m dari operator
  double approach_speed{0.25};
  double approach_timeout{30.0};
  double follow_speed{0.25};
  double follow_stop{0.5};  // flowchart: jaga jarak ~0.5 m saat follow
  double front_sector{30.0};
  int img_w{672};
  double hfov{90.0};
  bool simulate_bag{false};
  double simulate_sec{6.0};
  // Mode debug gesture-dulu: setelah TARGET_DIR balik lagi ke DETECT_GESTURE
  // (tidak lanjut ke SEARCH_BAG). Default true sesuai request "untuk skrg
  // deteksi gesture dulu". Set false bila mau full task sampai FOLLOW.
  bool gesture_only{true};
  // Tampilkan window OpenCV berisi log teks gesture (bbox + depth + arah).
  bool debug_window{true};
  // Mirror: kanan-asli tampil kanan (perspektif person: tangan kanan = KANAN).
  bool mirror{true};
  double goal_update_m{0.3};
  double nav_resend_sec{2.0};
  double nav_ready_timeout{15.0};
  double nav_min_goal_m{0.15};
};

// --- Tas dari /zed/bag_detections mentah -----------------------------------
struct BagInfo
{
  bool has_bbox{false};
  int bbox_x{0};
  int bbox_y{0};
  int bbox_w{0};
  int bbox_h{0};
  bool has_pos{false};  // 3D relatif robot (meter, ala maen_boss)
  double pos_x{0};
  double pos_y{0};
  double pos_z{0};
  double score{0.0};
  double range_m() const {return std::hypot(pos_x, pos_y);}
};

// --- Person dari /zed/person_keypoints mentah --------------------------------
struct PersonInfo
{
  std::string orientation;  // FRONT/BACK/SIDE_* (dari yaw/rel)
  std::string gesture;      // "LEFT_HAND_DOWN_LEFT | ..." / "NO_DIAGONAL_DOWN"
  std::optional<double> yaw;
  std::optional<double> rel;
  bool has_bbox{false};
  int bbox_x{0};
  int bbox_y{0};
  int bbox_w{0};
  int bbox_h{0};
  bool has_pos{false};  // 3D relatif robot (meter); false bila depth NaN
  double pos_x{0};
  double pos_y{0};
  double pos_z{0};
  std::vector<double> keypoints;  // 34 keypoint BODY_34 berpasangan x,y
};

struct RobotState
{
  double x{0}, y{0}, yaw{0};
  bool have_pose{false};
  double front_min_range_m{0};
  bool have_scan{false};
};

struct MotionCmd
{
  double vx{0};
  double w{0};
  bool stop{true};
};

struct FollowGoal
{
  bool valid{false};
  double x{0}, y{0}, theta{0};
};

// --- Snapshot mentah per step (diisi task_node dari latch topik + TF) --------
struct Snapshot
{
  std::string persons_raw;  // /zed/person_keypoints (latch)
  std::string bag;          // /zed/bag_detections (latch)
  std::optional<double> yaw;  // /zed/yaw_deg (latch, 0..360)
  bool have_scan{false};
  double front_min_range{0.0};
  bool have_pose{false};
  double pose_x{0.0};
  double pose_y{0.0};
  double pose_yaw{0.0};
  bool goal_reached{false};  // /luggage/goal_reached (latch)
  std::string hri_event;    // /hri/event (latch: WAKEWORD_DETECTED, etc.)
  double now{0.0};
};

// --- Dunia terpusat ----------------------------------------------------------
class World
{
public:
  World() = default;

  double elapsed(double now) const {return now - state_t0;}
  void gotoState(State s, double now, const std::string & text = "");
  void say(const std::string & text) {tts_text = text; tts_dirty = true;}

  State state{State::START};
  double state_t0{0.0};
  std::string status_text;
  bool status_dirty{false};

  // Suara TTS ke HRI
  std::string tts_text;
  bool tts_dirty{false};
  double last_recovery_speak{0.0};
  int recovery_attempts{0};
  bool return_goal_sent{false};
  std::string hri_event;

  // Person & target
  std::vector<PersonInfo> persons;
  PersonInfo target_person;
  bool have_target{false};
  bool gesture_seen{false};
  std::string target_dir;  // "kiri"/"kanan"/"bawah"
  bool target_dir_dirty{false};

  // Tas terdeteksi YOLO (semua, + distance): "bag:x,y,w,h|x,y,z|conf".
  std::vector<BagInfo> bags;

  // Tas: "x,y,w,h" kotak terpilih (zona kiri/kanan sesuai target_dir)
  std::string bag_box;
  double bag_bearing{0.0};
  double rotate_start_yaw{0.0};
  double rotate_target{0.0};

  // Operator (dilacak dari person BACK)
  bool op_visible{false};
  double op_range_m{-1.0};
  double op_bearing_deg{0.0};

  // Follow
  bool follow_bearing_valid{false};
  double follow_bearing_deg{0.0};
  double last_person_seen{0.0};
  int retries_left{3};
  double follow_t0{0.0};
  bool goal_reached{false};

  // Robot & nav
  RobotState robot;
  std::optional<double> yaw;
  bool nav_last_goal_valid{false};
  double nav_last_send{0.0};
  double nav_last_gx{0.0};
  double nav_last_gy{0.0};
  double nav_last_gyaw{0.0};

  MotionCmd cmd;
  std::string tick;  // status per-tick ("Mendekati ... 0.4 m")
  Params params;
};

// --- Parse & util (task_logic.cpp) -------------------------------------------
double wrap180(double angle_deg);
std::string trim(const std::string & s);
std::vector<std::string> split(const std::string & s, char delim);
// /zed/person_keypoints -> list person (gesture dihitung dari keypoint).
std::vector<PersonInfo> parsePersons(const std::string & data);
// /zed/bag_detections -> semua tas + distance ("bag:x,y,w,h|x,y,z|conf;...").
std::vector<BagInfo> parseBags(const std::string & data);
std::string gestureToDir(const std::string & gesture);

// --- Taktik (task_logic.cpp) -------------------------------------------------
bool isPointing(const PersonInfo & p);
std::string pointingDir(const PersonInfo & p);
int pickOperator(const std::vector<PersonInfo> & persons);
// Tas: "x,y,w,h" (simulasi) atau mentah "bag:...;..." + zona kiri/kanan.
std::optional<std::string> pickBag(
  const std::string & bag_msg, int img_w, const std::string & dir);
std::optional<double> bagBearing(
  const std::string & box, int img_w, double hfov_deg);
std::optional<double> operatorBearing(
  const std::vector<PersonInfo> & persons, int img_w, double hfov_deg);
std::optional<double> approachBearing(
  const std::vector<PersonInfo> & persons, int img_w, double hfov_deg);
FollowGoal followGoal(
  double robot_x, double robot_y, double robot_yaw,
  double bearing_deg, double front_range_m,
  double follow_stop_m, double min_goal_m);

// --- Fuse + state machine (task_logic.cpp) -----------------------------------
void integrate(World * world, const Snapshot & snap);
void decide(World * world, double now, bool nav_ready);

}  // namespace luggage_task

// --- Node ROS (task_node.cpp) -------------------------------------------------
using Nav2GoTo = nav2_msgs::action::NavigateToPose;
using NavGoalHandle = rclcpp_action::ClientGoalHandle<Nav2GoTo>;

class LuggageTask : public rclcpp::Node
{
public:
  LuggageTask();

private:
  void loadParam();
  void loadYamlFile();
  void setupRos();
  void syncParams();
  double nowSec() const;
  void step();
  void write(double now);
  void publishStatus(const std::string & text);
  void stop();
  void drive(double linear, double angular);
  double frontMinRange() const;
  void showGestureWindow();
  void onPersons(const std_msgs::msg::String::SharedPtr msg);
  void onBag(const std_msgs::msg::String::SharedPtr msg);
  void onYaw(const std_msgs::msg::Float64::SharedPtr msg);
  void onScan(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void onGoal(const std_msgs::msg::Bool::SharedPtr msg);
  void onImage(const sensor_msgs::msg::Image::SharedPtr msg);
  void onHriEvent(const std_msgs::msg::String::SharedPtr msg);
  bool robotPoseMap(double & x, double & y, double & yaw);
  bool navReady();
  void navSendGoal(double gx, double gy, double gyaw, double now);
  void navCancel();

  std::unique_ptr<luggage_task::World> world_;

  bool use_hri_{true};
  double person_timeout_{30.0};
  double hand_timeout_{30.0};
  double search_timeout_{40.0};
  double follow_duration_{0.0};
  double lost_timeout_{5.0};
  int reacquire_retries_{3};
  double rotate_speed_{0.3};
  double rotate_tol_{5.0};
  double approach_stop_m_{0.5};
  double approach_speed_{0.25};
  double approach_timeout_{30.0};
  double follow_speed_{0.25};
  double follow_stop_{0.5};
  double front_sector_{30.0};
  int img_w_{640};
  double hfov_{90.0};
  bool simulate_bag_{false};
  double simulate_sec_{6.0};
  std::string map_frame_{"map"};
  std::string base_frame_{"base_link"};
  std::string nav2_action_{"navigate_to_pose"};
  double goal_update_m_{0.3};
  double nav_resend_sec_{2.0};
  double nav_ready_timeout_{15.0};
  double nav_min_goal_m_{0.15};
  // Mode gesture-dulu + window log (request: deteksi menunjuk dulu).
  bool gesture_only_{true};
  bool debug_window_{true};
  bool mirror_{true};
  bool window_inited_{false};

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr target_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr hri_tts_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr persons_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr bag_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr yaw_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr goal_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr hri_event_sub_;
  // Preview awal dari zed (publish only): /zed/image (BGR).
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  cv::Mat last_image_;
  std::mutex image_mutex_;
  bool have_image_{false};
  int pending_img_w_{-1};  // lebar /zed/image aktual -> sinkron ke params.img_w
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp_action::Client<Nav2GoTo>::SharedPtr nav_client_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  NavGoalHandle::SharedPtr nav_goal_handle_;
  bool nav_goal_active_{false};
  bool nav_server_ready_{false};

  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr follow_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr follow_stop_client_;
  bool following_active_{false};

  std::string persons_msg_;
  std::string bag_msg_;
  std::optional<double> last_yaw_;
  sensor_msgs::msg::LaserScan::SharedPtr last_scan_;
  bool goal_reached_{false};
  std::string hri_event_msg_;

  std::string last_status_published_;
};

#endif  // LUGGAGE_TASK_TASK_HPP
