// Node ROS Task 1: baca topik mentah zed -> integrate() + decide()
// (task_logic.cpp) -> tulis cmd_vel / goal Nav2 / status.

#include "luggage_task/task.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "tf2/exceptions.h"

LuggageTask::LuggageTask()
: Node("luggage_task")
{
  world_ = std::make_unique<luggage_task::World>();
  loadParam();
  setupRos();
  syncParams();

  const double now = nowSec();
  world_->retries_left = reacquire_retries_;
  world_->gotoState(luggage_task::State::START, now, "START: inisialisasi robot + kamera");
  RCLCPP_INFO(
    get_logger(), "task start: gesture_only=%s mirror=%s img_w=%d (mirror=true: tangan kanan = KANAN)",
    gesture_only_ ? "true" : "false", mirror_ ? "true" : "false", img_w_);
  timer_ = create_wall_timer(
    std::chrono::duration<double>(0.1),
    std::bind(&LuggageTask::step, this));
}

void LuggageTask::loadParam()
{
  declare_parameter("person_timeout_sec", 30.0);
  declare_parameter("hand_timeout_sec", 30.0);
  declare_parameter("search_timeout_sec", 40.0);
  declare_parameter("follow_duration_sec", 0.0);
  declare_parameter("person_lost_timeout_sec", 5.0);
  declare_parameter("reacquire_retries", 3);
  declare_parameter("rotate_speed", 0.3);
  declare_parameter("rotate_tolerance_deg", 5.0);
  declare_parameter("approach_stop_m", 0.5);
  declare_parameter("approach_speed", 0.25);
  declare_parameter("approach_timeout_sec", 30.0);
  declare_parameter("follow_speed", 0.25);
  declare_parameter("follow_stop_distance_m", 0.5);
  declare_parameter("front_sector_deg", 30.0);
  declare_parameter("image_width_px", 672);
  declare_parameter("camera_hfov_deg", 90.0);
  declare_parameter("simulate_bag", false);
  declare_parameter("search_simulate_sec", 6.0);
  declare_parameter("map_frame", "map");
  declare_parameter("base_frame", "base_link");
  declare_parameter("nav2_action", "navigate_to_pose");
  declare_parameter("follow_goal_update_m", 0.3);
  declare_parameter("nav2_resend_sec", 2.0);
  declare_parameter("nav2_ready_timeout_sec", 15.0);
  declare_parameter("nav_min_goal_m", 0.15);
  // Mode gesture-dulu: true = berhenti di deteksi menunjuk (tidak ke SEARCH_BAG).
  declare_parameter("gesture_only", true);
  // Integrasi HRI suara: true = tunggu wakeword & komando suara.
  declare_parameter("use_hri", true);
  // Window OpenCV berisi log teks gesture (subs /zed/person_keypoints).
  declare_parameter("debug_window", true);
  // Mirror: default NYALA = perspektif person (tangan kanan = KANAN,
  // kanan-asli tampil kanan). Matikan bila mau label sisi robot.
  declare_parameter("mirror", true);
  rcl_interfaces::msg::ParameterDescriptor params_file_desc;
  params_file_desc.description =
    "Path file YAML params. Kosong = share/luggage_task/config/params.yaml. "
    "File dibaca LANGSUNG saat node start: edit yaml + restart, tanpa build.";
  declare_parameter("params_file", "", params_file_desc);

  loadYamlFile();

  use_hri_ = get_parameter("use_hri").as_bool();
  person_timeout_ = get_parameter("person_timeout_sec").as_double();
  hand_timeout_ = get_parameter("hand_timeout_sec").as_double();
  search_timeout_ = get_parameter("search_timeout_sec").as_double();
  follow_duration_ = get_parameter("follow_duration_sec").as_double();
  lost_timeout_ = get_parameter("person_lost_timeout_sec").as_double();
  reacquire_retries_ = get_parameter("reacquire_retries").as_int();
  rotate_speed_ = get_parameter("rotate_speed").as_double();
  rotate_tol_ = get_parameter("rotate_tolerance_deg").as_double();
  approach_stop_m_ = get_parameter("approach_stop_m").as_double();
  approach_speed_ = get_parameter("approach_speed").as_double();
  approach_timeout_ = get_parameter("approach_timeout_sec").as_double();
  follow_speed_ = get_parameter("follow_speed").as_double();
  follow_stop_ = get_parameter("follow_stop_distance_m").as_double();
  front_sector_ = get_parameter("front_sector_deg").as_double();
  img_w_ = get_parameter("image_width_px").as_int();
  hfov_ = get_parameter("camera_hfov_deg").as_double();
  simulate_bag_ = get_parameter("simulate_bag").as_bool();
  simulate_sec_ = get_parameter("search_simulate_sec").as_double();
  map_frame_ = get_parameter("map_frame").as_string();
  base_frame_ = get_parameter("base_frame").as_string();
  nav2_action_ = get_parameter("nav2_action").as_string();
  goal_update_m_ = get_parameter("follow_goal_update_m").as_double();
  nav_resend_sec_ = get_parameter("nav2_resend_sec").as_double();
  nav_ready_timeout_ = get_parameter("nav2_ready_timeout_sec").as_double();
  nav_min_goal_m_ = get_parameter("nav_min_goal_m").as_double();
  gesture_only_ = get_parameter("gesture_only").as_bool();
  debug_window_ = get_parameter("debug_window").as_bool();
  mirror_ = get_parameter("mirror").as_bool();
}

void LuggageTask::syncParams()
{
  auto & p = world_->params;
  p.use_hri = use_hri_;
  p.person_timeout = person_timeout_;
  p.hand_timeout = hand_timeout_;
  p.search_timeout = search_timeout_;
  p.follow_duration = follow_duration_;
  p.lost_timeout = lost_timeout_;
  p.reacquire_retries = reacquire_retries_;
  p.rotate_speed = rotate_speed_;
  p.rotate_tol = rotate_tol_;
  p.approach_stop_m = approach_stop_m_;
  p.approach_speed = approach_speed_;
  p.approach_timeout = approach_timeout_;
  p.follow_speed = follow_speed_;
  p.follow_stop = follow_stop_;
  p.front_sector = front_sector_;
  p.img_w = img_w_;
  p.hfov = hfov_;
  p.simulate_bag = simulate_bag_;
  p.simulate_sec = simulate_sec_;
  p.goal_update_m = goal_update_m_;
  p.nav_resend_sec = nav_resend_sec_;
  p.nav_ready_timeout = nav_ready_timeout_;
  p.nav_min_goal_m = nav_min_goal_m_;
  p.gesture_only = gesture_only_;
  p.debug_window = debug_window_;
  p.mirror = mirror_;
}

void LuggageTask::loadYamlFile()
{
  std::string path = get_parameter("params_file").as_string();
  if (path.empty()) {
    try {
      path = ament_index_cpp::get_package_share_directory("luggage_task") +
        "/config/params.yaml";
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "share dir tak ketemu (%s), pakai default kode.", e.what());
      return;
    }
  }
  std::ifstream f(path);
  if (!f.is_open()) {
    RCLCPP_WARN(get_logger(), "params file tak terbaca: %s, pakai default kode.", path.c_str());
    return;
  }
  size_t applied = 0;
  std::string line;
  std::string section;  // section top-level aktif (tanpa indent): hanya baca milik kami
  auto trimLine = [](std::string s) {
    const char * ws = " \t\r\n";
    const auto a = s.find_first_not_of(ws);
    if (a == std::string::npos) {return std::string();}
    const auto b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
  };
  while (std::getline(f, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) {line = line.substr(0, hash);}
    // Deteksi header section: tanpa indent di depan, diakhiri ':'.
    if (!line.empty() && line[0] != ' ' && line[0] != '\t') {
      const auto t = trimLine(line);
      if (!t.empty() && t.back() == ':' && t.find(' ') == std::string::npos) {
        section = t.substr(0, t.size() - 1);
        continue;
      }
    }
    line = trimLine(line);
    if (line.empty()) {continue;}
    if (section != "luggage_task") {continue;}
    const auto colon = line.find(':');
    if (colon == std::string::npos) {continue;}
    const std::string key = trimLine(line.substr(0, colon));
    std::string val = trimLine(line.substr(colon + 1));
    if (key.empty() || val.empty() || !has_parameter(key)) {continue;}
    try {
      const auto type = get_parameter(key).get_type();
      if (type == rclcpp::ParameterType::PARAMETER_BOOL) {
        set_parameter(rclcpp::Parameter(key, val == "true" || val == "1"));
      } else if (type == rclcpp::ParameterType::PARAMETER_INTEGER) {
        set_parameter(rclcpp::Parameter(key, std::stoi(val)));
      } else if (type == rclcpp::ParameterType::PARAMETER_DOUBLE) {
        set_parameter(rclcpp::Parameter(key, std::stod(val)));
      } else if (type == rclcpp::ParameterType::PARAMETER_STRING) {
        set_parameter(rclcpp::Parameter(key, val));
      } else {
        continue;
      }
      ++applied;
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "param '%s=%s' diabaikan (%s).", key.c_str(), val.c_str(), e.what());
    }
  }
  RCLCPP_INFO(get_logger(), "loaded %zu params from %s.", applied, path.c_str());
}

void LuggageTask::setupRos()
{
  using std::placeholders::_1;
  auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(
    luggage_task::topics::kCmdVel, cmd_qos);
  status_pub_ = create_publisher<std_msgs::msg::String>(
    luggage_task::topics::kStatus, 10);
  target_pub_ = create_publisher<std_msgs::msg::String>(
    luggage_task::topics::kTargetDir, 10);

  // Topik mentah zed (bbox + 3D + nama) + yaw + lidar + goal.
  persons_sub_ = create_subscription<std_msgs::msg::String>(
    luggage_task::topics::kPersons, 10, std::bind(&LuggageTask::onPersons, this, _1));
  bag_sub_ = create_subscription<std_msgs::msg::String>(
    luggage_task::topics::kBags, 10, std::bind(&LuggageTask::onBag, this, _1));
  yaw_sub_ = create_subscription<std_msgs::msg::Float64>(
    luggage_task::topics::kYaw, 10, std::bind(&LuggageTask::onYaw, this, _1));
  scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
    luggage_task::topics::kScan, rclcpp::SensorDataQoS(),
    std::bind(&LuggageTask::onScan, this, _1));
  goal_sub_ = create_subscription<std_msgs::msg::Bool>(
    luggage_task::topics::kGoal, 10, std::bind(&LuggageTask::onGoal, this, _1));
  hri_tts_pub_ = create_publisher<std_msgs::msg::String>(
    luggage_task::topics::kHriTts, 10);
  hri_event_sub_ = create_subscription<std_msgs::msg::String>(
    luggage_task::topics::kHriEvent, 10, std::bind(&LuggageTask::onHriEvent, this, _1));
  // Preview awal dari zed (zed hanya publish): /zed/image (BGR).
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    luggage_task::topics::kImage, rclcpp::SensorDataQoS(),
    std::bind(&LuggageTask::onImage, this, _1));

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  nav_client_ = rclcpp_action::create_client<Nav2GoTo>(this, nav2_action_);
  follow_client_ = create_client<std_srvs::srv::Trigger>(luggage_task::topics::kFollowStart);
  follow_stop_client_ = create_client<std_srvs::srv::Trigger>(luggage_task::topics::kFollowStop);
}

double LuggageTask::nowSec() const
{
  return now().seconds();
}

void LuggageTask::publishStatus(const std::string & text)
{
  std_msgs::msg::String msg;
  msg.data = text;
  status_pub_->publish(msg);
  RCLCPP_INFO(get_logger(), "%s", msg.data.c_str());
  last_status_published_ = text;
}

void LuggageTask::stop()
{
  cmd_pub_->publish(geometry_msgs::msg::Twist());
}

void LuggageTask::drive(double linear, double angular)
{
  geometry_msgs::msg::Twist msg;
  msg.linear.x = linear;
  msg.angular.z = angular;
  cmd_pub_->publish(msg);
}

double LuggageTask::frontMinRange() const
{
  if (!last_scan_ || last_scan_->ranges.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  const auto & ranges = last_scan_->ranges;
  const size_t n = ranges.size();
  const size_t half = static_cast<size_t>((front_sector_ / 360.0) * n);
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i <= half && i < n; ++i) {
    const double r = ranges[i];
    if (std::isfinite(r) && r >= last_scan_->range_min) {
      best = std::min(best, r);
    }
  }
  for (size_t i = n - half; i < n; ++i) {
    const double r = ranges[i];
    if (std::isfinite(r) && r >= last_scan_->range_min) {
      best = std::min(best, r);
    }
  }
  return best;
}

void LuggageTask::onPersons(const std_msgs::msg::String::SharedPtr msg)
{
  persons_msg_ = luggage_task::trim(msg->data);
}

void LuggageTask::onBag(const std_msgs::msg::String::SharedPtr msg)
{
  bag_msg_ = luggage_task::trim(msg->data);
}

void LuggageTask::onYaw(const std_msgs::msg::Float64::SharedPtr msg)
{
  double yaw = std::fmod(msg->data, 360.0);
  if (yaw < 0.0) {yaw += 360.0;}
  last_yaw_ = yaw;
}

void LuggageTask::onScan(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  last_scan_ = msg;
}

void LuggageTask::onGoal(const std_msgs::msg::Bool::SharedPtr msg)
{
  if (msg->data) {goal_reached_ = true;}
}

void LuggageTask::onHriEvent(const std_msgs::msg::String::SharedPtr msg)
{
  hri_event_msg_ = luggage_task::trim(msg->data);
  RCLCPP_INFO(get_logger(), "[HRI Event] %s", hri_event_msg_.c_str());
}

void LuggageTask::onImage(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // Decode manual tanpa cv_bridge (hindari konflik header OpenCV5).
  // Dukung bgr8 / rgb8.
  try {
    if (msg->encoding != "bgr8" && msg->encoding != "rgb8") {return;}
    if (msg->width == 0 || msg->height == 0) {return;}
    const size_t expect = static_cast<size_t>(msg->step) * msg->height;
    if (msg->data.size() < expect) {return;}
    cv::Mat raw(
      static_cast<int>(msg->height), static_cast<int>(msg->width), CV_8UC3,
      const_cast<uint8_t *>(msg->data.data()), msg->step);
    cv::Mat bgr;
    if (msg->encoding == "rgb8") {
      cv::cvtColor(raw, bgr, cv::COLOR_RGB2BGR);
    } else {
      bgr = raw.clone();
    }
    if (bgr.empty()) {return;}
    std::lock_guard<std::mutex> lock(image_mutex_);
    last_image_ = bgr;
    have_image_ = true;
    pending_img_w_ = static_cast<int>(msg->width);
  } catch (...) {}
}

bool LuggageTask::robotPoseMap(double & x, double & y, double & yaw)
{
  try {
    const auto t = tf_buffer_->lookupTransform(
      map_frame_, base_frame_, tf2::TimePointZero,
      tf2::Duration(std::chrono::milliseconds(100)));
    x = t.transform.translation.x;
    y = t.transform.translation.y;
    yaw = 2.0 * std::atan2(t.transform.rotation.z, t.transform.rotation.w);
    return true;
  } catch (const tf2::TransformException &) {
    return false;
  }
}

bool LuggageTask::navReady()
{
  if (nav_server_ready_) {return true;}
  if (nav_client_->wait_for_action_server(std::chrono::seconds(0))) {
    nav_server_ready_ = true;
    return true;
  }
  return false;
}

void LuggageTask::navSendGoal(double gx, double gy, double gyaw, double now)
{
  Nav2GoTo::Goal goal;
  goal.pose.header.frame_id = map_frame_;
  goal.pose.header.stamp = this->now();
  goal.pose.pose.position.x = gx;
  goal.pose.pose.position.y = gy;
  goal.pose.pose.orientation.z = std::sin(gyaw / 2.0);
  goal.pose.pose.orientation.w = std::cos(gyaw / 2.0);

  auto opts = rclcpp_action::Client<Nav2GoTo>::SendGoalOptions();
  opts.goal_response_callback =
    [this](NavGoalHandle::SharedPtr gh) {
      nav_goal_active_ = static_cast<bool>(gh);
      if (gh) {nav_goal_handle_ = gh;}
    };
  opts.result_callback =
    [this](const NavGoalHandle::WrappedResult &) {
      nav_goal_active_ = false;
    };
  nav_client_->async_send_goal(goal, opts);
  nav_goal_active_ = true;
  world_->nav_last_send = now;
  world_->nav_last_gx = gx;
  world_->nav_last_gy = gy;
  world_->nav_last_gyaw = gyaw;
  world_->nav_last_goal_valid = true;
}

void LuggageTask::navCancel()
{
  if (nav_goal_active_ && nav_goal_handle_) {
    try {nav_client_->async_cancel_goal(nav_goal_handle_);} catch (...) {}
  }
  nav_goal_active_ = false;
}

void LuggageTask::step()
{
  // read -> integrate -> decide -> write.
  const double now = nowSec();

  // Sinkron lebar gambar aktual (ZED VGA = 672) agar hitungan bearing
  // (tengah gambar) selalu cocok dgn keypoint/bbox asli.
  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    if (pending_img_w_ > 0 && pending_img_w_ != img_w_) {
      img_w_ = pending_img_w_;
      world_->params.img_w = img_w_;
      RCLCPP_INFO(get_logger(), "image_width_px disinkron ke %d (lebar /zed/image).", img_w_);
      pending_img_w_ = img_w_;
    }
  }

  luggage_task::Snapshot snap;
  snap.persons_raw = persons_msg_;
  snap.bag = bag_msg_;
  snap.yaw = last_yaw_;
  snap.have_scan = static_cast<bool>(last_scan_) && !last_scan_->ranges.empty();
  snap.front_min_range = frontMinRange();
  snap.have_pose = robotPoseMap(snap.pose_x, snap.pose_y, snap.pose_yaw);
  snap.goal_reached = goal_reached_;
  snap.hri_event = hri_event_msg_;
  hri_event_msg_.clear();
  snap.now = now;

  luggage_task::integrate(world_.get(), snap);
  const bool ready = navReady();
  luggage_task::decide(world_.get(), now, ready);
  // Peringatan bila sumber gambar/person mati (penyebab window hitam).
  if (!have_image_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "belum ada /zed/image (cek node zed hidup + kamera tidak dipakai proses lain)");
  }
  if (persons_msg_.empty()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "belum ada /zed/person_keypoints (cek body tracking + model AI zed)");
  }
  write(now);
}

void LuggageTask::write(double now)
{
  const luggage_task::State st = world_->state;
  const std::string tag = std::string("[") + luggage_task::stateName(st) + "] ";

  if (world_->status_dirty) {
    publishStatus(tag + world_->status_text);
    world_->status_dirty = false;
  } else if (!world_->tick.empty() && tag + world_->tick != last_status_published_) {
    publishStatus(tag + world_->tick);
  }

  if (world_->tts_dirty) {
    std_msgs::msg::String tts_msg;
    tts_msg.data = world_->tts_text;
    hri_tts_pub_->publish(tts_msg);
    world_->tts_dirty = false;
    RCLCPP_INFO(get_logger(), "[TTS Sent] %s", tts_msg.data.c_str());
  }

  if (world_->target_dir_dirty) {
    std_msgs::msg::String msg;
    msg.data = world_->target_dir;
    target_pub_->publish(msg);
    world_->target_dir_dirty = false;
  }

  if (st == luggage_task::State::RETURN_START) {
    if (!world_->return_goal_sent && nav_server_ready_) {
      navSendGoal(0.0, 0.0, 0.0, now);
      world_->return_goal_sent = true;
      publishStatus("[RETURN_START] Mengirim goal Nav2 kembali ke titik Start (0.0, 0.0)...");
    }
    return;
  }

  if (st == luggage_task::State::FOLLOW) {
    if (!following_active_) {
      if (follow_client_ && follow_client_->service_is_ready()) {
        auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
        follow_client_->async_send_request(req);
        following_active_ = true;
        publishStatus(tag + "Memulai person following dinamis Nav2 (/follow/person)...");
      } else {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Menunggu service /follow/person siap (pastikan node person_follower berjalan)...");
      }
    }
    return;
  }

  // Jika keluar dari state FOLLOW dan following masih aktif, hentikan person_follower
  if (following_active_) {
    if (follow_stop_client_ && follow_stop_client_->service_is_ready()) {
      auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
      follow_stop_client_->async_send_request(req);
    }
    following_active_ = false;
    publishStatus("[FOLLOW] Menghentikan person following (/follow/stop)...");
  }

  navCancel();
  if (st == luggage_task::State::DONE || st == luggage_task::State::ABORT) {
    stop();
    showGestureWindow();
    return;
  }
  const auto & mo = world_->cmd;
  if (mo.stop) {
    stop();
  } else {
    drive(mo.vx, mo.w);
  }
  showGestureWindow();
}

void LuggageTask::showGestureWindow()
{
  if (!debug_window_) {return;}
  try {
    // Preview awal dari zed (/zed/image): zed hanya publish, task yg nampilin.
    cv::Mat frame;
    bool has_frame = false;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      if (have_image_ && !last_image_.empty()) {
        frame = last_image_.clone();
        has_frame = true;
      }
    }
    // Ukuran asli frame kamera (ZED VGA = 672x376): canvas = native,
    // titik & bbox digambar mentah tanpa skala agar nempel ke joint.
    // Mirror: flip horizontal supaya kanan-asli tampil kanan.
    cv::Mat canvas;
    if (has_frame) {
      canvas = frame.clone();
      if (mirror_) {cv::flip(canvas, canvas, 1);}
    } else {
      canvas = cv::Mat(376, 672, CV_8UC3, cv::Scalar(20, 20, 20));
      cv::putText(
        canvas, "menunggu /zed/image ... (zed publish only)", cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
      cv::putText(
        canvas, "cek: 1) nav2:=true? 2) ros2 node list | grep zed", cv::Point(10, 60),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
      cv::putText(
        canvas, "3) log zed: open failed? grab gagal? detak?", cv::Point(10, 85),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    }
    const int W = canvas.cols;
    const int H = canvas.rows;
    // Overlay ala maen_boss: teks langsung di atas gambar (merah, besar).
    cv::Mat view = canvas;

    const int picked = luggage_task::pickOperator(world_->persons);
    // 34 titik keypoint mentah langsung dari ZED SDK (tanpa garis skeleton).
    // Koordinat sudah piksel asli frame -> tanpa skala.
    for (size_t i = 0; i < world_->persons.size(); ++i) {
      const auto & p = world_->persons[i];
      const bool is_pick = (static_cast<int>(i) == picked);
      const cv::Scalar col = is_pick ? cv::Scalar(0, 255, 0) : cv::Scalar(150, 150, 150);
      if (p.has_bbox) {
        cv::Rect r(p.bbox_x, p.bbox_y, p.bbox_w, p.bbox_h);
        r &= cv::Rect(0, 0, W, H);
        cv::rectangle(canvas, r, col, is_pick ? 2 : 1);
      }
      for (int k = 0; k < 34 && static_cast<size_t>(k * 2 + 1) < p.keypoints.size(); ++k) {
        const double kx_raw = p.keypoints[static_cast<size_t>(k * 2)];
        const double ky_raw = p.keypoints[static_cast<size_t>(k * 2 + 1)];
        if (!std::isfinite(kx_raw) || !std::isfinite(ky_raw)) {continue;}
        const int kx = static_cast<int>(std::round(kx_raw));
        const int ky = static_cast<int>(std::round(ky_raw));
        // Konvensi zed: <0 = tak terdeteksi.
        if (kx < 0 || ky < 0 || kx >= W || ky >= H) {continue;}
        // Bahu & pergelangan (dipakai gestur) digambar lebih besar.
        const bool big = (k == 5 || k == 7 || k == 12 || k == 14);
        cv::circle(
          canvas, cv::Point(kx, ky), (big && is_pick) ? 5 : 3,
          big ? cv::Scalar(0, 0, 255) : col, -1);
      }
    }

    // Tas YOLO (magenta) + distance. Koordinat sudah mirror ikut persons.
    for (size_t i = 0; i < world_->bags.size() && i < 5; ++i) {
      const auto & b = world_->bags[i];
      if (!b.has_bbox) {continue;}
      cv::Rect r(b.bbox_x, b.bbox_y, b.bbox_w, b.bbox_h);
      r &= cv::Rect(0, 0, W, H);
      cv::rectangle(canvas, r, cv::Scalar(255, 0, 255), 2);
      char tb[64];
      if (b.has_pos) {
        snprintf(tb, sizeof(tb), "tas %.2f m", b.range_m());
      } else {
        snprintf(tb, sizeof(tb), "tas ?");
      }
      int ty = std::max(15, r.y - 6);
      cv::putText(
        canvas, tb, cv::Point(std::max(0, r.x), ty),
        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
    }

    // Overlay minimal: hanya "BAWAH KIRI" / "BAWAH KANAN" (perspektif person:
    // tangan kanan = KANAN) + distance. Tanpa teks lain saat gambar ada.
    if (has_frame) {
      const cv::Scalar green(0, 255, 0);
      std::string big;
      if (!world_->persons.empty() && picked >= 0) {
        const auto & pp = world_->persons[static_cast<size_t>(picked)];
        const std::string dir = luggage_task::pointingDir(pp);
        const bool pointing = luggage_task::isPointing(pp);
        if (pointing && dir == "kiri") {big = "BAWAH KIRI";}
        else if (pointing && dir == "kanan") {big = "BAWAH KANAN";}
        if (!big.empty()) {
          cv::putText(
            view, big, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2, green, 3,
            cv::LINE_AA);
        }
        if (pp.has_pos) {
          char db[64];
          snprintf(db, sizeof(db), "%.2f m", std::hypot(pp.pos_x, pp.pos_y));
          cv::putText(
            view, db, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0,
            cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
      }
    }
    const std::string win = "luggage_task - gesture (/zed/person_keypoints)";
    if (!window_inited_) {
      cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
      window_inited_ = true;
    }
    cv::imshow(win, canvas);
    cv::waitKey(1);
  } catch (const std::exception & e) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 10000,
      "debug_window gagal (headless?): %s", e.what());
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LuggageTask>());
  rclcpp::shutdown();
  return 0;
}
