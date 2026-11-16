#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "nucleo_node.hpp"
#include "icecream.hpp"

#define STM32_IP_DEFAULT "192.168.0.4"
#define STM32_PORT_DEFAULT 2002
#define LOCAL_PORT_DEFAULT 2023

namespace {
std::shared_ptr<barelang::Kinematic> kinematic;
std::unique_ptr<udpServer> udpLink;
udpSender stmSender;

struct STM32toPC stmRaw{};
struct STM32toPC stm2pc{};
std::mutex stm_mtx;

void stm32Callback()
{
  std::lock_guard<std::mutex> lk(stm_mtx);
  stm2pc = stmRaw;
}
}

nucleo::nucleo()
: Node("nucleo")
{
  loadParam();
  setupHardware();
  setupRos();

  last_time = now();
  last_cmd_time = last_time;
  timer = create_wall_timer(
        std::chrono::duration<double>(1.0 / loop_hz),
        std::bind(&nucleo::controlLoop, this));
}

void nucleo::loadParam()
{
  declare_parameter("stm_ip", STM32_IP_DEFAULT);
  declare_parameter("stm_port", STM32_PORT_DEFAULT);
  declare_parameter("local_port", LOCAL_PORT_DEFAULT);
  declare_parameter("local_ip", "");
  declare_parameter("loop_hz", 33.33);
  declare_parameter("rpm_max", 428.0);
  declare_parameter("pwm_max", 225.0);
  declare_parameter("pwm_scale", 0.932);
  declare_parameter("odom_frame", "odom");
  declare_parameter("base_frame", "base_link");
  declare_parameter("yaw_offset_deg", 0.0);
  declare_parameter("use_yaw_topic", false);
  declare_parameter("yaw_topic", "/vision/yaw_deg");
  declare_parameter("yaw_topic_timeout", 0.5);
  declare_parameter("cmd_vel_timeout", 0.5);
  declare_parameter("joy_topic", "joy_vel");
  declare_parameter("test_topic", "test_vel");
  declare_parameter("lifter_cmd_topic", "lifter_target");
  declare_parameter("lifter_state_topic", "lifter_state");
  declare_parameter("joy_timeout", 0.05);
  declare_parameter("test_timeout", 0.5);
  declare_parameter("smoothing_enable", true);
  declare_parameter("linear_accel_limit", 2.5);
  declare_parameter("angular_accel_limit", 7.0);
  declare_parameter("stop_decel_scale", 2.0);
  declare_parameter("pulses_per_rev", 7500.0);
  declare_parameter("wheel_radius_cm", 5.0);
  declare_parameter("base_length_cm", 20.8);
  declare_parameter("wheel_yaw_deg", 0.0);
  declare_parameter("odom_yaw_source", "stm32");

  stm_ip = get_parameter("stm_ip").as_string();
  stm_port = get_parameter("stm_port").as_int();
  local_port = get_parameter("local_port").as_int();
  local_ip = get_parameter("local_ip").as_string();
  loop_hz = get_parameter("loop_hz").as_double();
  rpm_max = get_parameter("rpm_max").as_double();
  pwm_max = get_parameter("pwm_max").as_double();
  pwm_scale = get_parameter("pwm_scale").as_double();
  odom_frame = get_parameter("odom_frame").as_string();
  base_frame = get_parameter("base_frame").as_string();
  yaw_offset_deg = get_parameter("yaw_offset_deg").as_double();
  use_yaw_topic = get_parameter("use_yaw_topic").as_bool();
  yaw_topic = get_parameter("yaw_topic").as_string();
  yaw_topic_timeout = get_parameter("yaw_topic_timeout").as_double();
  cmd_vel_timeout = get_parameter("cmd_vel_timeout").as_double();
  joy_topic = get_parameter("joy_topic").as_string();
  test_topic = get_parameter("test_topic").as_string();
  lifter_cmd_topic = get_parameter("lifter_cmd_topic").as_string();
  lifter_state_topic = get_parameter("lifter_state_topic").as_string();
  joy_timeout = get_parameter("joy_timeout").as_double();
  test_timeout = get_parameter("test_timeout").as_double();
  smoothing_enable = get_parameter("smoothing_enable").as_bool();
  linear_accel_limit = get_parameter("linear_accel_limit").as_double();
  angular_accel_limit = get_parameter("angular_accel_limit").as_double();
  stop_decel_scale = get_parameter("stop_decel_scale").as_double();
  pulses_per_rev_ = get_parameter("pulses_per_rev").as_double();
  wheel_radius_cm_ = get_parameter("wheel_radius_cm").as_double();
  base_length_cm_ = get_parameter("base_length_cm").as_double();
}

void nucleo::setupHardware()
{
  kinematic = std::make_shared<barelang::Kinematic>(
    pulses_per_rev_, wheel_radius_cm_, base_length_cm_);
  kinematic->setInitPos(0, 0, 0);
  udpLink = std::make_unique<udpServer>(local_port, local_ip);
  stmSender = udpLink->addSender(stm_ip, stm_port);
  udpLink->addReceiver(stm_ip, stm_port, &stmRaw, sizeof(stmRaw), stm32Callback);
  udpLink->receiverStart();
}

void nucleo::setupRos()
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
  cmd_sub = create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel", cmd_qos, std::bind(&nucleo::onCmdVel, this, _1));
  joy_sub = create_subscription<geometry_msgs::msg::Twist>(
    joy_topic, cmd_qos, std::bind(&nucleo::onJoyVel, this, _1));
  test_sub = create_subscription<geometry_msgs::msg::Twist>(
    test_topic, cmd_qos, std::bind(&nucleo::onTestVel, this, _1));
  lifter_sub = create_subscription<std_msgs::msg::UInt8>(
    lifter_cmd_topic, 10, std::bind(&nucleo::onLifterCmd, this, _1));
  lifter_state_pub = create_publisher<std_msgs::msg::UInt8>(lifter_state_topic, 10);

  srv_lifter_up_ = create_service<std_srvs::srv::Trigger>(
    "/lifter/naik",
    std::bind(&nucleo::handleLifterUp, this, _1, _2));
  srv_lifter_down_ = create_service<std_srvs::srv::Trigger>(
    "/lifter/turun",
    std::bind(&nucleo::handleLifterDown, this, _1, _2));

  odom_pub = create_publisher<nav_msgs::msg::Odometry>("odom", 10);
  tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
  if (use_yaw_topic) {
    yaw_sub = create_subscription<std_msgs::msg::Float64>(
      yaw_topic, 10, std::bind(&nucleo::onYawTopic, this, _1));
  }
}

void nucleo::onCmdVel(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  std::lock_guard<std::mutex> lk(cmd_mtx);
  cmd_vel.vx = msg->linear.x;
  cmd_vel.vy = msg->linear.y;
  cmd_vel.w = msg->angular.z;
  last_cmd_time = now();
}

void nucleo::onJoyVel(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  std::lock_guard<std::mutex> lk(cmd_mtx);
  joy_vel.vx = msg->linear.x;
  joy_vel.vy = msg->linear.y;
  joy_vel.w = msg->angular.z;
  joy_time = now();
  joy_ready = true;
}

void nucleo::onTestVel(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  std::lock_guard<std::mutex> lk(cmd_mtx);
  test_vel.vx = msg->linear.x;
  test_vel.vy = msg->linear.y;
  test_vel.w = msg->angular.z;
  test_time = now();
  test_ready = true;
}

void nucleo::onYawTopic(const std_msgs::msg::Float64::SharedPtr msg)
{
  std::lock_guard<std::mutex> lk(cmd_mtx);
  ext_yaw_deg = msg->data;
  ext_yaw_time = now();
  ext_yaw_ready = true;
}

void nucleo::onLifterCmd(const std_msgs::msg::UInt8::SharedPtr msg)
{
  lifter_target_.store(msg->data);
  RCLCPP_INFO(get_logger(), "Lifter target set: %u", static_cast<unsigned int>(lifter_target_.load()));
}

void nucleo::handleLifterUp(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
  std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  (void)req;
  lifter_target_.store(1);
  res->success = true;
  res->message = "Lifter target UP (1) sent";
  RCLCPP_INFO(get_logger(), "Service /lifter/naik called: lifter_target = 1");
}

void nucleo::handleLifterDown(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
  std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  (void)req;
  lifter_target_.store(0);
  res->success = true;
  res->message = "Lifter target DOWN (0) sent";
  RCLCPP_INFO(get_logger(), "Service /lifter/turun called: lifter_target = 0");
}

double nucleo::normalizeDeg(double deg)
{
  while (deg > 180.0) {
    deg -= 360.0;
  }
  while (deg <= -180.0) {
    deg += 360.0;
  }
  return deg;
}

void nucleo::controlLoop()
{
  auto t_now = now();
  double dt = calcDt(t_now);

  struct STM32toPC snap;
  {
    std::lock_guard<std::mutex> lk(stm_mtx);
    snap = stm2pc;
  }
  updateOdom(snap, dt, t_now);

  bool timed_out = false;
  MotionTarget target = resolveTarget(t_now, timed_out);
  MotionTarget shaped = applySmoothing(target, dt, timed_out);
  struct PCtoSTM32 pc2stm = sendMotor(shaped);

  publishOdom(t_now);

  std_msgs::msg::UInt8 l_state;
  l_state.data = snap.lifterState;
  lifter_state_pub->publish(l_state);

  IC(target.vx, target.vy, target.w);
  IC(shaped.vx, shaped.vy, shaped.w);
  IC(pc2stm.MKiri, pc2stm.MKanan, pc2stm.MBelakang);
  IC(snap.Ebelakang, snap.Ekanan, snap.EKiri);
}

MotionTarget nucleo::resolveTarget(const rclcpp::Time & t_now, bool & timed_out)
{
  std::lock_guard<std::mutex> lk(cmd_mtx);
  MotionTarget target{};
  timed_out = false;
  if ((t_now - last_cmd_time).seconds() < cmd_vel_timeout) {
    target = cmd_vel;
  } else {
    timed_out = true;
  }
  if (joy_ready && (t_now - joy_time).seconds() < joy_timeout) {
    target = joy_vel;
    timed_out = false;
  }
  if (test_ready && (t_now - test_time).seconds() < test_timeout) {
    target = test_vel;
    timed_out = false;
  }
  return target;
}

double nucleo::slewToward(double cur, double tgt, double max_delta)
{
  double d = tgt - cur;
  if (d > max_delta) {d = max_delta;}
  if (d < -max_delta) {d = -max_delta;}
  return cur + d;
}

MotionTarget nucleo::applySmoothing(const MotionTarget & target, double dt, bool timed_out)
{
  if (!smoothing_enable || dt <= 0.0) {
    return target;
  }
  if (dt > 0.1) {dt = 0.1;}
  double lin_step = linear_accel_limit * dt;
  double ang_step = angular_accel_limit * dt;
  if (timed_out) {
    lin_step *= stop_decel_scale;
    ang_step *= stop_decel_scale;
  }
  MotionTarget out = smoothed_;
  out.vx = slewToward(smoothed_.vx, target.vx, lin_step);
  out.vy = slewToward(smoothed_.vy, target.vy, lin_step);
  out.w = slewToward(smoothed_.w, target.w, ang_step);
  if (target.vx == 0.0 && std::fabs(out.vx) < 0.01) {out.vx = 0.0;}
  if (target.vy == 0.0 && std::fabs(out.vy) < 0.01) {out.vy = 0.0;}
  if (target.w == 0.0 && std::fabs(out.w) < 0.02) {out.w = 0.0;}
  smoothed_ = out;
  return out;
}

double nucleo::calcDt(const rclcpp::Time & t_now)
{
  double dt = (t_now - last_time).seconds();
  if (dt <= 0) {
    dt = 1.0 / loop_hz;
  }
  last_time = t_now;
  return dt;
}

void nucleo::updateOdom(const struct STM32toPC & stm2pc, double dt, const rclcpp::Time & t_now)
{
  double stm_raw = normalizeDeg(static_cast<double>(stm2pc.Yaw) + yaw_offset_deg);
  double ext_copy = 0.0;
  bool ext_fresh = false;
  {
    std::lock_guard<std::mutex> lk(cmd_mtx);
    if (use_yaw_topic && ext_yaw_ready &&
      (t_now - ext_yaw_time).seconds() < yaw_topic_timeout)
    {
      ext_copy = normalizeDeg(ext_yaw_deg + yaw_offset_deg);
      ext_fresh = true;
    }
  }
  if (!have_odom_yaw_) {
    odom_yaw_deg_ = 0.0;
    have_odom_yaw_ = true;
    last_stm_raw_ = stm_raw;
    have_last_stm_ = true;
    if (ext_fresh) {
      last_ext_raw_ = ext_copy;
      have_last_ext_ = true;
    }
  } else if (use_yaw_topic) {
    if (ext_fresh) {
      if (!have_last_ext_) {
        last_ext_raw_ = ext_copy;
        have_last_ext_ = true;
      } else {
        double d = normalizeDeg(ext_copy - last_ext_raw_);
        if (std::fabs(d) < 30.0) {
          odom_yaw_deg_ = normalizeDeg(odom_yaw_deg_ + d);
        }
        last_ext_raw_ = ext_copy;
      }
      last_stm_raw_ = stm_raw;
      have_last_stm_ = true;
    } else {
      have_last_ext_ = false;
      if (!have_last_stm_) {
        last_stm_raw_ = stm_raw;
        have_last_stm_ = true;
      } else {
        double d = normalizeDeg(stm_raw - last_stm_raw_);
        if (std::fabs(d) < 30.0) {
          odom_yaw_deg_ = normalizeDeg(odom_yaw_deg_ + d);
        }
        last_stm_raw_ = stm_raw;
      }
    }
  } else {
    have_last_ext_ = false;
    if (!have_last_stm_) {
      last_stm_raw_ = stm_raw;
      have_last_stm_ = true;
    } else {
      double d = normalizeDeg(stm_raw - last_stm_raw_);
      if (std::fabs(d) < 30.0) {
        odom_yaw_deg_ = normalizeDeg(odom_yaw_deg_ + d);
      }
      last_stm_raw_ = stm_raw;
    }
  }
  kinematic->pulse[0] = stm2pc.EKiri;
  kinematic->pulse[1] = stm2pc.Ebelakang;
  kinematic->pulse[2] = stm2pc.Ekanan;
  kinematic->setImu(odom_yaw_deg_);
  if (stm2pc.PushButton) {
    kinematic->setInitPos(0, 0, 0);
    odom_yaw_deg_ = 0.0;
    last_stm_raw_ = stm_raw;
    have_last_stm_ = true;
    if (ext_fresh) {
      last_ext_raw_ = ext_copy;
      have_last_ext_ = true;
    }
  }
  kinematic->odom(dt, true);
}

struct PCtoSTM32 nucleo::sendMotor(const MotionTarget & target)
{
  barelang::Vel2D vel_local;
  vel_local.x = target.vx * 100.0;
  vel_local.y = target.vy * 100.0;
  vel_local.theta = target.w;
  kinematic->inverse(vel_local);

  struct PCtoSTM32 pc2stm{};
  pc2stm.MKiri = radToPwm(kinematic->motor_out.v1);
  pc2stm.MBelakang = radToPwm(kinematic->motor_out.v2);
  pc2stm.MKanan = radToPwm(kinematic->motor_out.v3);
  pc2stm.lifter_target = lifter_target_.load();
  stmSender.put(&pc2stm);
  return pc2stm;
}

int16_t nucleo::radToPwm(double rad_per_sec)
{
  double rpm = rad_per_sec * 60.0 / (2 * M_PI);
  return static_cast<int16_t>((rpm / rpm_max) * pwm_max * pwm_scale);
}

void nucleo::publishOdom(const rclcpp::Time & t_now)
{
  auto pose = kinematic->getPosition();
  auto vel = kinematic->getVelocity();
  double yaw_rad = pose.theta * M_PI / 180.0;

  nav_msgs::msg::Odometry odom;
  odom.header.stamp = t_now;
  odom.header.frame_id = odom_frame;
  odom.child_frame_id = base_frame;
  odom.pose.pose.position.x = pose.x;
  odom.pose.pose.position.y = pose.y;
  odom.pose.pose.orientation.z = std::sin(yaw_rad / 2);
  odom.pose.pose.orientation.w = std::cos(yaw_rad / 2);
  odom.pose.covariance[0] = 0.01;
  odom.pose.covariance[7] = 0.01;
  odom.pose.covariance[35] = 0.05;
  odom.twist.twist.linear.x = vel.x;
  odom.twist.twist.linear.y = vel.y;
  odom.twist.twist.angular.z = vel.theta;
  odom.twist.covariance[0] = 0.01;
  odom.twist.covariance[7] = 0.01;
  odom.twist.covariance[35] = 0.05;
  odom_pub->publish(odom);

  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = t_now;
  tf.header.frame_id = odom_frame;
  tf.child_frame_id = base_frame;
  tf.transform.translation.x = pose.x;
  tf.transform.translation.y = pose.y;
  tf.transform.rotation.z = odom.pose.pose.orientation.z;
  tf.transform.rotation.w = odom.pose.pose.orientation.w;
  tfBroadcaster->sendTransform(tf);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<nucleo>());
  rclcpp::shutdown();
  return 0;
}
