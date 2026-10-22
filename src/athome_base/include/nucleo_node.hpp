#ifndef NUCLEO_NUCLEO_NODE_HPP
#define NUCLEO_NUCLEO_NODE_HPP

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>

#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "tf2_ros/transform_broadcaster.h"

#include "Kinematic.hpp"
#include "udpSender.h"
#include "udpServer.h"

struct MotionTarget
{
  double vx{0};
  double vy{0};
  double w{0};
};

class nucleo : public rclcpp::Node
{
public:
  nucleo();

private:
  void loadParam();
  void setupHardware();
  void setupRos();
  void onCmdVel(const geometry_msgs::msg::Twist::SharedPtr msg);
  void onJoyVel(const geometry_msgs::msg::Twist::SharedPtr msg);
  void onTestVel(const geometry_msgs::msg::Twist::SharedPtr msg);
  void onYawTopic(const std_msgs::msg::Float64::SharedPtr msg);
  static double normalizeDeg(double deg);
  void controlLoop();
  MotionTarget resolveTarget(const rclcpp::Time & t_now, bool & timed_out);
  static double slewToward(double cur, double tgt, double max_delta);
  MotionTarget applySmoothing(const MotionTarget & target, double dt, bool timed_out);
  double calcDt(const rclcpp::Time & t_now);
  void updateOdom(const struct STM32toPC & stm2pc, double dt, const rclcpp::Time & t_now);
  struct PCtoSTM32 sendMotor(const MotionTarget & target);
  int16_t radToPwm(double rad_per_sec);
  void publishOdom(const rclcpp::Time & t_now);

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub;
  // CMD_JOY ala NucleoServer: override bila fresh (<joy_timeout).
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr joy_sub;
  // STM_TO_TEST ala NucleoServer: override uji manual bila fresh.
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr test_sub;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr yaw_sub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
  rclcpp::TimerBase::SharedPtr timer;

  std::mutex cmd_mtx;
  MotionTarget cmd_vel;
  MotionTarget joy_vel;
  MotionTarget test_vel;
  rclcpp::Time joy_time;
  rclcpp::Time test_time;
  bool joy_ready{false};
  bool test_ready{false};
  std::string joy_topic{"joy_vel"};
  std::string test_topic{"test_vel"};
  double joy_timeout{0.05};
  double test_timeout{0.5};
  MotionTarget smoothed_;
  double cmd_vel_timeout{0.5};
  bool smoothing_enable{true};
  double linear_accel_limit{2.5};
  double angular_accel_limit{7.0};
  double stop_decel_scale{2.0};
  double pulses_per_rev_{7500.0};
  double wheel_radius_cm_{5.0};
  double base_length_cm_{20.8};
  double ext_yaw_deg{0};
  bool ext_yaw_ready{false};
  rclcpp::Time ext_yaw_time;
  double odom_yaw_deg_{0};
  bool have_odom_yaw_{false};
  double last_ext_raw_{0};
  bool have_last_ext_{false};
  double last_stm_raw_{0};
  bool have_last_stm_{false};

  std::string stm_ip, odom_frame, base_frame, yaw_topic, local_ip;
  int stm_port{2002}, local_port{2023};
  double loop_hz{33.33}, rpm_max{428}, pwm_max{225}, pwm_scale{0.932};
  double yaw_offset_deg{0.0};
  double yaw_topic_timeout{0.5};
  bool use_yaw_topic{false};

  rclcpp::Time last_time;
  rclcpp::Time last_cmd_time;
};

#endif  // NUCLEO_NUCLEO_NODE_HPP
