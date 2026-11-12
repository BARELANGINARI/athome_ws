#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace restaurant
{

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw_deg_or_rad)
{
  // Jika nilai lebih besar dari 2*pi, asumsikan input berupa derajat
  double yaw_rad = (std::abs(yaw_deg_or_rad) > 6.2831853) ? (yaw_deg_or_rad * M_PI / 180.0) : yaw_deg_or_rad;
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw_rad * 0.5);
  q.w = std::cos(yaw_rad * 0.5);
  return q;
}

class RestaurantNode : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  RestaurantNode()
  : Node("restaurant_node")
  {
    // Parameters
    declare_parameter("origin_x", 0.0);
    declare_parameter("origin_y", 0.0);
    declare_parameter("origin_yaw", 0.0);
    declare_parameter("global_frame", "map");
    declare_parameter("nav_timeout_sec", 120.0);

    origin_x_ = get_parameter("origin_x").as_double();
    origin_y_ = get_parameter("origin_y").as_double();
    origin_yaw_ = get_parameter("origin_yaw").as_double();
    global_frame_ = get_parameter("global_frame").as_string();
    nav_timeout_sec_ = get_parameter("nav_timeout_sec").as_double();

    // Callback groups
    srv_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // Nav2 Action Client
    nav_client_ = rclcpp_action::create_client<NavigateToPose>(
      this, "navigate_to_pose", action_cb_group_);

    // Service /homing
    srv_homing_ = create_service<std_srvs::srv::Trigger>(
      "/homing",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleHoming(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    // Service alias /restaurant/homing
    srv_homing_alias_ = create_service<std_srvs::srv::Trigger>(
      "/restaurant/homing",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleHoming(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[RestaurantNode]\033[0m Node initialized. Service /homing ready (Home: x=%.2f, y=%.2f, yaw=%.2f).",
      origin_x_, origin_y_, origin_yaw_);
  }

private:
  void handleHoming(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_navigating_.exchange(true)) {
      res->success = false;
      res->message = "Robot is currently busy navigating.";
      RCLCPP_WARN(get_logger(), "[Homing] %s", res->message.c_str());
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[Homing]\033[0m Service called. Navigating robot back to origin (x=%.2f, y=%.2f)...",
      origin_x_, origin_y_);

    bool ok = navigateToOrigin();
    is_navigating_ = false;

    res->success = ok;
    res->message = ok ? "Robot successfully arrived at homing position."
                      : "Failed or timed out navigating to homing position.";
  }

  bool navigateToOrigin()
  {
    if (!nav_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "[Homing] Nav2 action server 'navigate_to_pose' is unavailable!");
      return false;
    }

    NavigateToPose::Goal goal;
    goal.pose.header.frame_id = global_frame_;
    goal.pose.header.stamp = now();
    goal.pose.pose.position.x = origin_x_;
    goal.pose.pose.position.y = origin_y_;
    goal.pose.pose.position.z = 0.0;
    goal.pose.pose.orientation = yawToQuaternion(origin_yaw_);

    auto promise = std::make_shared<std::promise<rclcpp_action::ResultCode>>();
    auto future = promise->get_future();

    auto send_opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_opts.goal_response_callback = [this](const GoalHandleNav::SharedPtr & handle) {
      if (!handle) {
        RCLCPP_WARN(get_logger(), "[Homing] Nav2 rejected homing goal!");
      } else {
        current_goal_handle_ = handle;
        RCLCPP_INFO(get_logger(), "[Homing] Nav2 accepted homing goal. Driving to home...");
      }
    };
    send_opts.result_callback = [promise](const GoalHandleNav::WrappedResult & res) {
      promise->set_value(res.code);
    };

    auto send_future = nav_client_->async_send_goal(goal, send_opts);
    if (send_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready || !send_future.get()) {
      RCLCPP_ERROR(get_logger(), "[Homing] Failed sending goal to Nav2 action server!");
      return false;
    }

    if (future.wait_for(std::chrono::seconds(static_cast<int>(nav_timeout_sec_))) == std::future_status::ready) {
      const auto code = future.get();
      if (code == rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_INFO(get_logger(), "\033[1;32m[Homing] Arrived at home position!\033[0m");
        return true;
      } else {
        RCLCPP_WARN(
          get_logger(), "[Homing] Navigation failed or canceled (code %d).",
          static_cast<int>(code));
        return false;
      }
    } else {
      RCLCPP_ERROR(
        get_logger(), "[Homing] Navigation timed out after %.1f seconds. Canceling goal...",
        nav_timeout_sec_);
      if (current_goal_handle_) {
        nav_client_->async_cancel_goal(current_goal_handle_);
      }
      return false;
    }
  }

  double origin_x_{0.0};
  double origin_y_{0.0};
  double origin_yaw_{0.0};
  std::string global_frame_{"map"};
  double nav_timeout_sec_{120.0};

  std::atomic<bool> is_navigating_{false};
  GoalHandleNav::SharedPtr current_goal_handle_;

  rclcpp::CallbackGroup::SharedPtr srv_cb_group_;
  rclcpp::CallbackGroup::SharedPtr action_cb_group_;

  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_homing_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_homing_alias_;
};

}  // namespace restaurant

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<restaurant::RestaurantNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
