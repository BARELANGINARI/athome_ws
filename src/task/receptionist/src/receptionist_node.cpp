#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace receptionist
{

struct Waypoint
{
  std::string name;
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw * 0.5);
  q.w = std::cos(yaw * 0.5);
  return q;
}

class ReceptionistNode : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  ReceptionistNode()
  : Node("receptionist_node"), is_navigating_(false)
  {
    // Declare parameters for frames and waypoints
    declare_parameter<std::string>("global_frame", "map");
    declare_parameter<std::string>("announce_topic", "/receptionist/status");

    // Door waypoint
    declare_parameter<double>("door.x", 0.0);
    declare_parameter<double>("door.y", 0.0);
    declare_parameter<double>("door.yaw", 0.0);

    // Living room waypoint
    declare_parameter<double>("living_room.x", 1.5);
    declare_parameter<double>("living_room.y", 0.0);
    declare_parameter<double>("living_room.yaw", 0.0);

    global_frame_ = get_parameter("global_frame").as_string();
    std::string announce_topic = get_parameter("announce_topic").as_string();

    status_pub_ = create_publisher<std_msgs::msg::String>(announce_topic, 10);
    nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");

    // Service: Go to Door
    srv_go_door_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/go_to_door",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleGoTo("door", res);
      });

    // Service: Go to Living Room
    srv_go_living_room_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/go_to_living_room",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleGoTo("living_room", res);
      });

    RCLCPP_INFO(get_logger(), "Receptionist Navigation Node ready. Services available:");
    RCLCPP_INFO(get_logger(), " - /receptionist/go_to_door");
    RCLCPP_INFO(get_logger(), " - /receptionist/go_to_living_room");
    publishStatus("Receptionist navigation node ready");
  }

  void publishStatus(const std::string & msg_str)
  {
    std_msgs::msg::String msg;
    msg.data = msg_str;
    status_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "[STATUS] %s", msg_str.c_str());
  }

  Waypoint getWaypoint(const std::string & prefix)
  {
    Waypoint wp;
    wp.name = prefix;
    wp.x = get_parameter(prefix + ".x").as_double();
    wp.y = get_parameter(prefix + ".y").as_double();
    wp.yaw = get_parameter(prefix + ".yaw").as_double();
    return wp;
  }

  void handleGoTo(const std::string & target_name, std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lock(nav_mutex_);
    if (is_navigating_) {
      res->success = false;
      res->message = "Robot is currently busy with another navigation task";
      publishStatus(res->message);
      return;
    }

    Waypoint wp = getWaypoint(target_name);
    publishStatus("Command received: Navigating to " + target_name + "...");
    is_navigating_ = true;

    bool ok = navigateTo(wp);
    is_navigating_ = false;

    res->success = ok;
    res->message = ok ? "Reached " + target_name : "Failed to reach " + target_name;
    publishStatus(res->message);
  }

  bool navigateTo(const Waypoint & wp)
  {
    if (!nav_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "Nav2 action server 'navigate_to_pose' unavailable");
      return false;
    }

    NavigateToPose::Goal goal;
    goal.pose.header.frame_id = global_frame_;
    goal.pose.header.stamp = now();
    goal.pose.pose.position.x = wp.x;
    goal.pose.pose.position.y = wp.y;
    goal.pose.pose.position.z = 0.0;
    goal.pose.pose.orientation = yawToQuaternion(wp.yaw);

    auto send_goal_future = nav_client_->async_send_goal(goal);
    if (send_goal_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Failed to send navigation goal to %s", wp.name.c_str());
      return false;
    }

    auto goal_handle = send_goal_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(get_logger(), "Navigation goal rejected by Nav2 for %s", wp.name.c_str());
      return false;
    }

    auto result_future = nav_client_->async_get_result(goal_handle);
    while (rclcpp::ok()) {
      if (result_future.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready) {
        auto wrapped_result = result_future.get();
        return wrapped_result.code == rclcpp_action::ResultCode::SUCCEEDED;
      }
    }
    return false;
  }

private:
  std::string global_frame_;
  bool is_navigating_;
  std::mutex nav_mutex_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_door_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_living_room_;
};

}  // namespace receptionist

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<receptionist::ReceptionistNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
