#include <chrono>
#include <memory>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_srvs/srv/trigger.hpp"

#include "icecream.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("return_to_origin_cli");

  auto srv_client = node->create_client<std_srvs::srv::Trigger>("/homing");
  if (!srv_client->wait_for_service(std::chrono::milliseconds(500))) {
    srv_client = node->create_client<std_srvs::srv::Trigger>("/return_to_origin");
  }

  if (srv_client->wait_for_service(std::chrono::milliseconds(1000))) {
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto future = srv_client->async_send_request(req);

    if (rclcpp::spin_until_future_complete(node, future) == rclcpp::FutureReturnCode::SUCCESS) {
      auto res = future.get();
      if (res) {
        IC(res->success, res->message);
        rclcpp::shutdown();
        return res->success ? 0 : 1;
      }
    }
  }

  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  auto action_client = rclcpp_action::create_client<NavigateToPose>(node, "navigate_to_pose");
  if (!action_client->wait_for_action_server(std::chrono::seconds(5))) {
    IC("Action server navigate_to_pose offline");
    rclcpp::shutdown();
    return 1;
  }

  NavigateToPose::Goal goal;
  goal.pose.header.frame_id = "map";
  goal.pose.header.stamp = node->now();
  goal.pose.pose.position.x = 0.0;
  goal.pose.pose.position.y = 0.0;
  goal.pose.pose.position.z = 0.0;
  goal.pose.pose.orientation.x = 0.0;
  goal.pose.pose.orientation.y = 0.0;
  goal.pose.pose.orientation.z = 0.0;
  goal.pose.pose.orientation.w = 1.0;

  auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();

  auto send_goal_future = action_client->async_send_goal(goal, send_goal_options);
  if (rclcpp::spin_until_future_complete(node, send_goal_future) != rclcpp::FutureReturnCode::SUCCESS) {
    IC("Failed to send goal to Nav2");
    rclcpp::shutdown();
    return 1;
  }

  GoalHandleNav::SharedPtr goal_handle = send_goal_future.get();
  if (!goal_handle) {
    IC("Goal rejected by Nav2");
    rclcpp::shutdown();
    return 1;
  }

  auto result_future = action_client->async_get_result(goal_handle);
  if (rclcpp::spin_until_future_complete(node, result_future) != rclcpp::FutureReturnCode::SUCCESS) {
    IC("Failed waiting for Nav2 result");
    rclcpp::shutdown();
    return 1;
  }

  auto wrapped_result = result_future.get();
  int ret_code = (wrapped_result.code == rclcpp_action::ResultCode::SUCCEEDED) ? 0 : 1;
  IC(static_cast<int>(wrapped_result.code));

  rclcpp::shutdown();
  return ret_code;
}
