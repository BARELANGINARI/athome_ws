#include <chrono>
#include <iostream>
#include <memory>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_srvs/srv/trigger.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("return_to_origin_cli");

  std::cout << "\033[96m[INFO] Returning robot to starting pose (origin: x=0.0, y=0.0, yaw=0.0)...\033[0m" << std::endl;

  auto srv_client = node->create_client<std_srvs::srv::Trigger>("/return_to_origin");
  if (srv_client->wait_for_service(std::chrono::milliseconds(1500))) {
    std::cout << "\033[92m[INFO] Calling /return_to_origin service...\033[0m" << std::endl;
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto future = srv_client->async_send_request(req);

    if (rclcpp::spin_until_future_complete(node, future) == rclcpp::FutureReturnCode::SUCCESS) {
      auto res = future.get();
      if (res && res->success) {
        std::cout << "\033[92m[SUCCESS] " << res->message << "\033[0m" << std::endl;
        rclcpp::shutdown();
        return 0;
      } else {
        std::string msg = res ? res->message : "Empty service response";
        std::cout << "\033[91m[FAILED] " << msg << "\033[0m" << std::endl;
        rclcpp::shutdown();
        return 1;
      }
    }
  }

  std::cout << "\033[93m[INFO] Service /return_to_origin unavailable. Sending goal to Nav2 action server...\033[0m" << std::endl;
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  auto action_client = rclcpp_action::create_client<NavigateToPose>(node, "navigate_to_pose");
  if (!action_client->wait_for_action_server(std::chrono::seconds(5))) {
    std::cout << "\033[91m[ERROR] Action server 'navigate_to_pose' inactive! Ensure Nav2 is running.\033[0m" << std::endl;
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

  std::cout << "\033[96m[INFO] Sending origin goal to Nav2...\033[0m" << std::endl;
  auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();

  auto send_goal_future = action_client->async_send_goal(goal, send_goal_options);
  if (rclcpp::spin_until_future_complete(node, send_goal_future) != rclcpp::FutureReturnCode::SUCCESS) {
    std::cout << "\033[91m[ERROR] Failed to send goal to action server!\033[0m" << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  GoalHandleNav::SharedPtr goal_handle = send_goal_future.get();
  if (!goal_handle) {
    std::cout << "\033[91m[ERROR] Nav2 rejected origin goal!\033[0m" << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  std::cout << "\033[92m[INFO] Goal accepted by Nav2. Navigating to origin...\033[0m" << std::endl;
  auto result_future = action_client->async_get_result(goal_handle);
  if (rclcpp::spin_until_future_complete(node, result_future) != rclcpp::FutureReturnCode::SUCCESS) {
    std::cout << "\033[91m[ERROR] Failed waiting for navigation result from Nav2!\033[0m" << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  auto wrapped_result = result_future.get();
  int ret_code = 1;
  switch (wrapped_result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      std::cout << "\033[92m[SUCCESS] Robot arrived at origin!\033[0m" << std::endl;
      ret_code = 0;
      break;
    case rclcpp_action::ResultCode::CANCELED:
      std::cout << "\033[93m[WARNING] Return-to-origin navigation canceled.\033[0m" << std::endl;
      ret_code = 1;
      break;
    default:
      std::cout << "\033[91m[FAILED] Return-to-origin navigation failed with status code "
                << static_cast<int>(wrapped_result.code) << ".\033[0m" << std::endl;
      ret_code = 1;
      break;
  }

  rclcpp::shutdown();
  return ret_code;
}
