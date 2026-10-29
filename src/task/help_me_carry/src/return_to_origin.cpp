// CLI / Skrip C++ untuk memerintahkan robot kembali ke titik awal (origin: x=0, y=0, yaw=0).
//
// Cara panggil:
//     ros2 run help_me_carry return_to_origin
//
// Alur kerja:
// 1. Coba service /return_to_origin (disediakan oleh place_navigator jika aktif).
// 2. Jika service aktif, panggil dan tunggu hingga selesai.
// 3. Fallback: jika service tidak ada, langsung kirim action goal NavigateToPose ke Nav2
//    dengan koordinat x=0.0, y=0.0, yaw=0.0 pada frame "map".

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

  std::cout << "\033[96m[INFO] Memerintahkan robot kembali ke titik awal (origin: x=0.0, y=0.0, yaw=0.0)...\033[0m" << std::endl;

  // 1. Coba lewat service /return_to_origin (place_navigator)
  auto srv_client = node->create_client<std_srvs::srv::Trigger>("/return_to_origin");
  if (srv_client->wait_for_service(std::chrono::milliseconds(1500))) {
    std::cout << "\033[92m[INFO] Menghubungi service /return_to_origin...\033[0m" << std::endl;
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto future = srv_client->async_send_request(req);

    if (rclcpp::spin_until_future_complete(node, future) == rclcpp::FutureReturnCode::SUCCESS) {
      auto res = future.get();
      if (res && res->success) {
        std::cout << "\033[92m[SUKSES] " << res->message << "\033[0m" << std::endl;
        rclcpp::shutdown();
        return 0;
      } else {
        std::string msg = res ? res->message : "Respon service kosong";
        std::cout << "\033[91m[GAGAL] " << msg << "\033[0m" << std::endl;
        rclcpp::shutdown();
        return 1;
      }
    }
  }

  // 2. Fallback: langsung kirim goal NavigateToPose ke Nav2 action server
  std::cout << "\033[93m[INFO] Service /return_to_origin tidak terdeteksi. Mengirim goal langsung ke Nav2 action server...\033[0m" << std::endl;
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  auto action_client = rclcpp_action::create_client<NavigateToPose>(node, "navigate_to_pose");
  if (!action_client->wait_for_action_server(std::chrono::seconds(5))) {
    std::cout << "\033[91m[ERROR] Action server 'navigate_to_pose' tidak aktif! Pastikan Nav2 sedang berjalan.\033[0m" << std::endl;
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

  std::cout << "\033[96m[INFO] Mengirim goal origin ke Nav2...\033[0m" << std::endl;
  auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();

  auto send_goal_future = action_client->async_send_goal(goal, send_goal_options);
  if (rclcpp::spin_until_future_complete(node, send_goal_future) != rclcpp::FutureReturnCode::SUCCESS) {
    std::cout << "\033[91m[ERROR] Gagal mengirim goal ke action server!\033[0m" << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  GoalHandleNav::SharedPtr goal_handle = send_goal_future.get();
  if (!goal_handle) {
    std::cout << "\033[91m[ERROR] Nav2 menolak goal origin!\033[0m" << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  std::cout << "\033[92m[INFO] Goal diterima Nav2. Robot sedang bergerak menuju titik awal...\033[0m" << std::endl;
  auto result_future = action_client->async_get_result(goal_handle);
  if (rclcpp::spin_until_future_complete(node, result_future) != rclcpp::FutureReturnCode::SUCCESS) {
    std::cout << "\033[91m[ERROR] Gagal menunggu hasil navigasi dari Nav2!\033[0m" << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  auto wrapped_result = result_future.get();
  int ret_code = 1;
  switch (wrapped_result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      std::cout << "\033[92m[SUKSES] Robot berhasil tiba kembali di posisi awal start (origin)!\033[0m" << std::endl;
      ret_code = 0;
      break;
    case rclcpp_action::ResultCode::CANCELED:
      std::cout << "\033[93m[PERINGATAN] Navigasi kembali ke titik awal dibatalkan.\033[0m" << std::endl;
      ret_code = 1;
      break;
    default:
      std::cout << "\033[91m[GAGAL] Navigasi kembali ke titik awal gagal dengan kode status "
                << static_cast<int>(wrapped_result.code) << ".\033[0m" << std::endl;
      ret_code = 1;
      break;
  }

  rclcpp::shutdown();
  return ret_code;
}
