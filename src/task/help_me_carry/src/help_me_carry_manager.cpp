#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;

enum class TaskState {
  INIT,
  WAIT_WAKEWORD,
  WAIT_START_COMMAND,
  FOLLOW_OPERATOR,
  RECOVER_OPERATOR,
  WAIT_BAG_POINTING,
  GRASP_BAG,
  RETURN_TO_START,
  COMPLETED,
  ABORTED
};

class HelpMeCarryOrchestrator : public rclcpp::Node {
public:
  using Trigger = std_srvs::srv::Trigger;
  using TriggerClient = rclcpp::Client<Trigger>;
  using TriggerResult = std::function<void(bool, const std::string &)>;

  HelpMeCarryOrchestrator()
  : Node("help_me_carry_orchestrator")
  {
    callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions options;
    options.callback_group = callback_group_;

    tts_pub_ = create_publisher<std_msgs::msg::String>("/hri/tts_say", 10);
    tts_done_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/tts_done", 10,
      std::bind(&HelpMeCarryOrchestrator::onTtsDone, this, std::placeholders::_1),
      options);
    stt_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/stt_transcript", 10,
      std::bind(&HelpMeCarryOrchestrator::onTranscript, this, std::placeholders::_1),
      options);
    emergency_sub_ = create_subscription<std_msgs::msg::String>(
      "/stt/keyword_interrupt", 10,
      std::bind(&HelpMeCarryOrchestrator::onTranscript, this, std::placeholders::_1),
      options);
    person_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/person_label", rclcpp::SensorDataQoS(),
      std::bind(&HelpMeCarryOrchestrator::onPersonLabel, this, std::placeholders::_1),
      options);

    follow_start_client_ = create_client<Trigger>(
      "/follow/person", rclcpp::ServicesQoS(), callback_group_);
    follow_stop_client_ = create_client<Trigger>(
      "/follow/stop", rclcpp::ServicesQoS(), callback_group_);
    align_bag_client_ = create_client<Trigger>(
      "/task/align_bag", rclcpp::ServicesQoS(), callback_group_);
    grab_bag_client_ = create_client<Trigger>(
      "/arm/ambil_tas", rclcpp::ServicesQoS(), callback_group_);
    homing_client_ = create_client<Trigger>(
      "/homing", rclcpp::ServicesQoS(), callback_group_);

    fsm_timer_ = create_wall_timer(
      100ms, std::bind(&HelpMeCarryOrchestrator::fsmTick, this), callback_group_);

    task_start_time_ = now();
    state_enter_time_ = now();
    RCLCPP_INFO(get_logger(), "Help Me Carry orchestrator ready.");
  }

private:
  static constexpr double GLOBAL_TIMEOUT_SEC = 300.0;
  static constexpr double OPERATOR_TIMEOUT_SEC = 3.0;

  std::atomic<TaskState> state_{TaskState::INIT};
  std::atomic<bool> busy_{false};
  std::atomic<bool> speaking_{false};
  std::atomic<bool> wakeword_requested_{false};
  std::atomic<bool> follow_requested_{false};
  std::atomic<bool> arrived_requested_{false};
  std::atomic<bool> emergency_stop_requested_{false};
  std::atomic<bool> operator_seen_{false};

  rclcpp::Time task_start_time_;
  rclcpp::Time state_enter_time_;
  rclcpp::Time last_operator_seen_time_;
  rclcpp::Time tts_cooldown_until_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_started_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_command_time_{0, 0, RCL_ROS_TIME};

  std::mutex data_mutex_;
  std::string last_command_;
  std::function<void()> pending_tts_done_;
  bool bag_prompt_sent_{false};
  bool recovery_prompt_sent_{false};

  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::TimerBase::SharedPtr fsm_timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tts_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr tts_done_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr stt_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr emergency_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_sub_;
  TriggerClient::SharedPtr follow_start_client_;
  TriggerClient::SharedPtr follow_stop_client_;
  TriggerClient::SharedPtr align_bag_client_;
  TriggerClient::SharedPtr grab_bag_client_;
  TriggerClient::SharedPtr homing_client_;

  void transitionTo(TaskState next_state, const std::string & reason)
  {
    RCLCPP_INFO(
      get_logger(), "[FSM] %d -> %d: %s",
      static_cast<int>(state_.load()), static_cast<int>(next_state), reason.c_str());
    state_enter_time_ = now();
    bag_prompt_sent_ = false;
    recovery_prompt_sent_ = false;
    state_ = next_state;
  }

  void speakAsync(const std::string & text, std::function<void()> on_done = nullptr)
  {
    std_msgs::msg::String message;
    message.data = text;
    tts_pub_->publish(message);
    RCLCPP_INFO(get_logger(), "[TTS] %s", text.c_str());

    speaking_ = true;
    busy_ = true;
    tts_started_time_ = now();
    tts_cooldown_until_ = now() + rclcpp::Duration::from_seconds(0.5);
    pending_tts_done_ = std::move(on_done);
  }

  void onTtsDone(const std_msgs::msg::String::SharedPtr = nullptr)
  {
    speaking_ = false;
    busy_ = false;
    tts_cooldown_until_ = now() + rclcpp::Duration::from_seconds(0.4);

    auto callback = std::move(pending_tts_done_);
    pending_tts_done_ = nullptr;
    if (callback) {
      callback();
    }
  }

  bool canListen() const
  {
    return !speaking_.load() && now() >= tts_cooldown_until_;
  }

  static std::string normalize(std::string text)
  {
    std::transform(
      text.begin(), text.end(), text.begin(),
      [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
      });
    return text;
  }


  // Wakeword sebelum mengoperasikan robot
  static bool matchWakeword(const std::string & text)
  {
    static const std::vector<std::string> corpus = {
      "hello robot", "halo robot", "hey robot", "hello barelang", "hi robot", "robot"
    };
    for (const auto & phrase : corpus) {
      if (text.find(phrase) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  static bool matchFollow(const std::string & text)
  {
    static const std::vector<std::string> corpus = {
      "follow me", "please follow me", "come with me", "lets go follow me",
      "start following me", "guide me follow me", "following me", "follow"
    };
    for (const auto & phrase : corpus) {
      if (text.find(phrase) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  static bool matchArrived(const std::string & text)
  {
    static const std::vector<std::string> corpus = {
      "arrived at the location", "we have arrived", "this is the location",
      "reached the car", "arrived at the car", "stop here we arrived",
      "i have arrived", "we are here", "here is the car", "arrived"
    };
    for (const auto & phrase : corpus) {
      if (text.find(phrase) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  void onTranscript(const std_msgs::msg::String::SharedPtr message)
  {
    if (!canListen()) {
      return;
    }

    const std::string command = normalize(message->data);
    if (command.empty()) {
      return;
    }

    const auto current_time = now();
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (command == last_command_ &&
        (current_time - last_command_time_).seconds() < 2.0)
      {
        return;
      }
      last_command_ = command;
      last_command_time_ = current_time;
    }

    const TaskState current_state = state_.load();
    if (matchWakeword(command) && current_state == TaskState::WAIT_WAKEWORD) {
      wakeword_requested_ = true;
    }
    if (matchFollow(command) && current_state == TaskState::WAIT_START_COMMAND) {
      follow_requested_ = true;
    }
    if (matchArrived(command) && current_state == TaskState::FOLLOW_OPERATOR) {
      arrived_requested_ = true;
    }
    if (command.find("stop") != std::string::npos &&
      current_state == TaskState::FOLLOW_OPERATOR)
    {
      emergency_stop_requested_ = true;
    }
  }

  void onPersonLabel(const std_msgs::msg::String::SharedPtr message)
  {
    if (message->data.find("NO_PERSON") == std::string::npos &&
      (message->data.find("person:") != std::string::npos ||
      message->data.find("operator:") != std::string::npos))
    {
      operator_seen_ = true;
      last_operator_seen_time_ = now();
    }
  }

  void callTrigger(
    const TriggerClient::SharedPtr & client,
    const std::string & service_name,
    TriggerResult result)
  {
    if (!client->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "Service %s belum siap.", service_name.c_str());
      result(false, "Service not ready");
      return;
    }

    auto request = std::make_shared<Trigger::Request>();
    client->async_send_request(
      request,
      [result](TriggerClient::SharedFuture future) {
        try {
          const auto response = future.get();
          result(response->success, response->message);
        } catch (const std::exception & exception) {
          result(false, exception.what());
        }
      });
  }

  void stopFollowing(std::function<void(bool)> on_done)
  {
    callTrigger(
      follow_stop_client_, "/follow/stop",
      [on_done](bool success, const std::string &) { on_done(success); });
  }

  void fsmTick()
  {
    const TaskState current_state = state_.load();
    if ((now() - task_start_time_).seconds() > GLOBAL_TIMEOUT_SEC &&
      current_state != TaskState::COMPLETED && current_state != TaskState::ABORTED)
    {
      RCLCPP_ERROR(get_logger(), "Global task timeout reached.");
      state_ = TaskState::ABORTED;
      stopFollowing([](bool) {});
      return;
    }

    if (current_state == TaskState::FOLLOW_OPERATOR && operator_seen_.load() &&
      (now() - last_operator_seen_time_).seconds() > OPERATOR_TIMEOUT_SEC)
    {
      operator_seen_ = false;
      transitionTo(TaskState::RECOVER_OPERATOR, "Operator lost");
    }

    if (emergency_stop_requested_.exchange(false)) {
      busy_ = true;
      stopFollowing([this](bool) {
        busy_ = false;
        speakAsync("Okay, stopping.");
        transitionTo(TaskState::WAIT_START_COMMAND, "Emergency stop");
      });
      return;
    }

    // Safety watchdog: jika /hri/tts_done tidak kunjung datang dalam 6 detik, lepas kuncian busy
    if (speaking_.load() && (now() - tts_started_time_).seconds() > 6.0) {
      RCLCPP_WARN(get_logger(), "TTS timeout (no /hri/tts_done received), resetting busy state.");
      onTtsDone(nullptr);
    }

    if (busy_.load()) {
      return;
    }

    switch (state_.load()) {
      case TaskState::INIT:
        if (tts_pub_->get_subscription_count() == 0) {
          // Tunggu sampai tts_engine_node siap mendengarkan sebelum aktif
          return;
        }
        transitionTo(TaskState::WAIT_WAKEWORD, "TTS ready, waiting for wakeword");
        break;

      case TaskState::WAIT_WAKEWORD:
        if (wakeword_requested_.exchange(false)) {
          speakAsync(
            "Hello, I'm Barelang Inari, I want to help you to carry the luggage. "
            "If you are ready, please guide me to the place, and say follow me.",
            [this]() { transitionTo(TaskState::WAIT_START_COMMAND, "Wakeword acknowledged"); });
        }
        break;

      case TaskState::WAIT_START_COMMAND:
        if (follow_requested_.exchange(false)) {
          busy_ = true;
          callTrigger(
            follow_start_client_, "/follow/person",
            [this](bool success, const std::string & message) {
              busy_ = false;
              if (!success) {
                RCLCPP_ERROR(get_logger(), "Follow gagal: %s", message.c_str());
                speakAsync("Person following is not ready.");
                return;
              }
              transitionTo(TaskState::FOLLOW_OPERATOR, "Follow started");
              speakAsync("Okay, I am following you.");
            });
        }
        break;

      case TaskState::FOLLOW_OPERATOR:
        if (arrived_requested_.exchange(false)) {
          busy_ = true;
          stopFollowing([this](bool success) {
            busy_ = false;
            if (!success) {
              transitionTo(TaskState::ABORTED, "Could not stop following");
              return;
            }
            transitionTo(TaskState::WAIT_BAG_POINTING, "Arrival command");
            speakAsync("Please point to the bag you want me to carry.");
          });
        }
        break;

      case TaskState::RECOVER_OPERATOR:
        if (!recovery_prompt_sent_) {
          recovery_prompt_sent_ = true;
          speakAsync("I lost sight of you. Please call me or wave your hand.");
        } else if (operator_seen_.load()) {
          busy_ = true;
          callTrigger(
            follow_start_client_, "/follow/person",
            [this](bool success, const std::string & message) {
              busy_ = false;
              if (success) {
                transitionTo(TaskState::FOLLOW_OPERATOR, "Operator reacquired");
              } else {
                RCLCPP_WARN(get_logger(), "Recovery follow gagal: %s", message.c_str());
              }
            });
        } else if ((now() - state_enter_time_).seconds() > 15.0) {
          transitionTo(TaskState::ABORTED, "Operator recovery timeout");
        }
        break;

      case TaskState::WAIT_BAG_POINTING:
        if (!bag_prompt_sent_) {
          bag_prompt_sent_ = true;
          speakAsync("Which bag should I take?");
          break;
        }
        busy_ = true;
        callTrigger(
          align_bag_client_, "/task/align_bag",
          [this](bool success, const std::string & message) {
            busy_ = false;
            if (success) {
              transitionTo(TaskState::GRASP_BAG, "Bag aligned");
            } else {
              RCLCPP_WARN(get_logger(), "Align bag gagal: %s", message.c_str());
              bag_prompt_sent_ = false;
            }
          });
        break;

      case TaskState::GRASP_BAG:
        busy_ = true;
        callTrigger(
          grab_bag_client_, "/arm/ambil_tas",
          [this](bool success, const std::string & message) {
            busy_ = false;
            if (success) {
              transitionTo(TaskState::RETURN_TO_START, "Bag grasped");
              speakAsync("I have the bag. I am returning to the starting point.");
            } else {
              RCLCPP_ERROR(get_logger(), "Grasp gagal: %s", message.c_str());
              transitionTo(TaskState::ABORTED, "Bag grasp failed");
            }
          });
        break;

      case TaskState::RETURN_TO_START:
        busy_ = true;
        callTrigger(
          homing_client_, "/homing",
          [this](bool success, const std::string & message) {
            busy_ = false;
            if (success) {
              transitionTo(TaskState::COMPLETED, "Homing completed");
              speakAsync("I have returned to the starting point. Task completed.");
            } else {
              RCLCPP_ERROR(get_logger(), "Homing gagal: %s", message.c_str());
              transitionTo(TaskState::ABORTED, "Homing failed");
            }
          });
        break;

      case TaskState::COMPLETED:
      case TaskState::ABORTED:
        break;
    }
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HelpMeCarryOrchestrator>();
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
