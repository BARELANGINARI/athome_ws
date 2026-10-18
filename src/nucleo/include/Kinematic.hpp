#ifndef KINEMATIC_HPP
#define KINEMATIC_HPP

#include <chrono>
#include <vector>

namespace barelang
{
struct Vel2D
{
  double x{0};
  double y{0};
  double theta{0};
};
struct Pose2D
{
  double x{0};
  double y{0};
  double theta{0};
};
struct WheelOut
{
  double v1{0}, v2{0}, v3{0};
};
class Kinematic
{
public:
  Kinematic(double ppr = 7500.0, double r_wheel_cm = 5.0, double base_cm = 20.8);
  void odom(double dt, bool use_imu = true);
  Pose2D getPosition()
  {
    return position;
  }
  Vel2D getVelocity()
  {
    return vel_filtered;
  }
  void setInitPos(double x, double y, double theta_deg)
  {
    position.x = x;
    position.y = y;
    position.theta = theta_deg;
    last_position = position;
  }

  void setImu(double theta_deg)
  {
    position.theta = theta_deg;
  }

  void inverse(const Vel2D &vel_local_cm);
  WheelOut motor_out;

  std::vector<int> pulse{0, 0, 0};

private:
  double ppr_, r_wheel_cm_, base_cm_;
  std::vector<int> prev_pulse{0, 0, 0};
  bool initialized_{false};
  Pose2D position;
  Pose2D last_position;
  Vel2D vel_filtered;
};
}

#endif
