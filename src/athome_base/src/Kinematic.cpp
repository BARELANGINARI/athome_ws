#include "Kinematic.hpp"
#include <cmath>
#include <cstdint>

namespace barelang
{
Kinematic::Kinematic(double ppr, double r_wheel_cm, double base_cm)
  : ppr_(ppr), r_wheel_cm_(r_wheel_cm), base_cm_(base_cm)
{
}

void Kinematic::inverse(const Vel2D &v)
{
  motor_out.v1 = ((-sqrt(3) * v.x / 2) + (v.y / 2) + base_cm_ * v.theta) / r_wheel_cm_;
  motor_out.v2 = (-v.y + base_cm_ * v.theta) / r_wheel_cm_;
  motor_out.v3 = ((sqrt(3) * v.x / 2) + (v.y / 2) + base_cm_ * v.theta) / r_wheel_cm_;
}

void Kinematic::odom(double dt, bool use_imu)
{
  if (!initialized_) {
    prev_pulse = pulse;
    last_position = position;
    initialized_ = true;
    return;
  }
  if (dt <= 0)
    dt = 0.033;

  double v[3];
  for (int i = 0; i < 3; i++) {
    const int64_t cur = static_cast<int64_t>(pulse[i]);
    const int64_t prv = static_cast<int64_t>(prev_pulse[i]);
    int64_t delta = cur - prv;

    if (delta < -2147483648LL) {
      delta += 4294967296LL;
    }

    const double ticks_per_cm = ppr_ / (2 * M_PI * r_wheel_cm_);
    const double max_cm = 1.0 * dt * 3.0 * 100.0;   // m -> cm, margin 3x
    const int64_t max_tick = static_cast<int64_t>(max_cm * ticks_per_cm) + 50;
    if (delta > max_tick || delta < -max_tick) {
      delta = 0;  
    }

    const double tick = static_cast<double>(delta);
    v[i] = (tick / ppr_) * (2 * M_PI * r_wheel_cm_);
  }

  const double vel_x = ((-sqrt(3) * v[0]) + (sqrt(3) * v[2])) / 3;
  const double vel_y = ((-2 * v[1]) + v[0] + v[2]) / 3;
  double step_theta_deg = ((v[0] + v[1] + v[2]) / (3 * base_cm_)) * 180 / M_PI;

  double radian = fmod(position.theta * M_PI / 180.0, M_PI * 2);
  if (radian < 0)
    radian += M_PI * 2;

  double gx = (std::cos(radian) * vel_x) - (std::sin(radian) * vel_y);
  double gy = (std::cos(radian) * vel_y) + (std::sin(radian) * vel_x);

  position.x += (gx / 100.0);
  position.y += (gy / 100.0);

  if (!use_imu) {
    position.theta = radian * 180 / M_PI;
    position.theta += (step_theta_deg);
  }
  vel_filtered.x = (position.x - last_position.x) / dt;
  vel_filtered.y = (position.y - last_position.y) / dt;
  double dtheta_deg = position.theta - last_position.theta;
  while (dtheta_deg > 180.0) {
    dtheta_deg -= 360.0;
  }
  while (dtheta_deg <= -180.0) {
    dtheta_deg += 360.0;
  }
  vel_filtered.theta = (dtheta_deg * M_PI / 180.0) / dt;

  last_position = position;
  prev_pulse = pulse;
}
}
