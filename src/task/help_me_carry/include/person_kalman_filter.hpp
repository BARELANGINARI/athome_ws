#pragma once

#include <cmath>
#include <algorithm>

namespace help_me_carry
{

class PersonKalmanFilter
{
public:
  explicit PersonKalmanFilter(double process_noise = 1.2, double measurement_noise = 0.15)
  : q_var_(process_noise),
    r_var_(measurement_noise * measurement_noise)
  {
    reset();
  }

  void reset()
  {
    is_initialized_ = false;
    x_ = 0.0;
    y_ = 0.0;
    vx_ = 0.0;
    vy_ = 0.0;

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        p_[i][j] = 0.0;
      }
    }
    p_[0][0] = 0.2;
    p_[1][1] = 0.2;
    p_[2][2] = 1.0;
    p_[3][3] = 1.0;
  }

  void init(double zx, double zy)
  {
    x_ = zx;
    y_ = zy;
    vx_ = 0.0;
    vy_ = 0.0;

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        p_[i][j] = 0.0;
      }
    }
    p_[0][0] = 0.05;
    p_[1][1] = 0.05;
    p_[2][2] = 0.5;
    p_[3][3] = 0.5;
    is_initialized_ = true;
  }

  void predict(double dt)
  {
    if (!is_initialized_) {
      return;
    }
    if (dt <= 0.0) {
      dt = 0.033;
    }
    if (dt > 1.5) {
      dt = 1.5; 
    }

    x_ += vx_ * dt;
    y_ += vy_ * dt;
    vx_ *= 0.95;
    vy_ *= 0.95;

    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;

    const double q_pos = 0.25 * dt4 * q_var_;
    const double q_pos_vel = 0.5 * dt3 * q_var_;
    const double q_vel = dt2 * q_var_;

    const double p00 = p_[0][0] + dt * (p_[2][0] + p_[0][2]) + dt2 * p_[2][2] + q_pos;
    const double p01 = p_[0][1] + dt * p_[2][1] + dt * p_[0][3] + dt2 * p_[2][3];
    const double p02 = p_[0][2] + dt * p_[2][2] + q_pos_vel;
    const double p03 = p_[0][3] + dt * p_[2][3];

    const double p10 = p_[1][0] + dt * p_[3][0] + dt * p_[1][2] + dt2 * p_[3][2];
    const double p11 = p_[1][1] + dt * (p_[3][1] + p_[1][3]) + dt2 * p_[3][3] + q_pos;
    const double p12 = p_[1][2] + dt * p_[3][2];
    const double p13 = p_[1][3] + dt * p_[3][3] + q_pos_vel;

    const double p20 = p_[2][0] + dt * p_[2][2] + q_pos_vel;
    const double p21 = p_[2][1] + dt * p_[2][3];
    const double p22 = p_[2][2] + q_vel;
    const double p23 = p_[2][3];

    const double p30 = p_[3][0] + dt * p_[3][2];
    const double p31 = p_[3][1] + dt * p_[3][3] + q_pos_vel;
    const double p32 = p_[3][2];
    const double p33 = p_[3][3] + q_vel;

    p_[0][0] = p00; p_[0][1] = p01; p_[0][2] = p02; p_[0][3] = p03;
    p_[1][0] = p10; p_[1][1] = p11; p_[1][2] = p12; p_[1][3] = p13;
    p_[2][0] = p20; p_[2][1] = p21; p_[2][2] = p22; p_[2][3] = p23;
    p_[3][0] = p30; p_[3][1] = p31; p_[3][2] = p32; p_[3][3] = p33;
  }

  void update(double zx, double zy)
  {
    if (!is_initialized_) {
      init(zx, zy);
      return;
    }

    const double yx = zx - x_;
    const double yy = zy - y_;

    const double s00 = p_[0][0] + r_var_;
    const double s01 = p_[0][1];
    const double s10 = p_[1][0];
    const double s11 = p_[1][1] + r_var_;

    double det = s00 * s11 - s01 * s10;
    if (std::abs(det) < 1e-6) {
      det = 1e-6;
    }
    const double inv_s00 = s11 / det;
    const double inv_s01 = -s01 / det;
    const double inv_s10 = -s10 / det;
    const double inv_s11 = s00 / det;

    double k[4][2];
    for (int i = 0; i < 4; ++i) {
      k[i][0] = p_[i][0] * inv_s00 + p_[i][1] * inv_s10;
      k[i][1] = p_[i][0] * inv_s01 + p_[i][1] * inv_s11;
    }

    x_ += k[0][0] * yx + k[0][1] * yy;
    y_ += k[1][0] * yx + k[1][1] * yy;
    vx_ += k[2][0] * yx + k[2][1] * yy;
    vy_ += k[3][0] * yx + k[3][1] * yy;

    double p_new[4][4];
    for (int j = 0; j < 4; ++j) {
      p_new[0][j] = (1.0 - k[0][0]) * p_[0][j] - k[0][1] * p_[1][j];
      p_new[1][j] = -k[1][0] * p_[0][j] + (1.0 - k[1][1]) * p_[1][j];
      p_new[2][j] = p_[2][j] - k[2][0] * p_[0][j] - k[2][1] * p_[1][j];
      p_new[3][j] = p_[3][j] - k[3][0] * p_[0][j] - k[3][1] * p_[1][j];
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        p_[i][j] = p_new[i][j];
      }
    }
  }

  double get_gate_radius(double min_gate = 0.9, double max_gate = 2.2) const
  {
    if (!is_initialized_) {
      return min_gate;
    }
    const double unc = std::sqrt(std::max(0.0, p_[0][0] + p_[1][1]));
    return std::min(max_gate, std::max(min_gate, 2.2 * unc));
  }

  double distance_to(double px, double py) const
  {
    return std::hypot(px - x_, py - y_);
  }

  bool is_initialized() const { return is_initialized_; }
  double x() const { return x_; }
  double y() const { return y_; }
  double vx() const { return vx_; }
  double vy() const { return vy_; }
  double speed() const { return std::hypot(vx_, vy_); }

private:
  bool is_initialized_{false};
  double x_{0.0};
  double y_{0.0};
  double vx_{0.0};
  double vy_{0.0};
  double p_[4][4]{};
  double q_var_{1.2};
  double r_var_{0.0225};
};

} // namespace help_me_carry
