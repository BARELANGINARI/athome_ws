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
    // Hitung selisih counter encoder dengan aman terhadap overflow/wrap.
    //
    // BUG LAMA (bikin posisi meleset acak):
    //   double tick = static_cast<double>(pulse[i] - prev_pulse[i]);
    // `pulse[i] - prev_pulse[i]` dievaluasi sebagai int (32-bit) DULUAN
    // baru dikonversi ke double. Kalau counter STM32 wrap dari 2^31-1 ke
    // negatif, pengurangan itu overflow signed int -> perilaku tidak
    // undefined, hasilnya bisa melompat jauh. Satu lompatan saja sudah
    // cukup membuat odom melenceng jauh dan AMCL ikut bingung.
    //
    // Dipakai double sebelum pengurangan supaya tidak ada overflow, dan
    // count sebagai 64-bit supaya aman sampai batas 2^31.
    const int64_t cur = static_cast<int64_t>(pulse[i]);
    const int64_t prv = static_cast<int64_t>(prev_pulse[i]);
    int64_t delta = cur - prv;

    // Kalau STM32 me-reset counter (delta negatif besar), anggap wrap
    // 32-bit dan perbaiki. Tanpa ini posisi akan meloncat ke belakang.
    if (delta < -2147483648LL) {
      delta += 4294967296LL;
    }

    // Sanity check berbasis WAKTU, bukan angka tetap.
    //
    // Batas clamp ini TIDAK boleh sempit. Perhitungan riil untuk robot ini
    // (ppr 7500, roda r 5 cm -> 238.7 tick/cm, loop 33 Hz):
    //   max_vel_x 0.14 m/s = 14 cm/s  = 3342 tick/detik = 101 tick/loop
    // Percobaan awal memakai 0.25 m/s (181 tick/loop) dan itu TERLALU
    // KETAT: burst kecil / slip roda sesaat langsung tertolak sehingga
    // odometry macet dan AMCL ikut kehilangan localize.
    //
    // Dipakai 1.0 m/s dengan margin 3x (= 3.0 m/s efektif): jauh di atas
    // gerak normal 101 tick/loop, tapi tetap menyaring lompatan abnormal.
    // STM32 reset counter (10000 -> 50 = -9950 tick) dan paket UDP korup
    // tetap tertangkap.
    //
    // PENTING: ticks_per_cm ada dalam per-SENTIMETER, jadi batas jarak
    // HARUS dikonversi ke sentimeter (x100). Tanpa itu max_cm hanya
    // 0.099 "cm" = 1 mm, clamp jadi 74 tick dan GERAK NORMAL (101 tick)
    // ikut ditolak -> odometry macet total.
    const double ticks_per_cm = ppr_ / (2 * M_PI * r_wheel_cm_);
    const double max_cm = 1.0 * dt * 3.0 * 100.0;   // m -> cm, margin 3x
    const int64_t max_tick = static_cast<int64_t>(max_cm * ticks_per_cm) + 50;
    if (delta > max_tick || delta < -max_tick) {
      delta = 0;   // frame diabaikan, lebih baik diam 1 siklus daripada lompat
    }

    const double tick = static_cast<double>(delta);
    // PENTING: jangan static_cast<float>. Float 32-bit cuma ~7 digit
    // bermakna; tick/ppr_ yang kecil ikut kehilangan presisi. Ini murni
    // kebocoran presisi yang tidak perlu terjadi.
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
