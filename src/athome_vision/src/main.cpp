#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <onnxruntime_cxx_api.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace zed_vision
{

constexpr float PI_F = 3.14159265358979323846f;
constexpr double kOffsetX = 13.0;
constexpr double kOffsetY = 3.0;
constexpr double kOffsetZ = 44.0;
constexpr double kOffsetScale = 100.0;

float wrap180(float angle)
{
  while (angle > 180.0f) {angle -= 360.0f;}
  while (angle < -180.0f) {angle += 360.0f;}
  return angle;
}

float wrap360(float angle)
{
  angle = std::fmod(angle, 360.0f);
  if (angle < 0.0f) {angle += 360.0f;}
  if (angle >= 360.0f || std::abs(angle) < 1e-6f) {angle = 0.0f;}
  return angle;
}

float quatToYaw(float x, float y, float z, float w)
{
  return std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
}

bool computeBbox(
  const std::vector<sl::float2> & points, int width, int height, cv::Rect & bbox)
{
  float min_x = static_cast<float>(width), min_y = static_cast<float>(height);
  float max_x = 0.0f, max_y = 0.0f;
  bool found = false;

  for (const auto & p : points) {
    if (std::isfinite(p.x) && std::isfinite(p.y) &&
        p.x >= 0 && p.y >= 0 && p.x < width && p.y < height)
    {
      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);
      found = true;
    }
  }
  if (!found) return false;

  const int pad_x = static_cast<int>((max_x - min_x) * 0.05f + 5.0f);
  const int pad_y = static_cast<int>((max_y - min_y) * 0.05f + 5.0f);
  const int x0 = std::max(0, static_cast<int>(min_x) - pad_x);
  const int y0 = std::max(0, static_cast<int>(min_y) - pad_y);
  const int x1 = std::min(width - 1, static_cast<int>(max_x) + pad_x);
  const int y1 = std::min(height - 1, static_cast<int>(max_y) + pad_y);
  if (x1 <= x0 || y1 <= y0) return false;

  bbox = cv::Rect(x0, y0, x1 - x0, y1 - y0);
  return true;
}

float getMedian(std::vector<float> & values)
{
  const size_t n = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + n, values.end());
  return values[n];
}

bool get3dCoordinates(const cv::Rect & box, const sl::Mat & cloud, sl::float4 & point)
{
  const unsigned int max_radius = 10;
  const unsigned int obj_size = std::min(box.width, box.height);
  const unsigned int radius = std::min(max_radius, obj_size / 2);
  const int center_i = box.x + box.width / 2;
  const int center_j = box.y + box.height / 2;
  const int w = static_cast<int>(cloud.getWidth());
  const int h = static_cast<int>(cloud.getHeight());

  std::vector<float> xs, ys, zs;
  for (unsigned int r = 0; r < radius; ++r) {
    for (int dy = -static_cast<int>(r); dy <= static_cast<int>(r); ++dy) {
      for (int dx = -static_cast<int>(r); dx <= static_cast<int>(r); ++dx) {
        const int i = center_i + dx;
        const int j = center_j + dy;
        sl::float4 out{NAN, NAN, NAN, NAN};
        if (i >= 0 && i < w && j >= 0 && j < h) {
          cloud.getValue(i, j, &out);
        }
        if (std::isfinite(out.z)) {
          xs.push_back(out.x);
          ys.push_back(out.y);
          zs.push_back(out.z);
        }
      }
    }
  }

  // Jika titik tengah tidak memiliki depth (misal permukaan tas kertas polos tanpa tekstur),
  // lakukan grid sampling di area kontur/lipatan dalam bounding box (15% - 85%)
  if (xs.empty() && box.width > 8 && box.height > 8) {
    const int start_x = box.x + static_cast<int>(box.width * 0.08f);
    const int end_x = box.x + static_cast<int>(box.width * 0.92f);
    const int start_y = box.y + static_cast<int>(box.height * 0.05f);
    const int end_y = box.y + static_cast<int>(box.height * 0.95f);
    const int step_x = std::max(2, (end_x - start_x) / 8);
    const int step_y = std::max(2, (end_y - start_y) / 8);

    for (int y = start_y; y < end_y; y += step_y) {
      for (int x = start_x; x < end_x; x += step_x) {
        if (x >= 0 && x < w && y >= 0 && y < h) {
          sl::float4 out{NAN, NAN, NAN, NAN};
          cloud.getValue(x, y, &out);
          if (std::isfinite(out.z)) {
            xs.push_back(out.x);
            ys.push_back(out.y);
            zs.push_back(out.z);
          }
        }
      }
    }
  }

  if (xs.empty()) {
    point.x = point.y = point.z = NAN;
    return false;
  }
  point.x = getMedian(xs);
  point.y = getMedian(ys);
  point.z = getMedian(zs);
  return true;
}

bool get3dWithOffset(
  const cv::Rect & rect, const sl::Mat & cloud, double & x, double & y, double & z)
{
  const int cx = rect.x + rect.width / 2;
  const int cy = rect.y + rect.height / 2;
  sl::float4 point{NAN, NAN, NAN, NAN};

  if (cx >= 0 && cx < static_cast<int>(cloud.getWidth()) &&
      cy >= 0 && cy < static_cast<int>(cloud.getHeight()))
  {
    cloud.getValue(cx, cy, &point);
  }
  if (!std::isfinite(point.z)) {
    get3dCoordinates(rect, cloud, point);
  }
  if (!std::isfinite(point.z)) return false;

  x = (point.x + kOffsetX) / kOffsetScale;
  y = (-point.y + kOffsetY) / kOffsetScale;
  z = (point.z + kOffsetZ) / kOffsetScale;
  return true;
}

struct DominantColorInfo
{
  bool valid{false};
  int best_idx{0};
  std::string name{"black"};
  double ratio{0.0};
  double mean_l{0.0}, mean_a{0.0}, mean_b{0.0};
  double mean_c{0.0};
  double mean_s{0.0}, mean_v{0.0};
  double chroma{0.0};
  cv::Vec3b mean_bgr{0, 0, 0};
  cv::Mat bucket_map;
};

int classifyShirtColorLab(const cv::Vec3b & lab)
{
  const double l = static_cast<double>(lab[0]) * (100.0 / 255.0);
  const double a = static_cast<double>(lab[1]) - 128.0;
  const double b = static_cast<double>(lab[2]) - 128.0;
  const double c = std::hypot(a, b);

  // Deteksi coklat redup/muted (R > G > B: a* >= 2, b* >= 4) sebelum jatuh ke ambang gray
  if (l >= 12.0 && l <= 66.0 && a >= 2.0 && b >= 4.0 && c >= 6.5) {
    double h_warm = std::atan2(b, a) * 180.0 / PI_F;
    if (h_warm >= 24.0 && h_warm <= 92.0 &&
        ((c < 38.0 && l < 60.0) || (c < 30.0 && l < 66.0)))
    {
      return 10; // brown
    }
  }

  // Di CIE Lab, hitam/abu/putih selalu punya krominansi C* rendah meskipun L* redup
  if (l < 5.0 && c < 8.0) return 0;               // black (noise floor / bayangan dalam)
  if (c < 9.5 || (l > 72.0 && c < 14.0)) {
    if (l < 22.0) return 0;                       // black
    if (l > 56.0) return 1;                       // white (termasuk putih indoor redup)
    return 2;                                     // gray
  }

  // Sudut Hue di bidang a*b* (0..360 derajat)
  double h_deg = std::atan2(b, a) * 180.0 / PI_F;
  if (h_deg < 0.0) h_deg += 360.0;

  if (h_deg <= 38.0) {
    if (l < 30.0 && c < 24.0 && h_deg >= 28.0) return 10; // dark brown
    return (l > 50.0 && h_deg <= 15.0) ? 9 : 3;           // pink terang vs red
  }
  if (h_deg <= 94.0) {
    // Coklat / tan / khaki berada di sektor hue hangat dengan L* dan C* sedang-rendah
    if ((c < 38.0 && l < 60.0) || (c < 30.0 && l < 68.0)) return 10; // brown
    if (h_deg <= 74.0) return 4;                  // orange
    return 5;                                     // yellow
  }
  if (h_deg <= 198.0) return 6;                   // green
  if (h_deg <= 298.0) return 7;                   // blue (sky blue hingga navy)
  if (h_deg <= 336.0) return 8;                   // purple
  return (l >= 44.0) ? 9 : 3;                     // pink vs dark red/crimson
}

int refineChromaticLabDeltaE(double l, double a, double b)
{
  const double c = std::hypot(a, b);
  if (c < 1e-3) return 3;

  double h_deg = std::atan2(b, a) * 180.0 / PI_F;
  if (h_deg < 0.0) h_deg += 360.0;
  if ((h_deg > 38.0 && h_deg <= 94.0 && ((c < 38.0 && l < 60.0) || (c < 30.0 && l < 68.0))) ||
      (h_deg >= 28.0 && h_deg <= 38.0 && l < 30.0 && c < 24.0))
  {
    return 10; // brown
  }

  struct LabProto {
    int idx;
    double l;
    double a;
    double b;
  };
  // Prototipe perseptual CIE Lab untuk 7 warna kromatik cerah/jenuh
  static constexpr std::array<LabProto, 7> kProtos = {{
    {3, 40.0,  55.0,  38.0},  // red
    {4, 58.0,  38.0,  60.0},  // orange
    {5, 80.0,  -8.0,  75.0},  // yellow
    {6, 55.0, -50.0,  40.0},  // green
    {7, 42.0,  16.0, -58.0},  // blue
    {8, 38.0,  52.0, -48.0},  // purple
    {9, 64.0,  48.0,  -4.0},  // pink
  }};

  // Normalisasi skala vektor (a*, b*) ke chroma referensi ~60 agar gelap/terang sebanding,
  // ditambah bobot ringan pada L* untuk membedakan pink vs red.
  const double scale = 60.0 / c;
  const double na = a * scale;
  const double nb = b * scale;
  const double nl = (l < 35.0) ? std::min(50.0, l * 1.8) : l;

  int best_idx = 3;
  double best_de = 1e9;
  for (const auto & p : kProtos) {
    const double dl = (nl - p.l) * 0.45;
    const double da = na - p.a;
    const double db = nb - p.b;
    const double de = std::sqrt(dl * dl + da * da + db * db);
    if (de < best_de) {
      best_de = de;
      best_idx = p.idx;
    }
  }
  return best_idx;
}

cv::Vec3f computeWhiteBalanceGain(const cv::Mat & bgr)
{
  if (bgr.empty()) return cv::Vec3f(1.0f, 1.0f, 1.0f);
  double sum_b = 0.0, sum_g = 0.0, sum_r = 0.0;
  int count = 0;
  for (int row = 0; row < bgr.rows; row += 4) {
    const cv::Vec3b * ptr = bgr.ptr<cv::Vec3b>(row);
    for (int col = 0; col < bgr.cols; col += 4) {
      const cv::Vec3b & p = ptr[col];
      const int max_c = std::max({p[0], p[1], p[2]});
      const int min_c = std::min({p[0], p[1], p[2]});
      if (min_c < 10 || max_c > 248) continue;
      sum_b += p[0];
      sum_g += p[1];
      sum_r += p[2];
      ++count;
    }
  }
  if (count < 50) return cv::Vec3f(1.0f, 1.0f, 1.0f);
  const double mb = sum_b / count;
  const double mg = sum_g / count;
  const double mr = sum_r / count;
  const double m_gray = (mb + mg + mr) / 3.0;
  return cv::Vec3f(
    std::clamp(static_cast<float>(m_gray / std::max(1.0, mb)), 0.78f, 1.32f),
    std::clamp(static_cast<float>(m_gray / std::max(1.0, mg)), 0.78f, 1.32f),
    std::clamp(static_cast<float>(m_gray / std::max(1.0, mr)), 0.78f, 1.32f));
}

cv::Mat applyWhiteBalance(const cv::Mat & bgr_roi, const cv::Vec3f & gain)
{
  if (bgr_roi.empty()) return bgr_roi;
  if (std::abs(gain[0] - 1.0f) < 0.01f &&
      std::abs(gain[1] - 1.0f) < 0.01f &&
      std::abs(gain[2] - 1.0f) < 0.01f)
  {
    return bgr_roi;
  }
  cv::Mat out(bgr_roi.size(), bgr_roi.type());
  for (int row = 0; row < bgr_roi.rows; ++row) {
    const cv::Vec3b * src = bgr_roi.ptr<cv::Vec3b>(row);
    cv::Vec3b * dst = out.ptr<cv::Vec3b>(row);
    for (int col = 0; col < bgr_roi.cols; ++col) {
      dst[col][0] = cv::saturate_cast<uchar>(src[col][0] * gain[0]);
      dst[col][1] = cv::saturate_cast<uchar>(src[col][1] * gain[1]);
      dst[col][2] = cv::saturate_cast<uchar>(src[col][2] * gain[2]);
    }
  }
  return out;
}

cv::Mat buildTorsoDepthMask(const cv::Rect & roi, const sl::Mat & cloud)
{
  const int cw = static_cast<int>(cloud.getWidth());
  const int ch = static_cast<int>(cloud.getHeight());
  if (roi.width <= 4 || roi.height <= 4 || cw <= 0 || ch <= 0) return cv::Mat();

  const int x0 = std::max(0, roi.x + static_cast<int>(roi.width * 0.30));
  const int x1 = std::min(cw, roi.x + static_cast<int>(roi.width * 0.70));
  const int y0 = std::max(0, roi.y + static_cast<int>(roi.height * 0.30));
  const int y1 = std::min(ch, roi.y + static_cast<int>(roi.height * 0.70));

  std::vector<float> center_depths;
  const int step_y = std::max(1, (y1 - y0) / 8);
  const int step_x = std::max(1, (x1 - x0) / 8);
  for (int y = y0; y < y1; y += step_y) {
    for (int x = x0; x < x1; x += step_x) {
      sl::float4 pt{NAN, NAN, NAN, NAN};
      cloud.getValue(x, y, &pt);
      if (std::isfinite(pt.x) && pt.x > 20.0f && pt.x < 600.0f) {
        center_depths.push_back(pt.x);
      }
    }
  }
  if (center_depths.size() < 6) return cv::Mat();

  const float torso_x_cm = getMedian(center_depths);
  cv::Mat mask(roi.size(), CV_8UC1, cv::Scalar(255));
  int kept = 0;
  const int total = roi.width * roi.height;
  for (int row = 0; row < roi.height; ++row) {
    const int py = roi.y + row;
    if (py < 0 || py >= ch) continue;
    uint8_t * mptr = mask.ptr<uint8_t>(row);
    for (int col = 0; col < roi.width; ++col) {
      const int px = roi.x + col;
      if (px < 0 || px >= cw) continue;
      sl::float4 pt{NAN, NAN, NAN, NAN};
      cloud.getValue(px, py, &pt);
      if (std::isfinite(pt.x) && pt.x > 10.0f && std::abs(pt.x - torso_x_cm) > 35.0f) {
        mptr[col] = 0;
      } else {
        ++kept;
      }
    }
  }
  if (kept < std::max(12, total / 6)) return cv::Mat();
  return mask;
}

DominantColorInfo analyzeDominantShirtColor(
  const cv::Mat & bgr_roi, const cv::Mat & depth_mask = cv::Mat())
{
  static constexpr std::array<const char *, 11> kColorNames = {
    "black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"
  };

  DominantColorInfo info;
  if (bgr_roi.empty()) return info;

  cv::Mat lab_roi, hsv_roi;
  cv::cvtColor(bgr_roi, lab_roi, cv::COLOR_BGR2Lab);
  cv::cvtColor(bgr_roi, hsv_roi, cv::COLOR_BGR2HSV);
  info.bucket_map = cv::Mat(bgr_roi.size(), CV_8UC1);

  std::array<double, 11> weights{};
  std::array<int, 11> counts{};
  std::array<double, 11> sum_b{}, sum_g{}, sum_r{};
  std::array<double, 11> sum_l{}, sum_la{}, sum_lb{};
  std::array<double, 11> sum_s{}, sum_v{};

  const int rows = bgr_roi.rows;
  const int cols = bgr_roi.cols;
  if (rows <= 0 || cols <= 0) return info;

  const int r0 = rows / 5;
  const int r1 = rows - r0;
  const int c0 = cols / 5;
  const int c1 = cols - c0;

  const bool use_mask = (!depth_mask.empty() && depth_mask.size() == bgr_roi.size());
  double total_weight = 0.0;
  for (int row = 0; row < rows; ++row) {
    const cv::Vec3b * bgr_ptr = bgr_roi.ptr<cv::Vec3b>(row);
    const cv::Vec3b * lab_ptr = lab_roi.ptr<cv::Vec3b>(row);
    const cv::Vec3b * hsv_ptr = hsv_roi.ptr<cv::Vec3b>(row);
    const uint8_t * mask_ptr = use_mask ? depth_mask.ptr<uint8_t>(row) : nullptr;
    uint8_t * map_ptr = info.bucket_map.ptr<uint8_t>(row);
    const bool center_row = (row >= r0 && row < r1);
    for (int col = 0; col < cols; ++col) {
      if (mask_ptr && mask_ptr[col] == 0) {
        map_ptr[col] = 255;
        continue;
      }
      const cv::Vec3b & p_bgr = bgr_ptr[col];
      const cv::Vec3b & p_lab = lab_ptr[col];
      const cv::Vec3b & p_hsv = hsv_ptr[col];
      const int idx = classifyShirtColorLab(p_lab);
      map_ptr[col] = static_cast<uint8_t>(idx);

      // Prioritaskan area tengah torso agar tidak terganggu tepi lengan/kursi
      const double w = (center_row && col >= c0 && col < c1) ? 2.0 : 1.0;
      weights[idx] += w;
      total_weight += w;

      counts[idx]++;
      sum_b[idx] += p_bgr[0];
      sum_g[idx] += p_bgr[1];
      sum_r[idx] += p_bgr[2];

      sum_l[idx] += static_cast<double>(p_lab[0]) * (100.0 / 255.0);
      sum_la[idx] += static_cast<double>(p_lab[1]) - 128.0;
      sum_lb[idx] += static_cast<double>(p_lab[2]) - 128.0;
      sum_s[idx] += p_hsv[1];
      sum_v[idx] += p_hsv[2];
    }
  }

  if (total_weight <= 0.0) return info;

  int best_idx = 0;
  for (int i = 1; i < 11; ++i) {
    if (weights[i] > weights[best_idx]) best_idx = i;
  }

  // Jika lipatan gelap atau piksel netral membuat black/gray menang sementara ada warna kromatik/coklat konsisten,
  // promosikan warna tersebut agar navy/maroon/dark green/brown tidak jadi black/gray.
  if (best_idx == 0 || best_idx == 2) {
    int best_chrom_idx = 3;
    double total_chrom_weight = 0.0;
    for (int i = 3; i < 11; ++i) {
      total_chrom_weight += weights[i];
      if (weights[i] > weights[best_chrom_idx]) best_chrom_idx = i;
    }
    const double min_chrom_c = (best_chrom_idx == 10) ? 6.5 : 12.0;
    if (counts[best_chrom_idx] > 0 &&
        weights[best_chrom_idx] >= total_weight * 0.20 &&
        weights[best_chrom_idx] >= total_chrom_weight * 0.55)
    {
      const double ca = sum_la[best_chrom_idx] / counts[best_chrom_idx];
      const double cb = sum_lb[best_chrom_idx] / counts[best_chrom_idx];
      if (std::hypot(ca, cb) >= min_chrom_c) {
        best_idx = best_chrom_idx;
      }
    }
  }

  const int best_cnt = counts[best_idx];
  if (best_cnt <= 0) return info;

  info.valid = true;
  info.mean_l = sum_l[best_idx] / best_cnt;
  info.mean_a = sum_la[best_idx] / best_cnt;
  info.mean_b = sum_lb[best_idx] / best_cnt;
  info.mean_c = std::hypot(info.mean_a, info.mean_b);

  // Verifikasi akhir: cek apakah cluster gray ternyata coklat redup (a* >= 2, b* >= 4.5, C* >= 6.0)
  int final_idx = best_idx;
  if (best_idx == 2 && info.mean_l >= 14.0 && info.mean_l <= 66.0 &&
      info.mean_a >= 2.0 && info.mean_b >= 4.5 && info.mean_c >= 6.0)
  {
    const double h_warm = std::atan2(info.mean_b, info.mean_a) * 180.0 / PI_F;
    if (h_warm >= 24.0 && h_warm <= 92.0) {
      final_idx = 10; // brown
    }
  } else if (best_idx >= 3 && info.mean_c >= 6.5) {
    final_idx = refineChromaticLabDeltaE(info.mean_l, info.mean_a, info.mean_b);
  }

  info.best_idx = best_idx;
  info.name = kColorNames[static_cast<size_t>(final_idx)];
  info.ratio = weights[best_idx] / total_weight;

  const uint8_t mb = static_cast<uint8_t>(std::clamp(std::lround(sum_b[best_idx] / best_cnt), 0L, 255L));
  const uint8_t mg = static_cast<uint8_t>(std::clamp(std::lround(sum_g[best_idx] / best_cnt), 0L, 255L));
  const uint8_t mr = static_cast<uint8_t>(std::clamp(std::lround(sum_r[best_idx] / best_cnt), 0L, 255L));
  info.mean_bgr = cv::Vec3b(mb, mg, mr);
  info.chroma = static_cast<double>(std::max({mb, mg, mr}) - std::min({mb, mg, mr}));

  info.mean_s = sum_s[best_idx] / best_cnt;
  info.mean_v = sum_v[best_idx] / best_cnt;
  return info;
}

cv::Scalar extractTorsoColor(const cv::Mat & bgr, const cv::Rect & bbox)
{
  if (bgr.empty() || bbox.width <= 0 || bbox.height <= 0) return cv::Scalar(0, 0, 0);

  const int tw = std::max(4, static_cast<int>(bbox.width * 0.60f));
  const int th = std::max(4, static_cast<int>(bbox.height * 0.38f));
  const int tx = std::clamp(bbox.x + static_cast<int>(bbox.width * 0.20f), 0, bgr.cols - tw);
  const int ty = std::clamp(bbox.y + static_cast<int>(bbox.height * 0.20f), 0, bgr.rows - th);

  cv::Rect roi(tx, ty, tw, th);
  roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
  if (roi.area() <= 0) return cv::Scalar(0, 0, 0);

  const DominantColorInfo dom = analyzeDominantShirtColor(bgr(roi));
  if (dom.valid) {
    cv::Mat one_px(1, 1, CV_8UC3, cv::Scalar(dom.mean_bgr[0], dom.mean_bgr[1], dom.mean_bgr[2]));
    cv::Mat one_hsv;
    cv::cvtColor(one_px, one_hsv, cv::COLOR_BGR2HSV);
    const cv::Vec3b & p = one_hsv.at<cv::Vec3b>(0, 0);
    return cv::Scalar(p[0], p[1], p[2]);
  }
  cv::Mat hsv;
  cv::cvtColor(bgr(roi), hsv, cv::COLOR_BGR2HSV);
  return cv::mean(hsv);
}


}

struct YoloDetection
{
  cv::Rect rect;
  std::string label;
  float score{0.0f};
  int class_id{-1};
};

struct PrevPerson
{
  int id{-1};
  cv::Rect bbox;
  cv::Rect shirt_bbox;
  cv::Rect pants_bbox;
  cv::Rect hat_bbox;
  cv::Rect glasses_bbox;
  bool has_pos{false};
  double x_m{0.0}, y_m{0.0}, range_m{0.0};
  bool is_sitting{false};
  bool is_sitting_posture{false};
  bool is_standing_posture{false};
  bool is_facing_back{false};
  std::vector<cv::Point2f> kpt;
  std::string clothing, pants, hair, gender, name;
  bool has_hat{false};
  bool has_glasses{false};
};

struct PrevBag
{
  cv::Rect bbox;
  bool has_pos{false};
  double x_m{0.0}, y_m{0.0}, range_m{0.0};
};

struct PrevItem
{
  std::string label;
  cv::Rect bbox;
  bool has_pos{false};
  double x_m{0.0}, y_m{0.0}, range_m{0.0};
  float score{0.0f};
};

struct PrevChair
{
  std::string label{"chair"};
  cv::Rect bbox;
  bool has_pos{false};
  double x_m{0.0}, y_m{0.0}, range_m{0.0};
  bool occupied{false};
  std::string side{"center"};
};

static std::vector<std::string> loadLabels(const std::string & path)
{
  std::vector<std::string> labels;
  if (path.empty()) return labels;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) labels.push_back(line);
  }
  return labels;
}

class OrtModel
{
public:
  OrtModel()
  : mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

  bool load(
    const std::string & path, std::string * err = nullptr,
    OrtCudnnConvAlgoSearch algo_search = OrtCudnnConvAlgoSearchHeuristic)
  {
    session_.reset();
    using_cuda_ = false;
    try {
      Ort::SessionOptions opts;
      opts.SetIntraOpNumThreads(2);
      opts.SetInterOpNumThreads(1);
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      try {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        cuda_opts.cudnn_conv_algo_search = algo_search;
        cuda_opts.do_copy_in_default_stream = 1;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        using_cuda_ = true;
      } catch (const std::exception & e) {
        using_cuda_ = false;
      }

      session_ = std::make_unique<Ort::Session>(env(), path.c_str(), opts);
      Ort::AllocatorWithDefaultOptions alloc;
      input_ = session_->GetInputNameAllocated(0, alloc).get();
      output_ = session_->GetOutputNameAllocated(0, alloc).get();

      if (using_cuda_) {
        std::cout << "[OrtModel] CUDA Execution Provider aktif: " << path << std::endl;
      } else {
        std::cout << "[OrtModel] CPU Execution Provider aktif: " << path << std::endl;
      }
    } catch (const std::exception & e) {
      if (err) *err = e.what();
      session_.reset();
      return false;
    }
    return true;
  }

  bool ready() const {return static_cast<bool>(session_);}

  std::vector<int64_t> inputShape()
  {
    if (!session_) return {};
    try {
      return session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    } catch (const std::exception &) {
      return {};
    }
  }

  std::vector<int64_t> outputShape()
  {
    if (!session_) return {};
    try {
      return session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    } catch (const std::exception &) {
      return {};
    }
  }

  std::vector<Ort::Value> run(
    const cv::Mat & blob, const std::array<int64_t, 4> & shape)
  {
    if (!session_ || blob.empty()) return {};

    Ort::Value input = Ort::Value::CreateTensor<float>(
      mem_, const_cast<float *>(blob.ptr<float>()), blob.total(),
      shape.data(), shape.size());
    const char * in_names[] = {input_.c_str()};
    const char * out_names[] = {output_.c_str()};
    try {
      return session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    } catch (const std::exception & e) {
      std::cerr << "[OrtModel] inference failed: " << e.what() << std::endl;
      return {};
    }
  }

private:
  static Ort::Env & env()
  {
    static Ort::Env instance(ORT_LOGGING_LEVEL_WARNING, "athome_vision");
    return instance;
  }

  std::unique_ptr<Ort::Session> session_;
  std::string input_, output_;
  Ort::MemoryInfo mem_;
  bool using_cuda_{false};
};

class YoloDetector
{
public:
  bool load(
    const std::string & model_path, const std::string & classes_path,
    float conf = 0.45f, float nms = 0.45f, int size = 640,
    const std::string & default_label = "object", std::string * err = nullptr)
  {
    conf_ = conf;
    nms_ = nms;
    input_size_ = size;
    default_label_ = default_label;
    names_ = loadLabels(classes_path);
    return model_.load(model_path, err);
  }

  bool ready() const {return model_.ready();}

  void detect(
    const cv::Mat & bgr, const std::vector<std::string> & want_labels,
    std::vector<YoloDetection> & out)
  {
    out.clear();
    if (!model_.ready() || bgr.empty()) return;

    const std::unordered_set<std::string> wanted(want_labels.begin(), want_labels.end());
    cv::Mat blob = cv::dnn::blobFromImage(
      bgr, 1.0 / 255.0, cv::Size(input_size_, input_size_), cv::Scalar(), true, false);

    std::array<int64_t, 4> shape{1, 3, input_size_, input_size_};
    std::vector<Ort::Value> outputs = model_.run(blob, shape);
    if (outputs.empty() || !outputs.front().IsTensor()) return;

    const auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    const float * data = outputs.front().GetTensorData<float>();
    const int channels = (out_shape.size() == 3) ? static_cast<int>(out_shape[1]) : 84;
    const int count = (out_shape.size() == 3) ? static_cast<int>(out_shape[2]) : 8400;
    const int classes = channels - 4;
    if (classes <= 0) return;

    cv::Mat det(channels, count, CV_32F, const_cast<float *>(data));
    cv::Mat transposed = det.t();
    const float x_factor = static_cast<float>(bgr.cols) / static_cast<float>(input_size_);
    const float y_factor = static_cast<float>(bgr.rows) / static_cast<float>(input_size_);

    static const std::vector<std::string> coco80 = {
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
      "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
      "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
      "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
      "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    auto labelOf = [&](int id) -> std::string {
        if (id >= 0 && id < static_cast<int>(names_.size())) return names_[static_cast<size_t>(id)];
        if (classes == 80 && id >= 0 && id < 80) return coco80[static_cast<size_t>(id)];
        return "";
      };

    const bool is_bag_target = (default_label_ == "bag") || (wanted.count("bag") > 0);

    auto isWanted = [&](int id) {
        if (wanted.empty()) return true;
        if (classes == 1) {
          if (is_bag_target) return true;
          const std::string name = labelOf(id);
          return (!name.empty() && wanted.count(name) > 0) ||
                 wanted.count(default_label_) > 0;
        }
        const std::string name = labelOf(id);
        if (!name.empty() && wanted.count(name) > 0) return true;
        if (classes == 80) {
          if (wanted.count("bag") > 0 && (id == 24 || id == 26 || id == 28)) return true;
          if ((wanted.count("chair") > 0 || wanted.count("sofa") > 0 || wanted.count("couch") > 0) &&
              (id == 56 || id == 57)) return true;
        }
        return false;
      };

    std::vector<int> wanted_ids;
    if (is_bag_target && classes > 1 && !wanted.empty()) {
      for (int c = 0; c < classes; ++c) {
        if (isWanted(c)) wanted_ids.push_back(c);
      }
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < count; ++i) {
      const float * row = transposed.ptr<float>(i);
      float score = 0.0f;
      int class_id = -1;
      if (classes == 1) {
        score = row[4];
        class_id = 0;
      } else if (is_bag_target && !wanted_ids.empty()) {
        float max_wanted_score = 0.0f;
        int best_wanted_id = -1;
        for (int wid : wanted_ids) {
          const float s = row[4 + wid];
          if (s > max_wanted_score) {
            max_wanted_score = s;
            best_wanted_id = wid;
          }
        }
        score = max_wanted_score;
        class_id = best_wanted_id;
      } else {
        cv::Mat scores_row(1, classes, CV_32F, const_cast<float *>(row + 4));
        cv::Point best;
        double max_score = 0.0;
        cv::minMaxLoc(scores_row, nullptr, &max_score, nullptr, &best);
        score = static_cast<float>(max_score);
        class_id = best.x;
      }
      if (class_id < 0 || score < conf_ || !isWanted(class_id)) continue;

      const float cx = row[0], cy = row[1], w = row[2], h = row[3];
      const int left = std::max(0, std::min(static_cast<int>((cx - 0.5f * w) * x_factor), bgr.cols - 1));
      const int top = std::max(0, std::min(static_cast<int>((cy - 0.5f * h) * y_factor), bgr.rows - 1));
      const int width = std::max(1, std::min(static_cast<int>(w * x_factor), bgr.cols - left));
      const int height = std::max(1, std::min(static_cast<int>(h * y_factor), bgr.rows - top));
      if (width <= 4 || height <= 4) continue;

      boxes.emplace_back(left, top, width, height);
      scores.push_back(score);
      class_ids.push_back(class_id);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, conf_, nms_, keep);
    for (int idx : keep) {
      YoloDetection det_result;
      det_result.rect = boxes[static_cast<size_t>(idx)];
      det_result.score = scores[static_cast<size_t>(idx)];
      det_result.class_id = class_ids[static_cast<size_t>(idx)];
      det_result.label = (classes == 1) ? default_label_ : labelOf(det_result.class_id);
      if (det_result.label.empty()) det_result.label = default_label_;
      if (det_result.label == "cracker_box" || det_result.label == "crackerbox") det_result.label = "cracker";
      if (det_result.label == "sugar_box" || det_result.label == "sugarbox") det_result.label = "sugar";
      out.push_back(det_result);
    }
  }

private:
  OrtModel model_;
  std::vector<std::string> names_;
  std::string default_label_{"object"};
  float conf_{0.45f};
  float nms_{0.45f};
  int input_size_{640};
};

class AttributeAnalyzer
{
public:
  bool loadGender(const std::string & path, std::string * err = nullptr)
  {
    const bool ok = gender_.load(path, err);
    if (ok) configureGender();
    return ok;
  }

  bool loadColor(
    const std::string & path, const std::string & labels_path, std::string * err = nullptr)
  {
    color_labels_ = loadLabels(labels_path);
    return color_.load(path, err);
  }

  bool loadHair(
    const std::string & path, const std::string & labels_path, std::string * err = nullptr)
  {
    hair_labels_ = loadLabels(labels_path);
    return hair_.load(path, err);
  }

  bool is_arcface_{false};

  bool loadFace(const std::string & path, std::string * err = nullptr)
  {
    const bool want_arcface = (path.find("arcface") != std::string::npos ||
                               path.find("w600k") != std::string::npos ||
                               path.find("insightface") != std::string::npos);
    // ArcFace w600k_r50 fails cuDNN Frontend Heuristic/Exhaustive on Conv_19 in ORT 1.22 + cuDNN 9.27,
    // but runs on GPU with OrtCudnnConvAlgoSearchDefault.
    const OrtCudnnConvAlgoSearch algo =
      want_arcface ? OrtCudnnConvAlgoSearchDefault : OrtCudnnConvAlgoSearchHeuristic;
    bool ok = face_.load(path, err, algo);
    if (ok) {
      is_arcface_ = want_arcface;
    }
    return ok;
  }

  bool faceReady() const {return face_.ready();}

  struct DetectedFace
  {
    cv::Rect bbox;
    cv::Point2f right_eye;
    cv::Point2f left_eye;
    cv::Point2f nose;
    cv::Point2f right_mouth;
    cv::Point2f left_mouth;
    float score{0.0f};
    bool valid{false};
  };

  struct GenderPrediction
  {
    std::string label{"unknown"};
    float prob_female{0.0f};
    float prob_male{0.0f};
    float margin{0.0f}; // prob_male - prob_female (positive = male, negative = female)
    float confidence{0.0f};
    bool is_high_quality{false};
  };

  static float computeBodyGenderScore(const std::vector<sl::float2> & kpt)
  {
    if (kpt.size() <= 22) return 0.0f;
    const auto valid = [](const sl::float2 & p) {
      return std::isfinite(p.x) && std::isfinite(p.y) && p.x > 0.0f && p.y > 0.0f;
    };
    if (!valid(kpt[5]) || !valid(kpt[12]) || !valid(kpt[18]) || !valid(kpt[22])) {
      return 0.0f;
    }

    const float shoulder_w = std::hypot(kpt[12].x - kpt[5].x, kpt[12].y - kpt[5].y);
    const float hip_w = std::hypot(kpt[22].x - kpt[18].x, kpt[22].y - kpt[18].y);
    if (shoulder_w < 15.0f || hip_w < 10.0f) return 0.0f;

    const float ratio = shoulder_w / hip_w;
    // Rasio antropometrik lebar bahu terhadap lebar pinggul:
    // Pria dewasa rata-rata: 1.45 - 1.75
    // Wanita dewasa rata-rata: 1.15 - 1.35
    if (ratio >= 1.48f) {
      return std::min(0.15f, (ratio - 1.48f) * 0.35f + 0.05f); // Positif = tend to male
    } else if (ratio <= 1.25f) {
      return -std::min(0.15f, (1.25f - ratio) * 0.35f + 0.05f); // Negatif = tend to female
    }
    return 0.0f;
  }

  bool loadFaceDetector(const std::string & path, float conf_thresh = 0.55f, std::string * err = nullptr)
  {
    try {
      if (!std::filesystem::exists(path)) {
        if (err) *err = "Face detector file does not exist: " + path;
        return false;
      }
      face_detect_conf_ = conf_thresh;
      face_detector_ = cv::FaceDetectorYN::create(
        path, "", cv::Size(320, 320), conf_thresh, 0.45f, 5000);
      return (face_detector_ != nullptr);
    } catch (const std::exception & e) {
      if (err) *err = e.what();
      return false;
    }
  }

  bool faceDetectorReady() const {return (face_detector_ != nullptr);}

  DetectedFace detectFace(const cv::Mat & bgr, const cv::Rect & person_bbox = cv::Rect())
  {
    DetectedFace result;
    if (!faceDetectorReady() || bgr.empty()) return result;

    cv::Rect search_roi(0, 0, bgr.cols, bgr.rows);
    if (person_bbox.width > 20 && person_bbox.height > 20) {
      const int hx = std::max(0, person_bbox.x - static_cast<int>(person_bbox.width * 0.15f));
      const int hy = std::max(0, person_bbox.y - static_cast<int>(person_bbox.height * 0.08f));
      const int hw = std::min(bgr.cols - hx, static_cast<int>(person_bbox.width * 1.30f));
      const int hh = std::min(bgr.rows - hy, static_cast<int>(person_bbox.height * 0.55f));
      search_roi = cv::Rect(hx, hy, hw, hh) & cv::Rect(0, 0, bgr.cols, bgr.rows);
    }

    if (search_roi.width < 16 || search_roi.height < 16) return result;

    try {
      cv::Mat roi_img = bgr(search_roi);
      float scale = 1.0f;
      cv::Mat det_img = roi_img;
      if (roi_img.cols < 180 || roi_img.rows < 180) {
        scale = 2.0f;
        cv::resize(roi_img, det_img, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
      }
      face_detector_->setInputSize(det_img.size());
      cv::Mat faces;
      face_detector_->detect(det_img, faces);

      if (faces.rows <= 0) return result;

      int best_idx = -1;
      float best_score = face_detect_conf_;
      for (int i = 0; i < faces.rows; ++i) {
        const float score = faces.at<float>(i, 14);
        if (score >= best_score) {
          best_score = score;
          best_idx = i;
        }
      }

      if (best_idx < 0) return result;

      result.valid = true;
      result.score = best_score;
      result.bbox = cv::Rect(
        static_cast<int>(faces.at<float>(best_idx, 0) / scale) + search_roi.x,
        static_cast<int>(faces.at<float>(best_idx, 1) / scale) + search_roi.y,
        static_cast<int>(faces.at<float>(best_idx, 2) / scale),
        static_cast<int>(faces.at<float>(best_idx, 3) / scale)
      );
      result.right_eye = cv::Point2f(
        faces.at<float>(best_idx, 4) / scale + search_roi.x,
        faces.at<float>(best_idx, 5) / scale + search_roi.y);
      result.left_eye = cv::Point2f(
        faces.at<float>(best_idx, 6) / scale + search_roi.x,
        faces.at<float>(best_idx, 7) / scale + search_roi.y);
      result.nose = cv::Point2f(
        faces.at<float>(best_idx, 8) / scale + search_roi.x,
        faces.at<float>(best_idx, 9) / scale + search_roi.y);
      result.right_mouth = cv::Point2f(
        faces.at<float>(best_idx, 10) / scale + search_roi.x,
        faces.at<float>(best_idx, 11) / scale + search_roi.y);
      result.left_mouth = cv::Point2f(
        faces.at<float>(best_idx, 12) / scale + search_roi.x,
        faces.at<float>(best_idx, 13) / scale + search_roi.y);

      if (result.right_eye.x > result.left_eye.x) {
        std::swap(result.right_eye, result.left_eye);
        std::swap(result.right_mouth, result.left_mouth);
      }
    } catch (...) {
      result.valid = false;
    }

    return result;
  }

  static bool validKeypoint(const sl::float2 & point)
  {
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           point.x >= 0.0f && point.y >= 0.0f;
  }

  static cv::Rect computeShirtRoi(
    const cv::Mat & bgr, const cv::Rect & bbox, const std::vector<sl::float2> & kpt)
  {
    if (bgr.empty()) return cv::Rect();

    // 1. Jika keypoint bahu tersedia (kpt[5] = LEFT_SHOULDER, kpt[12] = RIGHT_SHOULDER)
    if (kpt.size() > 12 && validKeypoint(kpt[5]) && validKeypoint(kpt[12])) {
      const float sx = (kpt[5].x + kpt[12].x) * 0.5f;
      const float sy = (kpt[5].y + kpt[12].y) * 0.5f;
      const float shoulder_w = std::abs(kpt[12].x - kpt[5].x);

      if (shoulder_w > 15.0f) {
        int tw = static_cast<int>(shoulder_w * 0.90f);
        int th = static_cast<int>(shoulder_w * 1.10f);
        int tx = static_cast<int>(sx - tw * 0.5f);
        int ty = static_cast<int>(sy + shoulder_w * 0.10f); // Area torso lebih luas agar tidak terdominasi teks/logo dada

        cv::Rect roi(tx, ty, tw, th);
        roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (bbox.width > 10 && bbox.height > 10) {
          roi &= bbox;
        }
        if (roi.width >= 10 && roi.height >= 10) return roi;
      }
    }

    // 2. Fallback jika keypoint bahu tidak terdeteksi, gunakan proporsi torso dari bbox
    if (bbox.width > 10 && bbox.height > 10) {
      const int tw = std::max(10, static_cast<int>(bbox.width * 0.65f));
      const int th = std::max(10, static_cast<int>(bbox.height * 0.40f));
      const int tx = std::clamp(bbox.x + static_cast<int>(bbox.width * 0.175f), 0, bgr.cols - tw);
      const int ty = std::clamp(bbox.y + static_cast<int>(bbox.height * 0.20f), 0, bgr.rows - th);

      cv::Rect roi(tx, ty, tw, th);
      roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
      return roi;
    }

    return cv::Rect();
  }

  static cv::Rect computePantsRoi(
    const cv::Mat & bgr, const cv::Rect & bbox, const std::vector<sl::float2> & kpt)
  {
    if (bgr.empty()) return cv::Rect();

    // 1. Jika keypoint pinggul tersedia (BODY_34: kpt[18]=LEFT_HIP, kpt[22]=RIGHT_HIP,
    //    kpt[19]=LEFT_KNEE, kpt[23]=RIGHT_KNEE)
    if (kpt.size() > 23 && validKeypoint(kpt[18]) && validKeypoint(kpt[22])) {
      const float hx = (kpt[18].x + kpt[22].x) * 0.5f;
      const float hy = (kpt[18].y + kpt[22].y) * 0.5f;
      const float hip_w = std::abs(kpt[22].x - kpt[18].x);

      if (hip_w > 10.0f) {
        float min_x = std::min(kpt[18].x, kpt[22].x);
        float max_x = std::max(kpt[18].x, kpt[22].x);
        float knee_y_sum = 0.0f;
        int knee_cnt = 0;

        if (validKeypoint(kpt[19])) {
          min_x = std::min(min_x, kpt[19].x);
          max_x = std::max(max_x, kpt[19].x);
          knee_y_sum += kpt[19].y;
          knee_cnt++;
        }
        if (validKeypoint(kpt[23])) {
          min_x = std::min(min_x, kpt[23].x);
          max_x = std::max(max_x, kpt[23].x);
          knee_y_sum += kpt[23].y;
          knee_cnt++;
        }

        int px = 0, py = 0, pw = 0, ph = 0;
        if (knee_cnt > 0) {
          const float ky = knee_y_sum / static_cast<float>(knee_cnt);
          const float leg_dy = std::max(0.0f, ky - hy);
          const float cx = (min_x + max_x) * 0.5f;
          const float fpw = std::max(hip_w * 1.15f, (max_x - min_x) * 0.90f);
          pw = static_cast<int>(fpw);
          px = static_cast<int>(cx - fpw * 0.5f);
          py = static_cast<int>(hy + std::max(2.0f, leg_dy * 0.08f));
          ph = static_cast<int>(std::max({hip_w * 0.85f, leg_dy * 1.05f, 16.0f}));
        } else {
          pw = static_cast<int>(hip_w * 1.20f);
          ph = static_cast<int>(hip_w * 1.35f);
          px = static_cast<int>(hx - pw * 0.5f);
          py = static_cast<int>(hy + hip_w * 0.10f);
        }

        cv::Rect roi(px, py, pw, ph);
        roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (bbox.width > 10 && bbox.height > 10) {
          roi &= bbox;
        }
        if (roi.width >= 10 && roi.height >= 10) return roi;
      }
    }

    // 2. Fallback jika keypoint pinggul tidak terdeteksi, gunakan proporsi bagian bawah bbox
    if (bbox.width > 10 && bbox.height > 10) {
      const int pw = std::max(10, static_cast<int>(bbox.width * 0.55f));
      const int ph = std::max(10, static_cast<int>(bbox.height * 0.34f));
      const int px = std::clamp(bbox.x + static_cast<int>(bbox.width * 0.225f), 0, bgr.cols - pw);
      const int py = std::clamp(bbox.y + static_cast<int>(bbox.height * 0.52f), 0, bgr.rows - ph);

      cv::Rect roi(px, py, pw, ph);
      roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
      if (roi.width >= 10 && roi.height >= 10) return roi;
    }

    return cv::Rect();
  }

  std::string detectGarmentColor(
    const cv::Mat & bgr, const cv::Rect & roi,
    const std::string & suffix, const std::string & fallback,
    const sl::Mat * cloud = nullptr,
    const cv::Vec3f & wb_gain = cv::Vec3f(1.0f, 1.0f, 1.0f))
  {
    if (bgr.empty() || roi.width <= 5 || roi.height <= 5) return fallback;

    cv::Rect safe_roi = roi & cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (safe_roi.area() <= 0) return fallback;

    cv::Mat cropped = zed_vision::applyWhiteBalance(bgr(safe_roi), wb_gain);
    cv::Mat depth_mask = (cloud != nullptr)
      ? zed_vision::buildTorsoDepthMask(safe_roi, *cloud)
      : cv::Mat();

    // 1. Cari warna dominan (tahan bayangan gelap, teks/logo, lampu ruangan, & background)
    const zed_vision::DominantColorInfo dom =
      zed_vision::analyzeDominantShirtColor(cropped, depth_mask);
    if (!dom.valid) return fallback;

    std::string chosen_color = dom.name;

    // 2. Inferensi model ONNX pada patch warna dominan yang sudah di-normalisasi kecerahannya
    //    (color_model.onnx tidak memiliki kelas "brown", jadi pertahankan "brown" dari CIE Lab)
    if (dom.name != "brown" && color_.ready() && !color_labels_.empty()) {
      cv::Mat masked_cropped = cropped.clone();
      for (int row = 0; row < masked_cropped.rows; ++row) {
        cv::Vec3b * dst = masked_cropped.ptr<cv::Vec3b>(row);
        const uint8_t * map_ptr = dom.bucket_map.ptr<uint8_t>(row);
        for (int col = 0; col < masked_cropped.cols; ++col) {
          if (map_ptr[col] != static_cast<uint8_t>(dom.best_idx)) {
            dst[col] = dom.mean_bgr;
          }
        }
      }

      cv::Mat ready_img;
      if (dom.mean_v < 140.0 &&
          (dom.best_idx >= 3 || dom.mean_c >= 10.5))
      {
        const double scale = std::min(4.0, 155.0 / std::max(8.0, dom.mean_v));
        cv::convertScaleAbs(masked_cropped, ready_img, scale, 0);
      } else if (dom.best_idx == 1 && dom.mean_v < 180.0) {
        const double scale = std::min(1.6, 200.0 / std::max(1.0, dom.mean_v));
        cv::convertScaleAbs(masked_cropped, ready_img, scale, 0);
      } else {
        ready_img = masked_cropped;
      }

      cv::Mat resized;
      cv::resize(ready_img, resized, cv::Size(224, 224));
      cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
      cv::Mat blob = cv::dnn::blobFromImage(
        resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);
      std::array<int64_t, 4> shape{1, 3, 224, 224};
      std::vector<Ort::Value> outputs = color_.run(blob, shape);
      if (!outputs.empty()) {
        const auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
        const int classes = (out_shape.size() >= 2)
          ? static_cast<int>(out_shape[1]) : static_cast<int>(color_labels_.size());
        std::vector<float> probs;
        softmax(outputs.front().GetTensorData<float>(), classes, probs);
        int best = 0;
        for (int i = 1; i < classes; ++i) {
          if (probs[static_cast<size_t>(i)] > probs[static_cast<size_t>(best)]) best = i;
        }

        if (best < static_cast<int>(color_labels_.size())) {
          std::string cand = color_labels_[static_cast<size_t>(best)];
          std::transform(cand.begin(), cand.end(), cand.begin(), ::tolower);
          if (cand == "grey") cand = "gray";

          const float prob = probs[static_cast<size_t>(best)];
          bool accept_model = false;
          if (cand == dom.name) {
            accept_model = (prob > 0.25f);
          } else if (cand == "black") {
            // Di Lab, warna gelap jenuh punya mean_c >= 10.5; jangan izinkan model menimpa menjadi black
            accept_model = (dom.best_idx == 0) ||
                           (dom.mean_l < 20.0 && dom.mean_c < 10.5 && prob > 0.55f);
          } else if (cand == "white") {
            accept_model = (dom.mean_l > 54.0 && dom.mean_c < 13.0 && prob > 0.45f);
          } else if (cand == "gray") {
            accept_model = (dom.mean_c < 13.0 && prob > 0.45f);
          } else if (dom.best_idx >= 3) {
            accept_model = (prob > 0.58f);
          } else {
            accept_model = (dom.mean_c >= 12.0 && prob > 0.65f);
          }

          if (accept_model) {
            chosen_color = cand;
          }
        }
      }
    }

    if (!suffix.empty() &&
        chosen_color.find(suffix) == std::string::npos &&
        chosen_color.find("clothes") == std::string::npos)
    {
      chosen_color += " " + suffix;
    }
    return chosen_color;
  }

  std::string clothingColor(
    const cv::Mat & bgr, const cv::Rect & roi,
    const sl::Mat * cloud = nullptr,
    const cv::Vec3f & wb_gain = cv::Vec3f(1.0f, 1.0f, 1.0f))
  {
    return detectGarmentColor(bgr, roi, "shirt", "casual attire", cloud, wb_gain);
  }

  std::string pantsColor(
    const cv::Mat & bgr, const cv::Rect & roi,
    const sl::Mat * cloud = nullptr,
    const cv::Vec3f & wb_gain = cv::Vec3f(1.0f, 1.0f, 1.0f))
  {
    return detectGarmentColor(bgr, roi, "pants", "", cloud, wb_gain);
  }

  std::string clothingColor(
    const cv::Mat & bgr, const cv::Rect & bbox, const std::vector<sl::float2> & kpt)
  {
    const cv::Rect roi = computeShirtRoi(bgr, bbox, kpt);
    return clothingColor(bgr, roi);
  }

  static cv::Rect computeHeadAccessoryRoi(
    const cv::Mat & bgr, const cv::Rect & bbox, const std::vector<sl::float2> & kpt)
  {
    if (bgr.empty()) return cv::Rect();

    // Gunakan keypoint kepala/bahu ZED BODY_34 jika tersedia (kpt[26]=HEAD, kpt[5]=L_SHOULDER, kpt[12]=R_SHOULDER)
    // Buat ROI persegi agar tidak terdistorsi saat di-resize ke 640x640 oleh YOLO
    if (kpt.size() > 26 && validKeypoint(kpt[26])) {
      float span = 90.0f;
      if (validKeypoint(kpt[5]) && validKeypoint(kpt[12])) {
        span = std::max(span, std::abs(kpt[12].x - kpt[5].x) * 1.05f);
      } else if (bbox.width > 10) {
        span = std::max(span, static_cast<float>(bbox.width) * 0.70f);
      }
      const int side = static_cast<int>(span * 1.35f);
      const int hx = static_cast<int>(kpt[26].x - side * 0.50f);
      const int hy = static_cast<int>(kpt[26].y - side * 0.62f);
      cv::Rect roi(hx, hy, side, side);
      roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
      if (roi.width >= 20 && roi.height >= 20) return roi;
    }

    if (bbox.width > 10 && bbox.height > 10) {
      const int pad_x = static_cast<int>(bbox.width * 0.10f);
      const int top_pad = static_cast<int>(bbox.height * 0.10f);
      const int hx = bbox.x - pad_x;
      const int hy = bbox.y - top_pad;
      const int hw = bbox.width + 2 * pad_x;
      const int hh = static_cast<int>(bbox.height * 0.45f);
      cv::Rect roi(hx, hy, hw, hh);
      roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
      if (roi.width >= 20 && roi.height >= 20) return roi;
    }
    return cv::Rect();
  }

  std::string hairColor(
    const cv::Mat & bgr, const cv::Rect & bbox,
    const std::vector<sl::float2> & kpt = {})
  {
    if (bgr.empty() || bbox.width <= 0 || bbox.height <= 0) return "dark hair";

    cv::Point2f eye_left, eye_right, nose;
    cv::Rect roi;
    if (faceDetectorReady()) {
      DetectedFace df = detectFace(bgr, bbox);
      if (df.valid && df.score >= 0.45f) {
        const float eye_dist = std::hypot(df.left_eye.x - df.right_eye.x, df.left_eye.y - df.right_eye.y);
        const float center_x = (df.right_eye.x + df.left_eye.x) * 0.5f;
        const float center_y = std::min(df.right_eye.y, df.left_eye.y) - eye_dist * 0.85f;
        const int hw = std::max(10, static_cast<int>(eye_dist * 1.5f));
        const int hh = std::max(10, static_cast<int>(eye_dist * 1.0f));
        const int hx = static_cast<int>(center_x - hw * 0.5f);
        const int hy = static_cast<int>(center_y - hh * 0.5f);
        roi = cv::Rect(hx, hy, hw, hh);
      }
    }
    if (roi.empty()) {
      if (faceLandmarks(kpt, eye_left, eye_right, nose)) {
        const float eye_dist = std::hypot(eye_right.x - eye_left.x, eye_right.y - eye_left.y);
        const float center_x = (eye_left.x + eye_right.x) * 0.5f;
        const float center_y = std::min(eye_left.y, eye_right.y) - eye_dist * 0.85f;
        const int hw = std::max(10, static_cast<int>(eye_dist * 1.5f));
        const int hh = std::max(10, static_cast<int>(eye_dist * 1.0f));
        const int hx = static_cast<int>(center_x - hw * 0.5f);
        const int hy = static_cast<int>(center_y - hh * 0.5f);
        roi = cv::Rect(hx, hy, hw, hh);
      } else {
        const int hw = std::max(6, static_cast<int>(bbox.width * 0.35f));
        const int hh = std::max(6, static_cast<int>(bbox.height * 0.12f));
        const int hx = bbox.x + static_cast<int>(bbox.width * 0.325f);
        const int hy = bbox.y + static_cast<int>(bbox.height * 0.02f);
        roi = cv::Rect(hx, hy, hw, hh);
      }
    }
    roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (roi.area() <= 0) return "dark hair";

    if (hair_.ready() && !hair_labels_.empty()) {
      cv::Mat resized;
      cv::resize(bgr(roi), resized, cv::Size(224, 224));
      cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
      cv::Mat blob = cv::dnn::blobFromImage(
        resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);
      std::array<int64_t, 4> shape{1, 3, 224, 224};
      std::vector<Ort::Value> outputs = hair_.run(blob, shape);
      if (!outputs.empty()) {
        const auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
        const int classes = (out_shape.size() >= 2)
          ? static_cast<int>(out_shape[1]) : static_cast<int>(hair_labels_.size());
        std::vector<float> probs;
        softmax(outputs.front().GetTensorData<float>(), classes, probs);
        int best = 0;
        for (int i = 1; i < classes; ++i) {
          if (probs[static_cast<size_t>(i)] > probs[static_cast<size_t>(best)]) best = i;
        }
        if (probs[static_cast<size_t>(best)] > 0.50f && best < static_cast<int>(hair_labels_.size())) {
          std::string result = hair_labels_[static_cast<size_t>(best)];
          std::transform(result.begin(), result.end(), result.begin(), ::tolower);
          return result;
        }
      }
    }

    cv::Mat hsv;
    cv::cvtColor(bgr(roi), hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Vec3b> hair_pixels;
    hair_pixels.reserve(roi.area());
    int dark_count = 0;
    for (int r = 0; r < hsv.rows; ++r) {
      const cv::Vec3b * row = hsv.ptr<cv::Vec3b>(r);
      for (int c = 0; c < hsv.cols; ++c) {
        const cv::Vec3b & p = row[c];
        // Filter out bright background wall pixels (high value, low saturation)
        if (p[2] > 190 && p[1] < 35) continue;
        hair_pixels.push_back(p);
        if (p[2] < 70) dark_count++;
      }
    }

    if (hair_pixels.empty()) return "dark hair";

    // If >= 35% of valid pixels are dark, it is dark / black hair
    if (dark_count * 100 / static_cast<int>(hair_pixels.size()) >= 35) {
      return "black hair";
    }

    double sum_h = 0, sum_s = 0, sum_v = 0;
    for (const auto & p : hair_pixels) {
      sum_h += p[0];
      sum_s += p[1];
      sum_v += p[2];
    }
    const double n = hair_pixels.size();
    const double h = sum_h / n, s = sum_s / n, v = sum_v / n;

    if (v < 65.0) return "black hair";
    if (s < 35.0 && v > 150.0) return "gray hair";
    if (h >= 15.0 && h <= 35.0 && s > 50.0 && v > 120.0) return "blond hair";
    if (h >= 8.0 && h <= 30.0 && v <= 120.0) return "brown hair";
    return "dark hair";
  }

  GenderPrediction predictGenderDetailed(
    const std::vector<sl::float2> & kpt, const cv::Mat & bgr, const cv::Rect & bbox)
  {
    GenderPrediction result;
    if (!gender_.ready() || bgr.empty() || bbox.width <= 0 || bbox.height <= 0) {
      return result;
    }

    const int size = gender_input_size_;
    cv::Mat blob;
    bool yunet_aligned = false;
    bool high_quality = false;

    if (gender_is_genderage_) {
      // 1. Prioritaskan YuNet face detector dengan Quality & Frontality Gate
      if (faceDetectorReady()) {
        DetectedFace df = detectFace(bgr, bbox);
        if (df.valid && df.score >= 0.50f) {
          const float eye_dist = std::hypot(df.left_eye.x - df.right_eye.x, df.left_eye.y - df.right_eye.y);

          // Quality Gate: Ukuran wajah minimal (eye_dist >= 12 px, bbox >= 35x35 px)
          const bool size_ok = (eye_dist >= 12.0f && df.bbox.width >= 35 && df.bbox.height >= 35);

          // Frontality Gate: Cek apakah wajah menghadap depan (yaw ratio <= 1.80)
          const float d_re = std::hypot(df.nose.x - df.right_eye.x, df.nose.y - df.right_eye.y);
          const float d_le = std::hypot(df.nose.x - df.left_eye.x, df.nose.y - df.left_eye.y);
          const float yaw_ratio = std::max(d_re, d_le) / std::max(1.0f, std::min(d_re, d_le));
          const bool front_ok = (yaw_ratio <= 1.80f);

          // Roll Gate: Sudut kemiringan kepala <= 32 derajat
          const float roll_deg = std::abs(
            std::atan2(df.left_eye.y - df.right_eye.y, df.left_eye.x - df.right_eye.x) * 180.0f / zed_vision::PI_F);
          const bool roll_ok = (roll_deg <= 32.0f);

          // Sharpness Gate: Uji variansi Laplacian untuk memfilter blur gerakan / kamera guncang
          bool sharp_ok = false;
          const cv::Rect f_box = df.bbox & cv::Rect(0, 0, bgr.cols, bgr.rows);
          if (f_box.width >= 24 && f_box.height >= 24) {
            cv::Mat gray_face, lap;
            cv::cvtColor(bgr(f_box), gray_face, cv::COLOR_BGR2GRAY);
            cv::Laplacian(gray_face, lap, CV_32F);
            cv::Scalar mean, stddev;
            cv::meanStdDev(lap, mean, stddev);
            const double sharpness = stddev[0] * stddev[0];
            sharp_ok = (sharpness >= 30.0);
          }

          if (size_ok && front_ok && roll_ok && sharp_ok) {
            const float s = static_cast<float>(size) / 112.0f;
            std::vector<cv::Point2f> from{
              df.right_eye, df.left_eye, df.nose, df.right_mouth, df.left_mouth};
            std::vector<cv::Point2f> to{
              {38.2946f * s, 51.6963f * s},
              {73.5318f * s, 51.5014f * s},
              {56.0252f * s, 71.7366f * s},
              {41.5493f * s, 92.3655f * s},
              {70.7299f * s, 92.2041f * s}};
            cv::Mat transform = cv::estimateAffinePartial2D(from, to, cv::noArray(), cv::LMEDS);
            if (transform.empty()) transform = cv::estimateAffine2D(from, to);
            if (!transform.empty()) {
              cv::Mat aligned;
              cv::warpAffine(bgr, aligned, transform, cv::Size(size, size),
                cv::INTER_LINEAR, cv::BORDER_REPLICATE);
              cv::Mat rgb;
              cv::cvtColor(aligned, rgb, cv::COLOR_BGR2RGB);
              blob = cv::dnn::blobFromImage(
                rgb, 1.0f, cv::Size(size, size), cv::Scalar(0, 0, 0), false, false);
              yunet_aligned = true;
              high_quality = true;
            }
          }
        }
      }

      // 2. Fallback ke landmark ZED skeleton jika YuNet tidak lolos gate
      if (!yunet_aligned) {
        cv::Point2f eye_left, eye_right, nose;
        if (!faceLandmarks(kpt, eye_left, eye_right, nose)) {
          return result;
        }
        const float eye_dist = std::hypot(eye_right.x - eye_left.x, eye_right.y - eye_left.y);
        if (eye_dist < 9.0f) {
          return result;
        }

        const float s = static_cast<float>(size) / 112.0f;
        std::vector<cv::Point2f> from{eye_left, eye_right, nose};
        std::vector<cv::Point2f> to{
          {38.2946f * s, 51.6963f * s},
          {73.5318f * s, 51.5014f * s},
          {56.0252f * s, 71.7366f * s}};
        cv::Mat transform = cv::estimateAffinePartial2D(from, to, cv::noArray(), cv::LMEDS);
        if (transform.empty()) transform = cv::getAffineTransform(from, to);

        cv::Mat aligned;
        if (!transform.empty()) {
          cv::warpAffine(bgr, aligned, transform, cv::Size(size, size),
            cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        } else {
          const cv::Rect roi = faceRegion(bbox, kpt, bgr.cols, bgr.rows) & cv::Rect(0, 0, bgr.cols, bgr.rows);
          if (roi.width < 14 || roi.height < 14) return result;
          cv::resize(bgr(roi), aligned, cv::Size(size, size));
        }

        cv::Mat rgb;
        cv::cvtColor(aligned, rgb, cv::COLOR_BGR2RGB);
        blob = cv::dnn::blobFromImage(
          rgb, 1.0f, cv::Size(size, size), cv::Scalar(0, 0, 0), false, false);
      }
    } else {
      cv::Rect roi;
      if (gender_nhwc_) {
        roi = faceRegion(bbox, kpt, bgr.cols, bgr.rows);
      } else {
        const int x1 = std::clamp(bbox.x, 0, bgr.cols);
        const int y1 = std::clamp(bbox.y, 0, bgr.rows);
        const int x2 = std::clamp(bbox.x + bbox.width, 0, bgr.cols);
        const int y2 = std::clamp(static_cast<int>(bbox.y + bbox.height * 0.55), 0, bgr.rows);
        if (y2 > y1 && x2 > x1) roi = cv::Rect(x1, y1, x2 - x1, y2 - y1);
      }

      roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
      if (roi.width < 14 || roi.height < 14) return result;
      blob = letterboxBlob(bgr, roi, size, gender_nhwc_, 1.0f / 255.0f);
    }

    if (blob.empty()) return result;

    const std::array<int64_t, 4> shape = gender_nhwc_
      ? std::array<int64_t, 4>{1, size, size, 3}
      : std::array<int64_t, 4>{1, 3, size, size};
    std::vector<Ort::Value> outputs = gender_.run(blob, shape);
    if (outputs.empty()) return result;

    const auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    const int out_len = (out_shape.size() >= 2)
      ? static_cast<int>(out_shape[out_shape.size() - 1]) : 2;
    const int classes = gender_is_genderage_ ? std::min(2, out_len) : out_len;
    std::vector<float> probs;
    probabilities(outputs.front().GetTensorData<float>(), classes, probs);
    if (probs.size() >= 2) {
      float f_p = probs[0];
      float m_p = probs[1];

      // Fusi dengan postur tubuh antropometrik ZED skeleton (rasio bahu/pinggul)
      const float body_score = computeBodyGenderScore(kpt);
      f_p = std::clamp(f_p - body_score * 0.5f, 0.01f, 0.99f);
      m_p = std::clamp(m_p + body_score * 0.5f, 0.01f, 0.99f);
      const float norm = f_p + m_p;
      f_p /= norm;
      m_p /= norm;

      result.prob_female = f_p;
      result.prob_male = m_p;
      result.margin = m_p - f_p;
      result.confidence = std::max(f_p, m_p);
      result.is_high_quality = high_quality;

      float female_thresh = 0.57f;
      float male_thresh = 0.53f;
      if (!yunet_aligned && gender_is_genderage_) {
        female_thresh = 0.68f;
        male_thresh = 0.56f;
      }
      if (f_p >= female_thresh && f_p > m_p) {
        result.label = "female";
      } else if (m_p >= male_thresh && m_p > f_p) {
        result.label = "male";
      }
    }
    return result;
  }

  std::string gender(
    const std::vector<sl::float2> & kpt, const cv::Mat & bgr, const cv::Rect & bbox)
  {
    return predictGenderDetailed(kpt, bgr, bbox).label;
  }

  static cv::Mat twoEyeSimilarityTransform(
    const cv::Point2f & p1, const cv::Point2f & p2, int out_size)
  {
    const float s = static_cast<float>(out_size) / 112.0f;
    const cv::Point2f q1(38.2946f * s, 51.6963f * s);
    const cv::Point2f q2(73.5318f * s, 51.5014f * s);
    const double dx = p2.x - p1.x;
    const double dy = p2.y - p1.y;
    const double d_sq = dx * dx + dy * dy;
    if (d_sq < 16.0) return {};
    const double qx = q2.x - q1.x;
    const double qy = q2.y - q1.y;
    const double a = (dx * qx + dy * qy) / d_sq;
    const double b = (dx * qy - dy * qx) / d_sq;
    const double tx = q1.x - (a * p1.x - b * p1.y);
    const double ty = q1.y - (b * p1.x + a * p1.y);
    cv::Mat M(2, 3, CV_64F);
    M.at<double>(0, 0) = a;
    M.at<double>(0, 1) = -b;
    M.at<double>(0, 2) = tx;
    M.at<double>(1, 0) = b;
    M.at<double>(1, 1) = a;
    M.at<double>(1, 2) = ty;
    return M;
  }

  std::vector<float> faceEmbedding(
    const cv::Mat & bgr, const cv::Rect & bbox, const std::vector<sl::float2> & kpt)
  {
    if (!face_.ready() || bgr.empty() || bbox.width <= 0 || bbox.height <= 0) return {};

    // 1. Prioritaskan YuNet face detector untuk landmark 5 titik sub-pixel
    if (faceDetectorReady()) {
      DetectedFace df = detectFace(bgr, bbox);
      if (df.valid && df.score >= 0.28f) {
        const float eye_dist = std::hypot(df.left_eye.x - df.right_eye.x, df.left_eye.y - df.right_eye.y);
        if (eye_dist >= 4.5f) {
          std::vector<cv::Point2f> from{
            df.right_eye, df.left_eye, df.nose, df.right_mouth, df.left_mouth};
          std::vector<cv::Point2f> to{
            {38.2946f, 51.6963f},
            {73.5318f, 51.5014f},
            {56.0252f, 71.7366f},
            {41.5493f, 92.3655f},
            {70.7299f, 92.2041f}};
          cv::Mat transform = cv::estimateAffinePartial2D(from, to, cv::noArray(), cv::LMEDS);
          if (transform.empty()) transform = cv::estimateAffine2D(from, to);
          if (!transform.empty()) {
            cv::Mat aligned;
            cv::warpAffine(bgr, aligned, transform, cv::Size(112, 112),
              cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            return embedFace(aligned);
          }
        }
      }
    }

    // 2. Fallback ke landmark ZED skeleton 2-eye similarity transform
    cv::Point2f eye_left, eye_right, nose;
    if (faceLandmarks(kpt, eye_left, eye_right, nose)) {
      const float eye_dist = std::hypot(eye_right.x - eye_left.x, eye_right.y - eye_left.y);
      if (eye_dist >= 4.5f && nose.y >= std::min(eye_left.y, eye_right.y) - eye_dist * 0.2f) {
        // Pure 2-eye similarity transform agar jitter keypoint hidung ZED tidak merusak skala wajah
        cv::Mat transform = twoEyeSimilarityTransform(eye_left, eye_right, 112);
        if (!transform.empty()) {
          cv::Mat aligned;
          cv::warpAffine(bgr, aligned, transform, cv::Size(112, 112),
            cv::INTER_LINEAR, cv::BORDER_REPLICATE);
          return embedFace(aligned);
        }
      }
    }

    return {};
  }

  std::vector<float> extractFaceFromImage(const cv::Mat & img)
  {
    if (!face_.ready() || img.empty()) return {};

    // 1. Prioritaskan deteksi wajah YuNet + 5-landmark affine alignment jika detector siap
    if (faceDetectorReady()) {
      DetectedFace df = detectFace(img);
      if (df.valid && df.score >= 0.35f) {
        const float eye_dist = std::hypot(df.left_eye.x - df.right_eye.x, df.left_eye.y - df.right_eye.y);
        if (eye_dist >= 6.5f) {
          std::vector<cv::Point2f> from{
            df.right_eye, df.left_eye, df.nose, df.right_mouth, df.left_mouth};
          std::vector<cv::Point2f> to{
            {38.2946f, 51.6963f},
            {73.5318f, 51.5014f},
            {56.0252f, 71.7366f},
            {41.5493f, 92.3655f},
            {70.7299f, 92.2041f}};
          cv::Mat transform = cv::estimateAffinePartial2D(from, to, cv::noArray(), cv::LMEDS);
          if (transform.empty()) transform = cv::estimateAffine2D(from, to);
          if (!transform.empty()) {
            cv::Mat aligned;
            cv::warpAffine(img, aligned, transform, cv::Size(112, 112),
              cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            return embedFace(aligned);
          }
        }
      }
    }

    // 2. Fallback: Center crop persegi
    cv::Rect roi(0, 0, img.cols, img.rows);
    if (img.cols > img.rows) {
      const int diff = img.cols - img.rows;
      roi = cv::Rect(diff / 2, 0, img.rows, img.rows);
    } else if (img.rows > img.cols) {
      const int side = img.cols;
      roi = cv::Rect(0, 0, side, std::min(img.rows, side));
    }
    cv::Mat cropped = img(roi);
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(112, 112));
    return embedFace(resized);
  }

  static float matchFace(const std::vector<float> & a, const std::vector<float> & b)
  {
    if (a.empty() || b.empty() || a.size() != b.size()) return -1.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;
  }

  static std::string detectGesture(const std::vector<sl::float2> & kpt)
  {
    if (kpt.size() < 15) return "NONE";
    const std::string left = classifyArm(kpt, 5, 7);
    const std::string right = classifyArm(kpt, 12, 14);
    if (right == "DOWN_RIGHT" || left == "DOWN_RIGHT") return "RIGHT";
    if (left == "DOWN_LEFT" || right == "DOWN_LEFT") return "LEFT";
    return "NONE";
  }

  static float keypointDistance(const sl::float2 & a, const sl::float2 & b)
  {
    return std::hypot(a.x - b.x, a.y - b.y);
  }

  static float poseScale(const std::vector<sl::float2> & kpt)
  {
    if (kpt.size() > 12 && validKeypoint(kpt[5]) && validKeypoint(kpt[12])) {
      const float shoulder_width = keypointDistance(kpt[5], kpt[12]);
      if (shoulder_width > 10.0f) return shoulder_width;
    }
    if (kpt.size() > 3 && validKeypoint(kpt[3]) && validKeypoint(kpt[0])) {
      const float torso_length = keypointDistance(kpt[3], kpt[0]);
      if (torso_length > 10.0f) return torso_length * 0.5f;
    }
    return 30.0f;
  }

  static bool isHandRaised(
    const std::vector<sl::float2> & kpt, int shoulder, int elbow, int wrist, float margin)
  {
    if (shoulder >= static_cast<int>(kpt.size()) ||
        elbow >= static_cast<int>(kpt.size()) ||
        wrist >= static_cast<int>(kpt.size()))
    {
      return false;
    }

    const auto & s = kpt[static_cast<size_t>(shoulder)];
    const auto & e = kpt[static_cast<size_t>(elbow)];
    const auto & w = kpt[static_cast<size_t>(wrist)];
    if (!validKeypoint(s) || !validKeypoint(e) || !validKeypoint(w)) return false;

    // Image y axis grows downward: a raised hand is above the shoulder and elbow.
    if (w.y > s.y - margin) return false;
    if (w.y > e.y - margin * 0.5f) return false;
    if (keypointDistance(w, e) < margin) return false;
    return true;
  }

  static std::string detectWave(const std::vector<sl::float2> & kpt)
  {
    if (kpt.size() < 24) return "NONE";
    const float margin = std::max(8.0f, poseScale(kpt) * 0.25f);
    const bool left = isHandRaised(kpt, 5, 6, 7, margin);
    const bool right = isHandRaised(kpt, 12, 13, 14, margin);
    if (!left && !right) return "NONE";
    if (left && right) {
      return kpt[7].y <= kpt[14].y ? "WAVE_LEFT" : "WAVE_RIGHT";
    }
    return left ? "WAVE_LEFT" : "WAVE_RIGHT";
  }

  static std::string detectOrientation(const sl::BodyData & body, const cv::Rect & bbox)
  {
    const auto & kpt = body.keypoint_2d;
    if (kpt.size() < 34) return "FRONT";

    int back_votes = 0;
    int front_votes = 0;
    bool is_narrow_side = false;

    const float box_w = static_cast<float>(std::max(1, bbox.width));

    // 1. Shoulders (LEFT_SHOULDER = 5, RIGHT_SHOULDER = 12)
    if (validKeypoint(kpt[5]) && validKeypoint(kpt[12])) {
      const float dx = kpt[12].x - kpt[5].x; // Right shoulder x - Left shoulder x in image
      const float shoulder_w = std::abs(dx);
      if (shoulder_w < 0.18f * box_w) {
        is_narrow_side = true;
      } else if (dx > 0.0f) {
        back_votes += 3;
      } else {
        front_votes += 3;
      }
    }

    // 2. 3D Shoulders (sl::BodyData::keypoint in RIGHT_HANDED_Z_UP_X_FWD: +Y is left)
    if (body.keypoint.size() >= 34) {
      const auto & p5 = body.keypoint[5];
      const auto & p12 = body.keypoint[12];
      if (std::isfinite(p5.y) && std::isfinite(p12.y)) {
        const float dy = p5.y - p12.y; // Left shoulder Y - Right shoulder Y in 3D (+Y = left)
        if (std::abs(dy) > 10.0f) {
          if (dy > 0.0f) {
            back_votes += 2;
          } else {
            front_votes += 2;
          }
        }
      }
    }

    // 3. Hips (LEFT_HIP = 18, RIGHT_HIP = 22)
    if (validKeypoint(kpt[18]) && validKeypoint(kpt[22])) {
      const float dx = kpt[22].x - kpt[18].x;
      const float hip_w = std::abs(dx);
      if (hip_w >= 0.12f * box_w) {
        if (dx > 0.0f) {
          back_votes += 2;
        } else {
          front_votes += 2;
        }
      }
    }

    // 4. Facial landmarks (NOSE = 27, LEFT_EYE = 28, RIGHT_EYE = 30)
    const bool has_nose = validKeypoint(kpt[27]);
    const bool has_left_eye = validKeypoint(kpt[28]);
    const bool has_right_eye = validKeypoint(kpt[30]);

    if (has_nose && (has_left_eye || has_right_eye)) {
      if (has_left_eye && has_right_eye) {
        if (kpt[28].x > kpt[30].x) {
          front_votes += 3;
        } else {
          back_votes += 1;
        }
      } else {
        front_votes += 2;
      }
    } else if (!has_nose && !has_left_eye && !has_right_eye) {
      back_votes += 1;
    }

    // 5. Global Root Orientation Quaternion from ZED SDK (heading along +X is 0 deg)
    const auto & q = body.global_root_orientation;
    const float q_norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (q_norm_sq > 0.5f && q_norm_sq < 1.5f) {
      const float yaw = zed_vision::wrap180(
        zed_vision::quatToYaw(q.x, q.y, q.z, q.w) * 180.0f / zed_vision::PI_F);
      const float abs_yaw = std::abs(yaw);
      if (abs_yaw <= 55.0f) {
        back_votes += 3;
      } else if (abs_yaw >= 125.0f) {
        front_votes += 3;
      } else if (abs_yaw >= 70.0f && abs_yaw <= 110.0f) {
        is_narrow_side = true;
      }
    }

    if (is_narrow_side && std::abs(back_votes - front_votes) <= 1) {
      return "SIDE";
    }
    return (back_votes > front_votes) ? "BACK" : "FRONT";
  }

  static bool isFacingBack(const sl::BodyData & body, const cv::Rect & bbox)
  {
    return detectOrientation(body, bbox) == "BACK";
  }

private:
  std::vector<float> runSingleFaceEmbed(const cv::Mat & image)
  {
    const double scale = is_arcface_ ? (1.0 / 127.5) : 1.0;
    const cv::Scalar mean = is_arcface_ ? cv::Scalar(127.5, 127.5, 127.5) : cv::Scalar(0, 0, 0);
    cv::Mat blob = cv::dnn::blobFromImage(
      image, scale, cv::Size(112, 112), mean, true, false);
    std::array<int64_t, 4> shape{1, 3, 112, 112};
    std::vector<Ort::Value> outputs = face_.run(blob, shape);
    if (outputs.empty()) return {};

    const float * data = outputs.front().GetTensorData<float>();
    const auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    const int dim = (out_shape.size() >= 2) ? static_cast<int>(out_shape[1]) : 512;
    std::vector<float> feature(static_cast<size_t>(dim));
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
      feature[static_cast<size_t>(i)] = data[i];
      norm_sq += data[i] * data[i];
    }
    const float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-6f));
    for (int i = 0; i < dim; ++i) {
      feature[static_cast<size_t>(i)] *= inv_norm;
    }
    return feature;
  }

  std::vector<float> embedFace(const cv::Mat & image)
  {
    if (!face_.ready() || image.empty()) return {};

    // Normalisasi pencahayaan ringan (CLAHE pada channel L) + flip-average (TTA)
    cv::Mat enhanced = image;
    if (image.channels() == 3) {
      cv::Mat lab;
      cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
      std::vector<cv::Mat> ch(3);
      cv::split(lab, ch);
      auto clahe = cv::createCLAHE(2.0, cv::Size(4, 4));
      clahe->apply(ch[0], ch[0]);
      cv::merge(ch, lab);
      cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
    }

    std::vector<float> f1 = runSingleFaceEmbed(enhanced);
    if (f1.empty()) return {};

    cv::Mat flipped;
    cv::flip(enhanced, flipped, 1);
    std::vector<float> f2 = runSingleFaceEmbed(flipped);
    if (f2.size() != f1.size()) return f1;

    float norm_sq = 0.0f;
    for (size_t i = 0; i < f1.size(); ++i) {
      f1[i] = f1[i] + f2[i];
      norm_sq += f1[i] * f1[i];
    }
    const float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-6f));
    for (size_t i = 0; i < f1.size(); ++i) {
      f1[i] *= inv_norm;
    }
    return f1;
  }

  static bool faceLandmarks(
    const std::vector<sl::float2> & kpt, cv::Point2f & eye_left,
    cv::Point2f & eye_right, cv::Point2f & nose)
  {
    if (kpt.size() <= 30) return false;
    const auto valid = [](const sl::float2 & p) {
        return std::isfinite(p.x) && std::isfinite(p.y) && p.x > 0.0f && p.y > 0.0f;
      };
    if (!valid(kpt[27]) || !valid(kpt[28]) || !valid(kpt[30])) return false;

    eye_left = cv::Point2f(kpt[28].x, kpt[28].y);
    eye_right = cv::Point2f(kpt[30].x, kpt[30].y);
    nose = cv::Point2f(kpt[27].x, kpt[27].y);
    if (eye_left.x > eye_right.x) std::swap(eye_left, eye_right);
    const float eye_dist = std::hypot(
      eye_right.x - eye_left.x, eye_right.y - eye_left.y);
    return eye_dist >= 4.0f;
  }

  void configureGender()
  {
    gender_nhwc_ = false;
    gender_input_size_ = 224;
    gender_is_genderage_ = false;
    const auto shape = gender_.inputShape();
    const auto out_shape = gender_.outputShape();
    if (out_shape.size() >= 2 && out_shape.back() == 3) {
      gender_is_genderage_ = true;
    }
    if (shape.size() != 4) return;
    if (shape[3] == 3) {
      gender_nhwc_ = true;
      if (shape[1] > 0) gender_input_size_ = static_cast<int>(shape[1]);
      else if (shape[2] > 0) gender_input_size_ = static_cast<int>(shape[2]);
    } else if (shape[1] == 3) {
      gender_nhwc_ = false;
      if (shape[2] > 0) gender_input_size_ = static_cast<int>(shape[2]);
    } else if (shape[3] == -1 && shape[1] > 3) {
      gender_nhwc_ = true;
      gender_input_size_ = static_cast<int>(shape[1]);
    } else if (shape[1] == -1 && shape[3] > 3) {
      gender_nhwc_ = false;
      gender_input_size_ = static_cast<int>(shape[3]);
    }
    if (!gender_nhwc_ && gender_input_size_ == 96) {
      gender_is_genderage_ = true;
    }
  }

public:
  static cv::Rect faceRegion(
    const cv::Rect & bbox, const std::vector<sl::float2> & kpt, int cols, int rows)
  {
    int x1, y1, x2, y2;
    cv::Point2f eye_left, eye_right, nose;
    if (faceLandmarks(kpt, eye_left, eye_right, nose)) {
      const float eye_dist = std::hypot(
        eye_right.x - eye_left.x, eye_right.y - eye_left.y);
      const float center_x = (eye_left.x + eye_right.x) * 0.5f;
      const float center_y = (eye_left.y + eye_right.y) * 0.5f + eye_dist * 0.15f;
      const float half = eye_dist * 1.6f;
      x1 = static_cast<int>(center_x - half);
      x2 = static_cast<int>(center_x + half);
      y1 = static_cast<int>(center_y - half);
      y2 = static_cast<int>(center_y + half);
    } else if (kpt.size() > 28 && std::isfinite(kpt[26].x) && std::isfinite(kpt[26].y) &&
        kpt[26].x > 0 && kpt[26].y > 0)
    {
      float span = bbox.width * 0.45f;
      if (kpt.size() > 3 && std::isfinite(kpt[3].y) && kpt[3].y > kpt[26].y) {
        span = std::max(span, (kpt[3].y - kpt[26].y) * 2.0f);
      }
      x1 = static_cast<int>(kpt[26].x - span * 0.5f);
      x2 = static_cast<int>(kpt[26].x + span * 0.5f);
      y1 = static_cast<int>(kpt[26].y - span * 0.65f);
      y2 = static_cast<int>(kpt[26].y + span * 0.65f);
    } else {
      y1 = bbox.y;
      y2 = bbox.y + static_cast<int>(bbox.height * 0.22);
      x1 = bbox.x + static_cast<int>(bbox.width * 0.20);
      x2 = bbox.x + static_cast<int>(bbox.width * 0.80);
    }
    x1 = std::clamp(x1, 0, cols);
    x2 = std::clamp(x2, 0, cols);
    y1 = std::clamp(y1, 0, rows);
    y2 = std::clamp(y2, 0, rows);
    if (x2 <= x1 || y2 <= y1) return {};
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
  }

  cv::Rect faceCropRegion(
    const cv::Rect & bbox, const std::vector<sl::float2> & kpt,
    int cols, int rows, const cv::Mat & bgr = cv::Mat())
  {
    if (!bgr.empty() && faceDetectorReady()) {
      DetectedFace df = detectFace(bgr, bbox);
      if (df.valid && df.bbox.width > 15 && df.bbox.height > 15) {
        int pad_x = static_cast<int>(df.bbox.width * 0.35f);
        int pad_y = static_cast<int>(df.bbox.height * 0.40f);
        cv::Rect r(
          std::max(0, df.bbox.x - pad_x),
          std::max(0, df.bbox.y - pad_y),
          std::min(cols - std::max(0, df.bbox.x - pad_x), df.bbox.width + pad_x * 2),
          std::min(rows - std::max(0, df.bbox.y - pad_y), df.bbox.height + pad_y * 2)
        );
        r &= cv::Rect(0, 0, cols, rows);
        if (r.width > 20 && r.height > 20) return r;
      }
    }

    cv::Rect face_roi = faceRegion(bbox, kpt, cols, rows);
    if (face_roi.width > 20 && face_roi.height > 20) {
      int pad_x = static_cast<int>(face_roi.width * 0.35f);
      int pad_y = static_cast<int>(face_roi.height * 0.40f);
      cv::Rect r(
        std::max(0, face_roi.x - pad_x),
        std::max(0, face_roi.y - pad_y),
        std::min(cols - std::max(0, face_roi.x - pad_x), face_roi.width + pad_x * 2),
        std::min(rows - std::max(0, face_roi.y - pad_y), face_roi.height + pad_y * 2)
      );
      return r & cv::Rect(0, 0, cols, rows);
    }

    cv::Rect fallback_roi = bbox;
    fallback_roi.height = std::min(rows - fallback_roi.y, static_cast<int>(fallback_roi.height * 0.55f));
    return fallback_roi & cv::Rect(0, 0, cols, rows);
  }

private:
  static cv::Mat letterboxBlob(
    const cv::Mat & bgr, const cv::Rect & roi, int size, bool nhwc,
    float scale_factor = 1.0f / 255.0f)
  {
    if (bgr.empty() || size <= 0) return {};
    const cv::Rect safe = roi & cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (safe.width <= 0 || safe.height <= 0) return {};

    const float scale = std::min(
      static_cast<float>(size) / static_cast<float>(safe.width),
      static_cast<float>(size) / static_cast<float>(safe.height));
    const int w = std::max(1, static_cast<int>(std::round(safe.width * scale)));
    const int h = std::max(1, static_cast<int>(std::round(safe.height * scale)));

    cv::Mat resized;
    cv::resize(bgr(safe), resized, cv::Size(w, h));
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(canvas(cv::Rect((size - w) / 2, (size - h) / 2, w, h)));

    cv::Mat rgb;
    if (!nhwc) cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    if (nhwc) {
      cv::Mat blob(1, size * size * 3, CV_32F);
      const uchar * src = canvas.ptr<uchar>();
      float * dst = blob.ptr<float>();
      for (int i = 0; i < size * size * 3; ++i) dst[i] = src[i] * scale_factor;
      return blob;
    }
    return cv::dnn::blobFromImage(
      rgb, scale_factor, cv::Size(size, size), cv::Scalar(0, 0, 0), false, false);
  }

  static void probabilities(
    const float * data, int count, std::vector<float> & probs)
  {
    probs.assign(static_cast<size_t>(std::max(count, 0)), 0.0f);
    if (count <= 0 || data == nullptr) return;
    float sum = 0.0f;
    bool normalized = true;
    for (int i = 0; i < count; ++i) {
      const float value = data[i];
      sum += value;
      if (!std::isfinite(value) || value < 0.0f || value > 1.0f) normalized = false;
    }
    if (normalized && std::fabs(sum - 1.0f) < 0.05f) {
      for (int i = 0; i < count; ++i) probs[static_cast<size_t>(i)] = data[i];
      return;
    }
    softmax(data, count, probs);
  }

  static std::string classifyArm(
    const std::vector<sl::float2> & kpt, int shoulder, int wrist)
  {
    constexpr float threshold = 0.5f;
    if (shoulder >= static_cast<int>(kpt.size()) || wrist >= static_cast<int>(kpt.size())) return "";
    const auto & s = kpt[static_cast<size_t>(shoulder)];
    const auto & w = kpt[static_cast<size_t>(wrist)];
    if (!std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(w.x) || !std::isfinite(w.y) ||
        s.x < 0.f || s.y < 0.f || w.x < 0.f || w.y < 0.f)
    {
      return "";
    }
    const float dx = w.x - s.x, dy = w.y - s.y;
    if (dy <= 0.0f || std::fabs(dx) < threshold * dy) return "";
    return dx < 0.0f ? "DOWN_LEFT" : "DOWN_RIGHT";
  }

  static void softmax(const float * logits, int count, std::vector<float> & probs)
  {
    probs.assign(static_cast<size_t>(std::max(count, 0)), 0.0f);
    if (count <= 0) return;
    float max_logit = -1e9f;
    for (int i = 0; i < count; ++i) max_logit = std::max(max_logit, logits[i]);
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
      probs[static_cast<size_t>(i)] = std::exp(logits[i] - max_logit);
      sum += probs[static_cast<size_t>(i)];
    }
    if (sum <= 0.0f) return;
    for (int i = 0; i < count; ++i) probs[static_cast<size_t>(i)] /= sum;
  }

  OrtModel gender_, color_, hair_, face_;
  std::vector<std::string> color_labels_, hair_labels_;
  bool gender_nhwc_{true};
  bool gender_is_genderage_{false};
  int gender_input_size_{224};
  cv::Ptr<cv::FaceDetectorYN> face_detector_;
  float face_detect_conf_{0.55f};
};

class ZedNode : public rclcpp::Node
{
public:
  ZedNode();
  ~ZedNode() override;

private:
  struct RegisteredPerson
  {
    std::string name;
    std::vector<float> feature;
    std::vector<std::vector<float>> face_gallery;
    int feature_samples{0};
    cv::Scalar torso_hsv{0, 0, 0};
    std::string clothing_color;
    std::string pants_color;
    std::string hair_color;
    std::string gender;
    bool has_hat{false};
    bool has_glasses{false};
    bool has_color{false};
    bool locked{false};
    bool has_face_verified{false};
    int last_tracking_id{-1};
    double last_x{0.0}, last_y{0.0};
    bool has_last_pos{false};
    rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
  };

  struct OperatorMemory
  {
    bool active{false};
    int stable_id{0};
    int last_zed_id{-1};
    double x{0.0}, y{0.0}, vx{0.0}, vy{0.0};
    cv::Scalar torso_hsv{0, 0, 0};
    bool has_color{false};
    rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
  };

  struct WaveMemory
  {
    std::string side;
    int streak{0};
    rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
  };

  struct BackMemory
  {
    int streak{0};
    rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
  };

  struct PersonNameMemory
  {
    std::string confirmed_name;
    std::string candidate_name;
    int candidate_streak{0};
    std::deque<std::string> history;
    rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
  };

  struct ShirtColorMemory
  {
    std::deque<std::string> history;
    rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
  };

  struct GenderMemory
  {
    std::deque<std::string> history;
    double accumulated_margin{0.0};
    int high_quality_samples{0};
    rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
  };

  std::string updateGenderVote(
    int id, const AttributeAnalyzer::GenderPrediction & pred, const rclcpp::Time & now_time)
  {
    auto & mem = gender_memory_[id];
    mem.last_seen = now_time;

    if (pred.label != "unknown") {
      const double weight = pred.is_high_quality ? 1.6 : 0.7;
      mem.accumulated_margin = std::clamp(
        mem.accumulated_margin * 0.88 + static_cast<double>(pred.margin) * weight,
        -5.0, 5.0
      );
      if (pred.is_high_quality) mem.high_quality_samples++;
      mem.history.push_back(pred.label);
      if (mem.history.size() > 9) mem.history.pop_front();
    }

    if (mem.accumulated_margin >= 0.65) return "male";
    if (mem.accumulated_margin <= -0.65) return "female";

    if (!mem.history.empty()) {
      int m_cnt = 0, f_cnt = 0;
      for (const auto & v : mem.history) {
        if (v == "male") m_cnt++;
        else if (v == "female") f_cnt++;
      }
      if (m_cnt >= f_cnt + 2) return "male";
      if (f_cnt >= m_cnt + 2) return "female";
    }

    return "unknown";
  }

  static std::string updateAttributeVote(
    std::unordered_map<int, ShirtColorMemory> & memory_map,
    int id, const std::string & raw_val, const rclcpp::Time & now_time)
  {
    auto & mem = memory_map[id];
    mem.history.push_back(raw_val);
    if (mem.history.size() > 7) mem.history.pop_front();
    mem.last_seen = now_time;

    std::unordered_map<std::string, int> counts;
    for (const auto & v : mem.history) counts[v]++;
    std::string best = raw_val;
    int best_cnt = 0;
    for (const auto & v : mem.history) {
      const int cnt = counts[v];
      if (cnt >= best_cnt) {
        best_cnt = cnt;
        best = v;
      }
    }
    return best;
  }

  void publishYaw();
  void loop();
  void processBags(const cv::Mat & bgr, const sl::Mat & cloud);
  void processDrinks(const cv::Mat & bgr, const sl::Mat & cloud);
  void processPersons(const cv::Mat & bgr, const sl::Mat & cloud);
  void processChairs(const cv::Mat & bgr, const sl::Mat & cloud);
  void showPreview(
    const cv::Mat & bgr,
    const std::unordered_map<int, cv::Rect> * live_bboxes = nullptr);

  std::string resolvePath(const std::string & path) const;
  void loadModels();
  void loadPreRegisteredHost(const std::string & custom_photo_path = "");
  void loadYolo(
    YoloDetector & detector, const std::string & model_param,
    const std::vector<std::string> & fallbacks, const std::string & name,
    float conf, float nms, int size);

  sl::Camera zed_;
  sl::Mat zed_image_;
  sl::Mat zed_point_cloud_;
  sl::Bodies bodies_;
  sl::BodyTrackingRuntimeParameters body_runtime_parameters_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr bag_dets_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr drink_dets_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr persons_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr recognized_name_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr guest_1_name_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr guest_2_name_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr gesture_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr attributes_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr empty_seat_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr chair_dets_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr order_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr person_back_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  bool publish_yaw_{true};
  bool yaw_360_{true};
  double camera_fps_{30.0};
  bool enable_body_tracking_{true};
  bool debug_window_{false};
  bool mirror_{true};

  AttributeAnalyzer analyzer_;
  std::vector<RegisteredPerson> registered_persons_;
  std::string pending_register_name_;
  std::string active_guest_name_;
  rclcpp::Time register_start_time_{0, 0, RCL_ROS_TIME};
  std::string guest_1_name_;
  std::string guest_2_name_;
  double face_threshold_{0.40};

  void publishGuestNames();

  std::vector<PrevPerson> preview_persons_;
  std::vector<PrevBag> preview_bags_;
  std::vector<PrevItem> preview_drinks_;
  std::vector<PrevChair> preview_chairs_;

  std::mutex preview_mutex_;
  std::vector<PrevPerson> disp_persons_;
  std::vector<PrevBag> disp_bags_;
  std::vector<PrevItem> disp_drinks_;
  std::vector<PrevChair> disp_chairs_;

  std::mutex frame_mutex_;
  std::condition_variable worker_cv_;
  std::atomic<bool> worker_running_{false};
  std::atomic<bool> worker_busy_{false};
  bool has_new_frame_{false};
  cv::Mat worker_bgr_;
  sl::Mat worker_cloud_;
  sl::Bodies worker_bodies_;
  std::thread worker_thread_;
  void inferenceWorkerLoop();

  YoloDetector yolo_;
  YoloDetector coco_bag_yolo_;
  YoloDetector botol_yolo_;
  YoloDetector cracker_yolo_;
  YoloDetector pringles_yolo_;
  YoloDetector sugarbox_yolo_;
  YoloDetector soda_yolo_;
  YoloDetector milk_yolo_;
  YoloDetector chair_yolo_;
  YoloDetector hat_yolo_;
  YoloDetector glasses_yolo_;
  float hat_conf_{0.60f};

  bool enable_yolo_{true};
  bool enable_bag_{true};
  bool enable_drinks_{true};
  bool enable_chair_{true};
  std::vector<std::string> bag_labels_;
  std::vector<std::string> chair_labels_;
  double yolo_min_area_{500.0};
  double drink_min_area_{180.0};
  int yolo_stride_{3};
  int yolo_frame_{0};
  int drink_frame_{0};
  int chair_frame_{0};
  int person_attr_frame_{0};
  std::vector<YoloDetection> last_bag_dets_;
  std::vector<YoloDetection> last_drink_dets_;
  std::vector<YoloDetection> last_chair_dets_;

  OperatorMemory op_mem_;
  std::unordered_map<int, WaveMemory> wave_memory_;
  bool enable_back_detection_{true};
  std::string person_back_topic_{"/vision/person_back"};
  std::unordered_map<int, BackMemory> back_memory_;
  std::unordered_map<int, PersonNameMemory> name_memory_;
  std::unordered_map<int, ShirtColorMemory> shirt_color_memory_;
  std::unordered_map<int, ShirtColorMemory> pants_color_memory_;
  std::unordered_map<int, ShirtColorMemory> hat_memory_;
  std::unordered_map<int, ShirtColorMemory> glasses_memory_;
  std::unordered_map<int, GenderMemory> gender_memory_;
  std::string task_mode_{"all"};
  bool is_help_me_carry_{false};
  bool is_receptionist_{false};
  bool is_restaurant_{false};
  double person_max_dist_{2.0};
  double person_seated_max_dist_{3.0};
  double chair_max_dist_{2.0};
  double bag_max_dist_{2.0};
  double drink_max_dist_{2.5};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_name_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr host_photo_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr capture_face_srv_;
  std::atomic<bool> capture_face_requested_{false};
  std::mutex state_mutex_;

  // State log ringkasan per frame
  bool latest_operator_found_{false};
  double latest_op_x_{0.0}, latest_op_y_{0.0}, latest_op_dist_{0.0};
  bool latest_back_found_{false};
  double latest_back_x_{0.0}, latest_back_y_{0.0}, latest_back_dist_{0.0};
  std::string latest_gesture_{""};
  bool latest_guest_found_{false};
  std::string latest_guest_tag_{""};
  double latest_guest_x_{0.0}, latest_guest_y_{0.0}, latest_guest_dist_{0.0};
  bool latest_customer_found_{false};
  double latest_cust_x_{0.0}, latest_cust_y_{0.0}, latest_cust_dist_{0.0};
  bool latest_wave_found_{false};
  double latest_wave_x_{0.0}, latest_wave_y_{0.0};
  rclcpp::Time last_summary_log_time_{0, 0, RCL_ROS_TIME};

  void printSummaryLog();
};

ZedNode::ZedNode()
: Node("vision")
{
  declare_parameter("task", "all");
  declare_parameter("camera_fps", 30.0);
  declare_parameter("depth_mode", std::string("performance"));
  declare_parameter("publish_yaw", true);
  declare_parameter("yaw_360", true);
  declare_parameter("enable_body_tracking", true);
  declare_parameter("depth_minimum_distance", 30.0);
  declare_parameter("person_max_dist", 2.0);
  declare_parameter("person_seated_max_dist", 3.0);
  declare_parameter("bag_max_dist", 2.0);
  declare_parameter("enable_yolo", true);
  declare_parameter("debug_window", false);
  declare_parameter("mirror", true);
  declare_parameter("yolo_model", std::string("models/yolov8n_sim.onnx"));
  declare_parameter("yolo_classes", std::string("models/coco_names.txt"));
  declare_parameter("yolo_conf", 0.35);
  declare_parameter("bag_conf", 0.35);
  declare_parameter("yolo_nms", 0.45);
  declare_parameter("yolo_size", 640);
  declare_parameter("yolo_stride", 3);
  declare_parameter("bag_labels", std::vector<std::string>{"bag", "paper_bag", "backpack", "handbag", "suitcase"});
  declare_parameter("yolo_min_area", 250.0);
  declare_parameter("drink_min_area", 180.0);
  declare_parameter("drink_conf", 0.48);
  declare_parameter("water_conf", 0.25);
  declare_parameter("drink_max_dist", 2.5);
  declare_parameter("botol_model", std::string("models/water.onnx"));
  declare_parameter("soda_model", std::string("models/soda.onnx"));
  declare_parameter("cracker_model", std::string("models/crackerbox.onnx"));
  declare_parameter("pringles_model", std::string("models/pringles.onnx"));
  declare_parameter("sugarbox_model", std::string("models/sugarbox.onnx"));
  declare_parameter("milk_model", std::string("models/milk.onnx"));
  declare_parameter("enable_drinks", true);
  declare_parameter("chair_model", std::string(""));
  declare_parameter("enable_chair", true);
  declare_parameter("chair_max_dist", 3.0);
  declare_parameter("chair_labels", std::vector<std::string>{"chair", "couch", "sofa", "bench"});
  declare_parameter("enable_bag", true);
  declare_parameter("gender_model", std::string("models/genderage.onnx"));
  declare_parameter("color_model", std::string("models/color_model.onnx"));
  declare_parameter("color_labels", std::string("models/color_names.txt"));
  declare_parameter("hair_model", std::string("models/hair_model.onnx"));
  declare_parameter("hair_labels", std::string("models/hair_names.txt"));
  declare_parameter("hat_model", std::string("models/hat.onnx"));
  declare_parameter("glasses_model", std::string("models/glasses.onnx"));
  declare_parameter("accessory_conf", 0.18);
  declare_parameter("hat_conf", 0.60);
  declare_parameter("face_model", std::string("models/arcface_w600k_r50.onnx"));
  declare_parameter("face_threshold", 0.40);
  declare_parameter("face_detect_model", std::string("models/face_detection_yunet.onnx"));
  declare_parameter("face_detect_conf", 0.55);
  declare_parameter("camera_resolution", std::string(""));
  declare_parameter("host_name", std::string("Revaldo"));
  declare_parameter("host_photo", std::string("models/host.jpg"));
  declare_parameter("host_shirt_color", std::string("black shirt"));
  declare_parameter("host_pants_color", std::string(""));
  declare_parameter("host_gender", std::string("male"));
  declare_parameter("host_hair_color", std::string("black"));
  declare_parameter("host_hat", false);
  declare_parameter("host_glasses", false);
  declare_parameter("enable_back_detection", true);
  declare_parameter("person_back_topic", std::string("/vision/person_back"));

  task_mode_ = get_parameter("task").as_string();
  std::string task = task_mode_;
  std::transform(task.begin(), task.end(), task.begin(), ::tolower);
  is_help_me_carry_ = false;
  is_receptionist_ = false;
  is_restaurant_ = false;
  if (task == "help_me_carry" || task == "hmc") {
    is_help_me_carry_ = true;
  } else if (task == "receptionist" || task == "rec") {
    is_receptionist_ = true;
  } else if (task == "restaurant" || task == "resto" || task == "waiter") {
    is_restaurant_ = true;
  } else {
    // Mode "all": aktifkan semua
    is_help_me_carry_ = true;
    is_receptionist_ = true;
    is_restaurant_ = true;
  }
  enable_bag_ = get_parameter("enable_bag").as_bool();
  RCLCPP_INFO(
    get_logger(), "Vision task mode: %s (enable_bag: %s)",
    task_mode_.c_str(), enable_bag_ ? "true" : "false");

  camera_fps_ = get_parameter("camera_fps").as_double();
  publish_yaw_ = get_parameter("publish_yaw").as_bool();
  yaw_360_ = get_parameter("yaw_360").as_bool();
  enable_body_tracking_ = get_parameter("enable_body_tracking").as_bool();
  if (camera_fps_ <= 0.0) throw std::runtime_error("camera_fps must be greater than zero");

  yaw_publisher_ = create_publisher<std_msgs::msg::Float64>("/vision/yaw_deg", 10);
  bag_dets_publisher_ = create_publisher<std_msgs::msg::String>("/vision/bag_detections", 10);
  drink_dets_publisher_ = create_publisher<std_msgs::msg::String>("/vision/drink_detections", 10);
  persons_publisher_ = create_publisher<std_msgs::msg::String>("/vision/person_label", 10);
  recognized_name_publisher_ =
    create_publisher<std_msgs::msg::String>("/vision/recognized_name", 10);
  guest_1_name_publisher_ =
    create_publisher<std_msgs::msg::String>("/vision/guest_1_name", 10);
  guest_2_name_publisher_ =
    create_publisher<std_msgs::msg::String>("/vision/guest_2_name", 10);
  gesture_publisher_ = create_publisher<std_msgs::msg::String>("/vision/gesture", 10);
  attributes_publisher_ = create_publisher<std_msgs::msg::String>("/vision/person_attributes", 10);
  empty_seat_publisher_ = create_publisher<std_msgs::msg::String>("/vision/empty_seat", 10);
  chair_dets_publisher_ = create_publisher<std_msgs::msg::String>("/vision/chair_detections", 10);
  order_publisher_ = create_publisher<std_msgs::msg::String>("/vision/order", 10);

  enable_back_detection_ = get_parameter("enable_back_detection").as_bool();
  person_back_topic_ = get_parameter("person_back_topic").as_string();
  person_back_publisher_ = create_publisher<std_msgs::msg::String>(person_back_topic_, 10);

  capture_face_srv_ = create_service<std_srvs::srv::Trigger>(
    "/vision/camera_capture_face",
    [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
           std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
      capture_face_requested_ = true;
      res->success = true;
      res->message = "Customer face capture triggered successfully";
      RCLCPP_INFO(get_logger(), "\033[1;32m[VISION]\033[0m Service /vision/camera_capture_face triggered");
    });

  person_name_sub_ = create_subscription<std_msgs::msg::String>(
    "/vision/set_person_name", 10,
    [this](const std_msgs::msg::String::SharedPtr msg) {
      std::string name = msg->data;
      while (!name.empty() &&
        (name.back() == ' ' || name.back() == '\r' || name.back() == '\n' || name.back() == '\t'))
      {
        name.pop_back();
      }
      while (!name.empty() && (name.front() == ' ' || name.front() == '\t')) name.erase(name.begin());
      std::lock_guard<std::mutex> state_lk(state_mutex_);

      // Support "rename:<old_name>:<new_name>" so 1.5m pre-scanned profiles can be bound to the spoken name at 0.7m
      if (name.rfind("rename:", 0) == 0) {
        const auto sep = name.find(':', 7);
        if (sep != std::string::npos) {
          const std::string old_name = name.substr(7, sep - 7);
          const std::string new_name = name.substr(sep + 1);
          if (!new_name.empty()) {
            bool renamed = false;
            for (auto & rp : registered_persons_) {
              if (rp.name == old_name) {
                rp.name = new_name;
                renamed = true;
                RCLCPP_INFO(get_logger(),
                  "\033[1;32m[RECEPTIONIST]\033[0m Renamed scanned profile '%s' -> '%s' (%s, %s, %s, face=%s)",
                  old_name.c_str(), new_name.c_str(),
                  rp.clothing_color.c_str(), rp.hair_color.c_str(), rp.gender.c_str(),
                  rp.feature.empty() ? "NO" : "OK");
                break;
              }
            }
            for (auto & nm_kv : name_memory_) {
              if (nm_kv.second.confirmed_name == old_name) {
                nm_kv.second.confirmed_name = new_name;
              }
            }
            if (pending_register_name_ == old_name) {
              pending_register_name_ = new_name;
            }
            if (active_guest_name_ == old_name) {
              active_guest_name_ = new_name;
            }
            if (guest_1_name_.empty() || guest_1_name_ == old_name) {
              guest_1_name_ = new_name;
            } else if ((guest_2_name_.empty() || guest_2_name_ == old_name) && new_name != guest_1_name_) {
              guest_2_name_ = new_name;
            }
            if (renamed) {
              return;
            }
            // Fallback: if old_name wasn't in registered_persons_, register new_name normally
            name = new_name;
          }
        }
      }

      pending_register_name_ = name;
      active_guest_name_ = name;
      capture_face_requested_ = true;
      const bool is_temp_pending = (name.rfind("__pending_", 0) == 0);
      const std::string host_param_name = get_parameter("host_name").as_string();
      if (!name.empty()) {
        register_start_time_ = now();
        if (!is_temp_pending && name != host_param_name) {
          if (guest_1_name_.empty()) {
            guest_1_name_ = name;
          } else if (guest_2_name_.empty() && name != guest_1_name_) {
            guest_2_name_ = name;
          }
        }

        int first_person_id = -1;
        {
          std::lock_guard<std::mutex> p_lk(preview_mutex_);
          if (!disp_persons_.empty()) {
            first_person_id = disp_persons_[0].id;
          }
        }

        // Reset lock agar registrasi baru dapat mengumpulkan multi-sample wajah & atribut terbaru
        bool exists = false;
        for (auto & rp : registered_persons_) {
          if (rp.name == name) {
            rp.last_tracking_id = first_person_id;
            rp.last_seen = register_start_time_;
            rp.feature_samples = 0;
            if (name != host_param_name) {
              rp.feature.clear();
              rp.face_gallery.clear();
              rp.clothing_color.clear();
              rp.hair_color.clear();
              rp.gender.clear();
            }
            rp.locked = false;
            rp.has_face_verified = false;
            exists = true;
            break;
          }
        }
        if (!exists && first_person_id >= 0) {
          RegisteredPerson rp;
          rp.name = name;
          rp.last_tracking_id = first_person_id;
          rp.last_seen = register_start_time_;
          rp.locked = false;
          registered_persons_.push_back(rp);
        }
      } else {
        pending_register_name_.clear();
        active_guest_name_.clear();
      }
      publishGuestNames();
    });

  host_photo_sub_ = create_subscription<std_msgs::msg::String>(
    "/vision/load_host_photo", 10,
    [this](const std_msgs::msg::String::SharedPtr msg) {
      std::string path = msg->data;
      while (!path.empty() &&
        (path.back() == ' ' || path.back() == '\r' || path.back() == '\n' || path.back() == '\t'))
      {
        path.pop_back();
      }
      while (!path.empty() && (path.front() == ' ' || path.front() == '\t')) path.erase(path.begin());
      std::lock_guard<std::mutex> state_lk(state_mutex_);
      RCLCPP_INFO(get_logger(),
        "\033[1;32m[HOST PHOTO]\033[0m Request received to load/detect Host from photo: '%s'", path.c_str());
      loadPreRegisteredHost(path);
    });

  sl::InitParameters init;
  init.sdk_gpu_id = 0;
  std::string dm = get_parameter("depth_mode").as_string();
  std::transform(dm.begin(), dm.end(), dm.begin(), ::tolower);
  if (dm == "ultra") {
    init.depth_mode = sl::DEPTH_MODE::ULTRA;
  } else if (dm == "quality") {
    init.depth_mode = sl::DEPTH_MODE::QUALITY;
  } else if (dm == "neural") {
    init.depth_mode = sl::DEPTH_MODE::NEURAL;
  } else if (dm == "neural_light") {
    init.depth_mode = sl::DEPTH_MODE::NEURAL_LIGHT;
  } else {
    init.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
  }
  init.coordinate_units = sl::UNIT::CENTIMETER;
  std::string res_str = get_parameter("camera_resolution").as_string();
  std::transform(res_str.begin(), res_str.end(), res_str.begin(), ::tolower);
  if (res_str == "hd720" || (res_str.empty() && is_receptionist_ && !is_help_me_carry_ && !is_restaurant_)) {
    init.camera_resolution = sl::RESOLUTION::HD720;
    RCLCPP_INFO(get_logger(), "\033[1;32m[CAMERA]\033[0m Resolution: HD720 (1280x720) optimized for Receptionist");
  } else if (res_str == "hd1080") {
    init.camera_resolution = sl::RESOLUTION::HD1080;
    RCLCPP_INFO(get_logger(), "\033[1;32m[CAMERA]\033[0m Resolution: HD1080 (1920x1080)");
  } else if (res_str == "hd2k") {
    init.camera_resolution = sl::RESOLUTION::HD2K;
    RCLCPP_INFO(get_logger(), "\033[1;32m[CAMERA]\033[0m Resolution: HD2K (2208x1242)");
  } else {
    init.camera_resolution = sl::RESOLUTION::VGA;
    RCLCPP_INFO(get_logger(), "\033[1;32m[CAMERA]\033[0m Resolution: VGA (672x376) optimized for %s",
      is_help_me_carry_ ? "Help Me Carry" : (is_restaurant_ ? "Restaurant" : "General Tracking"));
  }
  init.camera_fps = static_cast<int>(camera_fps_);
  init.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  init.depth_minimum_distance =
    static_cast<float>(get_parameter("depth_minimum_distance").as_double());

  RCLCPP_INFO(get_logger(), "ZED camera capturing started");
  sl::ERROR_CODE state = sl::ERROR_CODE::FAILURE;
  for (int attempt = 1; attempt <= 5; ++attempt) {
    state = zed_.open(init);
    if (state == sl::ERROR_CODE::SUCCESS) break;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  if (state != sl::ERROR_CODE::SUCCESS) {
    RCLCPP_ERROR(get_logger(), "ZED open failed: %s", sl::toString(state).c_str());
    throw std::runtime_error(std::string("ZED open failed: ") + sl::toString(state).c_str());
  }

  if (zed_.enablePositionalTracking(sl::PositionalTrackingParameters()) !=
      sl::ERROR_CODE::SUCCESS)
  {
    zed_.close();
    throw std::runtime_error("Enable positional tracking failed");
  }

  sl::BodyTrackingParameters body_params;
  body_params.detection_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_FAST;
  body_params.body_format = sl::BODY_FORMAT::BODY_34;
  body_params.enable_tracking = true;
  body_params.enable_body_fitting = false;
  if (enable_body_tracking_ && zed_.enableBodyTracking(body_params) != sl::ERROR_CODE::SUCCESS) {
    zed_.close();
    throw std::runtime_error("Enable body tracking failed");
  }
  body_runtime_parameters_.detection_confidence_threshold = 25;

  enable_yolo_ = get_parameter("enable_yolo").as_bool();
  debug_window_ = get_parameter("debug_window").as_bool();
  mirror_ = get_parameter("mirror").as_bool();
  bag_labels_ = get_parameter("bag_labels").as_string_array();
  chair_labels_ = get_parameter("chair_labels").as_string_array();
  yolo_min_area_ = get_parameter("yolo_min_area").as_double();
  drink_min_area_ = get_parameter("drink_min_area").as_double();
  yolo_stride_ = std::max(1, static_cast<int>(get_parameter("yolo_stride").as_int()));
  enable_chair_ = get_parameter("enable_chair").as_bool();
  enable_drinks_ = get_parameter("enable_drinks").as_bool();
  face_threshold_ = get_parameter("face_threshold").as_double();
  person_max_dist_ = get_parameter("person_max_dist").as_double();
  person_seated_max_dist_ = get_parameter("person_seated_max_dist").as_double();
  chair_max_dist_ = get_parameter("chair_max_dist").as_double();
  bag_max_dist_ = get_parameter("bag_max_dist").as_double();
  drink_max_dist_ = get_parameter("drink_max_dist").as_double();

  if (is_help_me_carry_ && !is_receptionist_ && !is_restaurant_) {
    enable_body_tracking_ = true;
    enable_bag_ = true;
    enable_back_detection_ = true;
    enable_drinks_ = false;
    enable_chair_ = false;
    RCLCPP_INFO(get_logger(), "\033[1;32m[HELP_ME_CARRY]\033[0m Active: Person, Bag, Back");
  } else if (is_receptionist_ && !is_help_me_carry_ && !is_restaurant_) {
    enable_body_tracking_ = true;
    enable_drinks_ = true;
    enable_bag_ = false;
    enable_back_detection_ = false;
    enable_chair_ = true;
    enable_yolo_ = true;
    RCLCPP_INFO(get_logger(), "\033[1;32m[RECEPTIONIST]\033[0m Active: Person, Chair, Soda, Milk, Water, Attributes (Color, Gender, Name)");
  } else if (is_restaurant_ && !is_help_me_carry_ && !is_receptionist_) {
    enable_body_tracking_ = true;
    enable_drinks_ = true;
    enable_bag_ = false;
    enable_back_detection_ = false;
    enable_chair_ = false;
    RCLCPP_INFO(get_logger(), "\033[1;32m[RESTAURANT]\033[0m Active: Soda, Pringles, Sugar, Cracker");
  }

  loadModels();

  worker_running_ = true;
  worker_thread_ = std::thread(&ZedNode::inferenceWorkerLoop, this);

  // Poll faster than camera_fps_ so zed_.grab() paces the loop cleanly at camera_fps_ without timer-boundary skips
  timer_ = create_wall_timer(
    std::chrono::milliseconds(5), std::bind(&ZedNode::loop, this));
}

ZedNode::~ZedNode()
{
  worker_running_ = false;
  worker_cv_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  zed_.close();
}

std::string ZedNode::resolvePath(const std::string & path) const
{
  if (path.empty()) return "";
  if (std::filesystem::path(path).is_absolute()) return path;

  auto tryResolve = [](const std::string & rel) -> std::string {
    if (rel.empty()) return "";
    try {
      const std::string share =
        ament_index_cpp::get_package_share_directory("athome_vision") + "/" + rel;
      if (std::filesystem::exists(share)) return share;
    } catch (...) {
    }
    if (std::filesystem::exists(rel)) return rel;
    if (std::filesystem::exists("src/athome_vision/" + rel)) return "src/athome_vision/" + rel;
    const std::string ws0 = "/home/dell/Documents/athome_ws/src/athome_vision/" + rel;
    if (std::filesystem::exists(ws0)) return ws0;
    const std::string ws = "/home/dell/Desktop/athome_ws/src/athome_vision/" + rel;
    if (std::filesystem::exists(ws)) return ws;
    const std::string ws2 = "/home/dell/Downloads/athome_ws/src/athome_vision/" + rel;
    if (std::filesystem::exists(ws2)) return ws2;
    return "";
  };

  std::string found = tryResolve(path);
  if (!found.empty()) return found;

  // Jika path berbentuk "models/<file>" tanpa subfolder task, cari di folder task terkait
  if (path.rfind("models/", 0) == 0 && path.find('/', 7) == std::string::npos) {
    const std::string fname = path.substr(7);
    std::vector<std::string> task_dirs;
    if (is_receptionist_ && !is_restaurant_ && !is_help_me_carry_) {
      task_dirs = {"receptionist", "restaurant", "help_me_carry"};
    } else if (is_restaurant_ && !is_receptionist_ && !is_help_me_carry_) {
      task_dirs = {"restaurant", "receptionist", "help_me_carry"};
    } else if (is_help_me_carry_ && !is_receptionist_ && !is_restaurant_) {
      task_dirs = {"help_me_carry", "receptionist", "restaurant"};
    } else {
      task_dirs = {"receptionist", "restaurant", "help_me_carry"};
    }
    for (const auto & dir : task_dirs) {
      found = tryResolve("models/" + dir + "/" + fname);
      if (!found.empty()) return found;
    }
  }

  return "share/athome_vision/" + path;
}

void ZedNode::loadYolo(
  YoloDetector & detector, const std::string & model_param,
  const std::vector<std::string> & fallbacks, const std::string & name,
  float conf, float nms, int size)
{
  if (model_param.empty()) return;
  std::string path = resolvePath(model_param);
  if (!std::filesystem::exists(path)) {
    for (const auto & fallback : fallbacks) {
      const std::string candidate = resolvePath(fallback);
      if (std::filesystem::exists(candidate)) {
        path = candidate;
        break;
      }
    }
  }
  if (!std::filesystem::exists(path)) {
    RCLCPP_WARN(get_logger(), "%s model not found: %s", name.c_str(), path.c_str());
    return;
  }
  std::string err;
  if (detector.load(path, "", conf, nms, size, name, &err)) {
    RCLCPP_INFO(get_logger(), "%s model loaded: %s", name.c_str(), path.c_str());
  } else {
    RCLCPP_ERROR(get_logger(), "%s model error: %s", name.c_str(), err.c_str());
  }
}

void ZedNode::loadModels()
{
  const float conf = static_cast<float>(get_parameter("yolo_conf").as_double());
  const float bag_conf = static_cast<float>(get_parameter("bag_conf").as_double());
  const float nms = static_cast<float>(get_parameter("yolo_nms").as_double());
  const float drink_conf = static_cast<float>(get_parameter("drink_conf").as_double());
  const int size = static_cast<int>(get_parameter("yolo_size").as_int());

  if (enable_yolo_) {
    const std::string model_param = get_parameter("yolo_model").as_string();
    std::string model_path = resolvePath(model_param);
    if (!std::filesystem::exists(model_path)) {
      for (const auto & fb : {"models/paper_bag.onnx", "models/yolov8n_sim.onnx"}) {
        const std::string candidate = resolvePath(fb);
        if (std::filesystem::exists(candidate)) {
          model_path = candidate;
          break;
        }
      }
    }
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
      RCLCPP_WARN(get_logger(), "YOLO model not found: %s", model_path.c_str());
      enable_yolo_ = false;
    } else {
      std::string err;
      std::string classes = resolvePath(get_parameter("yolo_classes").as_string());
      if (is_receptionist_ || model_path.find("yolov8n") != std::string::npos) {
        if (classes.find("paper_bag") != std::string::npos || !std::filesystem::exists(classes)) {
          classes = resolvePath("models/coco_names.txt");
        }
      }
      const float active_yolo_conf = is_receptionist_ ? conf : bag_conf;
      const std::string active_yolo_label = is_receptionist_ ? "chair" : "bag";
      if (!yolo_.load(model_path, classes, active_yolo_conf, nms, size, active_yolo_label, &err)) {
        RCLCPP_ERROR(get_logger(), "YOLO load error: %s", err.c_str());
        enable_yolo_ = false;
      } else {
        RCLCPP_INFO(get_logger(), "Primary %s YOLO loaded: %s", active_yolo_label.c_str(), model_path.c_str());
      }
    }

    // Muat model pelengkap ansambel tas hanya jika deteksi tas diaktifkan (misal Help Me Carry)
    if (enable_bag_) {
      std::string secondary_model;
      std::string secondary_classes;
      if (model_path.find("paper_bag") != std::string::npos) {
        secondary_model = resolvePath("models/yolov8n_sim.onnx");
        secondary_classes = resolvePath("models/coco_names.txt");
      } else {
        secondary_model = resolvePath("models/paper_bag.onnx");
        secondary_classes = resolvePath("models/paper_bag_classes.txt");
      }

      if (!secondary_model.empty() && std::filesystem::exists(secondary_model)) {
        std::string err;
        if (coco_bag_yolo_.load(secondary_model, secondary_classes, bag_conf, nms, size, "bag", &err)) {
          RCLCPP_INFO(get_logger(), "Ensemble Complementary Bag YOLO loaded: %s", secondary_model.c_str());
        }
      }
    }
  }

  if (enable_drinks_) {
    // Soda dimuat untuk Receptionist, Restaurant, maupun All
    loadYolo(soda_yolo_, get_parameter("soda_model").as_string(),
      {"models/soda.onnx"}, "soda", drink_conf, nms, size);

    // Milk dimuat untuk Receptionist, Restaurant, maupun All
    loadYolo(milk_yolo_, get_parameter("milk_model").as_string(),
      {"models/milk.onnx"}, "milk", drink_conf, nms, size);

    // Water dimuat untuk Receptionist maupun All (threshold lebih sensitif untuk plastik bening)
    if (is_receptionist_ || (!is_help_me_carry_ && !is_restaurant_)) {
      const float water_conf = static_cast<float>(get_parameter("water_conf").as_double());
      loadYolo(botol_yolo_, get_parameter("botol_model").as_string(),
        {"models/water.onnx", "models/botol.onnx"}, "water", water_conf, nms, size);
    }

    // Pringles, Cracker & Sugar hanya dimuat jika mode Restaurant atau All
    if (is_restaurant_ || (!is_help_me_carry_ && !is_receptionist_)) {
      loadYolo(pringles_yolo_, get_parameter("pringles_model").as_string(),
        {"models/pringles.onnx", "models/pringles_ chips _can.onnx"}, "pringles",
        drink_conf, nms, size);
      loadYolo(cracker_yolo_, get_parameter("cracker_model").as_string(),
        {"models/crackerbox.onnx", "models/cheezit.onnx", "models/best_cracker.onnx"}, "cracker", drink_conf, nms, size);
      loadYolo(sugarbox_yolo_, get_parameter("sugarbox_model").as_string(),
        {"models/sugarbox.onnx", "models/sugar_box.onnx", "models/sugar.onnx"}, "sugar",
        drink_conf, nms, size);
    }
  }

  if (enable_chair_) {
    loadYolo(chair_yolo_, get_parameter("chair_model").as_string(),
      {}, "chair", conf, nms, size);
  }

  if (is_receptionist_) {
    const float acc_conf = static_cast<float>(get_parameter("accessory_conf").as_double());
    hat_conf_ = static_cast<float>(get_parameter("hat_conf").as_double());
    auto loadOptionalAccessory = [&](YoloDetector & det, const std::string & param_name, const std::string & label, float model_conf) {
      const std::string path = resolvePath(get_parameter(param_name).as_string());
      if (!path.empty() && std::filesystem::exists(path)) {
        std::string err;
        if (det.load(path, "", model_conf, nms, size, label, &err)) {
          RCLCPP_INFO(get_logger(), "Accessory YOLO (%s) loaded (conf=%.2f): %s", label.c_str(), model_conf, path.c_str());
        } else {
          RCLCPP_WARN(get_logger(), "Accessory YOLO (%s) load error: %s", label.c_str(), err.c_str());
        }
      }
    };
    loadOptionalAccessory(hat_yolo_, "hat_model", "hat", hat_conf_);
    loadOptionalAccessory(glasses_yolo_, "glasses_model", "glasses", acc_conf);
  }

  // Atribut tamu & pelanggan (warna pakaian, gender, face/nama) untuk task receptionist & restaurant
  if (is_receptionist_ || is_restaurant_) {
    const std::string color_labels =
      resolvePath(get_parameter("color_labels").as_string());
    const std::string hair_labels =
      resolvePath(get_parameter("hair_labels").as_string());
    const std::string color_path = resolvePath(get_parameter("color_model").as_string());
    const std::string hair_path = resolvePath(get_parameter("hair_model").as_string());
    const std::string face_path = resolvePath(get_parameter("face_model").as_string());

    std::vector<std::string> gender_candidates;
    auto addGenderCandidate = [&](const std::string & path) {
        if (path.empty()) return;
        if (std::find(gender_candidates.begin(), gender_candidates.end(), path) !=
            gender_candidates.end()) return;
        gender_candidates.push_back(path);
      };
    addGenderCandidate(resolvePath(get_parameter("gender_model").as_string()));
    addGenderCandidate(resolvePath("models/genderage.onnx"));
    addGenderCandidate(resolvePath("models/gender_model_weights.onnx"));

    bool gender_loaded = false;
    std::string gender_err;
    for (const auto & path : gender_candidates) {
      if (!std::filesystem::exists(path)) continue;
      if (analyzer_.loadGender(path, &gender_err)) {
        RCLCPP_INFO(get_logger(), "Gender model loaded: %s", path.c_str());
        gender_loaded = true;
        break;
      }
    }
    if (!gender_loaded) {
      RCLCPP_WARN(get_logger(), "Gender model unavailable, gender will be unknown: %s",
        gender_err.c_str());
    }
    if (!color_path.empty()) {
      std::string err;
      if (!analyzer_.loadColor(color_path, color_labels, &err)) {
        RCLCPP_WARN(get_logger(), "Color model unavailable, using HSV: %s", err.c_str());
      } else {
        RCLCPP_INFO(get_logger(), "Clothing Color ResNet model loaded: %s (labels: %s)",
          color_path.c_str(), color_labels.c_str());
      }
    }
    if (!hair_path.empty()) {
      std::string err;
      if (!analyzer_.loadHair(hair_path, hair_labels, &err)) {
        RCLCPP_WARN(get_logger(), "Hair model unavailable, using HSV: %s", err.c_str());
      }
    }
    bool face_loaded = false;
    if (!face_path.empty()) {
      std::string err;
      if (analyzer_.loadFace(face_path, &err)) {
        RCLCPP_INFO(get_logger(), "Face SFace loaded: %s (threshold %.3f)",
          face_path.c_str(), face_threshold_);
        face_loaded = true;
      } else {
        RCLCPP_WARN(get_logger(), "Face model unavailable: %s", err.c_str());
      }
    }

    std::string face_detect_model = get_parameter("face_detect_model").as_string();
    float face_detect_conf = static_cast<float>(get_parameter("face_detect_conf").as_double());
    std::string face_detect_path = resolvePath(face_detect_model);
    if (!face_detect_path.empty() && std::filesystem::exists(face_detect_path)) {
      std::string err;
      if (analyzer_.loadFaceDetector(face_detect_path, face_detect_conf, &err)) {
        RCLCPP_INFO(get_logger(),
          "\033[1;32m[YUNET]\033[0m Face detector loaded: %s (conf: %.2f)",
          face_detect_path.c_str(), face_detect_conf);
      } else {
        RCLCPP_WARN(get_logger(), "Failed to load YuNet face detector: %s", err.c_str());
      }
    } else {
      RCLCPP_WARN(get_logger(), "YuNet model file not found: %s, using ZED keypoints fallback",
        face_detect_path.c_str());
    }

    if (face_loaded) {
      loadPreRegisteredHost();
    }
  }
}

void ZedNode::loadPreRegisteredHost(const std::string & custom_photo_path)
{
  std::string host_name = get_parameter("host_name").as_string();
  if (host_name.empty()) return;

  std::string raw_shirt = get_parameter("host_shirt_color").as_string();
  const std::string suffix = " shirt";
  if (raw_shirt.size() > suffix.size() &&
      raw_shirt.rfind(suffix) == raw_shirt.size() - suffix.size())
  {
    raw_shirt = raw_shirt.substr(0, raw_shirt.size() - suffix.size());
  }

  std::string host_photo = custom_photo_path.empty()
    ? get_parameter("host_photo").as_string()
    : custom_photo_path;
  std::vector<float> feat;
  cv::Mat img;
  if (!host_photo.empty()) {
    std::string full_path = resolvePath(host_photo);
    if (!std::filesystem::exists(full_path)) {
      std::string fallback = resolvePath("models/" + host_photo);
      if (std::filesystem::exists(fallback)) {
        full_path = fallback;
      }
    }
    if (std::filesystem::exists(full_path)) {
      img = cv::imread(full_path);
      if (!img.empty()) {
        feat = analyzer_.extractFaceFromImage(img);
      }
    }
  }

  RegisteredPerson host_person;
  host_person.name = host_name;
  host_person.clothing_color = raw_shirt;
  host_person.pants_color = get_parameter("host_pants_color").as_string();
  host_person.gender = get_parameter("host_gender").as_string();
  host_person.hair_color = get_parameter("host_hair_color").as_string();
  host_person.has_hat = get_parameter("host_hat").as_bool();
  host_person.has_glasses = get_parameter("host_glasses").as_bool();
  host_person.has_color = false;
  host_person.last_tracking_id = -1;
  host_person.last_seen = now();

  if (!img.empty()) {
    std::vector<YoloDetection> acc_dets;
    if (!host_person.has_hat && hat_yolo_.ready()) {
      acc_dets.clear();
      hat_yolo_.detect(img, {"hat"}, acc_dets);
      for (const auto & d : acc_dets) {
        if (d.score >= 0.25f) {
          host_person.has_hat = true;
          break;
        }
      }
    }
    if (!host_person.has_glasses && glasses_yolo_.ready()) {
      acc_dets.clear();
      glasses_yolo_.detect(img, {"glasses"}, acc_dets);
      for (const auto & d : acc_dets) {
        if (d.score >= 0.18f) {
          host_person.has_glasses = true;
          break;
        }
      }
    }
  }

  if (!feat.empty()) {
    host_person.feature = feat;
    host_person.face_gallery.push_back(feat);
    // Multi-scale augmentation untuk foto wajah Host agar cocok dengan wajah jarak jauh di living room (2.0-3.0m)
    for (int sz : {64, 48}) {
      cv::Mat img_scaled;
      cv::resize(img, img_scaled, cv::Size(sz, sz), 0, 0, cv::INTER_AREA);
      std::vector<float> feat_scaled = analyzer_.extractFaceFromImage(img_scaled);
      if (!feat_scaled.empty()) {
        host_person.face_gallery.push_back(feat_scaled);
      }
    }
    host_person.feature_samples = 30; // Pre-registered with maximum confidence
    host_person.locked = true;
    host_person.has_face_verified = true;
    RCLCPP_INFO(get_logger(),
      "\033[1;32m[RECEPTIONIST]\033[0m Host '%s' registered from photo (%dx%d, shirt: %s, pants: %s, gender: %s, hat: %s, glasses: %s, gallery: %zu templates)",
      host_name.c_str(), img.cols, img.rows, raw_shirt.c_str(),
      host_person.pants_color.empty() ? "-" : host_person.pants_color.c_str(),
      host_person.gender.c_str(),
      host_person.has_hat ? "yes" : "no",
      host_person.has_glasses ? "yes" : "no",
      host_person.face_gallery.size());
  } else {
    // Foto tidak ada: tetap daftarkan profil Host berbasis atribut pakaian & gender
    host_person.feature_samples = 0;
    host_person.locked = false; // Self-learn face when robot faces Host in Cycle 1
    host_person.has_face_verified = false;
    RCLCPP_INFO(get_logger(),
      "\033[1;32m[RECEPTIONIST]\033[0m Host '%s' pre-registered from attributes (shirt: %s, pants: %s, gender: %s, hat: %s, glasses: %s, photo pending self-learn)",
      host_name.c_str(), raw_shirt.c_str(),
      host_person.pants_color.empty() ? "-" : host_person.pants_color.c_str(),
      host_person.gender.c_str(),
      host_person.has_hat ? "yes" : "no",
      host_person.has_glasses ? "yes" : "no");
  }

  auto it = std::remove_if(registered_persons_.begin(), registered_persons_.end(),
    [&host_name](const RegisteredPerson & p) { return p.name == host_name; });
  registered_persons_.erase(it, registered_persons_.end());

  registered_persons_.push_back(host_person);
}

void ZedNode::publishYaw()
{
  if (!publish_yaw_) return;
  sl::SensorsData sensors;
  if (zed_.getSensorsData(sensors, sl::TIME_REFERENCE::IMAGE) != sl::ERROR_CODE::SUCCESS ||
      !sensors.imu.is_available)
  {
    return;
  }
  const auto q = sensors.imu.pose.getOrientation();
  const float yaw = zed_vision::quatToYaw(q.x, q.y, q.z, q.w) * 180.0f / zed_vision::PI_F;
  std_msgs::msg::Float64 msg;
  msg.data = static_cast<double>(
    yaw_360_ ? zed_vision::wrap360(yaw) : zed_vision::wrap180(yaw));
  yaw_publisher_->publish(msg);
}

void ZedNode::processBags(const cv::Mat & bgr, const sl::Mat & cloud)
{
  preview_bags_.clear();
  if (!enable_bag_) return;

  if (enable_yolo_ && yolo_frame_++ % yolo_stride_ == 0) {
    std::vector<YoloDetection> all_dets;

    // 1. Deteksi menggunakan model utama (paper_bag / model yang dikonfigurasi)
    if (yolo_.ready()) {
      std::vector<YoloDetection> d1;
      yolo_.detect(bgr, bag_labels_, d1);
      all_dets.insert(all_dets.end(), d1.begin(), d1.end());
    }

    // 2. Deteksi menggunakan model pelengkap ansambel (misal COCO backpack/handbag/suitcase atau paper_bag)
    if (coco_bag_yolo_.ready()) {
      std::vector<YoloDetection> d2;
      coco_bag_yolo_.detect(bgr, bag_labels_, d2);
      all_dets.insert(all_dets.end(), d2.begin(), d2.end());
    }

    // 3. NMS antar deteksi model untuk menggabungkan bounding box tas yang sama
    if (all_dets.size() > 1) {
      std::vector<cv::Rect> b_boxes;
      std::vector<float> b_scores;
      for (const auto & d : all_dets) {
        b_boxes.push_back(d.rect);
        b_scores.push_back(d.score);
      }
      std::vector<int> keep;
      const float nms_thresh = static_cast<float>(get_parameter("yolo_nms").as_double());
      const float bag_conf = static_cast<float>(get_parameter("bag_conf").as_double());
      cv::dnn::NMSBoxes(b_boxes, b_scores, bag_conf, nms_thresh, keep);

      std::vector<YoloDetection> merged_dets;
      for (int idx : keep) {
        merged_dets.push_back(all_dets[static_cast<size_t>(idx)]);
      }
      all_dets = std::move(merged_dets);
    }

    last_bag_dets_ = all_dets;
  }

  std::vector<YoloDetection> dets = last_bag_dets_;

  struct BagCandidate {
    YoloDetection det;
    bool ok{false};
    double x{0.0}, y{0.0}, z{0.0};
    double dist{1e9};
  };
  std::vector<BagCandidate> candidates;

  for (const auto & det : dets) {
    if (static_cast<double>(det.rect.area()) < yolo_min_area_) continue;

    // Filter bentuk rasio aspek yang fleksibel untuk segala jenis tas (tas berdiri, tas jinjing, koper rebah)
    const float aspect = static_cast<float>(det.rect.width) / static_cast<float>(std::max(1, det.rect.height));
    if (aspect > 3.5f || aspect < 0.25f) continue;

    const float centroid_bagX = det.rect.x + det.rect.width * 0.5f;
    const float centroid_bagY = det.rect.y + det.rect.height * 0.5f;
    sl::float4 point{NAN, NAN, NAN, NAN};

    if (centroid_bagX >= 0 && centroid_bagX < cloud.getWidth() &&
        centroid_bagY >= 0 && centroid_bagY < cloud.getHeight())
    {
      cloud.getValue(static_cast<int>(centroid_bagX), static_cast<int>(centroid_bagY), &point);
    }

    double x = 0.0, y = 0.0, z = 0.0;
    bool ok = false;

    if (std::isfinite(point.z)) {
      x = (point.x + zed_vision::kOffsetX) / zed_vision::kOffsetScale;
      y = (-point.y + zed_vision::kOffsetY) / zed_vision::kOffsetScale;
      z = (point.z + zed_vision::kOffsetZ) / zed_vision::kOffsetScale;
      ok = true;
    } else {
      if (zed_vision::get3dCoordinates(det.rect, cloud, point) && std::isfinite(point.z)) {
        x = (point.x + zed_vision::kOffsetX) / zed_vision::kOffsetScale;
        y = (-point.y + zed_vision::kOffsetY) / zed_vision::kOffsetScale;
        z = (point.z + zed_vision::kOffsetZ) / zed_vision::kOffsetScale;
        ok = true;
      }
    }

    // Filter dimensi fisik 3D: gunakan forward distance (x) yang valid.
    // Tolak hanya jika objek melebihi 1.2 meter (misal meja panjang atau dinding utuh)
    if (ok && std::isfinite(x) && x > 0.3) {
      const double real_w = (det.rect.width * x) / 530.0;
      const double real_h = (det.rect.height * x) / 530.0;
      if (real_w > 1.20 || real_h > 1.20) continue;
    }

    const double dist = (ok && std::isfinite(x) && std::isfinite(y)) ? std::hypot(x, y) : 1e9;
    if (ok && dist > bag_max_dist_) continue;
    if (!ok && (det.rect.area() < 3000 || det.rect.height < 45)) continue;
    candidates.push_back({det, ok, x, y, z, dist});
  }

  // Urutkan kandidat tas berdasarkan jarak 3D terdekat (dengan tie-breaker confidence jika jarak sangat dekat)
  if (!candidates.empty()) {
    std::sort(candidates.begin(), candidates.end(), [](const BagCandidate & a, const BagCandidate & b) {
      if (std::abs(a.dist - b.dist) < 0.20) {
        return a.det.score > b.det.score;
      }
      return a.dist < b.dist;
    });
  }

  // Ambil HANYA 2 tas terdekat
  const size_t max_bags = std::min<size_t>(2, candidates.size());
  std::string data;
  for (size_t i = 0; i < max_bags; ++i) {
    const auto & b = candidates[i];
    const double valid_dist = (b.dist < 1e8) ? b.dist : 0.0;
    preview_bags_.push_back({b.det.rect, b.ok, b.x, b.y, valid_dist});
    if (!data.empty()) data += ";";
    data += cv::format("bag:%d,%d,%d,%d|",
      b.det.rect.x, b.det.rect.y, b.det.rect.width, b.det.rect.height);
    data += b.ok ? cv::format("%.2f,%.2f,%.2f|%.2f", b.x, b.y, b.z, b.det.score)
                 : cv::format("nan,nan,nan|%.2f", b.det.score);
  }

  std_msgs::msg::String msg;
  msg.data = data;
  bag_dets_publisher_->publish(msg);
}

void ZedNode::processDrinks(const cv::Mat & bgr, const sl::Mat & cloud)
{
  if (bgr.empty()) return;

  std::vector<YoloDetection> dets = last_drink_dets_;
  if (enable_drinks_ && (drink_frame_++ % yolo_stride_) == (1 % yolo_stride_)) {
    dets.clear();
    std::vector<YoloDetector *> active_detectors;
    if (is_receptionist_ && !is_restaurant_) {
      // Receptionist: HANYA deteksi soda, milk, dan water
      active_detectors = {&soda_yolo_, &milk_yolo_, &botol_yolo_};
    } else if (is_restaurant_ && !is_receptionist_) {
      // Restaurant: HANYA deteksi soda, pringles, sugar, cracker, dan milk
      active_detectors = {&soda_yolo_, &pringles_yolo_, &sugarbox_yolo_, &cracker_yolo_, &milk_yolo_};
    } else {
      active_detectors = {&soda_yolo_, &pringles_yolo_, &sugarbox_yolo_, &cracker_yolo_, &milk_yolo_, &botol_yolo_};
    }

    // Jalankan seluruh model minuman secara paralel di GPU menggunakan std::async
    std::vector<std::future<std::vector<YoloDetection>>> futures;
    futures.reserve(active_detectors.size());
    for (YoloDetector * detector : active_detectors) {
      if (!detector || !detector->ready()) continue;
      futures.push_back(std::async(std::launch::async, [this, detector, &bgr]() {
        std::vector<YoloDetection> found;
        detector->detect(bgr, {}, found);
        std::vector<YoloDetection> filtered;
        filtered.reserve(found.size());
        for (auto & item : found) {
          if (item.label == "cracker_box" || item.label == "crackerbox") item.label = "cracker";
          if (item.label == "sugar_box" || item.label == "sugarbox") item.label = "sugar";
          if (item.label == "bottle" || item.label == "botol" ||
              item.label == "mineral_water" || item.label == "mineral water")
          {
            item.label = "water";
          }

          if (is_receptionist_ && !is_restaurant_) {
            if (item.label != "milk" && item.label != "water" && item.label != "soda") continue;
          } else if (is_restaurant_ && !is_receptionist_) {
            if (item.label != "soda" && item.label != "pringles" &&
                item.label != "sugar" && item.label != "cracker" && item.label != "milk")
            {
              continue;
            }
          }

          const int area = item.rect.width * item.rect.height;
          if (area < drink_min_area_ || area > static_cast<int>(bgr.cols * bgr.rows * 0.35)) continue;
          if (item.rect.width < 10 || item.rect.height < 12) continue;

          const float ar = static_cast<float>(item.rect.height) / static_cast<float>(std::max(1, item.rect.width));
          if (item.label == "soda" && (ar < 0.9f || ar > 3.2f)) continue;
          if (item.label == "water" && (ar < 0.50f || ar > 5.5f)) continue;
          if (item.label == "pringles" && (ar < 1.1f || ar > 3.8f)) continue;
          if ((item.label == "cracker" || item.label == "sugar") && (ar < 0.5f || ar > 2.6f)) continue;
          if (item.label == "milk" && (ar < 0.8f || ar > 3.5f)) continue;

          filtered.push_back(item);
        }
        return filtered;
      }));
    }

    std::vector<YoloDetection> raw_dets;
    for (auto & fut : futures) {
      auto res = fut.get();
      raw_dets.insert(raw_dets.end(), res.begin(), res.end());
    }

    // 3. Cross-Model Non-Maximum Suppression (Inter-Class NMS):
    // Urutkan semua deteksi dari berbagai model berdasarkan confidence tertinggi
    std::sort(raw_dets.begin(), raw_dets.end(), [](const YoloDetection & a, const YoloDetection & b) {
      return a.score > b.score;
    });

    for (const auto & cand : raw_dets) {
      bool suppressed = false;
      for (const auto & kept : dets) {
        const cv::Rect inter = cand.rect & kept.rect;
        if (inter.area() > 0) {
          const float iou = static_cast<float>(inter.area()) /
                            static_cast<float>(cand.rect.area() + kept.rect.area() - inter.area());
          const float min_overlap = static_cast<float>(inter.area()) /
                                    static_cast<float>(std::min(cand.rect.area(), kept.rect.area()));
          // Jika dua model berbeda mendeteksi objek yang sama/tumpang tindih, pertahankan HANYA model dengan confidence lebih tinggi
          if (iou > 0.35f || min_overlap > 0.55f) {
            suppressed = true;
            break;
          }
        }
      }
      if (!suppressed) {
        dets.push_back(cand);
      }
    }
    last_drink_dets_ = dets;
  }

  const float water_conf_param = static_cast<float>(get_parameter("water_conf").as_double());
  preview_drinks_.clear();
  std::string data;
  for (const auto & det : dets) {
    if (static_cast<double>(det.rect.area()) < drink_min_area_) continue;
    double x = 0.0, y = 0.0, z = 0.0;
    bool ok = zed_vision::get3dWithOffset(det.rect, cloud, x, y, z);
    double dist = ok ? std::hypot(x, y) : 0.0;

    // 4. Validasi jarak 3D dan dimensi fisik objek (mencegah false positive pada meja, dinding, atau orang)
    if (ok) {
      if (dist > drink_max_dist_) {
        if (det.label == "water") {
          // Botol air transparan: sinar stereo sering menembus botol dan mengenai dinding/lantai belakang (> drink_max_dist_).
          // Jangan dibuang! Jadikan fallback 2D (tanpa depth).
          ok = false;
        } else {
          continue;
        }
      } else if (std::isfinite(x) && x > 0.25) {
        const double real_w = (det.rect.width * x) / 530.0;
        const double real_h = (det.rect.height * x) / 530.0;
        // Objek makanan / kaleng minuman berukuran maksimal lebar 35cm, tinggi 45cm
        if (real_w > 0.35 || real_h > 0.45 || real_w < 0.015 || real_h < 0.025) {
          if (det.label == "water") {
            // Refraksi/kedalaman latar belakang membuat estimasi dimensi 3D water tidak wajar
            ok = false;
          } else {
            continue;
          }
        }
      }
    }

    if (!ok) {
      // Botol air mineral transparan sering menghasilkan depth NaN atau refraksi tembus;
      // izinkan fallback 2D untuk water mulai skor water_conf agar tetap terdeteksi.
      const float min_no_depth = (det.label == "water") ? water_conf_param : 0.65f;
      if (det.score < min_no_depth) continue;
      if (det.rect.area() < 1200 || det.rect.height < 40) continue;
    }

    preview_drinks_.push_back({det.label, det.rect, ok, x, y, dist, det.score});
  }

  // Urutkan berdasarkan jarak terdekat (prioritaskan yang memiliki kedalaman valid)
  std::sort(preview_drinks_.begin(), preview_drinks_.end(),
    [](const PrevItem & a, const PrevItem & b) {
      if (a.has_pos != b.has_pos) return a.has_pos > b.has_pos;
      if (std::abs(a.range_m - b.range_m) < 0.20) {
        return a.score > b.score;
      }
      return a.range_m < b.range_m;
    });

  for (const auto & item : preview_drinks_) {
    if (!data.empty()) data += ";";
    data += cv::format("%s:%d,%d,%d,%d|", item.label.c_str(),
      item.bbox.x, item.bbox.y, item.bbox.width, item.bbox.height);
    data += item.has_pos ? cv::format("%.2f,%.2f,%.2f|%.2f", item.x_m, item.y_m, item.range_m, item.score)
                         : cv::format("nan,nan,nan|%.2f", item.score);
  }

  std_msgs::msg::String msg;
  msg.data = data;
  drink_dets_publisher_->publish(msg);
}

void ZedNode::processPersons(const cv::Mat & bgr, const sl::Mat & cloud)
{
  preview_persons_.clear();

  if (bodies_.body_list.empty()) {
    back_memory_.clear();
    std_msgs::msg::String person_msg;
    person_msg.data = "NO_PERSON";
    persons_publisher_->publish(person_msg);
    std_msgs::msg::String gesture_msg;
    gesture_msg.data = "NONE";
    gesture_publisher_->publish(gesture_msg);
    std_msgs::msg::String attr_msg;
    attr_msg.data = "";
    attributes_publisher_->publish(attr_msg);
    std_msgs::msg::String name_msg;
    name_msg.data = "";
    recognized_name_publisher_->publish(name_msg);
    return;
  }

  struct PersonEntry
  {
    const sl::BodyData * body{nullptr};
    int id{-1};
    cv::Rect bbox;
    std::string label{"person"};
    double x{0.0}, y{0.0}, z{0.0};
    bool has_depth{false};
    double range_m{std::numeric_limits<double>::infinity()};
    double dist{std::numeric_limits<double>::infinity()};
    bool is_sitting{false};
  };

  std::vector<PersonEntry> entries;
  for (const auto & body : bodies_.body_list) {
    if (body.keypoint_2d.size() < 34) continue;
    cv::Rect bbox;
    if (!zed_vision::computeBbox(body.keypoint_2d, bgr.cols, bgr.rows, bbox)) continue;
    PersonEntry entry;
    entry.body = &body;
    entry.id = body.id;
    entry.bbox = bbox;
    entry.has_depth = zed_vision::get3dWithOffset(bbox, cloud, entry.x, entry.y, entry.z);
    entry.range_m = (entry.has_depth && std::isfinite(entry.x) && std::isfinite(entry.y))
      ? std::hypot(entry.x, entry.y)
      : (1e6 - static_cast<double>(bbox.area()));
    entry.dist = entry.range_m;

    // Deteksi postur duduk awal jika keypoint 3D tersedia
    bool is_seated = false;
    if (body.keypoint.size() >= 25) {
      const auto & k3 = body.keypoint;
      auto valid3d = [](const sl::float3 & p) {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
      };
      auto leg_bent = [&](int hip_i, int knee_i, int ankle_i) {
        const auto & hip = k3[hip_i];
        const auto & knee = k3[knee_i];
        const auto & ankle = k3[ankle_i];
        if (!valid3d(hip) || !valid3d(knee) || !valid3d(ankle)) return false;
        const float thigh_dz = hip.z - knee.z;
        const float shin_dz = knee.z - ankle.z;
        const float thigh_horiz = std::hypot(knee.x - hip.x, knee.y - hip.y);
        const float ux = hip.x - knee.x, uy = hip.y - knee.y, uz = hip.z - knee.z;
        const float vx = ankle.x - knee.x, vy = ankle.y - knee.y, vz = ankle.z - knee.z;
        const float un = std::sqrt(ux * ux + uy * uy + uz * uz);
        const float vn = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (un < 1e-3f || vn < 1e-3f) return false;
        const float cos_knee = (ux * vx + uy * vy + uz * vz) / (un * vn);
        return (thigh_dz < 28.0f && shin_dz > 15.0f &&
                thigh_horiz > thigh_dz * 0.85f && cos_knee > -0.68f);
      };
      if (leg_bent(18, 19, 20) || leg_bent(22, 23, 24)) {
        is_seated = true;
      }
    }
    entry.is_sitting = is_seated;

    const double max_allowed = is_seated ? person_seated_max_dist_ : person_max_dist_;
    if (entry.has_depth && entry.range_m > max_allowed) {
      continue;
    }
    if (!entry.has_depth && (bbox.height < 150 || bbox.area() < 8000)) {
      continue;
    }

    entries.push_back(entry);
  }

  std::sort(entries.begin(), entries.end(), [](const PersonEntry & a, const PersonEntry & b) {
    return a.range_m < b.range_m;
  });

  const auto current_time = now();
  const cv::Vec3f wb_gain = zed_vision::computeWhiteBalanceGain(bgr);

  // Ekstraksi atribut warna baju, warna celana, aksesoris kepala, rambut, gender (7-frame majority voting)
  std::vector<cv::Rect> entry_shirt_rois(entries.size());
  std::vector<cv::Rect> entry_pants_rois(entries.size());
  std::vector<cv::Rect> entry_hat_bboxes(entries.size());
  std::vector<cv::Rect> entry_glasses_bboxes(entries.size());
  std::vector<std::string> entry_clothes(entries.size());
  std::vector<std::string> entry_pants(entries.size());
  std::vector<bool> entry_has_hat(entries.size(), false);
  std::vector<bool> entry_has_glasses(entries.size(), false);
  std::vector<std::string> entry_hairs(entries.size());
  std::vector<std::string> entry_genders(entries.size());
  std::vector<bool> entry_facing_back(entries.size(), false);
  if (is_receptionist_) {
    const bool run_attr_infer = (person_attr_frame_++ % 3 == 0);
    static std::unordered_map<int, cv::Rect> hat_bbox_cache;
    static std::unordered_map<int, cv::Rect> glasses_bbox_cache;
    static std::unordered_map<int, rclcpp::Time> hat_hit_time;
    static std::unordered_map<int, rclcpp::Time> glasses_hit_time;

    auto detectAccessory = [&](
      YoloDetector & detector,
      std::unordered_map<int, ShirtColorMemory> & mem_map,
      std::unordered_map<int, cv::Rect> & bbox_cache,
      std::unordered_map<int, rclcpp::Time> & hit_time_map,
      int person_id, const cv::Rect & head_roi,
      float min_score, int min_yes_cnt, bool is_hat,
      cv::Rect & out_bbox) -> bool
    {
      if (!detector.ready()) return false;
      std::string raw_val;
      if (run_attr_infer && head_roi.width >= 20 && head_roi.height >= 20) {
        std::vector<YoloDetection> found;
        detector.detect(bgr(head_roi), {}, found);
        const YoloDetection * best_det = nullptr;
        for (const auto & det : found) {
          if (det.score < min_score) continue;
          if (is_hat) {
            // Topi harus berada di bagian atas kepala & tidak menutupi seluruh wajah (hindari false positive rambut mengembang)
            const float rel_cy = (det.rect.y + det.rect.height * 0.5f) / static_cast<float>(head_roi.height);
            const float rel_h  = static_cast<float>(det.rect.height) / static_cast<float>(head_roi.height);
            const float rel_w  = static_cast<float>(det.rect.width) / static_cast<float>(head_roi.width);
            if (rel_cy > 0.45f || rel_h > 0.55f || rel_w < 0.25f) continue;
          }
          if (!best_det || det.score > best_det->score) {
            best_det = &det;
          }
        }
        if (best_det != nullptr) {
          cv::Rect det_rect = best_det->rect;
          det_rect.x += head_roi.x;
          det_rect.y += head_roi.y;
          det_rect &= cv::Rect(0, 0, bgr.cols, bgr.rows);
          if (det_rect.width > 0 && det_rect.height > 0) {
            auto cit = bbox_cache.find(person_id);
            if (cit != bbox_cache.end() && cit->second.width > 0) {
              // Smoothing bbox ringan agar tidak bergetar
              det_rect.x = static_cast<int>(0.5f * cit->second.x + 0.5f * det_rect.x);
              det_rect.y = static_cast<int>(0.5f * cit->second.y + 0.5f * det_rect.y);
              det_rect.width = static_cast<int>(0.5f * cit->second.width + 0.5f * det_rect.width);
              det_rect.height = static_cast<int>(0.5f * cit->second.height + 0.5f * det_rect.height);
            }
            bbox_cache[person_id] = det_rect;
            hit_time_map[person_id] = current_time;
            raw_val = "yes";
          } else {
            raw_val = "no";
          }
        } else {
          raw_val = "no";
        }
      }

      if (!raw_val.empty()) {
        updateAttributeVote(mem_map, person_id, raw_val, current_time);
      }

      // Hitung jumlah "yes" dalam jendela history + hysteresis waktu (1.5 detik) agar tidak hilang-timbul
      int yes_cnt = 0;
      auto mit = mem_map.find(person_id);
      if (mit != mem_map.end()) {
        for (const auto & v : mit->second.history) {
          if (v == "yes") yes_cnt++;
        }
      }
      auto hit_it = hit_time_map.find(person_id);
      const bool recent_hit = (hit_it != hit_time_map.end() &&
        (current_time - hit_it->second).seconds() < 1.5);
      const bool active = is_hat
        ? (yes_cnt >= min_yes_cnt)
        : ((yes_cnt >= min_yes_cnt) || (yes_cnt >= 2 && recent_hit));

      if (active) {
        auto bit = bbox_cache.find(person_id);
        if (bit != bbox_cache.end() && bit->second.width > 0 && bit->second.height > 0) {
          out_bbox = bit->second;
        }
      }
      return active;
    };
    for (size_t i = 0; i < entries.size(); ++i) {
      const auto & e = entries[i];
      const auto & kpt2d = e.body->keypoint_2d;
      if (enable_back_detection_) {
        entry_facing_back[i] = AttributeAnalyzer::isFacingBack(*e.body, e.bbox);
      }
      entry_shirt_rois[i] = AttributeAnalyzer::computeShirtRoi(bgr, e.bbox, kpt2d);
      std::string raw_cloth;
      if (run_attr_infer && entry_shirt_rois[i].width > 5 && entry_shirt_rois[i].height > 5) {
        raw_cloth = analyzer_.clothingColor(bgr, entry_shirt_rois[i], &cloud, wb_gain);
      }
      if (!raw_cloth.empty() && raw_cloth != "casual attire") {
        entry_clothes[i] = updateAttributeVote(shirt_color_memory_, e.id, raw_cloth, current_time);
      } else {
        auto cit = shirt_color_memory_.find(e.id);
        entry_clothes[i] = (cit != shirt_color_memory_.end() && !cit->second.history.empty())
          ? cit->second.history.back() : raw_cloth;
      }

      entry_pants_rois[i] = AttributeAnalyzer::computePantsRoi(bgr, e.bbox, kpt2d);
      std::string raw_pants;
      if (run_attr_infer && entry_pants_rois[i].width > 5 && entry_pants_rois[i].height > 5) {
        raw_pants = analyzer_.pantsColor(bgr, entry_pants_rois[i], &cloud, wb_gain);
      }
      if (!raw_pants.empty()) {
        entry_pants[i] = updateAttributeVote(pants_color_memory_, e.id, raw_pants, current_time);
      } else {
        auto pit = pants_color_memory_.find(e.id);
        entry_pants[i] = (pit != pants_color_memory_.end() && !pit->second.history.empty())
          ? pit->second.history.back() : "";
      }

      if (hat_yolo_.ready() || glasses_yolo_.ready()) {
        const cv::Rect head_roi = AttributeAnalyzer::computeHeadAccessoryRoi(bgr, e.bbox, kpt2d);
        entry_has_hat[i] = detectAccessory(
          hat_yolo_, hat_memory_, hat_bbox_cache, hat_hit_time, e.id, head_roi,
          hat_conf_, 4, true, entry_hat_bboxes[i]);
        entry_has_glasses[i] = detectAccessory(
          glasses_yolo_, glasses_memory_, glasses_bbox_cache, glasses_hit_time, e.id, head_roi,
          0.18f, 2, false, entry_glasses_bboxes[i]);
      }

      if (i == 0 || !pending_register_name_.empty() || !registered_persons_.empty()) {
        static std::unordered_map<int, std::string> hair_cache;
        if (run_attr_infer) {
          const std::string h = analyzer_.hairColor(bgr, e.bbox, kpt2d);
          if (!h.empty()) {
            hair_cache[e.id] = h;
          }
        }
        auto hit = hair_cache.find(e.id);
        if (hit != hair_cache.end()) {
          entry_hairs[i] = hit->second;
        }
        auto git = gender_memory_.find(e.id);
        auto currentGenderVote = [&]() -> std::string {
          if (git == gender_memory_.end()) return "unknown";
          if (git->second.accumulated_margin >= 0.65) return "male";
          if (git->second.accumulated_margin <= -0.65) return "female";
          if (!git->second.history.empty()) return git->second.history.back();
          return "unknown";
        };

        if (run_attr_infer && !entry_facing_back[i]) {
          const auto pred = analyzer_.predictGenderDetailed(kpt2d, bgr, e.bbox);
          const std::string voted = updateGenderVote(e.id, pred, current_time);
          if (voted != "unknown") {
            entry_genders[i] = voted;
          } else {
            entry_genders[i] = currentGenderVote();
          }
        } else {
          entry_genders[i] = currentGenderVote();
        }
      }
    }
  }

  int op_match_idx = -1;
  if (op_mem_.active) {
    const double dt = (current_time - op_mem_.last_seen).seconds();
    if (dt <= 4.0) {
      const double pred_x = op_mem_.x + op_mem_.vx * std::min(dt, 2.0);
      const double pred_y = op_mem_.y + op_mem_.vy * std::min(dt, 2.0);
      const double max_match_dist = 0.65 + 0.6 * std::min(dt, 1.5);
      double best_cost = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < entries.size(); ++i) {
        const auto & cand = entries[i];
        if (!cand.has_depth) continue;
        const double dist_diff = std::hypot(cand.x - pred_x, cand.y - pred_y);
        if (dist_diff > max_match_dist) continue;
        double color_penalty = 0.0;
        if (op_mem_.has_color) {
          const cv::Scalar cand_hsv = zed_vision::extractTorsoColor(bgr, cand.bbox);
          const double dh = std::min(
            std::abs(cand_hsv[0] - op_mem_.torso_hsv[0]),
            180.0 - std::abs(cand_hsv[0] - op_mem_.torso_hsv[0]));
          const double ds = std::abs(cand_hsv[1] - op_mem_.torso_hsv[1]);
          color_penalty = (dh / 90.0) * 0.8 + (ds / 255.0) * 0.5;
          if (entries.size() > 1 && cand.id != op_mem_.last_zed_id && color_penalty > 0.65 && dist_diff > 0.45) {
            continue;
          }
        }
        const double id_bonus = (op_mem_.last_zed_id >= 0 && cand.id == op_mem_.last_zed_id) ? -0.35 : 0.0;
        const double cost = dist_diff + color_penalty + id_bonus;
        if (cost < best_cost) {
          best_cost = cost;
          op_match_idx = static_cast<int>(i);
        }
      }
    } else {
      op_mem_.active = false;
      op_mem_.has_color = false;
      op_mem_.last_zed_id = -1;
    }
  }

  if (!op_mem_.active && !entries.empty()) {
    op_match_idx = 0;
    op_mem_.active = true;
    op_mem_.stable_id = 0;
    op_mem_.last_zed_id = entries[0].id;
    op_mem_.vx = op_mem_.vy = 0.0;
    entries[0].label = "operator";
    if (entries[0].has_depth) {
      op_mem_.x = entries[0].x;
      op_mem_.y = entries[0].y;
    }
    op_mem_.torso_hsv = zed_vision::extractTorsoColor(bgr, entries[0].bbox);
    op_mem_.has_color = true;
    op_mem_.last_seen = current_time;
  }

  if (op_match_idx >= 0 && op_match_idx < static_cast<int>(entries.size())) {
    if (op_match_idx > 0) std::swap(entries[0], entries[static_cast<size_t>(op_match_idx)]);
    auto & op = entries.front();
    op_mem_.last_zed_id = op.id;
    op.id = op_mem_.stable_id;
    op.label = "operator";
    if (op.has_depth) {
      const double dt = std::max(0.01, (current_time - op_mem_.last_seen).seconds());
      if (dt < 1.0) {
        op_mem_.vx = 0.7 * op_mem_.vx + 0.3 * ((op.x - op_mem_.x) / dt);
        op_mem_.vy = 0.7 * op_mem_.vy + 0.3 * ((op.y - op_mem_.y) / dt);
      }
      op_mem_.x = op.x;
      op_mem_.y = op.y;
    }
    op_mem_.last_seen = current_time;
    const cv::Scalar hsv = zed_vision::extractTorsoColor(bgr, op.bbox);
    op_mem_.torso_hsv = 0.85 * op_mem_.torso_hsv + 0.15 * hsv;
    op_mem_.has_color = true;

    if (is_help_me_carry_ && op.has_depth) {
      latest_operator_found_ = true;
      latest_op_x_ = op.x;
      latest_op_y_ = op.y;
      latest_op_dist_ = op.dist;
    }
  }

  const bool should_capture =
    (!pending_register_name_.empty() && !entries.empty() && (is_receptionist_ || is_restaurant_)) ||
    (capture_face_requested_ && !entries.empty());

  if (should_capture) {
    capture_face_requested_ = false;
    size_t target_idx = 0;
    std::vector<float> feature;
    const std::string host_pname = get_parameter("host_name").as_string();
    const bool is_host_capture = (pending_register_name_ == host_pname);

    int best_score = -999999;
    for (size_t i = 0; i < entries.size(); ++i) {
      std::vector<float> cur_feat = analyzer_.faceEmbedding(
        bgr, entries[i].bbox, entries[i].body->keypoint_2d);
      if (cur_feat.empty()) continue;

      int score = 1000;
      // Prefer person closest to center of camera
      const int cx = entries[i].bbox.x + entries[i].bbox.width / 2;
      score -= std::abs(cx - bgr.cols / 2) * 2;

      if (is_host_capture) {
        // Exclude Guest 1 if already registered
        if (!guest_1_name_.empty()) {
          for (const auto & rp : registered_persons_) {
            if (rp.name == guest_1_name_ && !rp.feature.empty()) {
              float g1_sim = AttributeAnalyzer::matchFace(cur_feat, rp.feature);
              for (const auto & gf : rp.face_gallery) {
                if (!gf.empty()) g1_sim = std::max(g1_sim, AttributeAnalyzer::matchFace(cur_feat, gf));
              }
              if (g1_sim > 0.40f) {
                score -= 5000;
                break;
              }
            }
          }
        }
        // Host is at living room distance (>= 1.3m), and typically seated
        if (entries[i].range_m >= 1.3) {
          score += 300;
        } else {
          score -= 600; // Penalize person standing too close (< 1.3m, e.g. Guest 1 standing near robot)
        }
        if (entries[i].range_m >= 1.2 && entries[i].range_m <= 3.5) {
          score += 200;
        }
      } else {
        // Door guest / Restaurant: prefer closer person
        score += static_cast<int>(std::max(0.0, 100.0 - entries[i].range_m * 25.0));
      }

      if (score > best_score) {
        best_score = score;
        target_idx = i;
        feature = cur_feat;
      }
    }
    const bool feature_has_landmarks = !feature.empty();
    const PersonEntry * target = &entries[target_idx];

    // Simpan foto capture pelanggan / tamu untuk HMI Display (athome_face_display)
    try {
      cv::Rect crop_roi = analyzer_.faceCropRegion(
        target->bbox, target->body->keypoint_2d, bgr.cols, bgr.rows, bgr);
      if (crop_roi.width > 20 && crop_roi.height > 20) {
        cv::Mat crop_img = bgr(crop_roi).clone();
        std::vector<std::string> save_dirs = {
          "src/athome_face_display/web/captures",
          "/home/dell/Documents/athome_ws/src/athome_face_display/web/captures",
          "/home/dell/Downloads/athome_ws/src/athome_face_display/web/captures"
        };
        try {
          const std::string share_path = ament_index_cpp::get_package_share_directory("athome_face_display");
          save_dirs.push_back(share_path + "/web/captures");
        } catch (...) {}

        for (const auto & dir : save_dirs) {
          try {
            std::filesystem::create_directories(dir);
            cv::imwrite(dir + "/customer_latest.jpg", crop_img);
            std::string safe_name = pending_register_name_.empty() ? "customer" : pending_register_name_;
            std::replace(safe_name.begin(), safe_name.end(), ' ', '_');
            cv::imwrite(dir + "/customer_" + safe_name + ".jpg", crop_img);
          } catch (...) {}
        }
        RCLCPP_INFO(get_logger(),
          "\033[1;32m[%s]\033[0m Saved customer image to captures/customer_latest.jpg (%dx%d)",
          is_restaurant_ ? "RESTAURANT" : "RECEPTIONIST",
          crop_img.cols, crop_img.rows);
      }
    } catch (...) {}

    const cv::Scalar torso_hsv = zed_vision::extractTorsoColor(bgr, target->bbox);
    const std::string detected_shirt = entry_clothes[target_idx];
    const std::string detected_pants = entry_pants[target_idx];
    const std::string detected_hair = entry_hairs[target_idx];
    const std::string detected_gender = (entry_genders[target_idx] != "unknown") ? entry_genders[target_idx] : "";
    const bool detected_hat = entry_has_hat[target_idx];
    const bool detected_glasses = entry_has_glasses[target_idx];
    const double reg_elapsed = (register_start_time_.nanoseconds() > 0)
      ? (current_time - register_start_time_).seconds() : 0.0;

    auto updateProfileAttributes = [&](RegisteredPerson & p) {
      p.last_tracking_id = target->id;
      p.last_seen = current_time;
      if (target->has_depth) {
        p.last_x = target->x;
        p.last_y = target->y;
        p.has_last_pos = true;
      }
      if (target->bbox.width > 20) {
        p.torso_hsv = torso_hsv;
        p.has_color = true;
      }
      if (!detected_shirt.empty()) p.clothing_color = detected_shirt;
      if (!detected_pants.empty()) p.pants_color = detected_pants;
      if (!detected_hair.empty()) p.hair_color = detected_hair;
      if (!detected_gender.empty()) p.gender = detected_gender;
      if (!p.locked) {
        p.has_hat = detected_hat;
      } else if (detected_hat) {
        p.has_hat = true;
      }
      if (detected_glasses) p.has_glasses = true;

      if (!feature.empty()) {
        if (p.face_gallery.empty()) {
          p.face_gallery.push_back(feature);
        } else if (p.face_gallery.size() < 5) {
          float max_s = 0.0f;
          for (const auto & gf : p.face_gallery) {
            max_s = std::max(max_s, AttributeAnalyzer::matchFace(feature, gf));
          }
          if (max_s < 0.88f) p.face_gallery.push_back(feature);
        }
        if (!p.locked) {
          if (p.feature.empty()) {
            p.feature = feature;
            p.feature_samples = 1;
          } else if (feature_has_landmarks || p.feature_samples < 3) {
            float norm_sq = 0.0f;
            const float w_new = feature_has_landmarks ? 0.25f : 0.10f;
            for (size_t k = 0; k < feature.size(); ++k) {
              p.feature[k] = p.feature[k] * (1.0f - w_new) + feature[k] * w_new;
              norm_sq += p.feature[k] * p.feature[k];
            }
            const float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-6f));
            for (size_t k = 0; k < feature.size(); ++k) {
              p.feature[k] *= inv_norm;
            }
            p.feature_samples++;
          }
        }
        p.has_face_verified = true;
        if (feature_has_landmarks && p.feature_samples >= 2) {
          p.locked = true;
        }
      }
    };

    RegisteredPerson * reg_ptr = nullptr;
    for (auto & person : registered_persons_) {
      if (person.name == pending_register_name_) {
        reg_ptr = &person;
        break;
      }
    }
    if (!reg_ptr) {
      RegisteredPerson new_person;
      new_person.name = pending_register_name_;
      registered_persons_.push_back(std::move(new_person));
      reg_ptr = &registered_persons_.back();
    }

    updateProfileAttributes(*reg_ptr);

    // Kumpulkan minimal 5 sample wajah (atau timeout aman 2.5-4.0 detik) sebelum mengunci template (untuk restaurant langsung kunci pada sample pertama)
    const bool ready_to_lock =
      (is_restaurant_) ||
      (reg_ptr->feature_samples >= 5) ||
      (reg_elapsed >= 2.5 && reg_ptr->feature_samples >= 1) ||
      (reg_elapsed >= 4.0);
    if (ready_to_lock) {
      reg_ptr->feature_samples = 30;
      reg_ptr->locked = true;
      RCLCPP_INFO(get_logger(),
        "\033[1;32m[%s]\033[0m Registered '%s' (shirt=%s, pants=%s, hair=%s, gender=%s, hat=%s, glasses=%s, face=%s)",
        is_restaurant_ ? "RESTAURANT" : "RECEPTIONIST",
        pending_register_name_.c_str(),
        reg_ptr->clothing_color.c_str(),
        reg_ptr->pants_color.c_str(),
        reg_ptr->hair_color.c_str(),
        reg_ptr->gender.c_str(),
        reg_ptr->has_hat ? "yes" : "no",
        reg_ptr->has_glasses ? "yes" : "no",
        reg_ptr->feature.empty() ? "NO" : "OK");
      pending_register_name_.clear();
    }
  } else if (!pending_register_name_.empty() && (is_receptionist_ || is_restaurant_) && entries.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
      "Guest \"%s\" pending registration: no person detected",
      pending_register_name_.c_str());
  }

  std::vector<std::string> recognized_names(entries.size(), "");
  if (is_receptionist_ && !registered_persons_.empty()) {
    // Reset tracking ID & anchor posisi jika orang terlalu lama tidak terlihat (> 6.0 detik)
    for (auto & prof : registered_persons_) {
      if ((current_time - prof.last_seen).seconds() > 6.0) {
        prof.last_tracking_id = -1;
        prof.has_last_pos = false;
      }
    }

    // Ekstraksi fitur visual untuk semua orang yang terdeteksi (Hanya Receptionist)
    struct EntryFeature {
    std::vector<float> face_feat;
    cv::Scalar torso_hsv{0, 0, 0};
    bool has_torso{false};
    cv::Rect shirt_roi;
    std::string shirt_color;
    std::string pants_color;
    std::string hair_color;
    std::string gender;
    bool has_hat{false};
    bool has_glasses{false};
  };
  std::vector<EntryFeature> entry_feats(entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    if (analyzer_.faceReady()) {
      auto mit = name_memory_.find(entries[i].id);
      const bool already_confirmed = (mit != name_memory_.end() && !mit->second.confirmed_name.empty());
      if (!already_confirmed || (person_attr_frame_ % 4 == 0)) {
        entry_feats[i].face_feat = analyzer_.faceEmbedding(
          bgr, entries[i].bbox, entries[i].body->keypoint_2d);
      }
    }
    entry_feats[i].torso_hsv = zed_vision::extractTorsoColor(bgr, entries[i].bbox);
    entry_feats[i].has_torso = (entries[i].bbox.width > 20 && entries[i].bbox.height > 40);
    entry_feats[i].shirt_roi = entry_shirt_rois[i];
    entry_feats[i].shirt_color = entry_clothes[i];
    entry_feats[i].pants_color = entry_pants[i];
    entry_feats[i].hair_color = entry_hairs[i];
    entry_feats[i].gender = (entry_genders[i] == "unknown") ? std::string("") : entry_genders[i];
    entry_feats[i].has_hat = entry_has_hat[i];
    entry_feats[i].has_glasses = entry_has_glasses[i];
  }

  // --------------------------------------------------------------------------
  // MULTI-TARGET PERSON MATCHING (Global Permutation Optimization & Anti-Swap)
  // --------------------------------------------------------------------------
  std::vector<std::vector<float>> score_matrix(
    entries.size(), std::vector<float>(registered_persons_.size(), -100.0f));
  std::vector<std::vector<float>> face_sim(
    entries.size(), std::vector<float>(registered_persons_.size(), -1.0f));
  const float th = static_cast<float>(face_threshold_);
  const float claim_th = std::max(th + 0.08f, 0.44f);
  const float strong_th = std::max(th + 0.16f, 0.52f);
  const float break_th = th - 0.04f;

  // Hitung seluruh kesamaan wajah terlebih dahulu agar margin antar kandidat bisa diukur
  for (size_t i = 0; i < entries.size(); ++i) {
    if (entry_feats[i].face_feat.empty()) continue;
    for (size_t j = 0; j < registered_persons_.size(); ++j) {
      const auto & prof = registered_persons_[j];
      if (prof.feature.empty()) continue;
      float f_sim = AttributeAnalyzer::matchFace(entry_feats[i].face_feat, prof.feature);
      for (const auto & gf : prof.face_gallery) {
        if (!gf.empty()) {
          f_sim = std::max(f_sim, AttributeAnalyzer::matchFace(entry_feats[i].face_feat, gf));
        }
      }
      face_sim[i][j] = f_sim;
    }
  }

  auto bestOtherFaceSim = [&](size_t entry_i, size_t skip_j) -> float {
    float other_best = -1.0f;
    for (size_t k = 0; k < registered_persons_.size(); ++k) {
      if (k == skip_j) continue;
      other_best = std::max(other_best, face_sim[entry_i][k]);
    }
    return other_best;
  };

  std::vector<std::string> mem_confirmed(entries.size(), "");
  for (size_t i = 0; i < entries.size(); ++i) {
    auto mit = name_memory_.find(entries[i].id);
    if (mit != name_memory_.end() && !mit->second.confirmed_name.empty() &&
        (current_time - mit->second.last_seen).seconds() < 5.0)
    {
      mem_confirmed[i] = mit->second.confirmed_name;
    }
  }

  for (size_t i = 0; i < entries.size(); ++i) {
    for (size_t j = 0; j < registered_persons_.size(); ++j) {
      const auto & prof = registered_persons_[j];
      const bool has_face = (!entry_feats[i].face_feat.empty() && !prof.feature.empty());
      const float f_sim = has_face ? face_sim[i][j] : -1.0f;
      const float other_best = has_face ? bestOtherFaceSim(i, j) : -1.0f;
      const float face_margin = f_sim - other_best;
      float face_score = 0.0f;

      if (has_face) {
        if (f_sim >= strong_th) {
          face_score = 0.85f + 0.15f * std::min(1.0f, (f_sim - strong_th) / std::max(0.01f, 1.0f - strong_th));
        } else if (f_sim >= th) {
          face_score = 0.55f + 0.30f * ((f_sim - th) / std::max(0.01f, strong_th - th));
        } else if (f_sim < 0.22f) {
          face_score = 0.0f;
        } else {
          face_score = 0.05f + 0.50f * ((f_sim - 0.22f) / std::max(0.01f, th - 0.22f));
        }
      }

      // Skor warna baju
      float shirt_score = 0.50f;
      if (!prof.clothing_color.empty() && !entry_feats[i].shirt_color.empty()) {
        if (prof.clothing_color == entry_feats[i].shirt_color) {
          shirt_score = 1.0f;
        } else {
          double hsv_sim = 0.0;
          if (prof.has_color && entry_feats[i].has_torso) {
            double dh = std::min(std::abs(entry_feats[i].torso_hsv[0] - prof.torso_hsv[0]),
                                 180.0 - std::abs(entry_feats[i].torso_hsv[0] - prof.torso_hsv[0]));
            double ds = std::abs(entry_feats[i].torso_hsv[1] - prof.torso_hsv[1]);
            double dv = std::abs(entry_feats[i].torso_hsv[2] - prof.torso_hsv[2]);
            double color_dist = (dh / 90.0) * 0.6 + (ds / 255.0) * 0.2 + (dv / 255.0) * 0.2;
            hsv_sim = std::max(0.0, 1.0 - color_dist / 0.35);
          }
          shirt_score = static_cast<float>(hsv_sim * 0.25);
        }
      } else if (prof.has_color && entry_feats[i].has_torso) {
        double dh = std::min(std::abs(entry_feats[i].torso_hsv[0] - prof.torso_hsv[0]),
                             180.0 - std::abs(entry_feats[i].torso_hsv[0] - prof.torso_hsv[0]));
        double ds = std::abs(entry_feats[i].torso_hsv[1] - prof.torso_hsv[1]);
        double dv = std::abs(entry_feats[i].torso_hsv[2] - prof.torso_hsv[2]);
        double color_dist = (dh / 90.0) * 0.6 + (ds / 255.0) * 0.2 + (dv / 255.0) * 0.2;
        shirt_score = static_cast<float>(std::max(0.0, 1.0 - color_dist / 0.40));
      }

      // Skor celana, gender & warna rambut sebagai pembeda ekstra saat baju/wajah mirip
      float pants_score = 0.50f;
      if (!prof.pants_color.empty() && !entry_feats[i].pants_color.empty()) {
        pants_score = (prof.pants_color == entry_feats[i].pants_color) ? 1.0f : 0.10f;
      }
      float gender_score = 0.50f;
      if (!prof.gender.empty() && !entry_feats[i].gender.empty()) {
        gender_score = (prof.gender == entry_feats[i].gender) ? 1.0f : 0.05f;
      }
      float hair_score = 0.50f;
      if (!prof.hair_color.empty() && !entry_feats[i].hair_color.empty()) {
        hair_score = (prof.hair_color == entry_feats[i].hair_color) ? 1.0f : 0.25f;
      }

      const bool same_tracker_id = (prof.last_tracking_id >= 0 && prof.last_tracking_id == entries[i].id);
      const bool recent_seen = ((current_time - prof.last_seen).seconds() <= 2.5);
      const double spatial_dist = (prof.has_last_pos && entries[i].has_depth)
        ? std::hypot(entries[i].x - prof.last_x, entries[i].y - prof.last_y)
        : 999.0;
      const bool near_anchor = (prof.has_last_pos && entries[i].has_depth && recent_seen && spatial_dist < 0.35);
      const bool is_sticky = (!mem_confirmed[i].empty() && prof.name == mem_confirmed[i]);
      const bool has_continuity = (same_tracker_id || near_anchor || is_sticky);

      // Hard Veto: tolak wajah ambigu, tolak klaim baru di bawah claim_th, dan jaga sticky identity
      const float effective_claim_th = std::max(claim_th, 0.46f);
      const bool is_host_prof = (prof.name == get_parameter("host_name").as_string());
      bool veto = false;
      if (has_face) {
        if (!has_continuity) {
          const bool shirt_match = (!prof.clothing_color.empty() && !entry_feats[i].shirt_color.empty() &&
                                    prof.clothing_color == entry_feats[i].shirt_color);
          const bool pants_match = (!prof.pants_color.empty() && !entry_feats[i].pants_color.empty() &&
                                    prof.pants_color == entry_feats[i].pants_color);
          const bool pants_conflict = (!prof.pants_color.empty() && !entry_feats[i].pants_color.empty() &&
                                       prof.pants_color != entry_feats[i].pants_color);
          const bool shirt_ok = (prof.clothing_color.empty() || entry_feats[i].shirt_color.empty() || shirt_match);
          const bool attr_matches = (shirt_match && !pants_conflict) || (pants_match && shirt_ok);
          float req_claim_th = attr_matches
            ? th
            : (registered_persons_.size() <= 1
                ? th
                : (is_host_prof && entries[i].range_m >= 1.3 ? std::max(th, 0.38f) : effective_claim_th));
          if (entries[i].is_sitting || entries[i].range_m > 1.6) {
            req_claim_th = attr_matches
              ? std::max(0.28f, th - 0.10f)
              : std::max(0.32f, th - 0.06f);
          }
          if (f_sim < req_claim_th) {
            veto = true; // Klaim identitas baru butuh keyakinan wajah >= req_claim_th
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
              "[FACE] '%s' similarity %.3f < %.3f (veto)",
              prof.name.c_str(), f_sim, req_claim_th);
          } else if (other_best >= th) {
            // Jaga margin minimum agar dua orang yang duduk tidak mudah tertukar
            const float req_margin = attr_matches ? 0.03f : ((entries[i].is_sitting || entries[i].range_m >= 1.4) ? 0.05f : 0.08f);
            if (face_margin < req_margin) {
              veto = true; // Wajah ambigu antara dua profil -> jangan tebak nama baru
            }
          }
        } else {
          const float active_break_th = (registered_persons_.size() <= 1 || entries[i].is_sitting || entries[i].range_m > 1.6)
            ? 0.20f : break_th;
          if (f_sim < active_break_th) {
            veto = true; // Konflik wajah memutus ID/anchor lama
          } else if (other_best >= strong_th && (other_best - f_sim) >= 0.08f) {
            veto = true; // Profil lain menang telak -> lepas ID/anchor lama bila tracker ZED bertukar
          }
        }
      } else {
        if (!prof.feature.empty() && !has_continuity) {
          // Jangan veto jika atribut pakaian dan gender cocok dengan tamu terdaftar.
          // Jika salah satu profil belum memiliki rekaman warna baju (misal host dari foto), jangan veto.
          const bool cloth_ok = prof.clothing_color.empty() || entry_feats[i].shirt_color.empty() ||
                                prof.clothing_color == entry_feats[i].shirt_color;
          const bool pants_ok = prof.pants_color.empty() || entry_feats[i].pants_color.empty() ||
                                prof.pants_color == entry_feats[i].pants_color;
          const bool gender_ok = prof.gender.empty() || entry_feats[i].gender.empty() ||
                                 prof.gender == entry_feats[i].gender;
          const bool strong_attr_match = cloth_ok && pants_ok && gender_ok;

          if (!strong_attr_match && !entries[i].is_sitting) {
            veto = true;
          }
        }
      }

      // Penalti konflik warna baju & gender (hard veto hanya jika berdiri & klaim baru multi-profil tanpa wajah kuat)
      float attr_penalty = 0.0f;
      if (!prof.clothing_color.empty() && !entry_feats[i].shirt_color.empty() &&
          prof.clothing_color != entry_feats[i].shirt_color &&
          (!has_face || f_sim < strong_th || face_margin < 0.10f))
      {
        if (!has_continuity && registered_persons_.size() >= 2 && !has_face && !entries[i].is_sitting) {
          veto = true;
        } else {
          attr_penalty += 0.20f;
        }
      }
      if (!prof.pants_color.empty() && !entry_feats[i].pants_color.empty() &&
          prof.pants_color != entry_feats[i].pants_color &&
          (!has_face || f_sim < strong_th || face_margin < 0.10f))
      {
        attr_penalty += 0.15f;
      }
      if (!prof.gender.empty() && !entry_feats[i].gender.empty() &&
          prof.gender != entry_feats[i].gender &&
          (!has_face || f_sim < strong_th || face_margin < 0.10f))
      {
        if (!has_continuity && registered_persons_.size() >= 2 && !has_face && !entries[i].is_sitting) {
          veto = true;
        } else {
          attr_penalty += 0.15f;
        }
      }

      if (veto) {
        score_matrix[i][j] = -100.0f;
      } else {
        if (has_face) {
          if (entries[i].range_m > 1.6) {
            // Pada jarak jauh living room (> 1.6m), beri bobot warna baju & celana lebih tinggi
            // untuk membedakan orang yang duduk saat kursi ditukar
            score_matrix[i][j] = 0.48f * face_score + 0.25f * shirt_score + 0.13f * pants_score +
                                 0.07f * gender_score + 0.07f * hair_score;
          } else {
            score_matrix[i][j] = 0.74f * face_score + 0.08f * shirt_score + 0.06f * pants_score +
                                 0.06f * gender_score + 0.06f * hair_score;
          }
          if (f_sim >= strong_th && face_margin >= 0.07f) {
            score_matrix[i][j] += 0.25f; // Bonus hanya jika unggul mutlak atas kandidat lain
          }
        } else {
          score_matrix[i][j] = 0.42f * shirt_score + 0.25f * pants_score + 0.18f * gender_score + 0.15f * hair_score;
        }
        score_matrix[i][j] -= attr_penalty;
        if (prof.has_hat) {
          if (entry_feats[i].has_hat) {
            score_matrix[i][j] += 0.08f;
          } else if (!has_face || f_sim < strong_th) {
            score_matrix[i][j] -= 0.05f;
          }
        }
        if (prof.has_glasses) {
          if (entry_feats[i].has_glasses) {
            score_matrix[i][j] += 0.08f;
          } else if (!has_face || f_sim < strong_th) {
            score_matrix[i][j] -= 0.05f;
          }
        }

        if (is_sticky) {
          score_matrix[i][j] += 0.18f; // Sticky inertia ringan agar tidak mengalahkan skor wajah ArcFace
        }
        if (near_anchor) {
          score_matrix[i][j] += 0.10f;
        } else if (prof.has_last_pos && entries[i].has_depth && recent_seen &&
                   spatial_dist > 1.15 && !same_tracker_id && (!has_face || f_sim < strong_th))
        {
          score_matrix[i][j] -= 0.25f;
        }

        if (same_tracker_id) {
          score_matrix[i][j] += 0.10f;
        }
      }
    }
  }

  // TAHAP 2: Global Permutation Search (Maksimalisasi Joint Affinity Score)
  std::vector<int> best_assignment(registered_persons_.size(), -1);
  float max_total_score = -1e9f;

  std::vector<int> current_assignment(registered_persons_.size(), -1);
  std::vector<bool> entry_used(entries.size(), false);

  std::function<void(size_t, float)> search_perm = [&](size_t prof_idx, float current_score) {
    if (prof_idx == registered_persons_.size()) {
      if (current_score > max_total_score) {
        max_total_score = current_score;
        best_assignment = current_assignment;
      }
      return;
    }

    // Opsi 1: Profil prof_idx tidak di-assign ke siapapun (-1)
    current_assignment[prof_idx] = -1;
    search_perm(prof_idx + 1, current_score);

    // Opsi 2: Assign profil prof_idx ke salah satu entry yang belum terpakai
    for (size_t i = 0; i < entries.size(); ++i) {
      if (!entry_used[i]) {
        float s = score_matrix[i][prof_idx];
        if (s >= 0.45f) { // Ambang batas kelayakan kecocokan
          entry_used[i] = true;
          current_assignment[prof_idx] = static_cast<int>(i);
          search_perm(prof_idx + 1, current_score + s);
          entry_used[i] = false;
        }
      }
    }
  };

  search_perm(0, 0.0f);

  // TAHAP 3: Terapkan Hasil Assignment Optimal & Update Terkendali
  std::vector<std::string> raw_recognized_names(entries.size(), "");
  std::vector<float> raw_match_scores(entries.size(), -100.0f);
  std::vector<bool> raw_strong_face(entries.size(), false);
  std::vector<bool> raw_has_face(entries.size(), false);
  for (size_t j = 0; j < registered_persons_.size(); ++j) {
    int assigned_entry = best_assignment[j];
    if (assigned_entry >= 0 && score_matrix[assigned_entry][j] >= 0.45f) {
      raw_recognized_names[assigned_entry] = registered_persons_[j].name;
      raw_match_scores[assigned_entry] = score_matrix[assigned_entry][j];

      auto & prof = registered_persons_[j];
      const bool has_face_match = (!entry_feats[assigned_entry].face_feat.empty() && !prof.feature.empty());
      raw_has_face[assigned_entry] = has_face_match;
      const float f_sim = has_face_match ? face_sim[assigned_entry][j] : -1.0f;
      const float other_best = has_face_match ? bestOtherFaceSim(assigned_entry, j) : -1.0f;
      const float margin = f_sim - other_best;
      raw_strong_face[assigned_entry] = (has_face_match && f_sim >= strong_th && margin >= 0.07f);

      const float eff_claim = (registered_persons_.size() <= 1)
        ? th
        : (entries[assigned_entry].is_sitting ? std::max(th, claim_th - 0.05f) : claim_th);
      const float eff_margin = entries[assigned_entry].is_sitting ? 0.03f : 0.05f;
      const bool face_confirmed = (has_face_match && f_sim >= eff_claim && margin >= eff_margin);
      const bool fallback_no_face = (prof.feature.empty() && score_matrix[assigned_entry][j] >= 0.75f);
      if (face_confirmed || fallback_no_face) {
        prof.last_tracking_id = entries[assigned_entry].id;
        if (entries[assigned_entry].has_depth) {
          prof.last_x = entries[assigned_entry].x;
          prof.last_y = entries[assigned_entry].y;
          prof.has_last_pos = true;
        }
        if (face_confirmed) prof.has_face_verified = true;
        prof.last_seen = current_time;
      } else if ((prof.last_tracking_id >= 0 && prof.last_tracking_id == entries[assigned_entry].id) ||
                 (prof.has_last_pos && entries[assigned_entry].has_depth &&
                  std::hypot(entries[assigned_entry].x - prof.last_x, entries[assigned_entry].y - prof.last_y) < 0.50))
      {
        prof.last_seen = current_time;
      }

      // Anti-kontaminasi: tambah sudut baru ke gallery saat wajah yakin & margin lebar
      if (has_face_match && f_sim >= 0.48f && margin >= 0.10f && prof.face_gallery.size() < 8) {
        float max_gal_sim = AttributeAnalyzer::matchFace(entry_feats[assigned_entry].face_feat, prof.feature);
        for (const auto & gf : prof.face_gallery) {
          if (!gf.empty()) {
            max_gal_sim = std::max(max_gal_sim, AttributeAnalyzer::matchFace(entry_feats[assigned_entry].face_feat, gf));
          }
        }
        if (max_gal_sim < 0.88f) {
          prof.face_gallery.push_back(entry_feats[assigned_entry].face_feat);
        }
      }

      // Belajar atribut baju/celana/rambut/gender jika sebelumnya belum terekam
      if (prof.clothing_color.empty() && !entry_feats[assigned_entry].shirt_color.empty()) {
        if (has_face_match && f_sim >= eff_claim && margin >= 0.05f) {
          prof.clothing_color = entry_feats[assigned_entry].shirt_color;
          prof.torso_hsv = entry_feats[assigned_entry].torso_hsv;
          prof.has_color = true;
        }
      }
      if (prof.pants_color.empty() && !entry_feats[assigned_entry].pants_color.empty()) {
        if (has_face_match && f_sim >= eff_claim && margin >= 0.05f) {
          prof.pants_color = entry_feats[assigned_entry].pants_color;
        }
      }
      if (prof.hair_color.empty() && !entry_feats[assigned_entry].hair_color.empty()) {
        if (has_face_match && f_sim >= eff_claim && margin >= 0.05f) {
          prof.hair_color = entry_feats[assigned_entry].hair_color;
        }
      }
      if (prof.gender.empty() && !entry_feats[assigned_entry].gender.empty()) {
        if (has_face_match && f_sim >= eff_claim && margin >= 0.05f) {
          prof.gender = entry_feats[assigned_entry].gender;
        }
      }
      if (!prof.has_hat && entry_feats[assigned_entry].has_hat) {
        if (has_face_match && f_sim >= eff_claim && margin >= 0.05f) {
          prof.has_hat = true;
        }
      }
      if (!prof.has_glasses && entry_feats[assigned_entry].has_glasses) {
        if (has_face_match && f_sim >= eff_claim && margin >= 0.05f) {
          prof.has_glasses = true;
        }
      }

      // KUNCI: Template wajah TIDAK BOLEH di-update jika sudah terkunci (prof.locked == true)
      if (!prof.locked && !entry_feats[assigned_entry].face_feat.empty() && prof.feature_samples < 25) {
        if (face_sim[assigned_entry][j] >= 0.50f) {
          float norm_sq = 0.0f;
          for (size_t k = 0; k < prof.feature.size(); ++k) {
            prof.feature[k] = prof.feature[k] * 0.95f + entry_feats[assigned_entry].face_feat[k] * 0.05f;
            norm_sq += prof.feature[k] * prof.feature[k];
          }
          const float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-6f));
          for (size_t k = 0; k < prof.feature.size(); ++k) {
            prof.feature[k] *= inv_norm;
          }
          prof.feature_samples++;
        }
      }
    }
  }

  // TAHAP 4: Sliding-Window Majority Voting & Hysteresis Anti-Flicker per Body ID
  for (size_t i = 0; i < entries.size(); ++i) {
    const int body_id = entries[i].id;
    auto & mem = name_memory_[body_id];
    const std::string & raw_name = raw_recognized_names[i];

    if (!raw_name.empty()) {
      mem.history.push_back(raw_name);
      if (mem.history.size() > 12) {
        mem.history.pop_front();
      }

      // Hitung modus pada jendela 12 frame terakhir
      std::unordered_map<std::string, int> counts;
      for (const auto & h : mem.history) {
        counts[h]++;
      }
      std::string mode_name = raw_name;
      int mode_count = 0;
      for (const auto & kv : counts) {
        if (kv.second > mode_count) {
          mode_count = kv.second;
          mode_name = kv.first;
        }
      }

      if (mem.confirmed_name.empty()) {
        if (raw_name == mem.candidate_name) {
          mem.candidate_streak++;
        } else {
          mem.candidate_name = raw_name;
          mem.candidate_streak = 1;
        }
        // Konfirmasi awal: jika >= 2 profil terdaftar, utamakan konfirmasi berbasis wajah agar tidak tertukar
        const bool instant_confirm =
          raw_strong_face[i] ||
          (raw_has_face[i] && raw_match_scores[i] >= 0.68f) ||
          (registered_persons_.size() <= 1 && raw_match_scores[i] >= 0.45f);
        const int req_streak = (registered_persons_.size() >= 2 && !raw_has_face[i]) ? 3 : 2;
        if (instant_confirm || mode_count >= req_streak || mem.candidate_streak >= req_streak) {
          mem.confirmed_name = (mode_count >= req_streak) ? mode_name : raw_name;
          mem.candidate_streak = 0;
          mem.last_seen = current_time;
        }
      } else if (raw_name == mem.confirmed_name || mode_name == mem.confirmed_name) {
        mem.candidate_streak = 0;
        mem.last_seen = current_time;
      } else {
        // Hysteresis: ganti label terkunci jika nama baru mendominasi >= 70% jendela
        // atau >= 3 frame dengan kecocokan wajah sangat kuat (koreksi cepat bila sempat tertukar).
        if (raw_name == mem.candidate_name) {
          mem.candidate_streak++;
        } else {
          mem.candidate_name = raw_name;
          mem.candidate_streak = 1;
        }

        const int win_size = static_cast<int>(mem.history.size());
        const bool mode_dominant = (win_size >= 8 && mode_count * 10 >= win_size * 7);
        const bool strong_override = (raw_strong_face[i] && mode_count >= 3);
        if (mode_dominant || strong_override) {
          mem.confirmed_name = mode_name;
          mem.candidate_streak = 0;
          mem.last_seen = current_time;
        }
      }
    } else {
      // Wajah menoleh/tertutup sementara; tahan nama di kursi hingga 8.0 detik saat duduk (5.0s berdiri) agar tidak blink
      const double hold_sec = entries[i].is_sitting ? 8.0 : 5.0;
      if (!mem.confirmed_name.empty() && (current_time - mem.last_seen).seconds() < hold_sec) {
        // Pertahankan mem.confirmed_name
      } else {
        mem.confirmed_name.clear();
        mem.history.clear();
        mem.candidate_streak = 0;
      }
    }

    recognized_names[i] = mem.confirmed_name;
  }

  // Enforsi keunikan nama aktif di frame ini: satu nama tidak boleh muncul di dua orang sekaligus
  std::unordered_map<std::string, size_t> name_owner;
  auto matchPriority = [&](size_t idx) -> double {
    const bool active_match = (raw_recognized_names[idx] == recognized_names[idx]);
    if (active_match) {
      return 100.0 + static_cast<double>(raw_match_scores[idx]);
    }
    const double age = std::max(0.0, (current_time - name_memory_[entries[idx].id].last_seen).seconds());
    return -age;
  };
  for (size_t i = 0; i < entries.size(); ++i) {
    if (recognized_names[i].empty()) continue;
    auto it = name_owner.find(recognized_names[i]);
    if (it == name_owner.end()) {
      name_owner[recognized_names[i]] = i;
    } else {
      const size_t prev_idx = it->second;
      if (matchPriority(i) > matchPriority(prev_idx)) {
        name_memory_[entries[prev_idx].id].confirmed_name.clear();
        name_memory_[entries[prev_idx].id].candidate_streak = 0;
        recognized_names[prev_idx].clear();
        it->second = i;
      } else {
        name_memory_[entries[i].id].confirmed_name.clear();
        name_memory_[entries[i].id].candidate_streak = 0;
        recognized_names[i].clear();
      }
    }
  }

  // Bersihkan name_memory_ dan shirt_color_memory_ yang sudah tidak terlihat lebih dari 5.0 detik
  for (auto it = name_memory_.begin(); it != name_memory_.end(); ) {
    if ((current_time - it->second.last_seen).seconds() > 5.0) {
      it = name_memory_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = shirt_color_memory_.begin(); it != shirt_color_memory_.end(); ) {
    if ((current_time - it->second.last_seen).seconds() > 5.0) {
      it = shirt_color_memory_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = pants_color_memory_.begin(); it != pants_color_memory_.end(); ) {
    if ((current_time - it->second.last_seen).seconds() > 5.0) {
      it = pants_color_memory_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = hat_memory_.begin(); it != hat_memory_.end(); ) {
    if ((current_time - it->second.last_seen).seconds() > 5.0) {
      it = hat_memory_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = glasses_memory_.begin(); it != glasses_memory_.end(); ) {
    if ((current_time - it->second.last_seen).seconds() > 5.0) {
      it = glasses_memory_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = gender_memory_.begin(); it != gender_memory_.end(); ) {
    if ((current_time - it->second.last_seen).seconds() > 5.0) {
      it = gender_memory_.erase(it);
    } else {
      ++it;
    }
  }
  }

  std::ostringstream ss;
  bool first_person = true;
  std::string gesture_result = "NONE";
  std::string first_recognized_name;
  std::string wave_result = "NONE";
  size_t wave_idx = entries.size();
  size_t back_idx = entries.size();

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    const auto & entry = entries[idx];
    const auto & points = entry.body->keypoint_2d;
    std::string recognized_name = recognized_names[idx];

    bool is_facing_back = false;
    if (enable_back_detection_) {
      is_facing_back = is_receptionist_
        ? static_cast<bool>(entry_facing_back[idx])
        : AttributeAnalyzer::isFacingBack(*entry.body, entry.bbox);
      if (is_facing_back) {
        auto & memory = back_memory_[entry.id];
        memory.streak = std::min(memory.streak + 1, 10);
        memory.last_seen = current_time;
        if (memory.streak >= 2 && entry.has_depth && entry.range_m <= person_max_dist_ &&
            (back_idx == entries.size() || entry.range_m < entries[back_idx].range_m))
        {
          back_idx = idx;
        }
      } else {
        back_memory_.erase(entry.id);
      }
    }

    if (is_restaurant_) {
      const std::string wave = AttributeAnalyzer::detectWave(points);
      if (wave == "NONE") {
        wave_memory_.erase(entry.id);
      } else {
        auto & memory = wave_memory_[entry.id];
        if (memory.side == wave) {
          memory.streak = std::min(memory.streak + 1, 10);
        } else {
          memory.side = wave;
          memory.streak = 1;
        }
        memory.last_seen = current_time;
        if (memory.streak >= 3 && entry.has_depth &&
            (wave_idx == entries.size() || entry.range_m < entries[wave_idx].range_m))
        {
          wave_idx = idx;
          wave_result = wave;
        }
      }
    }

    // Jika sedang mendaftarkan nama tamu baru di pintu / restoran
    if (recognized_name.empty() && (is_receptionist_ || is_restaurant_) && !pending_register_name_.empty() && first_person) {
      recognized_name = pending_register_name_;
    }

    if (first_recognized_name.empty() && !recognized_name.empty()) {
      first_recognized_name = recognized_name;
    }

    PrevPerson person;
    person.id = entry.id;
    person.bbox = entry.bbox;
    person.shirt_bbox = is_receptionist_
      ? entry_shirt_rois[idx]
      : AttributeAnalyzer::computeShirtRoi(bgr, entry.bbox, points);
    person.pants_bbox = is_receptionist_
      ? entry_pants_rois[idx]
      : AttributeAnalyzer::computePantsRoi(bgr, entry.bbox, points);
    person.has_pos = entry.has_depth;
    person.x_m = entry.x;
    person.y_m = entry.y;
    person.range_m = entry.range_m;
    person.is_facing_back = is_facing_back;
    person.name = recognized_name;

    // Deteksi postur duduk/berdiri menggunakan keypoint 3D ZED (unit cm, +Z atas)
    if (entry.body->keypoint.size() >= 25) {
      const auto & k3 = entry.body->keypoint;
      auto valid3d = [](const sl::float3 & p) {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
      };
      auto eval_leg_3d = [&](int hip_i, int knee_i, int ankle_i, bool & is_straight, bool & is_bent) {
        is_straight = false;
        is_bent = false;
        const auto & hip = k3[hip_i];
        const auto & knee = k3[knee_i];
        const auto & ankle = k3[ankle_i];
        if (!valid3d(hip) || !valid3d(knee) || !valid3d(ankle)) return;
        const float thigh_dz = hip.z - knee.z;
        const float shin_dz = knee.z - ankle.z;
        const float thigh_horiz = std::hypot(knee.x - hip.x, knee.y - hip.y);
        const float ux = hip.x - knee.x, uy = hip.y - knee.y, uz = hip.z - knee.z;
        const float vx = ankle.x - knee.x, vy = ankle.y - knee.y, vz = ankle.z - knee.z;
        const float un = std::sqrt(ux * ux + uy * uy + uz * uz);
        const float vn = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (un < 1e-3f || vn < 1e-3f) return;
        const float cos_knee = (ux * vx + uy * vy + uz * vz) / (un * vn);
        if (thigh_dz > 24.0f && shin_dz > 20.0f &&
            thigh_dz > thigh_horiz * 1.4f && cos_knee < -0.80f)
        {
          is_straight = true;
        } else if (thigh_dz < 28.0f && shin_dz > 15.0f &&
                   thigh_horiz > thigh_dz * 0.85f && cos_knee > -0.68f)
        {
          is_bent = true;
        }
      };
      bool l_straight = false, l_bent = false, r_straight = false, r_bent = false;
      eval_leg_3d(18, 19, 20, l_straight, l_bent);
      eval_leg_3d(22, 23, 24, r_straight, r_bent);
      person.is_standing_posture = (l_straight || r_straight) && !(l_bent || r_bent);
      person.is_sitting_posture = (l_bent || r_bent) && !(l_straight || r_straight);
      if (person.is_sitting_posture) {
        person.is_sitting = true;
      }
    }
    for (const auto & point : points) {
      person.kpt.emplace_back(
        std::isfinite(point.x) ? point.x : NAN,
        std::isfinite(point.y) ? point.y : NAN);
    }

    if (is_receptionist_) {
      person.clothing = entry_clothes[idx];
      person.pants = entry_pants[idx];
      person.has_hat = entry_has_hat[idx];
      person.has_glasses = entry_has_glasses[idx];
      person.hat_bbox = entry_hat_bboxes[idx];
      person.glasses_bbox = entry_glasses_bboxes[idx];
      const bool classify = (first_person || !pending_register_name_.empty() || !recognized_name.empty());
      if (classify) {
        person.hair = entry_hairs[idx];
        person.gender = entry_genders[idx];
      }
      if (!recognized_name.empty()) {
        for (const auto & prof : registered_persons_) {
          if (prof.name == recognized_name && prof.locked) {
            if ((person.gender.empty() || person.gender == "unknown") && !prof.gender.empty()) {
              person.gender = prof.gender;
            }
            if ((person.clothing.empty() || person.clothing == "casual attire") && !prof.clothing_color.empty()) {
              person.clothing = prof.clothing_color;
            }
            if (person.hair.empty() && !prof.hair_color.empty()) {
              person.hair = prof.hair_color;
            }
            break;
          }
        }
      }
    }

    if (first_person) {
      if (!is_restaurant_) {
        gesture_result = AttributeAnalyzer::detectGesture(points);
        if (is_help_me_carry_ && gesture_result != "NONE") {
          latest_gesture_ = gesture_result;
        }
      }

      if (is_receptionist_ && entry.has_depth) {
        if (!latest_guest_found_ || (!recognized_name.empty() && latest_guest_tag_ == "Guest")) {
          latest_guest_found_ = true;
          latest_guest_tag_ = (!recognized_name.empty() ? recognized_name : "Guest");
          latest_guest_x_ = entry.x;
          latest_guest_y_ = entry.y;
          latest_guest_dist_ = entry.range_m;
        }
      } else if (is_restaurant_ && entry.has_depth) {
        if (!latest_customer_found_) {
          latest_customer_found_ = true;
          latest_cust_x_ = entry.x;
          latest_cust_y_ = entry.y;
          latest_cust_dist_ = entry.range_m;
        }
      }
    }

    preview_persons_.push_back(person);

    if (!first_person) ss << ";";
    std::string label;
    if (is_receptionist_ && !is_help_me_carry_) {
      if (!recognized_name.empty()) {
        label = recognized_name;
      } else {
        label = person.is_sitting ? "seated" : (first_person ? "guest" : "person");
      }
    } else {
      label = entry.label;
    }
    first_person = false;
    ss << label << ":" << entry.bbox.x << "," << entry.bbox.y << ","
       << entry.bbox.width << "," << entry.bbox.height << "|";
    if (entry.has_depth) {
      ss << std::fixed << std::setprecision(2) << entry.x << "," << entry.y << "," << entry.z;
    } else {
      ss << "nan,nan,nan";
    }
    ss << "|id=" << entry.id << "|sitting=" << (person.is_sitting ? "1" : "0")
       << "|standing=" << (person.is_standing_posture ? "1" : "0")
       << "|back=" << (is_facing_back ? "1" : "0") << "|kpt=";
    for (size_t k = 0; k < 34; ++k) {
      if (k > 0) ss << ",";
      if (k < points.size() && std::isfinite(points[k].x) && std::isfinite(points[k].y)) {
        ss << std::fixed << std::setprecision(2) << points[k].x << "," << points[k].y;
      } else {
        ss << "-1,-1";
      }
    }
  }

  if (is_receptionist_ && !preview_persons_.empty()) {
    // Ikat publikasi atribut ke tamu terdaftar (RegisteredPerson) yang aktif/terkunci
    // alih-alih selalu mengambil entries[0] (first_person).
    std::string pub_gender;
    std::string pub_cloth;
    std::string pub_pants;
    std::string pub_hair;
    bool pub_hat = false;
    bool pub_glasses = false;
    bool bound_to_registered = false;

    const std::string target_name = !pending_register_name_.empty()
      ? pending_register_name_
      : active_guest_name_;

    if (!target_name.empty()) {
      const RegisteredPerson * target_prof = nullptr;
      for (const auto & prof : registered_persons_) {
        if (prof.name == target_name) {
          target_prof = &prof;
          break;
        }
      }

      int target_entry_idx = -1;
      for (size_t i = 0; i < entries.size(); ++i) {
        if (recognized_names[i] == target_name) {
          target_entry_idx = static_cast<int>(i);
          break;
        }
      }
      if (target_entry_idx < 0 && target_prof != nullptr && target_prof->last_tracking_id >= 0) {
        for (size_t i = 0; i < entries.size(); ++i) {
          if (entries[i].id == target_prof->last_tracking_id) {
            target_entry_idx = static_cast<int>(i);
            break;
          }
        }
      }

      if (target_entry_idx >= 0) {
        const size_t tidx = static_cast<size_t>(target_entry_idx);
        pub_gender = entry_genders[tidx];
        pub_cloth = entry_clothes[tidx];
        pub_pants = entry_pants[tidx];
        pub_hair = entry_hairs[tidx];
        pub_hat = entry_has_hat[tidx];
        pub_glasses = entry_has_glasses[tidx];
        if (target_prof != nullptr) {
          if ((pub_gender.empty() || pub_gender == "unknown") && !target_prof->gender.empty()) {
            pub_gender = target_prof->gender;
          }
          if ((pub_cloth.empty() || pub_cloth == "casual attire") && !target_prof->clothing_color.empty()) {
            pub_cloth = target_prof->clothing_color;
          }
          if (pub_pants.empty() && !target_prof->pants_color.empty()) {
            pub_pants = target_prof->pants_color;
          }
          if (pub_hair.empty() && !target_prof->hair_color.empty()) {
            pub_hair = target_prof->hair_color;
          }
          if (!pub_hat && target_prof->has_hat) {
            pub_hat = true;
          }
          if (!pub_glasses && target_prof->has_glasses) {
            pub_glasses = true;
          }
        }
        bound_to_registered = true;
      } else if (target_prof != nullptr && target_prof->locked) {
        pub_gender = target_prof->gender;
        pub_cloth = target_prof->clothing_color;
        pub_pants = target_prof->pants_color;
        pub_hair = target_prof->hair_color;
        pub_hat = target_prof->has_hat;
        pub_glasses = target_prof->has_glasses;
        bound_to_registered = true;
      }
    }

    if (!bound_to_registered) {
      pub_gender = preview_persons_.front().gender;
      pub_cloth = preview_persons_.front().clothing;
      pub_pants = preview_persons_.front().pants;
      pub_hair = preview_persons_.front().hair;
      pub_hat = preview_persons_.front().has_hat;
      pub_glasses = preview_persons_.front().has_glasses;
    }

    if (pub_gender.empty()) pub_gender = "person";
    if (pub_cloth.empty()) pub_cloth = "casual attire";
    if (pub_hair.empty()) pub_hair = "dark hair";

    std_msgs::msg::String attr_msg;
    attr_msg.data = pub_gender + ", " + pub_cloth;
    if (!pub_pants.empty()) attr_msg.data += ", " + pub_pants;
    attr_msg.data += ", " + pub_hair;
    if (pub_hat) attr_msg.data += ", wearing hat";
    if (pub_glasses) attr_msg.data += ", wearing glasses";
    attributes_publisher_->publish(attr_msg);
  }

  if (is_restaurant_) {
    for (auto it = wave_memory_.begin(); it != wave_memory_.end(); ) {
      if ((current_time - it->second.last_seen).seconds() > 1.0) {
        it = wave_memory_.erase(it);
      } else {
        ++it;
      }
    }

    gesture_result = wave_result;
    if (wave_idx < entries.size()) {
      const auto & customer = entries[wave_idx];
      std::ostringstream order_ss;
      order_ss << std::fixed << std::setprecision(2)
               << customer.x << ", " << customer.y;
      std_msgs::msg::String order_msg;
      order_msg.data = order_ss.str();
      order_publisher_->publish(order_msg);
      if (wave_result != "NONE") {
        if (is_restaurant_) {
          latest_wave_found_ = true;
          latest_wave_x_ = customer.x;
          latest_wave_y_ = customer.y;
        } else {
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1500,
            "\033[1;36m[WAVE]\033[0m %s (%.2f, %.2f)",
            wave_result.c_str(), customer.x, customer.y);
        }
      }
    }
  }

  if (enable_back_detection_) {
    for (auto it = back_memory_.begin(); it != back_memory_.end(); ) {
      if ((current_time - it->second.last_seen).seconds() > 1.0) {
        it = back_memory_.erase(it);
      } else {
        ++it;
      }
    }

    if (back_idx < entries.size()) {
      const auto & back_person = entries[back_idx];
      std::ostringstream back_ss;
      back_ss << std::fixed << std::setprecision(2)
              << back_person.x << ", " << back_person.y;
      std_msgs::msg::String back_msg;
      back_msg.data = back_ss.str();
      person_back_publisher_->publish(back_msg);
      if (is_help_me_carry_) {
        latest_back_found_ = true;
        latest_back_x_ = back_person.x;
        latest_back_y_ = back_person.y;
        latest_back_dist_ = back_person.range_m;
      } else {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1200,
          "\033[1;35m[BACK]\033[0m %.2f, %.2f (%.2fm)",
          back_person.x, back_person.y, back_person.range_m);
      }
    }
  }

  std_msgs::msg::String gesture_msg;
  gesture_msg.data = gesture_result;
  gesture_publisher_->publish(gesture_msg);

  std_msgs::msg::String person_msg;
  person_msg.data = first_person ? "NO_PERSON" : ss.str();
  persons_publisher_->publish(person_msg);

  std_msgs::msg::String name_msg;
  name_msg.data = is_receptionist_ ? first_recognized_name : "";
  recognized_name_publisher_->publish(name_msg);

  if (is_receptionist_) {
    publishGuestNames();
  }
}

void ZedNode::publishGuestNames()
{
  std::string host = get_parameter("host_name").as_string();
  std::vector<std::string> reg_guests;
  for (const auto & p : registered_persons_) {
    if (p.name != host && !p.name.empty() && p.name.rfind("__pending_", 0) != 0) {
      reg_guests.push_back(p.name);
    }
  }
  if (reg_guests.size() >= 1 && guest_1_name_.empty()) {
    guest_1_name_ = reg_guests[0];
  }
  if (reg_guests.size() >= 2 && guest_2_name_.empty()) {
    guest_2_name_ = reg_guests[1];
  }

  if (guest_1_name_publisher_) {
    std_msgs::msg::String m1;
    m1.data = guest_1_name_;
    guest_1_name_publisher_->publish(m1);
  }
  if (guest_2_name_publisher_) {
    std_msgs::msg::String m2;
    m2.data = guest_2_name_;
    guest_2_name_publisher_->publish(m2);
  }
}

void ZedNode::processChairs(const cv::Mat & bgr, const sl::Mat & cloud)
{
  if (!enable_chair_) {
    last_chair_dets_.clear();
    preview_chairs_.clear();
    std_msgs::msg::String chair_msg;
    chair_msg.data = "";
    chair_dets_publisher_->publish(chair_msg);
    std_msgs::msg::String seat_msg;
    seat_msg.data = "none";
    empty_seat_publisher_->publish(seat_msg);
    return;
  }

  static rclcpp::Time last_chair_hit_time(0, 0, RCL_ROS_TIME);
  std::vector<YoloDetection> dets = last_chair_dets_;
  if ((chair_frame_++ % yolo_stride_) == (2 % yolo_stride_)) {
    std::vector<YoloDetection> fresh_dets;
    if (chair_yolo_.ready()) {
      chair_yolo_.detect(bgr, {}, fresh_dets);
    } else if (enable_yolo_ && yolo_.ready()) {
      yolo_.detect(bgr, chair_labels_, fresh_dets);
    }
    if (!fresh_dets.empty()) {
      dets = fresh_dets;
      last_chair_dets_ = fresh_dets;
      last_chair_hit_time = now();
    } else if ((now() - last_chair_hit_time).seconds() < 1.5) {
      // Pertahankan deteksi kursi terakhir hingga 1.5s jika YOLO miss sesaat saat robot berputar
      dets = last_chair_dets_;
    } else {
      dets.clear();
      last_chair_dets_.clear();
    }
  }

  struct ActiveChair
  {
    std::string label{"chair"};
    cv::Rect bbox;
    bool has_pos{false};
    double x{0.0}, y{0.0}, z{0.0};
    double range_m{0.0};
    bool occupied{false};
  };

  std::vector<ActiveChair> chairs;
  for (const auto & det : dets) {
    if (static_cast<double>(det.rect.area()) < yolo_min_area_) continue;
    double x = 0.0, y = 0.0, z = 0.0;
    bool ok = false;
    sl::float4 median_pt{NAN, NAN, NAN, NAN};
    if (zed_vision::get3dCoordinates(det.rect, cloud, median_pt) && std::isfinite(median_pt.z)) {
      x = (median_pt.x + zed_vision::kOffsetX) / zed_vision::kOffsetScale;
      y = (-median_pt.y + zed_vision::kOffsetY) / zed_vision::kOffsetScale;
      z = (median_pt.z + zed_vision::kOffsetZ) / zed_vision::kOffsetScale;
      ok = true;
    } else {
      ok = zed_vision::get3dWithOffset(det.rect, cloud, x, y, z);
    }
    const std::string det_label = (det.label == "couch" || det.label == "sofa") ? "sofa" : det.label;

    ActiveChair chair;
    chair.label = det_label.empty() ? "chair" : det_label;
    chair.bbox = det.rect;
    chair.has_pos = ok;
    if (ok) {
      chair.x = x;
      chair.y = y;
      chair.z = z;
      chair.range_m = std::hypot(x, y);
      if (chair.range_m > chair_max_dist_) {
        continue;
      }
    } else {
      if (det.rect.area() < 2500 || det.rect.height < 45) {
        continue;
      }
    }
    chairs.push_back(chair);
  }

  auto compute_match_score = [&](const ActiveChair & chair, const PrevPerson & person) -> double {
      // 1. WAJIB: Ada irisan 2D antara bounding box kursi dan orang
      const cv::Rect inter = chair.bbox & person.bbox;
      if (inter.area() <= 0) return 0.0;

      const double chair_area = static_cast<double>(chair.bbox.area());
      const double person_area = static_cast<double>(person.bbox.area());
      if (chair_area <= 0.0 || person_area <= 0.0) return 0.0;

      const double overlap_chair = static_cast<double>(inter.area()) / chair_area;
      const double overlap_person = static_cast<double>(inter.area()) / person_area;

      // 2. Keselarasan horizontal (X-axis)
      const int inter_x = std::max(0,
        std::min(chair.bbox.x + chair.bbox.width, person.bbox.x + person.bbox.width) -
        std::max(chair.bbox.x, person.bbox.x));
      const int min_w = std::min(chair.bbox.width, person.bbox.width);
      const double x_ratio = min_w > 0 ? static_cast<double>(inter_x) / min_w : 0.0;

      // 3. Pengecekan posisi tubuh orang terhadap kursi
      const int cx = person.bbox.x + person.bbox.width / 2;
      const int cy = person.bbox.y + person.bbox.height / 2;
      const int hip_y = person.bbox.y + static_cast<int>(person.bbox.height * 0.65);
      const bool hip_inside = chair.bbox.contains(cv::Point(cx, hip_y));
      const bool center_inside = chair.bbox.contains(cv::Point(cx, cy));

      // Pengecekan titik tengah kursi di dalam tubuh orang
      const int chair_cx = chair.bbox.x + chair.bbox.width / 2;
      const int chair_cy = chair.bbox.y + chair.bbox.height / 2;
      const bool chair_center_inside = person.bbox.contains(cv::Point(chair_cx, chair_cy));

      // Keypoints tubuh ZED
      bool kpt_inside = false;
      for (const auto & kp : person.kpt) {
        if (std::isfinite(kp.x) && std::isfinite(kp.y) &&
            chair.bbox.contains(cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y))))
        {
          kpt_inside = true;
          break;
        }
      }

      // Tolak orang yang postur 3D-nya jelas berdiri tegak dan hanya menyenggol tepi kursi
      if (person.is_standing_posture &&
          overlap_person < 0.30 && overlap_chair < 0.30 &&
          !hip_inside && !chair_center_inside)
      {
        return 0.0;
      }

      // 4. Cek konsistensi kedalaman 3D (longgarkan untuk sofa lebar, postur duduk, atau overlap kuat)
      if (chair.has_pos && person.has_pos) {
        const double dist_3d = std::hypot(chair.x - person.x_m, chair.y - person.y_m);
        const bool strong_match =
          (chair.label == "sofa") || person.is_sitting_posture ||
          ((overlap_chair > 0.20 || overlap_person > 0.20) &&
           (hip_inside || kpt_inside || chair_center_inside));
        const double max_dist_3d = strong_match ? 2.40 : 1.75;
        if (dist_3d > max_dist_3d) return 0.0;
      }

      // Syarat kelayakan:
      // Overlap 2D (> 4% dari kursi atau orang) ATAU keselarasan horizontal dengan posisi tubuh/postur duduk
      const bool qualified = (overlap_chair > 0.04 || overlap_person > 0.04) ||
                             (x_ratio > 0.20 &&
                              (hip_inside || center_inside || chair_center_inside ||
                               kpt_inside || person.is_sitting_posture));

      if (!qualified) return 0.0;

      // Skor kecocokan: semakin tinggi overlap, keselarasan horizontal, dan postur duduk, semakin besar skornya
      double score = overlap_chair * 2.0 + overlap_person + x_ratio;
      if (hip_inside || center_inside) score += 1.0;
      if (kpt_inside) score += 0.5;
      if (person.is_sitting_posture) score += 1.5;
      if (chair.has_pos && person.has_pos) {
        const double dist_3d = std::hypot(chair.x - person.x_m, chair.y - person.y_m);
        score += std::max(0.0, 2.0 - dist_3d);
      }
      return score;
    };

  struct MatchPair {
    size_t chair_idx;
    size_t person_idx;
    double score;
  };
  std::vector<MatchPair> match_candidates;
  for (size_t c = 0; c < chairs.size(); ++c) {
    for (size_t p = 0; p < preview_persons_.size(); ++p) {
      const double score = compute_match_score(chairs[c], preview_persons_[p]);
      if (score > 0.0) {
        match_candidates.push_back({c, p, score});
      }
    }
  }

  // Urutkan pasangan pencocokan dari skor tertinggi ke terendah
  std::sort(match_candidates.begin(), match_candidates.end(),
    [](const MatchPair & a, const MatchPair & b) {
      return a.score > b.score;
    });

  // 1 orang menempati max 1 kursi; sofa dapat menampung > 1 orang duduk
  std::vector<bool> person_assigned(preview_persons_.size(), false);
  for (const auto & match : match_candidates) {
    const bool is_sofa = (chairs[match.chair_idx].label == "sofa");
    if ((!chairs[match.chair_idx].occupied || is_sofa) && !person_assigned[match.person_idx]) {
      chairs[match.chair_idx].occupied = true;
      person_assigned[match.person_idx] = true;
      preview_persons_[match.person_idx].is_sitting = true;
    }
  }

  auto side_of = [&](const cv::Rect & chair_bbox, bool has_pos, double y) {
      if (has_pos) {
        if (y > 0.20) return std::string("left");
        if (y < -0.20) return std::string("right");
        return std::string("center");
      }
      const int cx = chair_bbox.x + chair_bbox.width / 2;
      const int width = bgr.cols > 0 ? bgr.cols : 640;
      if (cx < width / 3) return std::string(mirror_ ? "right" : "left");
      if (cx > 2 * width / 3) return std::string(mirror_ ? "left" : "right");
      return std::string("center");
    };

  preview_chairs_.clear();
  std::string data;
  std::string best_empty_side = "none";
  double best_empty_dist = 1e9;
  double best_empty_x = 0.0, best_empty_y = 0.0, best_empty_z = 0.0;
  double best_empty_depth = 0.0;
  bool best_has_pos = false;

  for (const auto & chair : chairs) {
    const bool ok = chair.has_pos;
    const double dist = ok ? chair.range_m : 0.0;
    const std::string side = side_of(chair.bbox, ok, chair.y);

    preview_chairs_.push_back(
      {chair.label, chair.bbox, ok, chair.x, chair.y, dist, chair.occupied, side});

    if (!chair.occupied) {
      if (ok && dist > 0.1 && dist < best_empty_dist) {
        best_empty_dist = dist;
        best_empty_side = side;
        best_empty_x = chair.x;
        best_empty_y = chair.y;
        best_empty_z = chair.z;
        best_empty_depth = dist;
        best_has_pos = true;
      } else if (best_empty_side == "none") {
        best_empty_side = side;
        if (ok) {
          best_empty_x = chair.x;
          best_empty_y = chair.y;
          best_empty_z = chair.z;
          best_empty_depth = dist;
          best_has_pos = true;
        }
      }
    }

    if (!data.empty()) data += ";";
    data += cv::format(
      "%s:%d,%d,%d,%d|", chair.label.c_str(), chair.bbox.x, chair.bbox.y,
      chair.bbox.width, chair.bbox.height);
    data += ok
      ? cv::format("%.2f,%.2f,%.2f|", chair.x, chair.y, chair.z)
      : "nan,nan,nan|";
    data += chair.occupied ? "occupied|" : "empty|";
    data += side;
  }

  std_msgs::msg::String chair_msg;
  chair_msg.data = data;
  chair_dets_publisher_->publish(chair_msg);

  std_msgs::msg::String seat_msg;
  if (best_empty_side != "none") {
    if (best_has_pos) {
      seat_msg.data = cv::format(
        "%s|%.2f,%.2f,%.2f|%.2f",
        best_empty_side.c_str(), best_empty_x, best_empty_y, best_empty_z, best_empty_depth);
    } else {
      seat_msg.data = cv::format("%s|nan,nan,nan|nan", best_empty_side.c_str());
    }
  } else {
    seat_msg.data = "none";
  }
  empty_seat_publisher_->publish(seat_msg);
}

void ZedNode::showPreview(
  const cv::Mat & bgr,
  const std::unordered_map<int, cv::Rect> * live_bboxes)
{
  if (!debug_window_ || bgr.empty()) return;

  try {
    cv::Mat canvas;
    if (mirror_) cv::flip(bgr, canvas, 1);
    else canvas = bgr.clone();
    const int W = canvas.cols;
    const int H = canvas.rows;
    const bool flip = mirror_;

    auto flipRect = [&](const cv::Rect & in) {
        cv::Rect r = in;
        if (flip) r.x = W - r.x - r.width;
        return r;
      };
    auto banner = [&](const std::string & text, int x, int y,
        const cv::Scalar & border, const cv::Scalar & bg,
        const cv::Scalar & fg, double scale = 0.55, int thickness = 1) {
        int baseline = 0;
        const cv::Size size = cv::getTextSize(
          text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
        const int bx = std::clamp(x, 0, std::max(0, W - size.width - 6));
        const int by = std::clamp(y - size.height - 4, 0, std::max(0, H - size.height - 6));
        cv::Rect bg_rect(bx, by, size.width + 6, size.height + 6);
        bg_rect &= cv::Rect(0, 0, W, H);
        if (bg_rect.width > 0 && bg_rect.height > 0) {
          cv::rectangle(canvas, bg_rect, bg, -1);
          cv::rectangle(canvas, bg_rect, border, 1, cv::LINE_AA);
        }
        cv::putText(canvas, text, cv::Point(bx + 3, by + size.height + 1),
          cv::FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv::LINE_AA);
      };

    std::vector<PrevPerson> snap_persons;
    std::vector<PrevBag> snap_bags;
    std::vector<PrevItem> snap_drinks;
    std::vector<PrevChair> snap_chairs;
    {
      std::lock_guard<std::mutex> p_lk(preview_mutex_);
      snap_persons = disp_persons_;
      snap_bags = disp_bags_;
      snap_drinks = disp_drinks_;
      snap_chairs = disp_chairs_;
    }

    if (live_bboxes != nullptr && !live_bboxes->empty()) {
      for (auto & p : snap_persons) {
        auto it = live_bboxes->find(p.id);
        if (it != live_bboxes->end() && it->second.width > 0 && it->second.height > 0) {
          const int dx = (it->second.x + it->second.width / 2) - (p.bbox.x + p.bbox.width / 2);
          const int dy = (it->second.y + it->second.height / 2) - (p.bbox.y + p.bbox.height / 2);
          auto shiftRect = [&](cv::Rect & rc) {
            if (rc.width > 0 && rc.height > 0) {
              rc.x += dx;
              rc.y += dy;
            }
          };
          shiftRect(p.shirt_bbox);
          shiftRect(p.pants_bbox);
          shiftRect(p.hat_bbox);
          shiftRect(p.glasses_bbox);
          p.bbox = it->second;
        }
      }
    }

    for (size_t i = 0; i < std::min<size_t>(4, snap_persons.size()); ++i) {
      const auto & p = snap_persons[i];
      if (p.has_pos && p.range_m > (p.is_sitting ? person_seated_max_dist_ : person_max_dist_)) continue;
      if (!p.has_pos && (p.bbox.height < 150 || p.bbox.area() < 8000)) continue;
      cv::Rect r = flipRect(p.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width <= 0 || r.height <= 0) continue;

      const cv::Scalar color = p.is_sitting ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
      cv::rectangle(canvas, r, color, 2);
      const int cx = r.x + r.width / 2;
      const int cy = r.y + r.height / 2;
      cv::circle(canvas, cv::Point(cx, cy), 5, cv::Scalar(0, 255, 0), -1);
      cv::line(canvas, cv::Point(cx, cy), cv::Point(W / 2, H), cv::Scalar(0, 69, 255), 3);

      // Bbox warna biru langsung di area baju untuk deteksi warna pakaian (Hanya di Receptionist)
      if (is_receptionist_ && p.shirt_bbox.width > 0 && p.shirt_bbox.height > 0) {
        cv::Rect shirt_r = flipRect(p.shirt_bbox);
        shirt_r &= cv::Rect(0, 0, W, H);
        if (shirt_r.width > 0 && shirt_r.height > 0) {
          cv::rectangle(canvas, shirt_r, cv::Scalar(255, 0, 0), 2);
          if (!p.clothing.empty()) {
            banner(p.clothing, shirt_r.x, shirt_r.y, cv::Scalar(255, 0, 0),
              cv::Scalar(180, 0, 0), cv::Scalar(255, 255, 255), 0.48, 1);
          }
        }
      }

      // Bbox warna magenta di area celana untuk deteksi warna celana (Hanya di Receptionist)
      if (is_receptionist_ && p.pants_bbox.width > 0 && p.pants_bbox.height > 0) {
        cv::Rect pants_r = flipRect(p.pants_bbox);
        pants_r &= cv::Rect(0, 0, W, H);
        if (pants_r.width > 0 && pants_r.height > 0) {
          cv::rectangle(canvas, pants_r, cv::Scalar(255, 0, 255), 2);
          if (!p.pants.empty()) {
            banner(p.pants, pants_r.x, pants_r.y, cv::Scalar(255, 0, 255),
              cv::Scalar(160, 0, 160), cv::Scalar(255, 255, 255), 0.48, 1);
          }
        }
      }

      // Bbox aksesoris kepala: topi & kacamata (Hanya di Receptionist)
      if (is_receptionist_ && p.has_hat && p.hat_bbox.width > 0 && p.hat_bbox.height > 0) {
        cv::Rect hat_r = flipRect(p.hat_bbox) & cv::Rect(0, 0, W, H);
        if (hat_r.width > 0 && hat_r.height > 0) {
          cv::rectangle(canvas, hat_r, cv::Scalar(0, 200, 255), 2);
          banner("hat", hat_r.x, hat_r.y, cv::Scalar(0, 200, 255),
            cv::Scalar(0, 140, 180), cv::Scalar(255, 255, 255), 0.45, 1);
        }
      }
      if (is_receptionist_ && p.has_glasses && p.glasses_bbox.width > 0 && p.glasses_bbox.height > 0) {
        cv::Rect gl_r = flipRect(p.glasses_bbox) & cv::Rect(0, 0, W, H);
        if (gl_r.width > 0 && gl_r.height > 0) {
          cv::rectangle(canvas, gl_r, cv::Scalar(255, 255, 0), 2);
          banner("glasses", gl_r.x, gl_r.y, cv::Scalar(255, 255, 0),
            cv::Scalar(160, 160, 0), cv::Scalar(0, 0, 0), 0.45, 1);
        }
      }

      // Tampilkan banner nama tamu / registered name / Guest
      if (is_receptionist_) {
        std::string label = !p.name.empty() ? p.name : "Guest";
        if (!p.gender.empty() && p.gender != "unknown") label += " (" + p.gender + ")";
        if (p.is_sitting) label += " [Seated]";
        if (p.is_facing_back) label += " [Back]";
        banner(label, r.x, r.y, color,
          p.is_sitting ? cv::Scalar(0, 0, 180) : cv::Scalar(0, 140, 0),
          cv::Scalar(255, 255, 255), 0.58, 2);
      } else if (is_restaurant_) {
        std::string label = "Customer";
        if (p.is_facing_back) label += " [Back]";
        banner(label, r.x, r.y, color, cv::Scalar(180, 100, 0), cv::Scalar(255, 255, 255), 0.55, 2);
      } else {
        std::string label = "Operator";
        if (p.is_facing_back) label += " [Back]";
        banner(label, r.x, r.y, color, cv::Scalar(0, 140, 0), cv::Scalar(255, 255, 255), 0.55, 2);
      }
    }

    if (enable_bag_ || is_receptionist_) {
      // Gambar garis crosshair horizontal & vertikal sinkron sebagai target center
      const int target_cx = W / 2;
      const int target_cy = H / 2;
      cv::line(canvas, cv::Point(target_cx, 0), cv::Point(target_cx, H), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
      cv::line(canvas, cv::Point(0, target_cy), cv::Point(W, target_cy), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
      cv::circle(canvas, cv::Point(target_cx, target_cy), 8, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    if (enable_bag_) {
      const int target_cx = W / 2;
      const int target_cy = H / 2;

      for (size_t i = 0; i < std::min<size_t>(2, snap_bags.size()); ++i) {
        const auto & b = snap_bags[i];
        if (b.has_pos && b.range_m > bag_max_dist_) continue;
        if (!b.has_pos && (b.bbox.height < 45 || b.bbox.area() < 3000)) continue;
        cv::Rect r = flipRect(b.bbox);
        r &= cv::Rect(0, 0, W, H);
        if (r.width <= 0 || r.height <= 0) continue;
        cv::rectangle(canvas, r, cv::Scalar(0, 69, 255), 2);
        const int cx = r.x + r.width / 2;
        const int cy = r.y + r.height / 2;
        const bool is_x_aligned = (std::abs(cx - target_cx) <= 3);
        const cv::Scalar align_color = is_x_aligned ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
        cv::circle(canvas, cv::Point(cx, cy), 6, align_color, -1);
        cv::line(canvas, cv::Point(cx, cy), cv::Point(target_cx, target_cy), align_color, 2, cv::LINE_AA);
      }
    }

    for (size_t i = 0; i < std::min<size_t>(6, snap_drinks.size()); ++i) {
      const auto & d = snap_drinks[i];
      if (d.has_pos && d.range_m > drink_max_dist_) continue;
      if (!d.has_pos && (d.bbox.height < 40 || d.bbox.area() < 1200)) continue;
      cv::Rect r = flipRect(d.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width <= 0 || r.height <= 0) continue;
      cv::rectangle(canvas, r, cv::Scalar(255, 255, 0), 2);
      const int cx = r.x + r.width / 2;
      const int cy = r.y + r.height / 2;
      cv::circle(canvas, cv::Point(cx, cy), 4, cv::Scalar(255, 255, 0), -1);
      cv::line(canvas, cv::Point(cx, cy), cv::Point(W / 2, H), cv::Scalar(0, 69, 255), 2);

      // Banner label teks singkat di atas bounding box khusus untuk model target:
      // pringles, cracker, soda, sugar, milk, water
      std::string norm_label = d.label;
      if (norm_label == "cracker_box" || norm_label == "crackerbox") norm_label = "cracker";
      if (norm_label == "sugar_box" || norm_label == "sugarbox") norm_label = "sugar";
      if (norm_label == "bottle" || norm_label == "botol" ||
          norm_label == "mineral_water" || norm_label == "mineral water")
      {
        norm_label = "water";
      }

      if (norm_label == "soda" || norm_label == "pringles" || norm_label == "cracker" ||
          norm_label == "sugar" || norm_label == "milk" || norm_label == "water")
      {
        std::string banner_text = norm_label;
        if (d.has_pos) {
          banner_text += cv::format(" (%.2fm)", d.range_m);
        }
        banner(banner_text, r.x, r.y, cv::Scalar(255, 255, 0),
          cv::Scalar(180, 180, 0), cv::Scalar(0, 0, 0), 0.45, 1);
      }
    }

    for (size_t i = 0; i < std::min<size_t>(4, snap_chairs.size()); ++i) {
      const auto & c = snap_chairs[i];
      if (c.has_pos && c.range_m > chair_max_dist_) continue;
      if (!c.has_pos && (c.bbox.height < 60 || c.bbox.area() < 5000)) continue;
      cv::Rect r = flipRect(c.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width <= 0 || r.height <= 0) continue;
      const cv::Scalar color = c.occupied ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
      cv::rectangle(canvas, r, color, 2);
      const int cx = r.x + r.width / 2;
      const int cy = r.y + r.height / 2;
      if (!c.occupied) {
        const int target_cx = W / 2;
        const int target_cy = H / 2;
        const int chair_align_tol_px = (W >= 1000) ? 10 : 5;
        const bool is_x_aligned = (std::abs(cx - target_cx) <= chair_align_tol_px);
        const cv::Scalar align_color = is_x_aligned ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
        cv::line(canvas, cv::Point(cx, r.y), cv::Point(cx, r.y + r.height), align_color, 1, cv::LINE_AA);
        cv::circle(canvas, cv::Point(cx, cy), 6, align_color, -1);
        cv::line(canvas, cv::Point(cx, cy), cv::Point(target_cx, target_cy), align_color, 2, cv::LINE_AA);
      } else {
        cv::circle(canvas, cv::Point(cx, cy), 4, color, -1);
        cv::line(canvas, cv::Point(cx, cy), cv::Point(W / 2, H), color, 2);
      }
    }

    int hud_y = 36;
    auto hud_item = [&](const std::string & title, double x, double y, double dist, bool valid,
        const cv::Scalar & color) {
        std::string header = title;
        if (valid && dist > 0.0) {
          header += cv::format(" (%.2fm)", dist);
        } else {
          header += " : -";
        }
        cv::putText(canvas, header, cv::Point(10, hud_y),
          cv::FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv::LINE_AA);

        if (valid) {
          cv::putText(canvas, cv::format("x: %.2fm | y: %.2fm", x, y), cv::Point(10, hud_y + 18),
            cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
        }
        hud_y += valid ? 40 : 25;
      };

    const PrevPerson p0 = (!snap_persons.empty() && snap_persons[0].has_pos)
      ? snap_persons[0] : PrevPerson{};
    std::string person_title;
    if (is_receptionist_) {
      if (!p0.name.empty()) {
        person_title = p0.name + (p0.is_facing_back ? " [Back]" : "");
      } else {
        person_title = std::string("Guest") + (p0.is_facing_back ? " [Back]" : "");
      }
    } else if (is_restaurant_) {
      person_title = std::string("Customer") + (p0.is_facing_back ? " [Back]" : "");
    } else {
      person_title = std::string("Operator") + (p0.is_facing_back ? " [Back]" : "");
    }
    hud_item(person_title, p0.x_m, p0.y_m, p0.range_m, p0.has_pos, cv::Scalar(0, 255, 0));

    if (enable_bag_) {
      // Selalu tampilkan status Bag 1 dan Bag 2 (kedeteksi maupun belum)
      const bool b1_ok = (snap_bags.size() >= 1 && snap_bags[0].has_pos);
      const double b1_x = b1_ok ? snap_bags[0].x_m : 0.0;
      const double b1_y = b1_ok ? snap_bags[0].y_m : 0.0;
      const double b1_d = b1_ok ? snap_bags[0].range_m : 0.0;
      hud_item("Bag 1", b1_x, b1_y, b1_d, b1_ok, cv::Scalar(0, 69, 255));

      const bool b2_ok = (snap_bags.size() >= 2 && snap_bags[1].has_pos);
      const double b2_x = b2_ok ? snap_bags[1].x_m : 0.0;
      const double b2_y = b2_ok ? snap_bags[1].y_m : 0.0;
      const double b2_d = b2_ok ? snap_bags[1].range_m : 0.0;
      hud_item("Bag 2", b2_x, b2_y, b2_d, b2_ok, cv::Scalar(0, 69, 255));
    }


    static bool window_ready = false;
    if (!window_ready) {
      cv::namedWindow("POV", cv::WINDOW_AUTOSIZE);
      window_ready = true;
    }
    cv::imshow("POV", canvas);
    const int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {
      rclcpp::shutdown();
    } else if (key == 'h' || key == 'H') {
      if (!snap_persons.empty() && snap_persons[0].bbox.width > 20 && snap_persons[0].bbox.height > 20) {
        cv::Rect r = snap_persons[0].bbox;
        r &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (r.width > 20 && r.height > 20) {
          cv::Mat host_crop = bgr(r).clone();
          std::vector<sl::float2> sl_kpt;
          sl_kpt.reserve(snap_persons[0].kpt.size());
          for (const auto & pt : snap_persons[0].kpt) {
            sl_kpt.emplace_back(pt.x, pt.y);
          }
          cv::Rect face_roi = analyzer_.faceCropRegion(r, sl_kpt, bgr.cols, bgr.rows, bgr);
          if (face_roi.width > 20 && face_roi.height > 20) {
            host_crop = bgr(face_roi).clone();
          }
          std::vector<std::string> host_dirs = {
            "src/athome_vision/models",
            "/home/dell/Documents/athome_ws/src/athome_vision/models",
            "models"
          };
          for (const auto & dir : host_dirs) {
            try {
              std::filesystem::create_directories(dir);
              cv::imwrite(dir + "/host.jpg", host_crop);
            } catch (...) {}
          }
          RCLCPP_INFO(get_logger(),
            "\033[1;32m[HOST PHOTO]\033[0m Host photo saved to models/host.jpg via 'h' key (%dx%d)",
            host_crop.cols, host_crop.rows);
          loadPreRegisteredHost();
        }
      }
    }
  } catch (const std::exception &) {
    try {cv::destroyAllWindows();} catch (...) {}
    debug_window_ = false;
  }
}

void ZedNode::loop()
{
  try {
    if (zed_.grab() != sl::ERROR_CODE::SUCCESS) return;
    publishYaw();

    if (zed_.retrieveImage(zed_image_, sl::VIEW::LEFT, sl::MEM::CPU) != sl::ERROR_CODE::SUCCESS) {
      return;
    }
    cv::Mat bgra(
      zed_image_.getHeight(), zed_image_.getWidth(), CV_8UC4,
      zed_image_.getPtr<sl::uchar1>(sl::MEM::CPU),
      zed_image_.getStepBytes(sl::MEM::CPU));
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    if (bgr.empty()) return;

    sl::Bodies cap_bodies;
    std::unordered_map<int, cv::Rect> live_bboxes;
    if (enable_body_tracking_) {
      if (zed_.retrieveBodies(cap_bodies, body_runtime_parameters_) != sl::ERROR_CODE::SUCCESS) {
        return;
      }
      for (const auto & body : cap_bodies.body_list) {
        if (body.keypoint_2d.size() < 34) continue;
        cv::Rect live_r;
        if (zed_vision::computeBbox(body.keypoint_2d, bgr.cols, bgr.rows, live_r)) {
          live_bboxes[body.id] = live_r;
        }
      }
    }

    // Hanya download point cloud GPU->CPU saat inference worker siap menerima frame baru,
    // agar loop kamera + showPreview tetap mulus 30 FPS tanpa patah-patah.
    if (!worker_busy_.load()) {
      if (zed_.retrieveMeasure(zed_point_cloud_, sl::MEASURE::XYZ, sl::MEM::CPU) ==
          sl::ERROR_CODE::SUCCESS)
      {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        bgr.copyTo(worker_bgr_);
        zed_point_cloud_.copyTo(worker_cloud_, sl::COPY_TYPE::CPU_CPU);
        worker_bodies_ = std::move(cap_bodies);
        has_new_frame_ = true;
        worker_cv_.notify_one();
      }
    }

    // Tampilkan preview langsung di 30 FPS dengan sinkronisasi bbox live frame ini
    showPreview(bgr, &live_bboxes);
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
      "Vision loop exception: %s", e.what());
  }
}

void ZedNode::inferenceWorkerLoop()
{
  cv::Mat bgr;
  sl::Mat cloud;
  while (worker_running_.load() && rclcpp::ok()) {
    {
      std::unique_lock<std::mutex> lk(frame_mutex_);
      worker_cv_.wait(lk, [this]() {
        return has_new_frame_ || !worker_running_.load() || !rclcpp::ok();
      });
      if (!worker_running_.load() || !rclcpp::ok()) break;
      if (!has_new_frame_) continue;
      worker_busy_ = true;
      worker_bgr_.copyTo(bgr);
      worker_cloud_.copyTo(cloud, sl::COPY_TYPE::CPU_CPU);
      bodies_ = std::move(worker_bodies_);
      has_new_frame_ = false;
    }

    if (bgr.empty()) {
      worker_busy_ = false;
      continue;
    }

    try {
      latest_operator_found_ = false;
      latest_back_found_ = false;
      latest_gesture_.clear();
      latest_guest_found_ = false;
      latest_customer_found_ = false;
      latest_wave_found_ = false;

      processBags(bgr, cloud);
      processDrinks(bgr, cloud);
      {
        std::lock_guard<std::mutex> state_lk(state_mutex_);
        processPersons(bgr, cloud);
      }
      processChairs(bgr, cloud);
      printSummaryLog();

      {
        std::lock_guard<std::mutex> p_lk(preview_mutex_);
        disp_persons_ = preview_persons_;
        disp_bags_ = preview_bags_;
        disp_drinks_ = preview_drinks_;
        disp_chairs_ = preview_chairs_;
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
        "Inference worker exception: %s", e.what());
    }
    worker_busy_ = false;
    // Beri jeda singkat agar ZED SDK grab() di GPU RTX 2060 tidak berebut resource
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
  }
}

void ZedNode::printSummaryLog()
{
  rclcpp::Time now_time = now();
  if ((now_time - last_summary_log_time_).seconds() < 1.2) {
    return;
  }

  auto find_drink = [&](const std::string & l1, const std::string & l2 = "", const std::string & l3 = "") -> const PrevItem* {
    const PrevItem * no_depth_match = nullptr;
    for (const auto & d : preview_drinks_) {
      if (d.label == l1 || (!l2.empty() && d.label == l2) || (!l3.empty() && d.label == l3)) {
        if (d.has_pos) return &d;
        if (!no_depth_match) no_depth_match = &d;
      }
    }
    return no_depth_match;
  };

  std::vector<std::string> lines;

  if (is_help_me_carry_ && !is_receptionist_ && !is_restaurant_) {
    // 1. Operator
    if (latest_operator_found_) {
      lines.push_back(cv::format("  Operator: %.2f, %.2f (%.2fm)",
        latest_op_x_, latest_op_y_, latest_op_dist_));
    } else {
      lines.push_back("  Operator: 0.00, 0.00 (0.00m)");
    }

    // 2. Bag
    const PrevBag* found_bag = nullptr;
    for (const auto & b : preview_bags_) {
      if (b.has_pos) {
        found_bag = &b;
        break;
      }
    }
    if (found_bag) {
      lines.push_back(cv::format("  Bag: %.2f, %.2f (%.2fm)",
        found_bag->x_m, found_bag->y_m, found_bag->range_m));
    } else {
      lines.push_back("  Bag: 0.00, 0.00 (0.00m)");
    }

    // 3. Back
    if (latest_back_found_) {
      lines.push_back(cv::format("  Back: %.2f, %.2f (%.2fm)",
        latest_back_x_, latest_back_y_, latest_back_dist_));
    } else {
      lines.push_back("  Back: 0.00, 0.00 (0.00m)");
    }

    // 4. Gesture (selalu tampil)
    if (!latest_gesture_.empty() && latest_gesture_ != "NONE") {
      lines.push_back(cv::format("  Gesture: %s", latest_gesture_.c_str()));
    } else {
      lines.push_back("  Gesture: NONE");
    }

    std::string out = "\033[1;32m[HELP_ME_CARRY]\033[0m";
    for (const auto & l : lines) {
      out += "\n" + l;
    }
    RCLCPP_INFO(get_logger(), "%s", out.c_str());
    last_summary_log_time_ = now_time;

  } else if (is_receptionist_ && !is_help_me_carry_ && !is_restaurant_) {
    // 1. Guest / Person
    if (latest_guest_found_) {
      lines.push_back(cv::format("  %s: %.2f, %.2f (%.2fm)",
        latest_guest_tag_.c_str(), latest_guest_x_, latest_guest_y_, latest_guest_dist_));
    } else {
      lines.push_back("  Guest: 0.00, 0.00 (0.00m)");
    }

    auto log_drink = [&](const char * name, const PrevItem * item) {
      if (item && item->has_pos) {
        lines.push_back(cv::format("  %s: %.2f, %.2f (%.2fm)",
          name, item->x_m, item->y_m, item->range_m));
      } else if (item) {
        lines.push_back(cv::format("  %s: detected [2D only, score=%.2f]", name, item->score));
      } else {
        lines.push_back(cv::format("  %s: 0.00, 0.00 (0.00m)", name));
      }
    };
    log_drink("soda", find_drink("soda"));
    log_drink("milk", find_drink("milk"));
    log_drink("water", find_drink("water", "bottle", "mineral water"));

    // 5. Chair
    const PrevChair * best_chair = nullptr;
    for (const auto & c : preview_chairs_) {
      if (!c.occupied && c.has_pos) {
        best_chair = &c;
        break;
      }
    }
    if (!best_chair && !preview_chairs_.empty() && preview_chairs_[0].has_pos) {
      best_chair = &preview_chairs_[0];
    }

    if (best_chair && best_chair->has_pos) {
      lines.push_back(cv::format("  Chair: %.2f, %.2f (%.2fm)",
        best_chair->x_m, best_chair->y_m, best_chair->range_m));
    } else {
      lines.push_back("  Chair: 0.00, 0.00 (0.00m)");
    }

    std::string out = "\033[1;36m[RECEPTIONIST]\033[0m";
    for (const auto & l : lines) {
      out += "\n" + l;
    }
    RCLCPP_INFO(get_logger(), "%s", out.c_str());
    last_summary_log_time_ = now_time;

  } else if (is_restaurant_ && !is_help_me_carry_ && !is_receptionist_) {
    // 1. Customer
    if (latest_customer_found_) {
      lines.push_back(cv::format("  Customer: %.2f, %.2f (%.2fm)",
        latest_cust_x_, latest_cust_y_, latest_cust_dist_));
    } else {
      lines.push_back("  Customer: 0.00, 0.00 (0.00m)");
    }

    // 2. soda
    const auto * soda = find_drink("soda");
    if (soda) {
      lines.push_back(cv::format("  soda: %.2f, %.2f (%.2fm)",
        soda->x_m, soda->y_m, soda->range_m));
    } else {
      lines.push_back("  soda: 0.00, 0.00 (0.00m)");
    }

    // 3. pringles
    const auto * pringles = find_drink("pringles");
    if (pringles) {
      lines.push_back(cv::format("  pringles: %.2f, %.2f (%.2fm)",
        pringles->x_m, pringles->y_m, pringles->range_m));
    } else {
      lines.push_back("  pringles: 0.00, 0.00 (0.00m)");
    }

    // 4. sugar
    const auto * sugar = find_drink("sugar", "sugarbox", "sugar_box");
    if (sugar) {
      lines.push_back(cv::format("  sugar: %.2f, %.2f (%.2fm)",
        sugar->x_m, sugar->y_m, sugar->range_m));
    } else {
      lines.push_back("  sugar: 0.00, 0.00 (0.00m)");
    }

    // 5. cracker
    const auto * cracker = find_drink("cracker");
    if (cracker) {
      lines.push_back(cv::format("  cracker: %.2f, %.2f (%.2fm)",
        cracker->x_m, cracker->y_m, cracker->range_m));
    } else {
      lines.push_back("  cracker: 0.00, 0.00 (0.00m)");
    }

    // 6. milk
    const auto * milk = find_drink("milk");
    if (milk) {
      lines.push_back(cv::format("  milk: %.2f, %.2f (%.2fm)",
        milk->x_m, milk->y_m, milk->range_m));
    } else {
      lines.push_back("  milk: 0.00, 0.00 (0.00m)");
    }

    // 7. Wave (selalu tampil)
    if (latest_wave_found_) {
      lines.push_back(cv::format("  Wave: %.2f, %.2f",
        latest_wave_x_, latest_wave_y_));
    } else {
      lines.push_back("  Wave: 0.00, 0.00");
    }

    std::string out = "\033[1;34m[RESTAURANT]\033[0m";
    for (const auto & l : lines) {
      out += "\n" + l;
    }
    RCLCPP_INFO(get_logger(), "%s", out.c_str());
    last_summary_log_time_ = now_time;

  } else {
    // Mode "all" / general preview: cetak apa pun yang terdeteksi
    if (latest_operator_found_) {
      lines.push_back(cv::format("  Operator: %.2f, %.2f (%.2fm)",
        latest_op_x_, latest_op_y_, latest_op_dist_));
    }
    for (const auto & b : preview_bags_) {
      if (b.has_pos) {
        lines.push_back(cv::format("  Bag: %.2f, %.2f (%.2fm)", b.x_m, b.y_m, b.range_m));
      }
    }
    for (const auto & d : preview_drinks_) {
      if (d.has_pos) {
        lines.push_back(cv::format("  %s: %.2f, %.2f (%.2fm)", d.label.c_str(), d.x_m, d.y_m, d.range_m));
      }
    }
    if (!lines.empty()) {
      std::string out = "\033[1;35m[VISION]\033[0m";
      for (const auto & l : lines) {
        out += "\n" + l;
      }
      RCLCPP_INFO(get_logger(), "%s", out.c_str());
      last_summary_log_time_ = now_time;
    }
  }
}

int main(int argc, char ** argv)
{
  // Remove long nanosecond timestamp to keep terminal logs concise and on a single line
  setenv("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}", 1);

  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<ZedNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    std::cout << "\033[91m[ZED Fatal Exception]\033[0m " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  } catch (...) {
    std::cout << "\033[91m[ZED Fatal Exception]\033[0m Unknown error occurred!" << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
