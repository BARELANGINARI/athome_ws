#include "person_det/person_detection.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace person_det {

struct PersonDetector::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "person_detector"};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::unique_ptr<Ort::Session> pose_session;
  std::string pose_in_name;
  std::string pose_out_name;

  std::unique_ptr<Ort::Session> gender_session;
  std::string gender_in_name;
  std::string gender_out_name;

  std::unique_ptr<Ort::Session> color_session;
  std::string color_in_name;
  std::string color_out_name;
  std::vector<std::string> color_labels;

  std::unique_ptr<Ort::Session> hair_session;
  std::string hair_in_name;
  std::string hair_out_name;
  std::vector<std::string> hair_labels;

  std::map<std::string, std::vector<std::vector<float>>> face_memory;
};

PersonDetector::PersonDetector()
: pimpl_(std::make_unique<Impl>())
{}

PersonDetector::~PersonDetector() = default;

static std::vector<std::string> loadLabels(const std::string & filepath)
{
  std::vector<std::string> labels;
  std::ifstream f(filepath);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  return labels;
}

bool PersonDetector::init(const std::string & models_dir)
{
  ready_ = false;
  fs::path dir(models_dir);

  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::AllocatorWithDefaultOptions alloc;

    // 1. YOLOv8 Pose Model
    fs::path pose_path = dir / "yolov8n-pose.onnx";
    if (fs::exists(pose_path)) {
      pimpl_->pose_session = std::make_unique<Ort::Session>(pimpl_->env, pose_path.c_str(), opts);
      pimpl_->pose_in_name = pimpl_->pose_session->GetInputNameAllocated(0, alloc).get();
      pimpl_->pose_out_name = pimpl_->pose_session->GetOutputNameAllocated(0, alloc).get();
    } else {
      std::cerr << "[PersonDetector] Warning: pose model not found at " << pose_path << std::endl;
      return false;
    }

    // 2. Gender Classification Model
    fs::path gender_path = dir / "gender_cls.onnx";
    if (fs::exists(gender_path)) {
      pimpl_->gender_session = std::make_unique<Ort::Session>(pimpl_->env, gender_path.c_str(), opts);
      pimpl_->gender_in_name = pimpl_->gender_session->GetInputNameAllocated(0, alloc).get();
      pimpl_->gender_out_name = pimpl_->gender_session->GetOutputNameAllocated(0, alloc).get();
    }

    // 3. Clothing Color Model
    fs::path color_path = dir / "color_model.onnx";
    fs::path color_labels_path = dir / "color_names.txt";
    if (fs::exists(color_path)) {
      pimpl_->color_session = std::make_unique<Ort::Session>(pimpl_->env, color_path.c_str(), opts);
      pimpl_->color_in_name = pimpl_->color_session->GetInputNameAllocated(0, alloc).get();
      pimpl_->color_out_name = pimpl_->color_session->GetOutputNameAllocated(0, alloc).get();
      if (fs::exists(color_labels_path)) {
        pimpl_->color_labels = loadLabels(color_labels_path.string());
      }
    }

    // 4. Hair Classification Model
    fs::path hair_path = dir / "hair_model.onnx";
    fs::path hair_labels_path = dir / "hair_names.txt";
    if (fs::exists(hair_path)) {
      pimpl_->hair_session = std::make_unique<Ort::Session>(pimpl_->env, hair_path.c_str(), opts);
      pimpl_->hair_in_name = pimpl_->hair_session->GetInputNameAllocated(0, alloc).get();
      pimpl_->hair_out_name = pimpl_->hair_session->GetOutputNameAllocated(0, alloc).get();
      if (fs::exists(hair_labels_path)) {
        pimpl_->hair_labels = loadLabels(hair_labels_path.string());
      }
    }

    // 5. Face memory JSON
    fs::path mem_path = dir / "face_memory.json";
    if (fs::exists(mem_path)) {
      loadFaceMemory(mem_path.string());
    }

    ready_ = true;
    return true;
  } catch (const std::exception & e) {
    std::cerr << "[PersonDetector] Error during init: " << e.what() << std::endl;
    return false;
  }
}

std::string PersonDetector::classifyOrientation(
  const std::vector<Keypoint> & kps, const cv::Rect & bbox) const
{
  if (kps.size() < 7) {
    return "FRONT";
  }

  const float width = static_cast<float>(std::max(bbox.width, 1));
  const bool nose_ok = kps[NOSE].conf > 0.6f;

  int eyes_count = 0;
  if (kps[L_EYE].conf > 0.5f) eyes_count++;
  if (kps[R_EYE].conf > 0.5f) eyes_count++;

  const bool both_shoulders = (kps[L_SHOULDER].conf > 0.5f && kps[R_SHOULDER].conf > 0.5f);

  bool narrow = true;
  if (both_shoulders) {
    float shoulder_width = std::abs(kps[L_SHOULDER].x - kps[R_SHOULDER].x);
    narrow = (shoulder_width < 0.25f * width);
  }

  if (narrow) {
    if (nose_ok && both_shoulders) {
      float mid_x = (kps[L_SHOULDER].x + kps[R_SHOULDER].x) / 2.0f;
      return (kps[NOSE].x > mid_x) ? "RIGHT SIDE" : "LEFT SIDE";
    }
    if (kps[R_EAR].conf > 0.5f && kps[L_EAR].conf <= 0.5f) {
      return "LEFT SIDE";
    }
    if (kps[L_EAR].conf > 0.5f && kps[R_EAR].conf <= 0.5f) {
      return "RIGHT SIDE";
    }
    return "SIDE";
  }

  if ((nose_ok && eyes_count >= 1) || kps[NOSE].conf > 0.75f) {
    return "FRONT";
  }

  return "BACK";
}

std::string PersonDetector::classifyGender(const cv::Mat & crop)
{
  if (!pimpl_->gender_session || crop.empty() || crop.cols <= 0 || crop.rows <= 0) {
    return "unknown";
  }

  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(224, 224));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  cv::Mat blob = cv::dnn::blobFromImage(
    resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);

  std::array<int64_t, 4> input_shape{1, 3, 224, 224};
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
    pimpl_->mem_info, blob.ptr<float>(), blob.total(),
    input_shape.data(), input_shape.size());

  const char * in_names[] = {pimpl_->gender_in_name.c_str()};
  const char * out_names[] = {pimpl_->gender_out_name.c_str()};

  std::vector<Ort::Value> outputs;
  try {
    outputs = pimpl_->gender_session->Run(
      Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
  } catch (...) {
    return "unknown";
  }

  if (outputs.empty()) return "unknown";

  const float * logits = outputs.front().GetTensorData<float>();
  float l0 = logits[0];
  float l1 = logits[1];

  float m = std::max(l0, l1);
  float exp0 = std::exp(l0 - m);
  float exp1 = std::exp(l1 - m);
  float p0 = exp0 / (exp0 + exp1);
  float p1 = exp1 / (exp0 + exp1);

  if (p0 > 0.7f) return "female";
  if (p1 > 0.7f) return "male";

  return "unknown";
}

std::string PersonDetector::classifyClothing(const cv::Mat & torso_crop)
{
  if (!pimpl_->color_session || torso_crop.empty() || torso_crop.cols <= 0 || torso_crop.rows <= 0) {
    return "unknown";
  }

  cv::Mat resized;
  cv::resize(torso_crop, resized, cv::Size(224, 224));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  cv::Mat blob = cv::dnn::blobFromImage(
    resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);

  std::array<int64_t, 4> input_shape{1, 3, 224, 224};
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
    pimpl_->mem_info, blob.ptr<float>(), blob.total(),
    input_shape.data(), input_shape.size());

  const char * in_names[] = {pimpl_->color_in_name.c_str()};
  const char * out_names[] = {pimpl_->color_out_name.c_str()};

  std::vector<Ort::Value> outputs;
  try {
    outputs = pimpl_->color_session->Run(
      Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
  } catch (...) {
    return "unknown";
  }

  if (outputs.empty()) return "unknown";

  const auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
  int num_classes = (shape.size() >= 2) ? static_cast<int>(shape[1]) : 46;
  const float * logits = outputs.front().GetTensorData<float>();

  float max_l = -1e9f;
  for (int i = 0; i < num_classes; ++i) {
    if (logits[i] > max_l) max_l = logits[i];
  }

  float sum_exp = 0.0f;
  std::vector<float> probs(num_classes);
  for (int i = 0; i < num_classes; ++i) {
    probs[i] = std::exp(logits[i] - max_l);
    sum_exp += probs[i];
  }

  int best_idx = 0;
  float best_p = 0.0f;
  for (int i = 0; i < num_classes; ++i) {
    probs[i] /= sum_exp;
    if (probs[i] > best_p) {
      best_p = probs[i];
      best_idx = i;
    }
  }

  if (best_p > 0.35f && best_idx < static_cast<int>(pimpl_->color_labels.size())) {
    std::string res = pimpl_->color_labels[best_idx];
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
  }

  return "unknown";
}

std::string PersonDetector::classifyHair(const cv::Mat & head_crop)
{
  if (!pimpl_->hair_session || head_crop.empty() || head_crop.cols <= 0 || head_crop.rows <= 0) {
    return "unknown";
  }

  cv::Mat resized;
  cv::resize(head_crop, resized, cv::Size(224, 224));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  cv::Mat blob = cv::dnn::blobFromImage(
    resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);

  std::array<int64_t, 4> input_shape{1, 3, 224, 224};
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
    pimpl_->mem_info, blob.ptr<float>(), blob.total(),
    input_shape.data(), input_shape.size());

  const char * in_names[] = {pimpl_->hair_in_name.c_str()};
  const char * out_names[] = {pimpl_->hair_out_name.c_str()};

  std::vector<Ort::Value> outputs;
  try {
    outputs = pimpl_->hair_session->Run(
      Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
  } catch (...) {
    return "unknown";
  }

  if (outputs.empty()) return "unknown";

  const auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
  int num_classes = (shape.size() >= 2) ? static_cast<int>(shape[1]) : 5;
  const float * logits = outputs.front().GetTensorData<float>();

  float max_l = -1e9f;
  for (int i = 0; i < num_classes; ++i) {
    if (logits[i] > max_l) max_l = logits[i];
  }

  float sum_exp = 0.0f;
  std::vector<float> probs(num_classes);
  for (int i = 0; i < num_classes; ++i) {
    probs[i] = std::exp(logits[i] - max_l);
    sum_exp += probs[i];
  }

  int best_idx = 0;
  float best_p = 0.0f;
  for (int i = 0; i < num_classes; ++i) {
    probs[i] /= sum_exp;
    if (probs[i] > best_p) {
      best_p = probs[i];
      best_idx = i;
    }
  }

  if (best_p > 0.45f && best_idx < static_cast<int>(pimpl_->hair_labels.size())) {
    std::string res = pimpl_->hair_labels[best_idx];
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
  }

  return "unknown";
}

std::vector<float> PersonDetector::extractFaceEmbedding(
  const cv::Mat & frame, const cv::Rect & bbox,
  const std::vector<Keypoint> & kps)
{
  // Extract robust facial & upper torso descriptor (spatial color HSV + keypoints descriptor)
  // Bounded within image
  cv::Rect safe_box = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
  if (safe_box.area() <= 0) return {};

  int head_h = std::max(4, safe_box.height / 3);
  int head_w = std::max(4, safe_box.width / 2);
  int head_x = std::clamp(safe_box.x + safe_box.width / 4, 0, frame.cols - head_w);
  int head_y = std::clamp(safe_box.y, 0, frame.rows - head_h);

  cv::Rect head_roi(head_x, head_y, head_w, head_h);
  head_roi &= cv::Rect(0, 0, frame.cols, frame.rows);
  if (head_roi.area() <= 0) return {};

  cv::Mat crop = frame(head_roi);
  cv::Mat hsv;
  cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

  // 32-bin Hue + 16-bin Saturation histogram descriptor
  int h_bins = 16, s_bins = 8;
  int hist_size[] = {h_bins, s_bins};
  float h_ranges[] = {0, 180};
  float s_ranges[] = {0, 256};
  const float * ranges[] = {h_ranges, s_ranges};
  int channels[] = {0, 1};

  cv::Mat hist;
  cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, hist_size, ranges, true, false);
  cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L2);

  std::vector<float> emb;
  emb.reserve(h_bins * s_bins + 10);
  for (int i = 0; i < hist.rows; ++i) {
    for (int j = 0; j < hist.cols; ++j) {
      emb.push_back(hist.at<float>(i, j));
    }
  }

  // Append keypoint geometric proportions
  if (kps.size() >= 7) {
    float eye_dist = std::abs(kps[L_EYE].x - kps[R_EYE].x) / static_cast<float>(bbox.width + 1);
    float shoulder_dist = std::abs(kps[L_SHOULDER].x - kps[R_SHOULDER].x) / static_cast<float>(bbox.width + 1);
    emb.push_back(eye_dist);
    emb.push_back(shoulder_dist);
  }

  // Normalize full embedding
  float norm = 0.0f;
  for (float v : emb) norm += v * v;
  norm = std::sqrt(norm);
  if (norm > 1e-9f) {
    for (float & v : emb) v /= norm;
  }

  return emb;
}

float PersonDetector::cosineSimilarity(const std::vector<float> & a, const std::vector<float> & b)
{
  if (a.empty() || b.empty()) return 0.0f;
  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
  if (denom < 1e-9f) return 0.0f;
  return dot / denom;
}

std::string PersonDetector::matchName(const std::vector<float> & emb, float threshold) const
{
  if (emb.empty() || pimpl_->face_memory.empty()) return "";

  std::string best_name = "";
  float best_sim = threshold;

  for (const auto & [name, embs] : pimpl_->face_memory) {
    for (const auto & e : embs) {
      float sim = cosineSimilarity(emb, e);
      if (sim > best_sim) {
        best_sim = sim;
        best_name = name;
      }
    }
  }

  return best_name;
}

bool PersonDetector::addFaceEmbedding(const std::string & name, const std::vector<float> & emb)
{
  if (name.empty() || emb.empty()) return false;
  auto & vec = pimpl_->face_memory[name];
  vec.push_back(emb);
  if (vec.size() > 5) {
    vec.erase(vec.begin());  // Keep last 5
  }
  return true;
}

bool PersonDetector::loadFaceMemory(const std::string & json_path)
{
  std::ifstream f(json_path);
  if (!f.is_open()) return false;

  std::stringstream ss;
  ss << f.rdbuf();
  std::string content = ss.str();
  if (content.empty()) return false;

  pimpl_->face_memory.clear();

  // Simple JSON parsing of "name": [[...], [...]]
  size_t pos = 0;
  while ((pos = content.find('"', pos)) != std::string::npos) {
    size_t end_name = content.find('"', pos + 1);
    if (end_name == std::string::npos) break;
    std::string name = content.substr(pos + 1, end_name - pos - 1);
    pos = end_name + 1;

    size_t start_arr = content.find('[', pos);
    if (start_arr == std::string::npos) break;

    size_t end_arr = content.find(']', start_arr);
    // Find closing of outer array if nested
    int depth = 0;
    size_t cur = start_arr;
    while (cur < content.size()) {
      if (content[cur] == '[') depth++;
      else if (content[cur] == ']') {
        depth--;
        if (depth == 0) {
          end_arr = cur;
          break;
        }
      }
      cur++;
    }

    std::string block = content.substr(start_arr, end_arr - start_arr + 1);
    // Parse inner vector of floats
    size_t in_pos = 0;
    while ((in_pos = block.find('[', in_pos + 1)) != std::string::npos) {
      size_t in_end = block.find(']', in_pos);
      if (in_end == std::string::npos) break;
      std::string nums_str = block.substr(in_pos + 1, in_end - in_pos - 1);
      std::stringstream ns(nums_str);
      std::string val;
      std::vector<float> emb;
      while (std::getline(ns, val, ',')) {
        try {
          emb.push_back(std::stof(val));
        } catch (...) {}
      }
      if (!emb.empty()) {
        pimpl_->face_memory[name].push_back(emb);
      }
      in_pos = in_end;
    }

    pos = end_arr + 1;
  }

  std::cout << "[PersonDetector] Loaded face memory for " << pimpl_->face_memory.size() << " people." << std::endl;
  return true;
}

bool PersonDetector::saveFaceMemory(const std::string & json_path) const
{
  std::ofstream f(json_path);
  if (!f.is_open()) return false;

  f << "{\n";
  size_t idx = 0;
  for (const auto & [name, embs] : pimpl_->face_memory) {
    f << "  \"" << name << "\": [\n";
    for (size_t i = 0; i < embs.size(); ++i) {
      f << "    [";
      for (size_t j = 0; j < embs[i].size(); ++j) {
        f << embs[i][j];
        if (j + 1 < embs[i].size()) f << ", ";
      }
      f << "]";
      if (i + 1 < embs.size()) f << ",";
      f << "\n";
    }
    f << "  ]";
    if (++idx < pimpl_->face_memory.size()) f << ",";
    f << "\n";
  }
  f << "}\n";
  return true;
}

std::vector<PersonCandidate> PersonDetector::detect(const cv::Mat & frame, float conf_thresh)
{
  std::vector<PersonCandidate> results;
  if (!ready_ || !pimpl_->pose_session || frame.empty()) {
    return results;
  }

  const int input_size = 320;
  cv::Mat blob = cv::dnn::blobFromImage(
    frame, 1.0 / 255.0, cv::Size(input_size, input_size), cv::Scalar(), true, false);

  std::array<int64_t, 4> input_shape{1, 3, input_size, input_size};
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
    pimpl_->mem_info, blob.ptr<float>(), blob.total(),
    input_shape.data(), input_shape.size());

  const char * in_names[] = {pimpl_->pose_in_name.c_str()};
  const char * out_names[] = {pimpl_->pose_out_name.c_str()};

  std::vector<Ort::Value> outputs;
  try {
    outputs = pimpl_->pose_session->Run(
      Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
  } catch (...) {
    return results;
  }

  if (outputs.empty() || !outputs.front().IsTensor()) {
    return results;
  }

  // Output shape: [1, 56, 2100]
  const auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
  int channels = (shape.size() >= 2) ? static_cast<int>(shape[1]) : 56;
  int num_dets = (shape.size() >= 3) ? static_cast<int>(shape[2]) : 2100;
  const float * data = outputs.front().GetTensorData<float>();

  const float x_factor = static_cast<float>(frame.cols) / static_cast<float>(input_size);
  const float y_factor = static_cast<float>(frame.rows) / static_cast<float>(input_size);

  std::vector<cv::Rect> candidate_boxes;
  std::vector<float> candidate_confs;
  std::vector<std::vector<Keypoint>> candidate_kps;

  for (int i = 0; i < num_dets; ++i) {
    // Channel 4 is person class score
    float score = data[4 * num_dets + i];
    if (score < conf_thresh) continue;

    float cx = data[0 * num_dets + i];
    float cy = data[1 * num_dets + i];
    float w = data[2 * num_dets + i];
    float h = data[3 * num_dets + i];

    int left = std::max(0, std::min(static_cast<int>((cx - 0.5f * w) * x_factor), frame.cols - 1));
    int top = std::max(0, std::min(static_cast<int>((cy - 0.5f * h) * y_factor), frame.rows - 1));
    int width = std::max(1, std::min(static_cast<int>(w * x_factor), frame.cols - left));
    int height = std::max(1, std::min(static_cast<int>(h * y_factor), frame.rows - top));

    // Extract 17 keypoints
    std::vector<Keypoint> kps(17);
    int valid_kps = 0;
    for (int k = 0; k < 17; ++k) {
      float kx = data[(5 + k * 3 + 0) * num_dets + i] * x_factor;
      float ky = data[(5 + k * 3 + 1) * num_dets + i] * y_factor;
      float kc = data[(5 + k * 3 + 2) * num_dets + i];
      kps[k] = {kx, ky, kc};
      if (kc > 0.5f) valid_kps++;
    }

    // Heuristic: at least 5 keypoints above 0.5 confidence
    if (valid_kps < 5) continue;

    candidate_boxes.emplace_back(left, top, width, height);
    candidate_confs.push_back(score);
    candidate_kps.push_back(std::move(kps));
  }

  // Non-Maximum Suppression
  std::vector<int> indices;
  cv::dnn::NMSBoxes(candidate_boxes, candidate_confs, conf_thresh, 0.45f, indices);

  results.reserve(indices.size());
  for (int idx : indices) {
    PersonCandidate cand;
    cand.bbox = candidate_boxes[idx];
    cand.conf = candidate_confs[idx];
    cand.area = cand.bbox.area();
    cand.keypoints = candidate_kps[idx];

    // Orientation
    cand.orientation = classifyOrientation(cand.keypoints, cand.bbox);

    // Crops
    int x1 = cand.bbox.x;
    int y1 = cand.bbox.y;
    int x2 = cand.bbox.x + cand.bbox.width;
    int y2 = cand.bbox.y + cand.bbox.height;
    int h = cand.bbox.height;
    int w = cand.bbox.width;

    // Upper body crop for gender
    int upper_y2 = std::min(frame.rows, static_cast<int>(y1 + h * 0.55));
    if (upper_y2 > y1 && x2 > x1) {
      cv::Mat upper_crop = frame(cv::Rect(x1, y1, x2 - x1, upper_y2 - y1));
      cand.gender = classifyGender(upper_crop);
    }

    // Torso crop for clothing color
    int torso_y1 = std::min(frame.rows, static_cast<int>(y1 + h * 0.2));
    if (y2 > torso_y1 && x2 > x1) {
      cv::Mat torso_crop = frame(cv::Rect(x1, torso_y1, x2 - x1, y2 - torso_y1));
      cand.clothing = classifyClothing(torso_crop);
    }

    // Head crop for hair
    int head_y2 = std::min(frame.rows, static_cast<int>(y1 + h * 0.2));
    int head_x1 = std::max(0, x1 + w / 4);
    int head_x2 = std::min(frame.cols, x1 + 3 * w / 4);
    if (head_y2 > y1 && head_x2 > head_x1) {
      cv::Mat head_crop = frame(cv::Rect(head_x1, y1, head_x2 - head_x1, head_y2 - y1));
      cand.hair = classifyHair(head_crop);
    }

    // Face / Re-ID Embedding
    cand.embedding = extractFaceEmbedding(frame, cand.bbox, cand.keypoints);
    cand.matched_name = matchName(cand.embedding, 0.5f);

    results.push_back(std::move(cand));
  }

  return results;
}

void PersonDetector::drawCandidate(
  cv::Mat & frame, const PersonCandidate & cand, bool is_nearest) const
{
  int x1 = cand.bbox.x;
  int y1 = cand.bbox.y;
  int x2 = cand.bbox.x + cand.bbox.width;
  int y2 = cand.bbox.y + cand.bbox.height;

  // Draw main bounding box
  cv::Scalar box_color = is_nearest ? cv::Scalar(0, 255, 0) : cv::Scalar(200, 200, 0);
  cv::rectangle(frame, cand.bbox, box_color, 2);

  // Info label below bounding box: "FRONT | male | shirt:black | hair:black hair"
  std::string info_text = cand.orientation + " | " + cand.gender + " | shirt:" + cand.clothing + " | hair:" + cand.hair;
  int baseline = 0;
  cv::Size text_size = cv::getTextSize(info_text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
  int ty = y2 + text_size.height + 10;
  if (ty + 6 > frame.rows) {
    ty = y1 + text_size.height + 16;
  }

  cv::rectangle(
    frame,
    cv::Point(x1, ty - text_size.height - 4),
    cv::Point(x1 + text_size.width + 8, ty + 4),
    cv::Scalar(0, 255, 0), -1);

  cv::putText(
    frame, info_text, cv::Point(x1 + 4, ty),
    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

  // Draw nearest person large name banner if matched
  if (is_nearest && !cand.matched_name.empty()) {
    std::string name_text = cand.matched_name;
    cv::Size nsize = cv::getTextSize(name_text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, &baseline);
    int nty = y1 - 10;
    if (nty - nsize.height - 8 < 0) {
      nty = y1 + nsize.height + 8;
    }
    cv::rectangle(
      frame,
      cv::Point(x1, nty - nsize.height - 6),
      cv::Point(x1 + nsize.width + 8, nty + 6),
      cv::Scalar(0, 180, 0), -1);

    cv::putText(
      frame, name_text, cv::Point(x1 + 4, nty),
      cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  }
}

}  // namespace person_det

