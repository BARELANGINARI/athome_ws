#ifndef PERSON_DET_PERSON_DETECTION_HPP_
#define PERSON_DET_PERSON_DETECTION_HPP_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace person_det {

enum KeypointIndex {
  NOSE = 0,
  L_EYE = 1,
  R_EYE = 2,
  L_EAR = 3,
  R_EAR = 4,
  L_SHOULDER = 5,
  R_SHOULDER = 6,
  L_ELBOW = 7,
  R_ELBOW = 8,
  L_WRIST = 9,
  R_WRIST = 10,
  L_HIP = 11,
  R_HIP = 12,
  L_KNEE = 13,
  R_KNEE = 14,
  L_ANKLE = 15,
  R_ANKLE = 16
};

struct Keypoint {
  float x{0.0f};
  float y{0.0f};
  float conf{0.0f};
};

struct PersonCandidate {
  cv::Rect bbox;
  float conf{0.0f};
  int area{0};
  std::vector<Keypoint> keypoints;
  std::string orientation{"FRONT"};
  std::string gender{"unknown"};
  std::string clothing{"unknown"};
  std::string hair{"unknown"};
  std::string matched_name{""};
  std::vector<float> embedding;
};

class PersonDetector {
public:
  PersonDetector();
  ~PersonDetector();

  bool init(const std::string & models_dir);
  bool isReady() const { return ready_; }

  std::vector<PersonCandidate> detect(const cv::Mat & frame, float conf_thresh = 0.6f);

  std::string classifyOrientation(
    const std::vector<Keypoint> & kps, const cv::Rect & bbox) const;

  std::string classifyGender(const cv::Mat & crop);
  std::string classifyClothing(const cv::Mat & torso_crop);
  std::string classifyHair(const cv::Mat & head_crop);

  std::vector<float> extractFaceEmbedding(
    const cv::Mat & frame, const cv::Rect & bbox,
    const std::vector<Keypoint> & kps);

  std::string matchName(const std::vector<float> & emb, float threshold = 0.5f) const;
  bool addFaceEmbedding(const std::string & name, const std::vector<float> & emb);

  bool loadFaceMemory(const std::string & json_path);
  bool saveFaceMemory(const std::string & json_path) const;

  void drawCandidate(
    cv::Mat & frame, const PersonCandidate & cand, bool is_nearest = false) const;

  static float cosineSimilarity(const std::vector<float> & a, const std::vector<float> & b);

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
  bool ready_{false};
};

}  // namespace person_det

#endif  // PERSON_DET_PERSON_DETECTION_HPP_

