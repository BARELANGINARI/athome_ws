#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "person_det/person_detection.hpp"

namespace fs = std::filesystem;

int main(int argc, char ** argv)
{
  int cam_idx = 0;
  if (argc > 1) {
    try {
      cam_idx = std::stoi(argv[1]);
    } catch (...) {}
  }

  std::string models_dir = "models";
  if (argc > 2) {
    models_dir = argv[2];
  } else if (!fs::exists(models_dir) && fs::exists("src/person_det/models")) {
    models_dir = "src/person_det/models";
  }

  std::cout << "========================================================\n"
            << "  Person Detection & Attribute Recognition (C++ Standalone)\n"
            << "  Target Camera Index : " << cam_idx << "\n"
            << "  Models Dir          : " << models_dir << "\n"
            << "========================================================\n";

  person_det::PersonDetector detector;
  if (!detector.init(models_dir)) {
    std::cerr << "[ERROR] Could not load ONNX models from: " << models_dir << std::endl;
    return -1;
  }
  std::cout << "[OK] ONNX models loaded successfully." << std::endl;

  cv::VideoCapture cap(cam_idx, cv::CAP_V4L2);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  if (!cap.isOpened()) {
    std::cerr << "[WARN] Failed to open camera device " << cam_idx << ", scanning fallback devices...\n";
    for (int idx = 0; idx <= 9; ++idx) {
      if (idx == cam_idx) continue;
      cap.open(idx, cv::CAP_V4L2);
      if (cap.isOpened()) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cam_idx = idx;
        std::cout << "[OK] Connected to fallback camera index " << cam_idx << "\n";
        break;
      }
    }
  }

  if (!cap.isOpened()) {
    std::cerr << "[ERROR] No camera device found (checked indices 0..9).\n";
    return -1;
  }

  std::cout << "Starting video stream. Press 'q' or ESC to exit.\n";

  cv::Mat frame;
  while (true) {
    if (!cap.read(frame) || frame.empty()) {
      break;
    }

    // Crop left half if stereo side-by-side (e.g. ZED 2i)
    if (frame.cols >= frame.rows * 2) {
      frame = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows)).clone();
    }

    auto candidates = detector.detect(frame, 0.6f);

    const person_det::PersonCandidate * nearest = nullptr;
    int max_area = -1;
    for (const auto & cand : candidates) {
      if (cand.area > max_area) {
        max_area = cand.area;
        nearest = &cand;
      }
    }

    for (const auto & cand : candidates) {
      bool is_n = (nearest != nullptr && &cand == nearest);
      detector.drawCandidate(frame, cand, is_n);
    }

    cv::imshow("Person Detection (C++)", frame);
    char key = static_cast<char>(cv::waitKey(1));
    if (key == 'q' || key == 27) {
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  std::cout << "Program exited cleanly.\n";
  return 0;
}

