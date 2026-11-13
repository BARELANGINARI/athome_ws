#include <iostream>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <ctime>
#include <filesystem>

using namespace std;
using namespace sl;

string Filename(const string & directory)
{
  time_t now = time(0);
  tm * ltm = localtime(&now);

  char filename[100];
  sprintf(filename, "zed_%02d-%02d-%02d.mp4", ltm->tm_hour, ltm->tm_min, ltm->tm_sec);

  return directory + "/" + string(filename);
}

int main(int argc, char ** argv)
{
  string save_path = (argc > 1) ? argv[1] : "videos";

  std::error_code ec;
  std::filesystem::create_directories(save_path, ec);
  if (ec)
  {
    cerr << "Error: Could not create directory " << save_path << ": " << ec.message() << endl;
    return EXIT_FAILURE;
  }

  sl::Camera zed;

  InitParameters init_parameters;
  init_parameters.depth_mode = DEPTH_MODE::NONE;
  init_parameters.camera_resolution = RESOLUTION::VGA;
  init_parameters.camera_fps = 60;

  auto returned_state = zed.open(init_parameters);
  if (returned_state != ERROR_CODE::SUCCESS)
  {
    cout << "Error " << returned_state << ", exit program." << endl;
    return EXIT_FAILURE;
  }

  bool recording = false;
  cv::VideoWriter video;
  double fps = 60.0;

  while (true)
  {
    if (zed.grab() == ERROR_CODE::SUCCESS)
    {
      sl::Mat image;
      zed.retrieveImage(image, VIEW::LEFT);

      cv::Mat cvImage = cv::Mat(
        (int)image.getHeight(), (int)image.getWidth(), CV_8UC4,
        image.getPtr<sl::uchar1>(sl::MEM::CPU)).clone();
      cv::cvtColor(cvImage, cvImage, cv::COLOR_BGRA2BGR);

      cv::imshow("CAMERA", cvImage);

      char c = (char)cv::waitKey(1);

      if (c == 27)
      {
        break;
      }

      if (c == 's' && !recording)
      {
        string path = Filename(save_path);
        std::cout << "Recording started for " << path << "\n";
        video.open(
          path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
          cv::Size(cvImage.cols, cvImage.rows));

        if (!video.isOpened())
        {
          cerr << "Error: Could not open video writer." << endl;
          return EXIT_FAILURE;
        }

        recording = true;
      }

      if (recording)
        video.write(cvImage);

      if (c == 'x')
      {
        std::cout << "Recording finished.\n";
        recording = false;
        video.release();
      }
    }
  }

  zed.close();
  video.release();
  cv::destroyAllWindows();

  return 0;
}
