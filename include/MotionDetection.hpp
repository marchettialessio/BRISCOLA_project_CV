#ifndef MOTION_DETECTION_HPP
#define MOTION_DETECTION_HPP

#include <opencv2/opencv.hpp>

// Detect motion in the frame using background subtraction and display the motion mask
// Returns the side of the motion in the frame: 1 = North, -1 = South, 0 = no motion
int detectMotion(cv::Ptr<cv::BackgroundSubtractorMOG2> &subtractor, cv::Mat &frame, int &north, int &south, bool showWindows = true);

#endif // MOTION_DETECTION_HPP