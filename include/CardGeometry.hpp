#pragma once

#include <opencv2/opencv.hpp>

// Geometry and appearance helpers on the detected card rectangles,
// shared by the dataset builder and the inference

// get the short side of a rectangle
float shortSide(const cv::RotatedRect &rect);

// Horizontal card: the long side is closer to horizontal
bool isHorizontal(const cv::RotatedRect &rect);

// Straightens the rotated rectangle into a vertical image of fixed size
cv::Mat warpCard(const cv::Mat &frame, const cv::RotatedRect &rect, cv::Size size = cv::Size(200, 300));

// Small grayscale version of the crop, to compare appearance
cv::Mat thumbnail(const cv::Mat &crop);

// Similarity between two thumbnails, tolerant to 180 degree rotation
double similarity(const cv::Mat &a, const cv::Mat &b);
