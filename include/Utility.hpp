#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <regex>

// Draw the detected rectangles on the frame
void drawRectangles(cv::Mat &frame, const std::vector<cv::RotatedRect> &rectangles);

// Calculate the overlap ratio between two rotated rectangles
double overlapRatio(const cv::RotatedRect &rect1, const cv::RotatedRect &rect2);

// Extract the round number(es. game3round10.mp4 -> 10)
int roundNumber(const std::filesystem::path &video);

// Check if a cli argument is an integer
bool checkInteger(const std::string_view value);

#endif // UTILITY_HPP