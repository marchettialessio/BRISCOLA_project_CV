#ifndef CARD_DETECTION_HPP
#define CARD_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include "DatasetBuilder.hpp"

// Filter the detected rectangles based on overlap ratio and keep the larger ones and get the final cards
void filterRectangles(std::vector<cv::RotatedRect> &candidates, std::vector<cv::RotatedRect> &filteredRectangles);

// Check if the contour is a rectangle based on area, aspect ratio, and mean brightness
bool isRectangle(cv::Mat &grayFrame, std::vector<cv::Point> contour, cv::RotatedRect &candidate,
				 bool briscola = false);

// Detect the candidates from the contours and store them in the candidates vector, also update the rectanglesCounter
void detectCandidates(std::vector<std::vector<cv::Point>> contours, std::vector<cv::RotatedRect> &candidates,
					  cv::Mat grayFrame, int &rectanglesCounter, bool briscola = false);

// Find cards in the frame by processing the image, detecting contours, filtering rectangles, and drawing them on the frame
void findCards(cv::Mat &frame, cv::Mat &grayFrame, cv::Mat kernel, DatasetBuilder *builder = nullptr, int motion = 0);

// Find the face-up Briscola card among relaxed rectangle candidates
bool findBriscola(cv::Mat &frame, cv::Mat &grayFrame, cv::Mat kernel, cv::RotatedRect &briscola);

#endif // CARD_DETECTION_HPP