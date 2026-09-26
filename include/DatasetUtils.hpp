#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// rank 1-10, suit 0-3 (coins, cups, spades, clubs), classId = suit*10 + (rank - 1)
struct CardLabel
{
    int rank = 0;
    int suit = -1;
    int classId = -1;
    std::string suitName;
};

// Content of a round in the label file
struct RoundLabel
{
    int round = 0;
    CardLabel north, south, briscola;
    bool leaderNorth = true;
};

// i want to parse the label file and return the vector of RoundLabel, one for each round
std::vector<RoundLabel> parseLabels(const std::filesystem::path &file);

// search the directory of labels for the selected game
std::filesystem::path findLabelFile(const std::filesystem::path &labelsDir, int game);

// Straightens the rotated rectangle into a vertical image of fixed size
cv::Mat warpCard(const cv::Mat &frame, const cv::RotatedRect &rect, cv::Size size = cv::Size(200, 300));

// get the short side of a rectangle
float shortSide(const cv::RotatedRect &rect);

// Horizontal card: the long side is closer to horizontal
bool isHorizontal(const cv::RotatedRect &rect);

// Small grayscale version of the crop, to compare appearance
cv::Mat thumbnail(const cv::Mat &crop);

// Similarity between two thumbnails, tolerant to 180 degree rotation
double similarity(const cv::Mat &a, const cv::Mat &b);

// Genera il set dalle scansioni Trentine 
int buildTrentineSet(const std::filesystem::path &scansDir, const std::filesystem::path &outDir,
                     cv::Size size = cv::Size(200, 300));
