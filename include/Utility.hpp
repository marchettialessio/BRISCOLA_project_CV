#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <optional>
#include <regex>
#include "Labels.hpp"

struct Player {
    std::string position;       // "North" or "South"
    int points;
};

// Draw the detected rectangles on the frame
void drawRectangles(cv::Mat &frame, const std::vector<cv::RotatedRect> &rectangles);

// Calculate the overlap ratio between two rotated rectangles
double overlapRatio(const cv::RotatedRect &rect1, const cv::RotatedRect &rect2);

// Extract the round number(es. game3round10.mp4 -> 10)
int roundNumber(const std::filesystem::path &video);

// Check if a cli argument is an integer
bool checkInteger(const std::string_view value);

// Get the round results based on the leader and the cards played and print them
bool getRoundResults(int round, RoundLabel &results, GameLabel &gameResults, std::string leader, const CardLabel &firstCard, const CardLabel &secondCard, 
    const CardLabel &briscola);

// Count correct results for the round
void countCorrectResults(const RoundLabel label, const RoundLabel roundResults, int &correctCards, int &correctPlayers, int &correctBriscola);

// Compute statistics for the game
void computeGameStatistics(const GameLabel label, const std::string winner, const GameLabel gameResults, 
                            const int correctCards, const int correctPlayers, const int correctBriscola, std::ofstream &statisticsFile);

#endif // UTILITY_HPP