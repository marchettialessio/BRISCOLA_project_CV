#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <regex>
#include <optional>
#include <charconv>

struct Card {
    std::string id;     // e.g., "10-clubs"
    int number;         // e.g., 10
    std::string seed;   // e.g., "clubs"
    int points;         // e.g., 4
};

struct RoundResults {
    int points;
    std::string winner;
};

struct Player {
    std::string position;       // "North" or "South"
    int points;
    std::optional<Card> card;       // The card played by the player in the current round
};

// Draw the detected rectangles on the frame
void drawRectangles(cv::Mat &frame, const std::vector<cv::RotatedRect> &rectangles);

// Calculate the overlap ratio between two rotated rectangles
double overlapRatio(const cv::RotatedRect &rect1, const cv::RotatedRect &rect2);

// Extract the round number(es. game3round10.mp4 -> 10)
int roundNumber(const std::filesystem::path &video);

// Check if a cli argument is an integer
bool checkInteger(const std::string_view value);

// Create a Card object from a template image path
struct Card createCard(const std::string templatePath);

// Get the round results based on the leader and the cards played and print them
RoundResults getRoundResults(int round, std::string leader, const Card &firstCard, const Card &secondCard, const Card &briscola, int &pointsPlayerNorth, int &pointsPlayerSouth);

#endif // UTILITY_HPP