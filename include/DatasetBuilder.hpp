#pragma once

#include <filesystem>
#include <fstream>
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

// Associates the rectangles found by findCards with the round's cards and saves crop + label
class DatasetBuilder
{
public:
    DatasetBuilder(const std::filesystem::path &outDir, int game, int cropEvery = 3, int stableFrames = 5);

    //starting the generation of the dataset for a new round
    void startRound(const RoundLabel &label);
    void processFrame(const cv::Mat &frame, const std::vector<cv::RotatedRect> &rects);
    void endRound();

private:
    enum class Role
    {
        Unknown,
        Background,
        First, // first card played
        Second // second card played
    };

    // This struct represent a tracked rectangle
    struct Track
    {
        cv::RotatedRect rect;
        int firstSeen = 0;   // frame in which the track was born
        int stillCounter = 0;   // consecutive frames in which the card is still
        int missing = 0;     // consecutive frames without a match
        int lastSaved = -1000; // frame in which the last crop was saved
        float refArea = 0.0f; // area at confirmation time
        cv::Mat refThumb;     // appearance of the first crop, to discard crops of other cards
        Role role = Role::Unknown;
        std::vector<std::pair<int, cv::Mat>> crops; // (frame, crop) awaiting round validation, to save 
    };

    int matchTrack(const cv::RotatedRect &rect, const std::vector<bool> &used) const;
    void saveCrops(const Track &track, const CardLabel &label, const std::string &player);

    std::filesystem::path imagesDir;
    std::ofstream csv;
    int game;
    int cropEvery; // frames between saved crops for the same card
    int stableFrames; // frames before a card is confirmed and assigned a role
    int backgroundFrames = 15; // cards confirmed within these frames = briscola / background
    int maxMissing = 10;       // frames before removing a track without a role
    double minSimilarity = 0.5; // minimum correlation with the card's first crop

    RoundLabel current;
    std::vector<Track> tracks;
    int frameIdx = 0; // current frame index
    int nextRole = 0; // 0 = next card is First, 1 = Second, 2 = none

    int savedCrops = 0;
    int discardedCrops = 0;
    int validRounds = 0;
    int rejectedRounds = 0;
};
