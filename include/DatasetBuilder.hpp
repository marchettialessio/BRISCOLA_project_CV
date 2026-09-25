#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "DatasetUtils.hpp"

// Associates the rectangles found by findCards with the round's cards and saves crop + label
class DatasetBuilder
{
public:
    DatasetBuilder(const std::filesystem::path &outDir, int game, int cropEvery = 3, int stableFrames = 3);

    //starting the generation of the dataset for a new round
    void startRound(const RoundLabel &label);
    // motion = side of the motion in the frame: 1 North, -1 South, 0 none
    void processFrame(const cv::Mat &frame, const std::vector<cv::RotatedRect> &rects, int motion = 0);
    void endRound();

    // Also usable at inference time: side north/south of the first card played,
    // empty string if not found
    std::string leaderSide() const;

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
        cv::Point2f anchor;   // center at confirmation: a card on the table does not move
        int northVotes = 0;   // frames with North motion while the card was being placed
        int southVotes = 0;   // frames with South motion while the card was being placed
        cv::Mat refThumb;     // appearance of the first crop, to discard crops of other cards
        Role role = Role::Unknown;
        std::vector<std::pair<int, cv::Mat>> crops; // (frame, crop) awaiting round validation, to save 
    };


    // I want to find the track that matches the current rect, if any
    int matchTrack(const cv::RotatedRect &rect, const std::vector<bool> &used) const;

    // find the track that corresponds to a given role
    const Track *findRole(Role role) const;
    bool isNorth(const Track &track, const Track *other) const;
    void saveCrops(const Track &track, const CardLabel &label, const std::string &player);

    std::filesystem::path imagesDir;
    std::ofstream csv;
    int game;
    int cropEvery; // frames between saved crops for the same card
    int stableFrames; // frames before a card is confirmed and assigned a role
    double anchorRadius = 0.15;  // match radius for confirmed cards (fraction of the short side)
    double noSpawnRadius = 0.25; // no new track is spawned within this radius of a confirmed card
    int minVoteMargin = 2;      // minimum North/South vote difference to trust detectMotion
    int maxMissing = 10;       // frames before removing a track without a role
    double minSimilarity = 0.5; // minimum correlation with the card's first crop

    RoundLabel current;
    std::vector<Track> tracks;
    int frameIdx = 0; // current frame index
    int frameHeight = 0;
    int nextRole = 0; // 0 = next card is First, 1 = Second, 2 = none

    int savedCrops = 0;
    int discardedCrops = 0;
    int validRounds = 0;
    int rejectedRounds = 0;
};
