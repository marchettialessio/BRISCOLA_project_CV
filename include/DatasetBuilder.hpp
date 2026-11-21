#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "CardTracker.hpp"
#include "Labels.hpp"

// Associates the rectangles found by findCards with the round's cards (through CardTracker)
// and saves crop + label
class DatasetBuilder
{
public:
    DatasetBuilder(const std::filesystem::path &outDir, int game, int cropEvery = 3, int stableFrames = 3);

    //starting the generation of the dataset for a new round
    void startRound(const RoundLabel &label);
    // motion = side of the motion in the frame: 1 North, -1 South, 0 none
    void processFrame(const cv::Mat &frame, const std::vector<cv::RotatedRect> &rects, int motion = 0);
    // saves the crops of the two played cards if the tracker result is consistent with the label
    void endRound();

    // Also usable at inference time: side north/south of the first card played,
    // empty string if not found
    std::string leaderSide() const;

private:
    void saveCrops(const CardTracker::Track &track, const CardLabel &label, const std::string &player);

    CardTracker tracker;
    RoundLabel current;
    std::filesystem::path imagesDir;
    std::ofstream csv;
    int game;

    int savedCrops = 0;
    int validRounds = 0;
    int rejectedRounds = 0;
};

// Genera il set dalle scansioni Trentine
int buildTrentineSet(const std::filesystem::path &scansDir, const std::filesystem::path &outDir,
                     cv::Size size = cv::Size(200, 300));
