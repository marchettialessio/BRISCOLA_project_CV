#pragma once

#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

// Follows the rectangles found by findCards across the frames of a round, assigns a role to each card
// (briscola, first/second card played) and collects its crops.
// Used both to build the dataset and at inference time.
class CardTracker
{
public:
    enum class Role
    {
        Unknown,
        Background, // extra cards (e.g. pickup at end of round)
        Briscola,   // horizontal card under the deck
        First,      // first card played
        Second      // second card played
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
        std::vector<std::pair<int, cv::Mat>> crops; // (frame, crop) of the card
    };

    CardTracker(int cropEvery = 3, int stableFrames = 3);

    // new round: forget all the tracks
    void reset();
    // motion = side of the motion in the frame: 1 North, -1 South, 0 none
    void processFrame(const cv::Mat &frame, const std::vector<cv::RotatedRect> &rects, int motion = 0);

    // find the track that corresponds to a given role, nullptr if not found
    const Track *findRole(Role role) const;
    // Card side: true if the card comes from North. other = the other played card, can be nullptr
    bool isNorth(const Track &track, const Track *other) const;

    int discardedCrops() const { return discarded; }

private:
    // I want to find the track that matches the current rect, if any
    int matchTrack(const cv::RotatedRect &rect, const std::vector<bool> &used) const;

    int cropEvery; // frames between saved crops for the same card
    int stableFrames; // frames before a card is confirmed and assigned a role
    double anchorRadius = 0.15;  // match radius for confirmed cards (fraction of the short side)
    double noSpawnRadius = 0.25; // no new track is spawned within this radius of a confirmed card
    int minVoteMargin = 2;      // minimum North/South vote difference to trust detectMotion
    int maxMissing = 10;       // frames before removing a track without a role
    double minSimilarity = 0.5; // minimum correlation with the card's first crop

    std::vector<Track> tracks;
    int frameIdx = 0; // current frame index
    int frameHeight = 0;
    int nextRole = 0; // 0 = next card is First, 1 = Second, 2 = none
    bool briscolaFound = false;

    int discarded = 0; // crops discarded by the similarity check
};
