#pragma once

#include <filesystem>
#include <string>
#include <vector>

// rank 1-10, suit 0-3 (coins, cups, spades, clubs), classId = suit*10 + (rank - 1)
struct CardLabel
{
    int rank = 0;
    int suit = -1;
    int classId = -1;
    std::string suitName;

    // false for a card that was not found/recognized
    bool valid() const { return classId >= 0; }
};

// Content of a round: used both for the ground truth and for the predictions
struct RoundLabel
{
    int round = 0;
    CardLabel north, south, briscola;
    bool leaderNorth = true;
    bool winnerNorth = true;
    int points = 0; // points collected by the winner
};

// Whole game: the rounds plus the final result (reported only in the .txt label file)
struct GameLabel
{
    std::vector<RoundLabel> rounds;
    bool winnerNorth = true;
    int totalNorth = 0;
    int totalSouth = 0;
};

std::string toLower(std::string s);

// Converts value (ex. "6 , clubs") to CardLabel
bool parseCard(const std::string &value, CardLabel &card);

// Inverse of classId = suit*10 + (rank - 1), used to decode the classifier output
CardLabel cardFromClassId(int classId);

// i want to parse the label file and return the game: one RoundLabel for each round + final result
GameLabel parseGame(const std::filesystem::path &file);

// only the rounds of the label file
std::vector<RoundLabel> parseLabels(const std::filesystem::path &file);

// search the directory of labels for the selected game
std::filesystem::path findLabelFile(const std::filesystem::path &labelsDir, int game);
