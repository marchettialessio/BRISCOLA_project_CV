#include "Utility.hpp"
#include <fstream>

// Draw the detected rectangles on the frame
void drawRectangles(cv::Mat &frame, const std::vector<cv::RotatedRect> &rectangles)
{
    for (const auto &rectangle : rectangles)
    {
        std::array<cv::Point2f, 4> vertices;
        rectangle.points(vertices.data());
        for (int i = 0; i < 4; ++i)
        {
            cv::line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
}

// Calculate the overlap ratio between two rotated rectangles
double overlapRatio(const cv::RotatedRect &rect1, const cv::RotatedRect &rect2)
{
    std::vector<cv::Point2f> intersection;
    int result = cv::rotatedRectangleIntersection(rect1, rect2, intersection);

    if (result == cv::INTERSECT_NONE || intersection.empty())
        return 0.0;

    double intersectionArea = cv::contourArea(intersection);
    double smallerArea = std::min(rect1.size.area(), rect2.size.area());

    return smallerArea > 0.0 ? intersectionArea / smallerArea : 0.0;
}

// Extract the round number(es. game3round10.mp4 -> 10)
int roundNumber(const std::filesystem::path &video)
{
    std::smatch match;
    const std::string name = video.filename().string();

    if (std::regex_search(name, match, std::regex("round(\\d+)")))
        return std::stoi(match[1]);
    return -1;
}

// Check if a cli argument is an integer
bool checkInteger(const std::string_view value) {
    int result{};

    const auto [pointer, error] = std::from_chars(value.data(), value.data() + value.size(), result);

    const bool validSyntax =
        error == std::errc{} &&
        pointer == value.data() + value.size();

    return validSyntax && result >= 1 && result <= 4;
}

bool getRoundResults(int round, RoundLabel &results, GameLabel &gameResults, std::string leader, const CardLabel &firstCard, const CardLabel &secondCard, const CardLabel &briscola) {
    int roundPoints = firstCard.points() + secondCard.points();
    std::string winner;
    if (secondCard.suit != firstCard.suit) {
        if (secondCard.suit == briscola.suit) {
            winner = leader == "North" ? "South" : "North";     // The leader loses
        }
        winner = leader;    // The leader wins
    } else {
        if (firstCard.points() > secondCard.points()) {
            winner = leader;        // The leader wins
        } else {
            winner = leader == "North" ? "South" : "North";     // The leader loses
        }
    }

    if (winner == "North") {
        gameResults.totalNorth += roundPoints;
        results.winnerNorth = true;
    } else {
        gameResults.totalSouth += roundPoints;
        results.winnerNorth = false;
    }
    results.points = roundPoints;

    std::cout << "Round " << round << " results: " << winner << " wins with " << roundPoints << " points." << std::endl;
    return true;
}

// Count correct results for the round
void countCorrectResults(const RoundLabel label, const RoundLabel roundResults, int &correctCards, int &correctPlayers, int &correctBriscola) {
    if (label.north.classId == roundResults.north.classId) {
        correctCards++;
        std::cout << "Correct card for North player." << std::endl;
    }
    if (label.south.classId == roundResults.south.classId) {
        correctCards++;
        std::cout << "Correct card for South player." << std::endl;
    }
    if (label.leaderNorth == roundResults.leaderNorth) {
        correctPlayers++;
        std::cout << "Correct leader detection." << std::endl;
    }
    if (label.winnerNorth == roundResults.winnerNorth) {
        correctPlayers++;
        std::cout << "Correct winner detection." << std::endl;
    }
    if (label.briscola.classId == roundResults.briscola.classId) {
        correctBriscola++;
        std::cout << "Correct briscola detection." << std::endl;
    }
}

// Compute statistics for the game
void computeGameStatistics(const GameLabel label, const std::string winner, const GameLabel gameResults, 
                            const int correctCards, const int correctPlayers, const int correctBriscola, std::ofstream &statisticsFile) {
    int correctResults = 0;
    if (label.winnerNorth && winner == "North" || !label.winnerNorth && winner == "South") {
        correctResults++;
        std::cout << "Correct game winner detection." << std::endl;
    }
    if (label.totalNorth == gameResults.totalNorth) {
        correctResults++;
        std::cout << "Correct total points for North player." << std::endl;
    }
    if (label.totalSouth == gameResults.totalSouth) {
        correctResults++;
        std::cout << "Correct total points for South player." << std::endl;
    }

    float cardAccuracy = (static_cast<float>(correctCards) / 40.0f);
    float playerAccuracy = (static_cast<float>(correctPlayers) / 40.0f);
    float briscolaAccuracy = (static_cast<float>(correctBriscola) / 40.0f);
    float gameAccuracy = (static_cast<float>(correctResults) / 3.0f);

    if (statisticsFile.is_open()) {
        statisticsFile << "Card Accuracy: " << cardAccuracy << std::endl;
        statisticsFile << "Player Accuracy: " << playerAccuracy << std::endl;
        statisticsFile << "Briscola Accuracy: " << briscolaAccuracy << std::endl;
        statisticsFile << "Game Accuracy: " << gameAccuracy << std::endl;
    } else {
        std::cerr << "Unable to open statistics file." << std::endl;
    }
}