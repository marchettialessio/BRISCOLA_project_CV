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

struct Card createCard(const std::string templatePath) {
    std::regex pattern(R"((10|[1-9])-(clubs|coins|cups|spades))");
    std::smatch match;

    if (std::regex_match(templatePath, match, pattern)) {
        std::cout << "Template path: " << templatePath << " Matches: " << match[0] << ", " << match[1] << ", " << match[2] << std::endl;
        std::string id = match[0].str();
        int number = std::stoi(match[1].str());
        std::string seed = match[2].str();
        int points = 0;

        switch (number) {
            case 1:
                points = 11;
                break;
            case 3:
                points = 10;
                break;
            case 8:
                points = 2;
                break;
            case 9:
                points = 3;
                break;
            case 10:
                points = 4;
                break;
            default:
                points = 0;
        }

        return {id, number, seed, points};
    } else {
        throw std::invalid_argument("Invalid template path format: " + templatePath);
    }
}

RoundResults getRoundResults(int round, std::string leader, const Card &firstCard, const Card &secondCard, int &pointsPlayerNorth, int &pointsPlayerSouth) {
    std::string winner;
    int roundPoints = firstCard.points + secondCard.points;
    if (secondCard.seed != firstCard.seed) {
        winner = leader;
    } else {
        if (firstCard.points > secondCard.points) {
            winner = leader;
        } else {
            winner = leader == "North" ? "South" : "North";
        }
    }

    if (winner == "North") {
        pointsPlayerNorth += roundPoints;
    } else {
        pointsPlayerSouth += roundPoints;
    }

    std::cout << "Round " << round << " results: " << winner << " wins with " << roundPoints << " points." << std::endl;

    return {roundPoints, winner};
}