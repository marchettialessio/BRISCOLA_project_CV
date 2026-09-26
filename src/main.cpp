#include <iostream>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <regex>
#include <map>
#include "DatasetBuilder.hpp"

// false in modalità dataset: nessuna finestra, elaborazione più veloce
bool showWindows = true;

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

// Filter the detected rectangles based on overlap ratio and keep the larger ones and get the final cards
void filterRectangles(std::vector<cv::RotatedRect> &candidates, std::vector<cv::RotatedRect> &filteredRectangles)
{
    for (const auto &candidate : candidates)
    {
        bool keepCurrent = true;

        // Check for overlap with already filtered rectangles
        cv::RotatedRect currentRect = candidate;
        for (auto it = filteredRectangles.begin(); it != filteredRectangles.end();)
        {
            cv::RotatedRect otherRect = cv::RotatedRect(*it);

            // Calculate the overlap ratio between the current rectangle and the other rectangle
            double overlap = overlapRatio(currentRect, otherRect);

            // If the overlap ratio is greater than 0.85, keep the larger rectangle and discard the smaller one
            if (overlap > 0.85)
            {
                if (currentRect.size.area() > otherRect.size.area())
                {
                    it = filteredRectangles.erase(it);
                    continue;
                }

                keepCurrent = false;
                break;
            }

            ++it;
        }
        if (keepCurrent)
            filteredRectangles.push_back(currentRect);
    }
}

// Check if the contour is a rectangle based on area, aspect ratio, and mean brightness
bool isRectangle(cv::Mat &grayFrame, std::vector<cv::Point> contour, cv::RotatedRect &candidate)
{
    double area = cv::contourArea(contour);

    double frameArea = grayFrame.rows * grayFrame.cols;

    // Filter out small contours based on area
    if (area < 0.01 * frameArea || area > 0.3 * frameArea)
        return false;

    double perimeter = cv::arcLength(contour, true);

    // Approximate the contour to a polygon
    std::vector<cv::Point> polygon;
    cv::approxPolyDP(contour, polygon, 0.02 * perimeter, true);

    /*
    // Display the original contour and the approximated polygon
    cv::Mat contourImage = cv::Mat::zeros(grayFrame.size(), CV_8UC3);
    cv::drawContours(contourImage, contour, -1, cv::Scalar(0, 255, 0));
    cv::imshow("Contour", contourImage);

    cv::Mat polygonImage = cv::Mat::zeros(grayFrame.size(), CV_8UC3);
    cv::drawContours(polygonImage, polygon, -1, cv::Scalar(255, 0, 0));
    cv::imshow("Polygon", polygonImage);
    */

    // Filter out non-quadrilateral contours
    if (polygon.size() < 4 || polygon.size() > 15)
        return false;

    candidate = cv::minAreaRect(polygon);

    float width = candidate.size.width;
    float height = candidate.size.height;

    if (width <= 0 || height <= 0)
        return false;

    float aspectRatio = std::max(width, height) / std::min(width, height);

    // Filter out contours that do not match the expected aspect ratio of a card
    if (aspectRatio < 1.4f || aspectRatio > 2.2f)
        return false;

    // Check if the detected rectangle is within the frame boundaries
    cv::Rect bounds = candidate.boundingRect();

    cv::Rect safeBounds = bounds & cv::Rect(0, 0, grayFrame.cols, grayFrame.rows);
    if (safeBounds.empty())
        return false; // non valid bounding box, skip this contour

    // Check the mean brightness of the detected rectangle area to filter out dark areas like the back of the cards
    double meanBrightness = cv::mean(grayFrame(safeBounds))[0];
    if (meanBrightness < 120.0)
        return false;

    return true;
}

// Detect the candidates from the contours and store them in the candidates vector, also update the rectanglesCounter
void detectCandidates(std::vector<std::vector<cv::Point>> contours, std::vector<cv::RotatedRect> &candidates, cv::Mat grayFrame, int &rectanglesCounter)
{
    for (const auto &contour : contours)
    {
        // Check if the contour is a rectangle
        cv::RotatedRect candidate;
        if (isRectangle(grayFrame, contour, candidate))
        {
            rectanglesCounter++;
            candidates.push_back(candidate);
        }
    }
}

// Find cards in the frame by processing the image, detecting contours, filtering rectangles, and drawing them on the frame
void findCards(cv::Mat &frame, cv::Mat &grayFrame, cv::Mat kernel, DatasetBuilder *builder = nullptr, int motion = 0)
{
    // Convert the frame to grayscale and apply Gaussian blur
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayFrame, grayFrame, cv::Size(5, 5), 0);

    // Detect edges using Canny edge detection
    cv::Mat edges;
    cv::Canny(grayFrame, edges, 25.0, 90.0);

    // Apply morphological closing to fill gaps in the edges
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

    // Find contours in the edges
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Second edge map: light blur and no closing
    cv::Mat rawGray, rawEdges;
    cv::cvtColor(frame, rawGray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(rawGray, rawGray, cv::Size(3, 3), 0);
    cv::Canny(rawGray, rawEdges, 40.0, 90.0);

    std::vector<std::vector<cv::Point>> rawContours;
    cv::findContours(rawEdges, rawContours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    // duplicates between the two maps are removed by filterRectangles
    contours.insert(contours.end(), rawContours.begin(), rawContours.end());

    // Find new candidates
    int candidatesCounter = 0;
    std::vector<cv::RotatedRect> candidates;

    detectCandidates(contours, candidates, grayFrame, candidatesCounter);
    std::cout << "Detected candidates: " << candidatesCounter << std::endl;

    // Filter the detected candidates to get the cards
    std::vector<cv::RotatedRect> filteredRectangles;
    filterRectangles(candidates, filteredRectangles);

    int rectanglesCounter = static_cast<int>(filteredRectangles.size()); // final rectangles counter
    std::cout << "Filtered rectangles: " << rectanglesCounter << std::endl;

    // Associate the rectangles with the round's cards (before drawing on the frame)
    if (builder)
        builder->processFrame(frame, filteredRectangles, motion);

    // Draw the filtered rectangles on the frame
    drawRectangles(frame, filteredRectangles);
}

// Detect motion in the frame using background subtraction and display the motion mask
// Returns the side of the motion in the frame: 1 = North, -1 = South, 0 = no motion
int detectMotion(cv::Ptr<cv::BackgroundSubtractorMOG2> &subtractor, cv::Mat &frame, cv::Mat kernel, int &north, int &south)
{
    // Apply background subtraction to detect motion
    cv::Mat mask;
    subtractor->apply(frame, mask);

    // Apply Gaussian blur to the mask to reduce noise
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

    // Threshold the mask to create a binary image
    cv::threshold(mask, mask, 200, 255, cv::THRESH_BINARY);

    // Apply morpholgical opening to remove noise and small objects from the mask
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // Calculate the moments of the mask to find the centroid of the motion
    cv::Moments moments = cv::moments(mask, true);

    // m00 = n° white pixels in the mask
    if (moments.m00 == 0.0)
        return 0; // No motion detected, skip further processing

    double whiteRatio = moments.m00 / mask.total();

    if (whiteRatio < 0.02)
        return 0; // Not enough motion detected, skip further processing

    // Calculate the centroid of the motion m10 = sum of x coordinates of white pixels, m01 = sum of y coordinates of white pixels
    cv::Point2f centroid(moments.m10 / moments.m00,
                         moments.m01 / moments.m00);

    if (showWindows)
        cv::imshow("Motion", mask);

    if (centroid.y < frame.rows / 2)
    {
        std::cout << "Motion detected in the North region, North: " << north << std::endl;
        north++;
        return 1;
    }

    std::cout << "Motion detected in the South region, South: " << south << std::endl;
    south++;
    return -1;
}

bool processVideo(const std::string &path, std::ofstream &outputFile, DatasetBuilder *builder = nullptr)
{
    cv::VideoCapture video(path);

    if (!video.isOpened())
    {
        std::cerr << "Could not open the video: " << path << std::endl;
        return false;
    }

    cv::Mat frame, grayFrame;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Ptr<cv::BackgroundSubtractorMOG2> subtractor = cv::createBackgroundSubtractorMOG2(500, 16, true);
    int north = 0, south = 0; // Votes for the first card detection
    bool detected = false; // Flag to indicate if the first card has been detected

    while (true)
    {
        if (!video.read(frame) || frame.empty())
        {
            std::cout << "End of video or failed to read frame" << std::endl;
            break;
        }

        // If the first card has not been detected yet, perform motion detection to find the first card
        // In dataset mode the Leader comes from the label: motion detection is not needed
        // Motion side is also used by the builder to know from which side each card arrives
        const int motion = detectMotion(subtractor, frame, kernel2, north, south);
        if (!builder && !detected) {
            if (north >= 5)
            {
                std::cout << "Leader: North" << std::endl;
                detected = true; // Set the flag to true after the first card is detected

                if (outputFile.is_open())
                {
                    outputFile << "Leader: North" << std::endl;
                    outputFile.flush(); // Ensure the output is written to the file immediately
                    return true; // Exit the function after writing to the output file
                }
                else
                {
                    std::cerr << "Failed to write to output file." << std::endl;
                }
            }
            else if (south >= 5)
            {
                std::cout << "Leader: South" << std::endl;
                detected = true; // Set the flag to true after the first card is detected

                if (outputFile.is_open())
                {
                    outputFile << "Leader: South" << std::endl;
                    outputFile.flush(); // Ensure the output is written to the file immediately
                    return true; // Exit the function after writing to the output file
                }
                else
                {
                    std::cerr << "Failed to write to output file." << std::endl;
                }
            }
        }

        // Find cards in the current frame
        findCards(frame, grayFrame, kernel, builder, motion);

        if (showWindows)
        {
            cv::imshow("Video Frame", frame);

            // Press ESC to skip video
            if (cv::waitKey(builder ? 1 : 30) == 27)
                break;
        }
    }

    video.release();
    if (showWindows)
        cv::destroyAllWindows();

    return true;
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

int main(int argc, char **argv)
{
    const std::filesystem::path root(PROJECT_SOURCE_DIR);

    if (argc > 1 && std::string(argv[1]) == "--trentine")
        return buildTrentineSet(root / "Briscola" / "Briscola_Trentine", root / "dataset") > 0 ? 0 : 1;

    std::filesystem::path videoFolder = root / "Briscola" / "game1";
    std::filesystem::path outputFilePath = root / "output" / "game1output.txt";

    std::string game = "1";
    if (argc > 1)
    {
        game = argv[1];
        videoFolder = root / "Briscola" / ("game" + game);
        outputFilePath = root / "output" / ("game" + game + "output.txt");
    }

    std::ofstream outputFile(outputFilePath);

    const bool datasetMode = argc > 2 && std::string(argv[2]) == "--dataset";
    showWindows = !datasetMode;

    // ordering videos by round number
    std::vector<std::filesystem::path> videos;
    for (const auto &video : std::filesystem::directory_iterator(videoFolder))
    {
        if (!video.is_regular_file() || video.path().extension() != ".mp4")
            continue;
        videos.push_back(video.path());
    }
    std::sort(videos.begin(), videos.end(), [](const auto &a, const auto &b)
              { return roundNumber(a) < roundNumber(b); });

    std::unique_ptr<DatasetBuilder> builder;

    // load game labels, indexed by round
    std::map<int, RoundLabel> labels;
    if (datasetMode)
    {
        // I find the label file for the selected game
        const std::filesystem::path labelFile = findLabelFile(root / "labels", std::stoi(game));
        if (labelFile.empty())
        {
            std::cerr << "No label file for game " << game << std::endl;
            return 1;
        }

        // I parse all the labels for each round and store them
        for (const auto &label : parseLabels(labelFile))
            labels[label.round] = label;

        builder = std::make_unique<DatasetBuilder>(root / "dataset", std::stoi(game));
    }

    for (const auto &video : videos)
    {
        const int round = roundNumber(video);

        if (outputFile.is_open())
        {
            outputFile << "Round " << round << std::endl;
            outputFile.flush(); // Ensure the output is written to the file immediately
        }
        else
        {
            std::cerr << "Failed to open output file." << std::endl;
        }

        if (builder)
        {
            if (!labels.count(round))
            {
                std::cerr << "Missing label for round " << round << ", video skipped" << std::endl;
                continue;
            }
            //i start to build the dataset for the current round
            builder->startRound(labels[round]);
        }

        std::cout << "Processing video: " << video << std::endl;
        if (!processVideo(video.string(), outputFile, builder.get()))
            break;

        if (builder)
            builder->endRound();

        // Press ESC to stop
        if (showWindows && cv::waitKey(builder ? 1 : 1000) == 27)
            break;
    }

    return 0;
}
