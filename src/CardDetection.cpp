#include "CardDetection.hpp"
#include "Utility.hpp"

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
bool isRectangle(cv::Mat &grayFrame, std::vector<cv::Point> contour, cv::RotatedRect &candidate, bool briscola)
{
    double area = cv::contourArea(contour);

    double frameArea = grayFrame.rows * grayFrame.cols;
    double frame_ratio;
    double area_ratio;
    double min_ratio_length;
    double max_ratio_length;
    if(briscola){
        frame_ratio=0.001;
        area_ratio=0.3;
        min_ratio_length=0.8f;
        max_ratio_length=1.8f;
    }
    else{
        frame_ratio=0.01;
        area_ratio=0.3;
        min_ratio_length=1.4f;
        max_ratio_length=2.2f;
    }

    // Filter out small contours based on area
    if (area < frame_ratio * frameArea || area > area_ratio * frameArea)
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
    if (aspectRatio < min_ratio_length || aspectRatio > max_ratio_length)
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
void detectCandidates(std::vector<std::vector<cv::Point>> contours, std::vector<cv::RotatedRect> &candidates, cv::Mat grayFrame, int &rectanglesCounter, bool briscola)
{
    for (const auto &contour : contours)
    {
        // Check if the contour is a rectangle
        cv::RotatedRect candidate;
        if (isRectangle(grayFrame, contour, candidate, briscola))
        {
            rectanglesCounter++;
            candidates.push_back(candidate);
        }
    }
}

// Find cards in the frame by processing the image, detecting contours, filtering rectangles, and drawing them on the frame
std::vector<cv::RotatedRect> findCards(cv::Mat &frame, cv::Mat &grayFrame, cv::Mat kernel, DatasetBuilder *builder, int motion)
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

    detectCandidates(contours, candidates, grayFrame, candidatesCounter, false);
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
    return filteredRectangles;
}

bool findBriscola(cv::Mat &frame, cv::Mat &grayFrame, cv::Mat kernel, cv::RotatedRect &briscola)
{
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayFrame, grayFrame, cv::Size(5, 5), 0);

    cv::Mat edges;
    cv::Canny(grayFrame, edges, 25.0, 90.0);
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::RotatedRect> candidates;
    int candidatesCounter = 0;
    detectCandidates(contours, candidates, grayFrame, candidatesCounter, true);

    std::vector<cv::RotatedRect> filteredRectangles;
    filterRectangles(candidates, filteredRectangles);
    if (filteredRectangles.empty())
        return false;

    briscola = filteredRectangles.front();
    for (const auto &rectangle : filteredRectangles)
    {
        if (rectangle.size.area() < briscola.size.area())
            briscola = rectangle;
    }
    return true;
}