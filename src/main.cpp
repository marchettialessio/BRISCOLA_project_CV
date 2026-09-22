#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

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
        frame_ratio=0.002;
        area_ratio=0.5;
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
void findCards(cv::Mat &frame, cv::Mat &grayFrame, cv::Mat kernel)
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

    // Draw the filtered rectangles on the frame
    drawRectangles(frame, filteredRectangles);
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

    drawRectangles(frame, filteredRectangles);

    briscola = filteredRectangles.front();
    for (const auto &rectangle : filteredRectangles)
    {
        if (rectangle.size.area() < briscola.size.area())
            briscola = rectangle;
    }
    return true;
}

// Detect motion in the frame using background subtraction and display the motion mask
void detectMotion(cv::Ptr<cv::BackgroundSubtractorMOG2> &subtractor, cv::Mat &frame, cv::Mat kernel, int &north, int &south)
{
    // Apply background subtraction to detect motion
    cv::Mat mask;
    subtractor->apply(frame, mask);

    // Apply Gaussian blur to the mask to reduce noise
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

    // Threshold the mask to create a binary image
    cv::threshold(mask, mask, 220, 255, cv::THRESH_BINARY);

    // Apply morpholgical opening to remove noise and small objects from the mask
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // Calculate the moments of the mask to find the centroid of the motion
    cv::Moments moments = cv::moments(mask, true);

    // m00 = n° white pixels in the mask
    if (moments.m00 == 0.0)
        return; // No motion detected, skip further processing

    double whiteRatio = moments.m00 / mask.total();

    if (whiteRatio < 0.02)
        return; // Not enough motion detected, skip further processing

    // Calculate the centroid of the motion m10 = sum of x coordinates of white pixels, m01 = sum of y coordinates of white pixels
    cv::Point2f centroid(moments.m10 / moments.m00,
                         moments.m01 / moments.m00);

    if (centroid.y < frame.rows / 2)
    {
        std::cout << "Motion detected in the North region, North: " << north << std::endl;
        north++;
    }
    else
    {
        std::cout << "Motion detected in the South region, South: " << south << std::endl;
        south++;
    }

    cv::imshow("Motion", mask);

    // MAYBE TO REINFORCE IT BY TRACKING THE DIRECTION OF THE MOTION
    // IT GIVES PROBLEMS ON GAME 3 ROUND 4(?), THE FIRST CARD IS DETECTED IN THE SOUTH BUT DOES NOT GET TOO MANY VOTES
    // TO PERFORM ALSO MORE PARAMETER TUNING
}

bool processVideo(const std::string &path, bool firstRound)
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
    int north = 0, south = 0;
    bool detected = false; // Flag to indicate if the first card has been detected
    bool briscolaFixed = false;
    cv::RotatedRect briscola;
    int frameIndex = 0;


    while (true)
    {
        if (!video.read(frame) || frame.empty())
        {
            std::cout << "End of video or failed to read frame" << std::endl;
            break;
        }

        if (firstRound && frameIndex < 5 && !briscolaFixed)
        {
            briscolaFixed = findBriscola(frame, grayFrame, kernel, briscola);
            if (briscolaFixed)
                drawRectangles(frame, {briscola});
        }
        else
        {
            
            if (!detected)
            {
                detectMotion(subtractor, frame, kernel2, north, south);
                if (north >= 5){
                    std::cout << "Leader: North" << std::endl;
                    detected = true; // Set the flag to true after the first card is detected

                    cv::Mat motionFound = cv::Mat::zeros(frame.size(), CV_8UC3);
                    cv::putText(motionFound, "North", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                    cv::imshow("Motion", motionFound);
                }
                else if (south >= 5){
                    std::cout << "Leader: South" << std::endl;
                    detected = true; // Set the flag to true after the first card is detected

                    cv::Mat motionFound = cv::Mat::zeros(frame.size(), CV_8UC3);
                    cv::putText(motionFound, "South", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                    cv::imshow("Motion", motionFound);
                }
            }
            
            findCards(frame, grayFrame, kernel);
        }
        
        cv::imshow("Video Frame", frame);
        ++frameIndex;

        // Press ESC to skip video
        if (cv::waitKey(30) == 27)
            break;
    }

    video.release();
    cv::destroyAllWindows();

    return true;
}

int main(int argc, char **argv)
{   
    std::filesystem::path videoFolder;
    if(argc>1){
        videoFolder = std::filesystem::path(PROJECT_SOURCE_DIR) / argv[1];
        if (!std::filesystem::is_directory(videoFolder))
            videoFolder = std::filesystem::path(PROJECT_SOURCE_DIR) / "BRISCOLA" / argv[1];
    }
    else{
        videoFolder = std::filesystem::path(PROJECT_SOURCE_DIR) / "BRISCOLA" / "game3";
    }   
    //const std::filesystem::path path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Briscola" / "game3" / "game3round1.mp4";

    std::vector<std::filesystem::path> videos;
    for (const auto &entry : std::filesystem::directory_iterator(videoFolder))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".mp4")
            continue;
        videos.push_back(entry.path());
    }

    std::sort(videos.begin(), videos.end());
    bool firstRound = true;
    for (const auto &video : videos)
    {

        std::cout << "Processing video: " << video << std::endl;
        if (!processVideo(video.string(), firstRound))
            break;
        firstRound = false;

        // Press ESC to stop
        if (cv::waitKey(1000) == 27)
            break;
    }

    return 0;
}