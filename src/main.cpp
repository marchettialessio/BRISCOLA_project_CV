#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

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

bool isRectangle(cv::Mat &grayFrame, std::vector<cv::Point> contour, cv::RotatedRect &detectedRectangle)
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

    detectedRectangle = cv::minAreaRect(polygon);

    float width = detectedRectangle.size.width;
    float height = detectedRectangle.size.height;

    if (width <= 0 || height <= 0)
        return false;

    float aspectRatio = std::max(width, height) / std::min(width, height);

    // Filter out contours that do not match the expected aspect ratio of a card
    if (aspectRatio < 1.4f || aspectRatio > 2.2f)
        return false;

    // Check if the detected rectangle is within the frame boundaries
    cv::Rect bounds = detectedRectangle.boundingRect();

    cv::Rect safeBounds = bounds & cv::Rect(0, 0, grayFrame.cols, grayFrame.rows);
    if (safeBounds.empty())
        return false; // non valid bounding box, skip this contour

    // Check the mean brightness of the detected rectangle area to filter out dark areas like the back of the cards
    double meanBrightness = cv::mean(grayFrame(safeBounds))[0];
    if (meanBrightness < 120.0)
        return false;

    return true;
}

int main(int argc, char **argv)
{
    std::filesystem::path path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Briscola" / "game3" / "game3round11.mp4";

    cv::VideoCapture video(path);

    if (!video.isOpened())
    {
        std::cerr << "Could not open the video: " << path << std::endl;
        return -1;
    }

    cv::Mat frame, grayFrame, edges;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    std::vector<std::vector<cv::Point>> contours;

    int rectanglesCounter = 0;                       // rectangles counter
    std::vector<cv::RotatedRect> detectedRectangles; // found rectangles

    while (true)
    {
        if (!video.read(frame) || frame.empty())
        {
            std::cout << "End of video or failed to read frame." << std::endl;
            break;
        }

        // Convert the frame to grayscale and apply Gaussian blur
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grayFrame, grayFrame, cv::Size(5, 5), 0);

        // Detect edges using Canny edge detection
        cv::Canny(grayFrame, edges, 25.0, 90.0);

        // Apply morphological closing to fill gaps in the edges
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

        // Find contours in the edges
        cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        rectanglesCounter = 0;      // reset counter for each frame
        detectedRectangles.clear(); // clear previous rectangles for each frame

        for (const auto &contour : contours)
        {
            // Check if the contour is a rectangle and get its vertices
            cv::RotatedRect detectedRectangle;
            std::array<cv::Point2f, 4> vertices;
            if (isRectangle(grayFrame, contour, detectedRectangle))
            {
                rectanglesCounter++;
                detectedRectangles.push_back(detectedRectangle);
            }
        }
        std::cout << "Detected rectangles: " << rectanglesCounter << std::endl;

        // Final rectangles list
        std::vector<cv::RotatedRect> filteredRectangles;

        // Filter out overlapping rectangles based on the overlap ratio
        for (const auto &rectangle : detectedRectangles)
        {
            bool keepCurrent = true;

            // Check for overlap with already filtered rectangles
            cv::RotatedRect currentRect = rectangle;
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
                filteredRectangles.push_back(rectangle);
        }

        // Draw the filtered rectangles on the frame
        for (const auto &rectangle : filteredRectangles)
        {
            std::array<cv::Point2f, 4> vertices;
            rectangle.points(vertices.data());
            for (int i = 0; i < 4; ++i)
            {
                cv::line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        }
        rectanglesCounter = static_cast<int>(filteredRectangles.size()); // conteggio finale
        std::cout << "Filtered rectangles: " << rectanglesCounter << std::endl;

        cv::imshow("Video Frame", frame);

        // Press ESC to stop
        if (cv::waitKey(30) == 27)
            break;
    }

    video.release();
    cv::destroyAllWindows();

    return 0;
}