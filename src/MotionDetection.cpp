#include "MotionDetection.hpp"

// Detect motion in the frame using background subtraction and display the motion mask
// Returns the side of the motion in the frame: 1 = North, -1 = South, 0 = no motion
int detectMotion(cv::Ptr<cv::BackgroundSubtractorMOG2> &subtractor, cv::Mat &frame, int &north, int &south, bool showWindows)
{
    // Apply background subtraction to detect motion
    cv::Mat mask;
    subtractor->apply(frame, mask);

    // Apply Gaussian blur to the mask to reduce noise
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

    // Threshold the mask to create a binary image
    cv::threshold(mask, mask, 200, 255, cv::THRESH_BINARY);

    // Apply morpholgical opening to remove noise and small objects from the mask
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));

    // Calculate the moments of the mask to find the centroid of the motion
    cv::Moments moments = cv::moments(mask, true);

    // m00 = n° white pixels in the mask
    if (moments.m00 == 0.0)
        return 0; // No motion detected, skip further processing

    double whiteRatio = moments.m00 / mask.total();

    if (whiteRatio < 0.01)
        return 0; // Not enough motion detected, skip further processing

    // Calculate the centroid of the motion m10 = sum of x coordinates of white pixels, m01 = sum of y coordinates of white pixels
    cv::Point2f centroid(moments.m10 / moments.m00,
                         moments.m01 / moments.m00);
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