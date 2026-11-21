#include "CardGeometry.hpp"

#include <algorithm>
#include <array>

// get the short side of a rectangle
float shortSide(const cv::RotatedRect &rect)
{
    return std::min(rect.size.width, rect.size.height);
}

// Horizontal card: the long side is closer to horizontal (the briscola under the deck lies sideways,
// played cards are placed vertically)
bool isHorizontal(const cv::RotatedRect &rect)
{
    std::array<cv::Point2f, 4> pts;
    rect.points(pts.data());
    const cv::Point2f a = pts[1] - pts[0];
    const cv::Point2f b = pts[2] - pts[1];
    const cv::Point2f longSide = cv::norm(a) > cv::norm(b) ? a : b;
    return std::abs(longSide.x) > std::abs(longSide.y);
}

// Small grayscale version of the crop, to compare appearance
cv::Mat thumbnail(const cv::Mat &crop)
{
    cv::Mat gray, small;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, small, cv::Size(40, 60), 0, 0, cv::INTER_AREA);
    small.convertTo(small, CV_32F);
    return small;
}

// Similarity between two thumbnails, tolerant to 180 degree rotation
double similarity(const cv::Mat &a, const cv::Mat &b)
{
    cv::Mat flipped, score;
    cv::flip(b, flipped, -1);

    cv::matchTemplate(a, b, score, cv::TM_CCOEFF_NORMED);
    const double direct = score.at<float>(0, 0);
    cv::matchTemplate(a, flipped, score, cv::TM_CCOEFF_NORMED);
    return std::max(direct, static_cast<double>(score.at<float>(0, 0)));
}

cv::Mat warpCard(const cv::Mat &frame, const cv::RotatedRect &rect, cv::Size size)
{
    std::array<cv::Point2f, 4> pts;
    rect.points(pts.data());

    // Starts from the vertex where the short side originates, so the card comes out vertical
    const int start = cv::norm(pts[1] - pts[0]) < cv::norm(pts[2] - pts[1]) ? 0 : 1;

    std::array<cv::Point2f, 4> source;
    for (int i = 0; i < 4; ++i)
        source[i] = pts[(start + i) % 4];

    const float w = static_cast<float>(size.width - 1);
    const float h = static_cast<float>(size.height - 1);
    const std::array<cv::Point2f, 4> destination = {cv::Point2f(0, 0), cv::Point2f(w, 0), cv::Point2f(w, h), cv::Point2f(0, h)};

    cv::Mat transform = cv::getPerspectiveTransform(source.data(), destination.data());
    cv::Mat warped;
    cv::warpPerspective(frame, warped, transform, size);
    return warped;
}
