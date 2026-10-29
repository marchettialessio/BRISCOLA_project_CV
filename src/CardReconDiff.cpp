#include "CardReconDiff.hpp"

#include <array>
#include <filesystem>
#include <limits>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

static std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& poligon)
{
    std::vector<cv::Point2f> points = poligon;
    std::vector<cv::Point2f> ordinati(4);
    float minSum = points[0].x + points[0].y;
    float maxSum = minSum;
    float minDiff = points[0].x - points[0].y;
    float maxDiff = minDiff;

    int topLeft = 0;
    int bottomRight = 0;
    int topRight = 0;
    int bottomLeft = 0;

    for (int i = 0; i < 4; ++i)
    {
        float sum = points[i].x + points[i].y;
        float diff = points[i].x - points[i].y;

        if (sum < minSum)
        {
            minSum = sum;
            topLeft = i;
        }
        if (sum > maxSum)
        {
            maxSum = sum;
            bottomRight = i;
        }
        if (diff < minDiff)
        {
            minDiff = diff;
            topRight = i;
        }
        if (diff > maxDiff)
        {
            maxDiff = diff;
            bottomLeft = i;
        }
    }

    ordinati[0] = points[topLeft];
    ordinati[1] = points[topRight];
    ordinati[2] = points[bottomRight];
    ordinati[3] = points[bottomLeft];
    return ordinati;
}

static cv::Mat rectangleOmography(const cv::Mat& image, const std::vector<cv::Point2f>& poligon,
                                  cv::Size outputSize = cv::Size(500, 300), int orientation = -1)
{
    std::vector<cv::Point2f> source = orderPoints(poligon);
    float width = std::max(cv::norm(source[1] - source[0]), cv::norm(source[2] - source[3]));
    float height = std::max(cv::norm(source[3] - source[0]), cv::norm(source[2] - source[1]));

    if (width < 10.0f || height < 10.0f)
    {
        return cv::Mat();
    }

    const int outputWidth = outputSize.width;
    const int outputHeight = outputSize.height;
    std::vector<cv::Point2f> destination;

    if (orientation == 1)
    {
        destination = {
            {static_cast<float>(outputWidth - 1), 0.0f},
            {static_cast<float>(outputWidth - 1), static_cast<float>(outputHeight - 1)},
            {0.0f, static_cast<float>(outputHeight - 1)},
            {0.0f, 0.0f}
        };
    }
    else if (orientation == 0 || (orientation < 0 && width > height))
    {
        destination = {
            {0.0f, 0.0f},
            {static_cast<float>(outputWidth - 1), 0.0f},
            {static_cast<float>(outputWidth - 1), static_cast<float>(outputHeight - 1)},
            {0.0f, static_cast<float>(outputHeight - 1)}
        };
    }
    else
    {
        destination = {
            {static_cast<float>(outputWidth - 1), 0.0f},
            {static_cast<float>(outputWidth - 1), static_cast<float>(outputHeight - 1)},
            {0.0f, static_cast<float>(outputHeight - 1)},
            {0.0f, 0.0f}
        };
    }

    cv::Mat transform = cv::getPerspectiveTransform(source, destination);
    cv::Mat warped;
    cv::warpPerspective(image, warped, transform, cv::Size(outputWidth, outputHeight));
    return warped;
}

std::string ReconCard(const cv::Mat& frame, cv::RotatedRect card, const fs::path templateFolder)
{
    constexpr double maxDifference = 60.0;

    struct TemplateFeatures
    {
        fs::path path;
        cv::Mat grayImage;
        cv::Mat hsvImage;
    };

    const cv::Size templateSize(500, 300);
    std::vector<TemplateFeatures> templates;

    for (const auto& entry : fs::directory_iterator(templateFolder))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty())
        {
            continue;
        }

        cv::Mat templateGray;
        cv::cvtColor(img, templateGray, cv::COLOR_BGR2GRAY);
        cv::resize(templateGray, templateGray, templateSize);
        cv::Mat templateHsv;
        cv::cvtColor(img, templateHsv, cv::COLOR_BGR2HSV);
        cv::resize(templateHsv, templateHsv, templateSize);
        templates.push_back({entry.path(), templateGray, templateHsv});
    }

    std::array<cv::Point2f, 4> vertices;
    card.points(vertices.data());
    std::vector<cv::Point2f> polygon(vertices.begin(), vertices.end());

    cv::Mat rectified = rectangleOmography(frame, polygon);
    if (rectified.empty())
    {
        std::cerr << "Errore: impossibile rettificare la carta." << std::endl;
        return "";
    }
    cv::imshow("Omografia carta", rectified);

    cv::Mat rectifiedGray;
    cv::cvtColor(rectified, rectifiedGray, cv::COLOR_BGR2GRAY);
    cv::resize(rectifiedGray, rectifiedGray, templateSize);
    cv::Mat rectifiedHsv;
    cv::cvtColor(rectified, rectifiedHsv, cv::COLOR_BGR2HSV);
    cv::resize(rectifiedHsv, rectifiedHsv, templateSize);

    cv::Mat mirroredGray;
    cv::flip(rectifiedGray, mirroredGray, 1);
    cv::Mat mirroredHsv;
    cv::flip(rectifiedHsv, mirroredHsv, 1);

    fs::path bestMatch;
    double bestDifference = std::numeric_limits<double>::max();
    double bestGrayDifference = 0.0;
    double bestHsvDifference = 0.0;
    bool bestWasMirrored = false;

    for (const TemplateFeatures& templ : templates)
    {
        cv::Mat difference;
        cv::absdiff(rectifiedGray, templ.grayImage, difference);
        const double grayDifference = cv::mean(difference)[0];
        cv::absdiff(rectifiedHsv, templ.hsvImage, difference);
        const double hsvDifference = cv::mean(difference).val[0] / 3.0
                                   + cv::mean(difference).val[1] / 3.0
                                   + cv::mean(difference).val[2] / 3.0;
        double currentDifference = (grayDifference + hsvDifference) / 2.0;
        if (currentDifference < bestDifference)
        {
            bestDifference = currentDifference;
            bestGrayDifference = grayDifference;
            bestHsvDifference = hsvDifference;
            bestMatch = templ.path;
            bestWasMirrored = false;
        }

        cv::absdiff(mirroredGray, templ.grayImage, difference);
        const double mirroredGrayDifference = cv::mean(difference)[0];
        cv::absdiff(mirroredHsv, templ.hsvImage, difference);
        const double mirroredHsvDifference = cv::mean(difference).val[0] / 3.0
                                           + cv::mean(difference).val[1] / 3.0
                                           + cv::mean(difference).val[2] / 3.0;
        currentDifference = (mirroredGrayDifference + mirroredHsvDifference) / 2.0;
        if (currentDifference < bestDifference)
        {
            bestDifference = currentDifference;
            bestGrayDifference = mirroredGrayDifference;
            bestHsvDifference = mirroredHsvDifference;
            bestMatch = templ.path;
            bestWasMirrored = true;
        }
    }

    std::cout << "miglior match = " << bestMatch.stem().string() << " (grigio = " << bestGrayDifference
            << ", HSV = " << bestHsvDifference
            << ", combinata = " << bestDifference << ")" << std::endl;
    if (bestDifference > maxDifference || bestGrayDifference>60)
    {
        std::cerr << "Match scartato: differenza sopra la soglia " << maxDifference << std::endl;
        return "";
    }

    return bestMatch.stem().string();
}

std::string ReconBriscola(const cv::Mat& frame, cv::RotatedRect briscola, const fs::path templateFolder,
                          double& differenceScore)
{
    constexpr double maxDifference = 0.7;

    struct TemplateFeatures
    {
        fs::path path;
        cv::Mat grayImage;
        cv::Mat colorImage;
    };

    const cv::Size templateSize(500, 300);
    std::vector<TemplateFeatures> templates;

    for (const auto& entry : fs::directory_iterator(templateFolder))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty())
        {
            continue;
        }

            cv::resize(img, img, templateSize);
        cv::Mat templateGray;
        cv::cvtColor(img, templateGray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(templateGray, templateGray, cv::Size(3, 3), 1.2);
        cv::Mat smallTemplateGray;
        cv::pyrDown(templateGray, smallTemplateGray);
        cv::GaussianBlur(img, img, cv::Size(3, 3), 1.2);
        cv::Mat smallTemplateColor;
        cv::pyrDown(img, smallTemplateColor);
        templates.push_back({entry.path(), smallTemplateGray, smallTemplateColor});
    }

    std::array<cv::Point2f, 4> vertices;
    briscola.points(vertices.data());
    std::vector<cv::Point2f> polygon(vertices.begin(), vertices.end());

        const std::vector<cv::Point2f> ordered = orderPoints(polygon);
        const double sideA = (cv::norm(ordered[1] - ordered[0]) + cv::norm(ordered[2] - ordered[3])) / 2.0;
        const double sideB = (cv::norm(ordered[2] - ordered[1]) + cv::norm(ordered[3] - ordered[0])) / 2.0;
        const bool sideAIsShort = sideA <= sideB;
        const double cardShortSidePx = std::min(sideA, sideB);
        const double visibleSidePx = std::max(sideA, sideB);
        if (cardShortSidePx <= 0.0)
        {
            differenceScore = std::numeric_limits<double>::max();
            return "";
        }

        const double scale = templateSize.height / cardShortSidePx;
        const cv::Size patchSize(cvRound(visibleSidePx * scale), cvRound(cardShortSidePx * scale));
        if (patchSize.width <= 0 || patchSize.height <= 0 ||
            patchSize.width > templateSize.width || patchSize.height > templateSize.height)
        {
            differenceScore = std::numeric_limits<double>::max();
            return "";
        }

        cv::Mat rectified = rectangleOmography(frame, polygon, patchSize, sideAIsShort ? 1 : 0);
    if (rectified.empty())
    {
        std::cerr << "Errore: impossibile rettificare la carta." << std::endl;
        return "";
    }
    cv::imshow("Omografia briscola", rectified);

    cv::Mat rectifiedGray;
    cv::cvtColor(rectified, rectifiedGray, cv::COLOR_BGR2GRAY);
    const int marginX = static_cast<int>(rectifiedGray.cols * 0.08);
    const int marginY = static_cast<int>(rectifiedGray.rows * 0.08);
    const cv::Rect patchBounds(marginX, marginY,
        rectifiedGray.cols - 2 * marginX, rectifiedGray.rows - 2 * marginY);
    cv::Mat patch = rectifiedGray(patchBounds);
    cv::GaussianBlur(patch, patch, cv::Size(3, 3), 1.2);
    cv::Mat smallPatch;
    cv::pyrDown(patch, smallPatch);
    cv::Mat colorPatch = rectified(patchBounds);
    cv::GaussianBlur(colorPatch, colorPatch, cv::Size(3, 3), 1.2);
    cv::Mat smallColorPatch;
    cv::pyrDown(colorPatch, smallColorPatch);
    cv::Mat rotatedPatch;
    cv::rotate(smallPatch, rotatedPatch, cv::ROTATE_180);
    const std::array<cv::Mat, 2> patches = {smallPatch, rotatedPatch};
    cv::Mat rotatedColorPatch;
    cv::rotate(smallColorPatch, rotatedColorPatch, cv::ROTATE_180);
    const std::array<cv::Mat, 2> colorPatches = {smallColorPatch, rotatedColorPatch};

    fs::path bestMatch;
    double bestDifference = std::numeric_limits<double>::max();
    cv::Point bestPosition;
    int bestRotation = 0;

    for (const TemplateFeatures& templ : templates)
    {
        for (std::size_t rotation = 0; rotation < patches.size(); ++rotation)
        {
            if (patches[rotation].cols > templ.grayImage.cols || patches[rotation].rows > templ.grayImage.rows)
            {
                continue;
            }

            cv::Mat grayResult, colorResult, combinedResult;
            cv::Point currentPosition;
            cv::matchTemplate(templ.grayImage, patches[rotation], grayResult, cv::TM_SQDIFF_NORMED);
            cv::matchTemplate(templ.colorImage, colorPatches[rotation], colorResult, cv::TM_SQDIFF_NORMED);
            cv::addWeighted(grayResult, 0.5, colorResult, 0.5, 0.0, combinedResult);
            double currentDifference;
            cv::minMaxLoc(combinedResult, &currentDifference, nullptr, &currentPosition, nullptr);

            if (currentDifference < bestDifference)
            {
                bestDifference = currentDifference;
                bestMatch = templ.path;
                bestPosition = currentPosition;
                bestRotation = static_cast<int>(rotation) * 180;
            }
        }
    }

    differenceScore = bestDifference;
    std::cout << "Miglior match briscola = " << bestMatch.stem().string()
              << " (differenza = " << bestDifference
              << ", posizione = " << bestPosition.x
              << ", rotazione = " << bestRotation << " gradi)" << std::endl;
    if (bestMatch.empty() || bestDifference > maxDifference)
    {
        std::cerr << "Match briscola scartato: differenza sopra la soglia " << maxDifference << std::endl;
        return "";
    }

    return bestMatch.stem().string();
}
