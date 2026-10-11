#ifndef CARD_RECON_DIFF_HPP
#define CARD_RECON_DIFF_HPP

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

std::string ReconCard(const cv::Mat& frame, cv::RotatedRect card, const std::filesystem::path templateFolder);

#endif // CARD_RECON_DIFF_HPP