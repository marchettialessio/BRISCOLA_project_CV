#ifndef CARD_RECON_DIFF_HPP
#define CARD_RECON_DIFF_HPP

#include "Utility.hpp"

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

std::string ReconCard(const cv::Mat& frame, cv::RotatedRect card, const std::filesystem::path templateFolder);
std::string ReconBriscola(const cv::Mat& frame, cv::RotatedRect briscola, const std::filesystem::path templateFolder,
						  double& difference);

#endif // CARD_RECON_DIFF_HPP