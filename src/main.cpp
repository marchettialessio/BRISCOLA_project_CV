#include <iostream>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <regex>
#include <map>
#include "DatasetBuilder.hpp"
#include "Utility.hpp"
#include "CardDetection.hpp"
#include "MotionDetection.hpp"

// false in modalità dataset: nessuna finestra, elaborazione più veloce
bool showWindows = true;

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
        const int motion = detectMotion(subtractor, frame, kernel2, north, south, showWindows);
        if (!builder && !detected) {
            if (north >= 5)
            {
                std::cout << "Leader: North" << std::endl;
                detected = true; // Set the flag to true after the first card is detected

                if (outputFile.is_open())
                {
                    outputFile << "Leader: North" << std::endl;
                    outputFile.flush(); // Ensure the output is written to the file immediately
                    // return true; // Exit the function after writing to the output file
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
                    // To be commented in the final version
                    // return true; // Process the next round after writing to the output file
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
