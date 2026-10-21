#include <iostream>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <regex>
#include <map>
#include "DatasetBuilder.hpp"
#include "Utility.hpp"
#include "CardReconDiff.hpp"
#include "CardDetection.hpp"
#include "MotionDetection.hpp"

bool showWindows = true; // Global variable to control window display

bool processVideo(const std::string &path, std::ofstream &outputFile, DatasetBuilder *builder, bool firstRound,
                 const std::filesystem::path &templateFolder, int round, Player &playerNorth, Player &playerSouth)
{
    cv::VideoCapture video(path);
    if (!video.isOpened())
    {
        std::cerr << "Could not open the video: " << path << std::endl;
        return false;
    }

    cv::Mat frame, grayFrame;
    // kernel used for card detection, used to link edges (CLOSING)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // Background subtractor for motion detection
    cv::Ptr<cv::BackgroundSubtractorMOG2> subtractor = cv::createBackgroundSubtractorMOG2(500, 16, true);
    int north = 0, south = 0; // Votes for the first card detection
    std::string leader; // Leader side (North or South)

    bool briscolaFixed = false;
    cv::RotatedRect briscola;

    std::map<std::string, int> cardVotes;
    std::optional<Card> firstCard;
    std::optional<Card> secondCard;

    int frameIndex = 0;
    cv::namedWindow("Video Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Frame", 800, 450);

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
        const int motion = detectMotion(subtractor, frame, north, south, showWindows);
        if (!builder && leader == "") {
            if (north >= 5)
            {
                leader = "North";
                std::cout << "Leader: " << leader << std::endl;
            }
            else if (south >= 5)
            {
                leader = "South";
                std::cout << "Leader: " << leader << std::endl;
            }
        }

        if (firstRound && frameIndex < 5 && !briscolaFixed)
        {
            briscolaFixed = findBriscola(frame, grayFrame, kernel, briscola);
            if (frameIndex == 4 && !briscolaFixed)
            {
                std::cout << "Briscola non trovata nei primi 5 frame." << std::endl;
            }
        }

        // Find cards in the current frame
        cv::Mat recognitionFrame = frame.clone();
        std::vector<cv::RotatedRect> finalRect = findCards(frame, grayFrame, kernel, builder, motion);
        if (briscolaFixed && frameIndex < 5)
        {
            drawRectangles(frame, {briscola});
            const cv::Rect bounds = briscola.boundingRect();
            std::cout << "Briscola trovata al frame " << frameIndex + 1 << std::endl;
        }

        for (const auto &rect : finalRect)
        {
            std::string recognized = ReconCard(recognitionFrame, rect, templateFolder);
            if (recognized == "" || firstCard && recognized == firstCard->id)
                continue;

            ++cardVotes[recognized];
            std::cout << "Voto per " << recognized << ": " << cardVotes[recognized] << std::endl;

            if (cardVotes[recognized] >= 3)
            {
                if (!firstCard)
                {
                    // Create the first card object and assign it to the appropriate player
                    try {
                        firstCard = createCard(recognized);
                        std::cout << "Prima carta riconosciuta: " << secondCard->id 
                                  << " Number: " << secondCard->number 
                                  << " Seed: " << secondCard->seed 
                                  << " Points: " << secondCard->points << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << e.what() << std::endl;
                    }

                    if (leader == "North") {
                        playerNorth.card = *firstCard;
                        std::cout << "Assigned first card to North player" << std::endl;
                    } else if (leader == "South") {
                        playerSouth.card = *firstCard;
                        std::cout << "Assigned first card to South player" << std::endl;
                    }

                    cardVotes.clear();
                }
                else
                {
                    // Create the second card object and assign it to the appropriate player
                    try {
                        secondCard = createCard(recognized);
                        std::cout << "Seconda carta riconosciuta: " << secondCard->id 
                                  << " Number: " << secondCard->number 
                                  << " Seed: " << secondCard->seed 
                                  << " Points: " << secondCard->points << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << e.what() << std::endl;
                    }

                    if (leader == "North") {
                        playerSouth.card = *secondCard;
                        std::cout << "Assigned second card to North player" << std::endl;
                    } else if (leader == "South") {
                        playerNorth.card = *secondCard;
                        std::cout << "Assigned second card to South player" << std::endl;
                    }
                }
            }
        }
        if (secondCard) {
            RoundResults results = getRoundResults(round, leader, *firstCard, *secondCard, playerNorth.points, playerSouth.points);
            if(!outputFile.is_open()) {
                std::cerr << "Unable to open output file." << std::endl;
            } else {
                outputFile << "Round " << round << '\n'
                           << "North : " << playerNorth.card->number << " , " << playerNorth.card->seed << '\n'
                           << "South : " << playerSouth.card->number << " , " << playerSouth.card->seed << '\n'
                           << "Briscola : " << '\n'
                           << "Leader : " << leader << '\n'
                           << "Winner : " << results.winner << '\n'
                           << "Points : " << results.points << std::endl;

                std::cout << "Printed round results" << std::endl;
            }

            // TODO: delete this in the final version, it's just for testing
            playerNorth.card = {};
            playerSouth.card = {};
            break;
        }

        playerNorth.card = {};
        playerSouth.card = {};

        ++frameIndex;
        cv::imshow("Video Frame", frame);
        // Press ESC to skip video
        if (cv::waitKey(builder ? 1 : 30) == 27)
            break;
        
    }

    video.release();
    cv::destroyAllWindows();
    return true;
}

int main(int argc, char **argv)
{
    const std::filesystem::path root(PROJECT_SOURCE_DIR);

    if (argc > 1 && std::string(argv[1]) == "--trentine")
        return buildTrentineSet(root / "Briscola" / "Briscola_Trentine", root / "dataset") > 0 ? 0 : 1;

    const bool datasetMode = argc > 2 && std::string(argv[2]) == "--dataset";
    showWindows = !datasetMode;

    std::string game = "1";
    if (argc > 1 && checkInteger(argv[1])) {
        game = argv[1];
    }
    std::filesystem::path videoFolder = root / "Briscola" / ("game" + game);
    std::filesystem::path outputFilePath = root / "output" / ("game" + game + "output.txt");

    std::ofstream outputFile(outputFilePath);

    std::vector<std::filesystem::path> videos;
    for (const auto &video : std::filesystem::directory_iterator(videoFolder))
    {
        if (!video.is_regular_file() || video.path().extension() != ".mp4")
            continue;
        videos.push_back(video.path());
    }
    // ordering videos by round number
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
    
    std::filesystem::path templateFolder = root / "resources" / "cards";

    bool firstRound = true;
    int round = 0;
    Player playerNorth = {"North", 0, {}};
    Player playerSouth = {"South", 0, {}};

    for (const auto &video : videos)
    {
        round = roundNumber(video);

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
        if (!processVideo(video.string(), outputFile, builder.get(), firstRound, templateFolder, round, playerNorth, playerSouth))
            break;
        
        firstRound = false;

        if (builder)
            builder->endRound();

        // Press ESC to stop
        if (cv::waitKey(builder ? 1 : 1000) == 27)
            break;
    }

    if (!outputFile.is_open()) {
            std::cerr << "Unable to open output file." << std::endl;
        } else {
            std::string winner = (playerNorth.points > playerSouth.points) ? "North" : "South";
            outputFile << "----" << '\n'
                       << "Winner : " << winner << '\n'
                       << "Total Points South: " << playerSouth.points << '\n'
                       << "Total Points North: " << playerNorth.points << std::endl;

            std::cout << "Printed final results" << std::endl;
        }

    return 0;
}