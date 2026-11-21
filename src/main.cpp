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

RoundLabel processVideo(const std::string &path, std::ofstream &outputFile, DatasetBuilder *builder, bool firstRound,
                 const std::filesystem::path &templateFolder, int round, GameLabel &gameResults,
                 std::optional<CardLabel> &briscolaCard)
{
    cv::VideoCapture video(path);
    if (!video.isOpened())
    {
        std::cerr << "Could not open the video: " << path << std::endl;
        return RoundLabel();
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
    cv::RotatedRect bestBriscolaRect;
    std::string bestBriscolaMatch;
    double bestBriscolaDifference = std::numeric_limits<double>::max();

    std::map<std::string, int> cardVotes;
    std::optional<CardLabel> firstCard;
    std::optional<CardLabel> secondCard;
    // briscolaCard is owned by main: found in the first round and kept for the following ones

    int frameIndex = 0;
    cv::namedWindow("Video Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Frame", 800, 450);

    RoundLabel roundResults;
    if (briscolaCard)
        roundResults.briscola = *briscolaCard;
    bool printedResults = false;

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
        const int motion = detectMotion(subtractor, frame, north, south);
        if (!builder && leader == "") {
            if (north >= 5)
            {
                leader = "North";
                roundResults.leaderNorth = true;
                std::cout << "Leader: " << leader << std::endl;
            }
            else if (south >= 5)
            {
                leader = "South";
                roundResults.leaderNorth = false;
                std::cout << "Leader: " << leader << std::endl;
            }
        }

        cv::Mat recognitionFrame = frame.clone();
        cv::RotatedRect candidateBriscola;
        const bool briscolaCandidateFound = firstRound && frameIndex < 5 && !briscolaCard
                                         && findBriscola(recognitionFrame, grayFrame, kernel, candidateBriscola);

        // Find cards in the current frame
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
            if (recognized == "" || firstCard && recognized == firstCard->name())
                continue;

            ++cardVotes[recognized];
            std::cout << "Voto per " << recognized << ": " << cardVotes[recognized] << std::endl;

            if (cardVotes[recognized] >= 3)
            {
                if (!firstCard)
                {
                    // Create the first card object and assign it to the appropriate player
                    const CardLabel card = cardFromName(recognized);
                    if (!card.valid()) {
                        std::cerr << "Invalid card name: " << recognized << std::endl;
                        continue;
                    }
                    firstCard = card;
                    std::cout << "Prima carta riconosciuta: " << firstCard->toString()
                              << " (classId " << firstCard->classId
                              << ", points " << firstCard->points() << ")" << std::endl;

                    if (leader == "North") {
                        roundResults.north = *firstCard;
                        std::cout << "Assigned first card to North player" << std::endl;
                    } else if (leader == "South") {
                        roundResults.south = *firstCard;
                        std::cout << "Assigned first card to South player" << std::endl;
                    }

                    cardVotes.clear();
                }
                else
                {
                    // Create the second card object and assign it to the appropriate player
                    const CardLabel card = cardFromName(recognized);
                    if (!card.valid()) {
                        std::cerr << "Invalid card name: " << recognized << std::endl;
                        continue;
                    }
                    secondCard = card;
                    std::cout << "Seconda carta riconosciuta: " << secondCard->toString()
                              << " (classId " << secondCard->classId
                              << ", points " << secondCard->points() << ")" << std::endl;

                    if (leader == "North") {
                        roundResults.south = *secondCard;
                        std::cout << "Assigned second card to North player" << std::endl;
                    } else if (leader == "South") {
                        roundResults.north = *secondCard;
                        std::cout << "Assigned second card to South player" << std::endl;
                    }
                }
            }
        }
        if (firstRound && frameIndex < 5 && !briscolaCard)
        {
            if (briscolaCandidateFound)
            {
                double candidateDifference;
                const std::string candidateMatch = ReconBriscola(
                    recognitionFrame, candidateBriscola, templateFolder, candidateDifference);
                if (!candidateMatch.empty() && candidateDifference < bestBriscolaDifference)
                {
                    bestBriscolaDifference = candidateDifference;
                    bestBriscolaMatch = candidateMatch;
                    bestBriscolaRect = candidateBriscola;
                }
            }

            if (frameIndex == 4)
            {
                if (!bestBriscolaMatch.empty())
                {
                    briscolaCard = cardFromName(bestBriscolaMatch);
                    std::cout << "Briscola riconosciuta: " << briscolaCard->toString()
                              << " (differenza = " << bestBriscolaDifference << ")" << std::endl;
                    roundResults.briscola = *briscolaCard;
                    drawRectangles(frame, {bestBriscolaRect});
                }
                else
                {
                    std::cout << "Briscola non trovata nei primi 5 frame." << std::endl;
                }
            }
        }
        // TODO: delete this in the final version, it's just for testing, to skip to the next round without visualizing all the frames
        if (secondCard && briscolaCard && !printedResults) {
            getRoundResults(round, roundResults, gameResults, leader, *firstCard, *secondCard, *briscolaCard);
            if(!outputFile.is_open()) {
                std::cerr << "Unable to open output file." << std::endl;
            } else {
                outputFile << "Round " << round << '\n'
                           << "North : " << roundResults.north.toString() << '\n'
                           << "South : " << roundResults.south.toString() << '\n'
                           << "Briscola : " << briscolaCard->toString() << '\n'
                           << "Leader : " << leader << '\n'
                           << "Winner : " << (roundResults.winnerNorth ? "North" : "South") << '\n'
                           << "Points : " << roundResults.points << std::endl;

                std::cout << "Printed round results" << std::endl;
            }

            printedResults = true;
            break;
        }

        ++frameIndex;
        cv::imshow("Video Frame", frame);
        // Press ESC to skip video
        if (cv::waitKey(builder ? 1 : 30) == 27)
            break;
        
    }
    // TODO: in the final version, delete printedResults, beacuse you print the result only at the end of the video
    if (!printedResults && !(firstCard && secondCard && briscolaCard)) {
        std::cerr << "Round " << round << ": missing first card, second card or briscola, results not computed" << std::endl;
    }
    else if (!printedResults) {
            getRoundResults(round, roundResults, gameResults, leader, *firstCard, *secondCard, *briscolaCard);
            if(!outputFile.is_open()) {
                std::cerr << "Unable to open output file." << std::endl;
            } else {
                outputFile << "Round " << round << '\n'
                           << "North : " << roundResults.north.toString() << '\n'
                           << "South : " << roundResults.south.toString() << '\n'
                           << "Briscola : " << briscolaCard->toString() << '\n'
                           << "Leader : " << leader << '\n'
                           << "Winner : " << (roundResults.winnerNorth ? "North" : "South") << '\n'
                           << "Points : " << roundResults.points << std::endl;

                std::cout << "Printed round results" << std::endl;
            }

            printedResults = true;
        }

    video.release();
    cv::destroyAllWindows();
    return roundResults;
}

int main(int argc, char **argv)
{
    // TODO: change the path of the videoFolder, maybe put the video in GitHub realeses
    // TODO: output and statistics files should be CSV files
    // TODO: maybe stop the detection of the motion after the first card is detected

    const std::filesystem::path root(PROJECT_SOURCE_DIR);

    // if I want to build the Trentine dataset
    if (argc > 1 && std::string(argv[1]) == "--trentine")
        return buildTrentineSet(root / "Briscola" / "Briscola_Trentine", root / "dataset") > 0 ? 0 : 1;

    const bool datasetMode = argc > 2 && std::string(argv[2]) == "--dataset";
    showWindows = !datasetMode;

    const bool statisticsMode = argc > 2 && std::string(argv[2]) == "--statistics";

    std::string game = "1";
    if (argc > 1 && checkInteger(argv[1])) {
        game = argv[1];
    }

    std::filesystem::path videoFolder = root / "Briscola" / ("game" + game);
    std::filesystem::path outputFilePath = root / "output" / ("game" + game + "output.txt");
    std::filesystem::path statisticsFilePath = root / "output" / ("game" + game + "statistics.txt");

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

    // load game labels, indexed by round, needed for dataset building mode
    std::map<int, RoundLabel> roundLabels;    
    GameLabel gameLabel;

    if (statisticsMode || datasetMode)
    {
        // I find the label file for the selected game, it is useful both for the dataset building and both for the final results printing
        const std::filesystem::path labelFile = findLabelFile(root / "labels", std::stoi(game));
        if (labelFile.empty())
        {
            std::cerr << "No label file for game " << game << std::endl;
            return 1;
        }

        // I parse all the labels for each round and store them
        gameLabel = parseGame(labelFile);
        for (const auto &label : gameLabel.rounds) {
            roundLabels[label.round] = label;
        }

        if (datasetMode) {
            builder = std::make_unique<DatasetBuilder>(root / "dataset", std::stoi(game));
        }
    }
    
    std::filesystem::path templateFolder = root / "resources" / "cards";

    bool firstRound = true;
    int round = 0;
    GameLabel gameResults; // Store the results for the entire game
    std::optional<CardLabel> briscolaCard; // Detected in the first round, shared by all rounds

    RoundLabel results; // Store the results for each round
    int correctCards = 0;
    int correctPlayers = 0;
    int correctBriscola = 0;

    for (const auto &video : videos)
    {
        round = roundNumber(video);

        if (builder)
        {
            if (!roundLabels.count(round))
            {
                std::cerr << "Missing label for round " << round << ", video skipped" << std::endl;
                continue;
            }
            //i start to build the dataset for the current round
            builder->startRound(roundLabels[round]);
        }

        std::cout << "Processing video: " << video << std::endl;
        results = processVideo(video.string(), outputFile, builder.get(), firstRound, templateFolder, round, gameResults, briscolaCard);

        if (statisticsMode)
        {
            // Compare results with labels[round] and increment the counters for correct results
            countCorrectResults(roundLabels[round], results, correctCards, correctPlayers, correctBriscola);
        }
        
        firstRound = false;

        if (builder)
            builder->endRound();

        // Press ESC to stop
        if (cv::waitKey(builder ? 1 : 1000) == 27)
            break;
    }

    std::string winner = (gameResults.totalNorth > gameResults.totalSouth) ? "North" : "South";
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open output file." << std::endl;
    } else {
        outputFile << "----" << '\n'
                    << "Winner : " << winner << '\n'
                    << "Total Points South: " << gameResults.totalSouth << '\n'
                    << "Total Points North: " << gameResults.totalNorth << std::endl;

        std::cout << "Printed final results" << std::endl;
    }

    if (statisticsMode) {
        // Count correct game results (final winner, points north, points south)
        // Compute statistics for the game
        std::ofstream statisticsFile(statisticsFilePath);
        computeGameStatistics(gameLabel, winner, gameResults, correctCards, correctPlayers, correctBriscola, statisticsFile);
    }

    return 0;
}