#include "DatasetBuilder.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

DatasetBuilder::DatasetBuilder(const std::filesystem::path &outDir, int game, int cropEvery, int stableFrames)
    : tracker(cropEvery, stableFrames), imagesDir(outDir / "images"), game(game)
{
    std::filesystem::create_directories(imagesDir);

    // re-running the game overwrite the CSV
    const std::filesystem::path csvPath = outDir / ("labels_game" + std::to_string(game) + ".csv");
    csv.open(csvPath);
    if (!csv.is_open())
        std::cerr << "Could not create CSV: " << csvPath << std::endl;
    else
    // CSV header
        csv << "filename,class_id" << std::endl;
}

void DatasetBuilder::startRound(const RoundLabel &label)
{
    current = label;
    tracker.reset();
}

void DatasetBuilder::processFrame(const cv::Mat &frame, const std::vector<cv::RotatedRect> &rects, int motion)
{
    tracker.processFrame(frame, rects, motion);
}

std::string DatasetBuilder::leaderSide() const
{
    using Role = CardTracker::Role;
    const CardTracker::Track *first = tracker.findRole(Role::First);
    if (!first)
        return "";
    return tracker.isNorth(*first, tracker.findRole(Role::Second)) ? "north" : "south";
}

void DatasetBuilder::saveCrops(const CardTracker::Track &track, const CardLabel &label, const std::string &player)
{
    for (const auto &[frameNumber, crop] : track.crops)
    {
        std::ostringstream name;
        name << "g" << game << "_r" << std::setw(2) << std::setfill('0') << current.round
             << "_" << player << "_f" << std::setw(4) << std::setfill('0') << frameNumber << ".png";

        cv::imwrite((imagesDir / name.str()).string(), crop);

        csv << name.str() << "," << label.classId << "\n";
        savedCrops++;
    }
    csv.flush();
}

void DatasetBuilder::endRound()
{
    using Role = CardTracker::Role;
    const CardTracker::Track *first = tracker.findRole(Role::First);
    const CardTracker::Track *second = tracker.findRole(Role::Second);

    // If one of the two played cards is missing, discard the round
    if (!first || !second)
    {
        std::cout << "Round discarded, played cards found: " << (first ? 1 : 0) + (second ? 1 : 0) << "/2" << std::endl;
        rejectedRounds++;
        return;
    }

    const bool firstIsNorth = tracker.isNorth(*first, second);

    // Check: the first card played must be the leader label
    if (firstIsNorth != current.leaderNorth)
    {
        std::cout << "DISCARDED, position inconsistent with Leader ("
                  << (current.leaderNorth ? "North" : "South") << ")" << std::endl;
        rejectedRounds++;
        return;
    }

    const CardTracker::Track *north = firstIsNorth ? first : second;
    const CardTracker::Track *south = firstIsNorth ? second : first;

    saveCrops(*north, current.north, "north");
    saveCrops(*south, current.south, "south");
    validRounds++;

    std::cout << "OK, crop north " << north->crops.size() << ", south " << south->crops.size()
              << " (total " << savedCrops << ", discarded crops " << tracker.discardedCrops() << ", valid rounds " << validRounds
              << ", discarded " << rejectedRounds << ")" << std::endl;
}

int buildTrentineSet(const std::filesystem::path &scansDir, const std::filesystem::path &outDir, cv::Size size)
{
    const std::filesystem::path imagesDir = outDir / "images";
    std::filesystem::create_directories(imagesDir);

    const std::filesystem::path csvPath = outDir / "labels_trentine.csv";
    std::ofstream csv(csvPath);
    if (!csv.is_open())
    {
        std::cerr << "Could not create CSV: " << csvPath << std::endl;
        return 0;
    }
    csv << "filename,class_id" << std::endl;

    int saved = 0;
    for (const auto &entry : std::filesystem::directory_iterator(scansDir))
    {
        const std::string ext = toLower(entry.path().extension().string());
        if (!entry.is_regular_file() || (ext != ".jpg" && ext != ".png"))
            continue;

        std::string stem = entry.path().stem().string();
        std::replace(stem.begin(), stem.end(), '-', ',');
        CardLabel card;
        if (!parseCard(stem, card))
        {
            std::cerr << "Invalid scan name, skipped: " << entry.path().filename() << std::endl;
            continue;
        }

        cv::Mat image = cv::imread(entry.path().string());
        if (image.empty())
            continue;
        if (image.cols > image.rows)
            cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
        cv::resize(image, image, size, 0, 0, cv::INTER_AREA);

        const std::string name = "trentine_" + std::to_string(card.rank) + "_" + card.suitName + ".png";
        cv::imwrite((imagesDir / name).string(), image);
        csv << name << "," << card.classId << "\n";
        saved++;
    }

    std::cout << "Trentine set: " << saved << " images saved in " << imagesDir << std::endl;
    return saved;
}
