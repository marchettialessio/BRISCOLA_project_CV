#include "DatasetBuilder.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>

std::string trim(const std::string &s)
{
    const auto begin = s.find_first_not_of(" \t\r\n");
    // I'm searching first char != from " \t\r\n"
    if (begin == std::string::npos)
        return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                   { return std::tolower(c); });
    return s;
}

// method to convert suit name to index
int suitIndex(const std::string &name)
{
    if (name == "coins")
        return 0;
    if (name == "cups")
        return 1;
    if (name == "spades")
        return 2;
    if (name == "clubs")
        return 3;
    return -1;
}

// get the short side of a rectangle
float shortSide(const cv::RotatedRect &rect)
{
    return std::min(rect.size.width, rect.size.height);
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

// Converts value (ex. "6 , clubs") to CardLabel
bool parseCard(const std::string &value, CardLabel &card)
{
    const auto comma = value.find(',');
    if (comma == std::string::npos)
        return false;

    try
    {
        card.rank = std::stoi(trim(value.substr(0, comma)));
    }
    catch (...)
    {
        return false;
    }

    card.suitName = toLower(trim(value.substr(comma + 1)));
    card.suit = suitIndex(card.suitName);

    //check correctness
    if (card.rank < 1 || card.rank > 10 || card.suit < 0)
        return false;

    //simple calculation for the class ID, unique for each card
    card.classId = card.suit * 10 + (card.rank - 1);
    return true;
}

// i want to parse the label file and return the vector of RoundLabel, one for each round
std::vector<RoundLabel> parseLabels(const std::filesystem::path &file)
{
    std::vector<RoundLabel> rounds;
    std::ifstream in(file);

    if (!in.is_open())
    {
        std::cerr << "Could not open label file: " << file << std::endl;
        return rounds;
    }

    std::string line;

    // I read the file line by line
    while (std::getline(in, line))
    {
        line = trim(line);
        if (line.empty())
            continue;

        // New round
        if (line.rfind("Round", 0) == 0)
        {
            RoundLabel label;
            label.round = std::stoi(trim(line.substr(5)));
            rounds.push_back(label);
            continue;
        }

        const auto colon = line.find(':');
        if (colon == std::string::npos || rounds.empty())
            continue;

        // parse key and value, divided by the colon
        const std::string key = trim(line.substr(0, colon));
        const std::string value = trim(line.substr(colon + 1));
        RoundLabel &label = rounds.back();

        bool correct = true;

        if (key == "North")
            correct = parseCard(value, label.north);
        else if (key == "South")
            correct = parseCard(value, label.south);
        else if (key == "Briscola")
            correct = parseCard(value, label.briscola);
        else if (key == "Leader")
            label.leaderNorth = (value == "North");
        // Winner, points, total points are not needed for the dataset

        if (!correct)
            std::cerr << "Invalid label line (round " << label.round << "): " << line << std::endl;
    }

    return rounds;
}

std::filesystem::path findLabelFile(const std::filesystem::path &labelsDir, int game)
{
    const std::string prefix = "game" + std::to_string(game) + "output";

    for (const auto &entry : std::filesystem::directory_iterator(labelsDir))
    {
        const std::string name = entry.path().filename().string();
        // I check if it is regular, if the file starts with the right prefix and if the extension is .txt
        if (entry.is_regular_file() && name.rfind(prefix, 0) == 0 && entry.path().extension() == ".txt")
            return entry.path();
    }

    return {};
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

DatasetBuilder::DatasetBuilder(const std::filesystem::path &outDir, int game, int cropEvery, int stableFrames)
    : imagesDir(outDir / "images"), game(game), cropEvery(cropEvery), stableFrames(stableFrames)
{
    std::filesystem::create_directories(imagesDir);

    // re-running the game overwrite the CSV
    const std::filesystem::path csvPath = outDir / ("labels_game" + std::to_string(game) + ".csv");
    csv.open(csvPath);
    if (!csv.is_open())
        std::cerr << "Could not create CSV: " << csvPath << std::endl;
    else
    // CSV header
        csv << "filename,game,round,player,rank,suit,suit_name,class_id" << std::endl;
}

void DatasetBuilder::startRound(const RoundLabel &label)
{
    current = label;
    tracks.clear();
    frameIdx = 0;
    nextRole = 0;
}

// I want to find the track that matches the current rect, if any
int DatasetBuilder::matchTrack(const cv::RotatedRect &rect, const std::vector<bool> &used) const
{
    int best = -1;
    double bestDist = 0.0;

    for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
    {
        if (used[i])
            continue;

        const cv::RotatedRect &other = tracks[i].rect;

        // Distance between the centers of the two rectangles
        const double dist = cv::norm(rect.center - other.center);
        const double areaRatio = rect.size.area() / std::max(other.size.area(), 1.0f);

        // Same card: nearby center; tolerant area because the hand can enlarge the contour
        if (dist < 0.5 * shortSide(other) && areaRatio > 0.5 && areaRatio < 2.0)
        {
            if (best < 0 || dist < bestDist)
            {
                best = i;
                bestDist = dist;
            }
        }
    }

    return best;
}

void DatasetBuilder::processFrame(const cv::Mat &frame, const std::vector<cv::RotatedRect> &rects)
{
    // I have to undestand if a track has been matched
    std::vector<bool> used(tracks.size(), false);

    for (const auto &rect : rects)
    {
        // I search a track that matches the current rect
        const int idx = matchTrack(rect, used);

        // if no match, new track
        if (idx < 0)
        {
            // New track
            Track track;
            track.rect = rect;
            track.firstSeen = frameIdx;
            track.stillCounter = 1;
            tracks.push_back(track);
            used.push_back(true);
            continue;
        }

        used[idx] = true;
        Track &track = tracks[idx];

        // Card is still if the center moves little relative to the last seen position
        const bool still = cv::norm(rect.center - track.rect.center) < 0.05 * shortSide(track.rect);
        // Update the still counter, otherwise reset to 1
        track.stillCounter = still ? track.stillCounter + 1 : 1;
        track.missing = 0;
        track.rect = rect;

        // COonfirmation: if the card has been still for stableFrames frames, it is confirmed and assigned a role
        if (track.role == Role::Unknown && track.stillCounter >= stableFrames)
        {
            if (track.firstSeen < backgroundFrames)
                track.role = Role::Background; // already on the table at the start of the video (briscola)
            else if (nextRole == 0)
            {
                track.role = Role::First;
                nextRole++;
            }
            else if (nextRole == 1)
            {
                track.role = Role::Second;
                nextRole++;
            }
            else
                track.role = Role::Background; // extra cards (e.g. pickup at end of round)

            track.refArea = rect.size.area(); // reference area for the confirmed card
        }

        // save crop for a track every cropEvery frames for played cards, only if still
        const bool played = track.role == Role::First || track.role == Role::Second;
        // max is to guard divsision by zero, should never happen
        // ratio between current rectangle area, and first area saved
        const double areaRatio = rect.size.area() / std::max(track.refArea, 1.0f);
        const bool clean = areaRatio > 0.85 && areaRatio < 1.18; // discard contours merged with hand or other card
        if (played && clean && track.stillCounter >= stableFrames && frameIdx - track.lastSaved >= cropEvery)
        {
            cv::Mat crop = warpCard(frame, rect);
            cv::Mat thumb = thumbnail(crop);

            // The first crop acts as reference: crops too different are another card or heavily occluded
            if (track.refThumb.empty())
                track.refThumb = thumb;

            if (similarity(track.refThumb, thumb) >= minSimilarity)
                track.crops.emplace_back(frameIdx, crop);
            else
                discardedCrops++;

            track.lastSaved = frameIdx;
        }
    }

    // Tracks not seen in this frame; those without a role are removed after maxMissing frames.
    // Tracks with a role remain, so that if the card reappears it doesn't become a new card.
    for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
    {
        // A few lost frames don't reset stability
        if (!used[i] && ++tracks[i].missing > 3)
            tracks[i].stillCounter = 0;
    }

    // remove tracks without a role that have been missing for too long
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [this](const Track &t)
                                { return t.role == Role::Unknown && t.missing > maxMissing; }),
                 tracks.end());

    frameIdx++;
}

void DatasetBuilder::saveCrops(const Track &track, const CardLabel &label, const std::string &player)
{
    for (const auto &[frameNumber, crop] : track.crops)
    {
        std::ostringstream name;
        name << "g" << game << "_r" << std::setw(2) << std::setfill('0') << current.round
             << "_" << player << "_f" << std::setw(4) << std::setfill('0') << frameNumber << ".png";

        cv::imwrite((imagesDir / name.str()).string(), crop);

        csv << name.str() << "," << game << "," << current.round << "," << player << ","
            << label.rank << "," << label.suit << "," << label.suitName << "," << label.classId << "\n";
        savedCrops++;
    }
    csv.flush();
}

void DatasetBuilder::endRound()
{
    const Track *first = nullptr;
    const Track *second = nullptr;

    for (const auto &track : tracks)
    {
        if (track.role == Role::First)
            first = &track;
        else if (track.role == Role::Second)
            second = &track;
    }

    // If one of the two played cards is missing, discard the round
    if (!first || !second)
    {
        std::cout << "Round discarded, played cards found: " << (first ? 1 : 0) + (second ? 1 : 0) << "/2" << std::endl;
        rejectedRounds++;
        return;
    }

    // Spatial assignment: the topmost card is North
    const bool firstIsNorth = first->rect.center.y < second->rect.center.y;

    // Temporal check: the first card played must be the leader label
    if (firstIsNorth != current.leaderNorth)
    {
        std::cout << "DISCARDED, position inconsistent with Leader ("
                  << (current.leaderNorth ? "North" : "South") << ")" << std::endl;
        rejectedRounds++;
        return;
    }

    const Track *north = firstIsNorth ? first : second;
    const Track *south = firstIsNorth ? second : first;

    saveCrops(*north, current.north, "north");
    saveCrops(*south, current.south, "south");
    validRounds++;

    std::cout << "OK, crop north " << north->crops.size() << ", south " << south->crops.size()
              << " (total " << savedCrops << ", discarded crops " << discardedCrops << ", valid rounds " << validRounds
              << ", discarded " << rejectedRounds << ")" << std::endl;
}
