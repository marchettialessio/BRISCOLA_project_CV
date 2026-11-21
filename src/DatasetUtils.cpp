#include "DatasetUtils.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>

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
