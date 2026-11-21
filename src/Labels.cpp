#include "Labels.hpp"

#include <algorithm>
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

CardLabel cardFromClassId(int classId)
{
    static const char *suitNames[] = {"coins", "cups", "spades", "clubs"};

    CardLabel card;
    if (classId < 0 || classId >= 40)
        return card; // invalid card

    card.classId = classId;
    card.suit = classId / 10;
    card.rank = classId % 10 + 1;
    card.suitName = suitNames[card.suit];
    return card;
}

CardLabel cardFromName(const std::string &name)
{
    std::string value = name;
    std::replace(value.begin(), value.end(), '-', ',');

    CardLabel card;
    if (!parseCard(value, card))
        return CardLabel{}; // invalid card
    return card;
}

int CardLabel::points() const
{
    switch (rank)
    {
    case 1:
        return 11;
    case 3:
        return 10;
    case 8:
        return 2;
    case 9:
        return 3;
    case 10:
        return 4;
    default:
        return 0;
    }
}

std::string CardLabel::name() const
{
    return std::to_string(rank) + "-" + suitName;
}

std::string CardLabel::toString() const
{
    return std::to_string(rank) + " , " + suitName;
}

// i want to parse the label file and return the game: one RoundLabel for each round + final result
GameLabel parseGame(const std::filesystem::path &file)
{
    GameLabel game;
    std::vector<RoundLabel> &rounds = game.rounds;
    std::ifstream in(file);

    if (!in.is_open())
    {
        std::cerr << "Could not open label file: " << file << std::endl;
        return game;
    }

    std::string line;
    // after the "----" line the file reports the final result of the game
    bool finalResult = false;

    // I read the file line by line
    while (std::getline(in, line))
    {
        line = trim(line);
        if (line.empty())
            continue;

        if (line.rfind("--", 0) == 0)
        {
            finalResult = true;
            continue;
        }

        // New round
        if (line.rfind("Round", 0) == 0)
        {
            RoundLabel label;
            label.round = std::stoi(trim(line.substr(5)));
            rounds.push_back(label);
            continue;
        }

        const auto colon = line.find(':');
        if (colon == std::string::npos)
            continue;

        // parse key and value, divided by the colon
        const std::string key = trim(line.substr(0, colon));
        const std::string value = trim(line.substr(colon + 1));

        if (finalResult)
        {
            if (key == "Winner")
                game.winnerNorth = (value == "North");
            else if (key == "Total Points North")
                game.totalNorth = std::stoi(value);
            else if (key == "Total Points South")
                game.totalSouth = std::stoi(value);
            continue;
        }

        if (rounds.empty())
            continue;
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
        else if (key == "Winner")
            label.winnerNorth = (value == "North");
        else if (key == "Points")
            label.points = std::stoi(value);

        if (!correct)
            std::cerr << "Invalid label line (round " << label.round << "): " << line << std::endl;
    }

    return game;
}

std::vector<RoundLabel> parseLabels(const std::filesystem::path &file)
{
    return parseGame(file).rounds;
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

