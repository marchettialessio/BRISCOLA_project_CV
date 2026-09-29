#include "CardTracker.hpp"

#include "CardGeometry.hpp"

#include <algorithm>
#include <iostream>

CardTracker::CardTracker(int cropEvery, int stableFrames)
    : cropEvery(cropEvery), stableFrames(stableFrames)
{
}

void CardTracker::reset()
{
    tracks.clear();
    frameIdx = 0;
    nextRole = 0;
    briscolaFound = false;
}

int CardTracker::matchTrack(const cv::RotatedRect &rect, const std::vector<bool> &used) const
{
    int best = -1;
    double bestDist = 0.0;

    for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
    {
        if (used[i])
            continue;

        const cv::RotatedRect &other = tracks[i].rect;
        const bool confirmed = tracks[i].role != Role::Unknown;

        // Confirmed cards are still: compare with the confirmation center and a tight radius,
        // otherwise the second card, placed overlapping, gets absorbed by the first card's track
        const cv::Point2f reference = confirmed ? tracks[i].anchor : other.center;
        const double maxDist = (confirmed ? anchorRadius : 0.5) * shortSide(other);

        // Distance between the centers of the two rectangles
        const double dist = cv::norm(rect.center - reference);
        const double areaRatio = rect.size.area() / std::max(other.size.area(), 1.0f);

        // Same card: nearby center; tolerant area because the hand can enlarge the contour
        if (dist < maxDist && areaRatio > 0.5 && areaRatio < 2.0)
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

void CardTracker::processFrame(const cv::Mat &frame, const std::vector<cv::RotatedRect> &rects, int motion)
{
    frameHeight = frame.rows;

    // I have to undestand if a track has been matched
    std::vector<bool> used(tracks.size(), false);

    for (const auto &rect : rects)
    {
        // I search a track that matches the current rect
        const int idx = matchTrack(rect, used);

        // if no match, new track
        if (idx < 0)
        {
            // Duplicate of a card already seen in this frame (outer border + inner frame)
            bool duplicate = false;
            for (int j = 0; j < static_cast<int>(tracks.size()); ++j)
                if (used[j] && cv::norm(rect.center - tracks[j].rect.center) < 0.5 * shortSide(tracks[j].rect))
                    duplicate = true;
            // Rectangle slightly shifted from a confirmed card: it is the same card, not a new one
            for (const auto &other : tracks)
                if (other.role != Role::Unknown && cv::norm(rect.center - other.anchor) < noSpawnRadius * shortSide(other.rect))
                    duplicate = true;
            if (duplicate)
                continue;

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

        // Until the card is still it is being placed: the motion tells which side it comes from
        if (track.role == Role::Unknown)
        {
            if (motion > 0)
                track.northVotes++;
            else if (motion < 0)
                track.southVotes++;
        }

        // Card is still if the center moves little relative to the last seen position
        const bool still = cv::norm(rect.center - track.rect.center) < 0.05 * shortSide(track.rect);
        // Update the still counter, otherwise reset to 1
        track.stillCounter = still ? track.stillCounter + 1 : 1;
        track.missing = 0;
        track.rect = rect;

        // COonfirmation: if the card has been still for stableFrames frames, it is confirmed and assigned a role
        if (track.role == Role::Unknown && track.stillCounter >= stableFrames)
        {
            if (isHorizontal(rect))
            {
                // briscola lies sideways under the deck; only one per round
                track.role = briscolaFound ? Role::Background : Role::Briscola;
                briscolaFound = true;
            }
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
            track.anchor = rect.center;
            std::cout << "DBG confirm f" << frameIdx << " role " << static_cast<int>(track.role) << " born f" << track.firstSeen
                      << " center " << rect.center << " size " << rect.size << " votes N" << track.northVotes << " S" << track.southVotes << std::endl;
        }

        // save crop for a track every cropEvery frames for played cards and briscola, only if still
        const bool keepCrops = track.role == Role::First || track.role == Role::Second || track.role == Role::Briscola;
        // max is to guard divsision by zero, should never happen
        // ratio between current rectangle area, and first area saved
        const double areaRatio = rect.size.area() / std::max(track.refArea, 1.0f);
        const bool clean = areaRatio > 0.85 && areaRatio < 1.18; // discard contours merged with hand or other card
        if (keepCrops && clean && track.stillCounter >= stableFrames && frameIdx - track.lastSaved >= cropEvery)
        {
            cv::Mat crop = warpCard(frame, rect);
            cv::Mat thumb = thumbnail(crop);

            // The first crop acts as reference: crops too different are another card or heavily occluded
            if (track.refThumb.empty())
                track.refThumb = thumb;

            if (similarity(track.refThumb, thumb) >= minSimilarity)
                track.crops.emplace_back(frameIdx, crop);
            else
                discarded++;

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

const CardTracker::Track *CardTracker::findRole(Role role) const
{
    for (const auto &track : tracks)
        if (track.role == role)
            return &track;
    return nullptr;
}

// Card side: detectMotion votes collected while it was being placed 
bool CardTracker::isNorth(const Track &track, const Track *other) const
{
    if (std::abs(track.northVotes - track.southVotes) >= minVoteMargin)
        return track.northVotes > track.southVotes;
        //not enogh votes in first
    if (other && std::abs(other->northVotes - other->southVotes) >= minVoteMargin)
        return other->southVotes > other->northVotes;
        // not enough votes in second, use position to decide
    if (other)
        return track.rect.center.y < other->rect.center.y;
    return track.rect.center.y < frameHeight / 2.0f;
}
