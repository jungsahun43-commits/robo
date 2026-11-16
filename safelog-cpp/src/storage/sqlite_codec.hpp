#pragma once

// Private conversions between contract types and SQLite column text.
// Not part of the public storage API.

#include "safelog/contracts/types.hpp"

#include <string>
#include <string_view>

namespace safelog::storage::detail {

// UTC ISO-8601 with a fixed 9-digit fraction, e.g. "2026-11-16T09:30:00.123456700Z".
// Fixed width keeps lexical order equal to chronological order in ORDER BY.
std::string formatTime(TimePoint value);
// Accepts "YYYY-MM-DDTHH:MM:SS[.fraction]Z". Throws std::runtime_error on malformed text.
TimePoint parseTime(std::string_view text);

// Inverses of safelog::to_string. Throw std::runtime_error for unknown text.
UserRole parseUserRole(std::string_view text);
FindingStatus parseFindingStatus(std::string_view text);
PhotoKind parsePhotoKind(std::string_view text);
AiAnalysisType parseAiAnalysisType(std::string_view text);
AiReviewDecision parseAiReviewDecision(std::string_view text);

} // namespace safelog::storage::detail
