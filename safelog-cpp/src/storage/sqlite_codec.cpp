#include "sqlite_codec.hpp"

#include <chrono>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>

namespace safelog::storage::detail {
namespace {

using namespace std::chrono;

[[noreturn]] void badTime(std::string_view text) {
  throw std::runtime_error("Invalid timestamp in database: '" + std::string(text) + "'");
}

// Parses exactly `count` decimal digits starting at `pos`.
long long digits(std::string_view text, std::size_t pos, std::size_t count) {
  if (pos + count > text.size()) badTime(text);
  long long value = 0;
  for (std::size_t i = pos; i < pos + count; ++i) {
    const char c = text[i];
    if (c < '0' || c > '9') badTime(text);
    value = value * 10 + (c - '0');
  }
  return value;
}

void expect(std::string_view text, std::size_t pos, char c) {
  if (pos >= text.size() || text[pos] != c) badTime(text);
}

template <typename E>
E parseEnum(std::string_view text, std::initializer_list<E> values, const char* typeName) {
  for (const E value : values)
    if (to_string(value) == text) return value;
  throw std::runtime_error(std::string("Unknown ") + typeName + " in database: '" + std::string(text) + "'");
}

} // namespace

std::string formatTime(TimePoint value) {
  const auto day = floor<days>(value);
  const year_month_day ymd{day};
  const auto ns = duration_cast<nanoseconds>(value - day);  // [0, 24h)
  const auto secs = duration_cast<seconds>(ns);
  const long long total = secs.count();
  char buffer[48];
  std::snprintf(buffer, sizeof buffer, "%04d-%02u-%02uT%02lld:%02lld:%02lld.%09lldZ",
                static_cast<int>(ymd.year()), static_cast<unsigned>(ymd.month()),
                static_cast<unsigned>(ymd.day()), total / 3600, total / 60 % 60, total % 60,
                static_cast<long long>((ns - secs).count()));
  return buffer;
}

TimePoint parseTime(std::string_view text) {
  // YYYY-MM-DDTHH:MM:SS
  const auto y = digits(text, 0, 4);
  expect(text, 4, '-');
  const auto mo = digits(text, 5, 2);
  expect(text, 7, '-');
  const auto d = digits(text, 8, 2);
  expect(text, 10, 'T');
  const auto h = digits(text, 11, 2);
  expect(text, 13, ':');
  const auto mi = digits(text, 14, 2);
  expect(text, 16, ':');
  const auto s = digits(text, 17, 2);

  std::size_t pos = 19;
  long long fractionNs = 0;
  if (pos < text.size() && text[pos] == '.') {
    ++pos;
    const std::size_t start = pos;
    long long scale = 100'000'000;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
      fractionNs += (text[pos] - '0') * scale;  // digits beyond nanoseconds are truncated
      scale /= 10;
      ++pos;
    }
    if (pos == start) badTime(text);
  }
  expect(text, pos, 'Z');
  if (pos + 1 != text.size()) badTime(text);

  const year_month_day ymd{year{static_cast<int>(y)}, month{static_cast<unsigned>(mo)},
                           day{static_cast<unsigned>(d)}};
  if (!ymd.ok() || h > 23 || mi > 59 || s > 60) badTime(text);
  const auto exact = sys_days{ymd} + hours{h} + minutes{mi} + seconds{s} + nanoseconds{fractionNs};
  return time_point_cast<TimePoint::duration>(exact);
}

UserRole parseUserRole(std::string_view text) {
  return parseEnum(text, {UserRole::Inspector, UserRole::Assignee, UserRole::Manager}, "user role");
}

FindingStatus parseFindingStatus(std::string_view text) {
  return parseEnum(text, {FindingStatus::Open, FindingStatus::InProgress, FindingStatus::PendingReview,
                          FindingStatus::Verified}, "finding status");
}

PhotoKind parsePhotoKind(std::string_view text) {
  return parseEnum(text, {PhotoKind::Before, PhotoKind::After}, "photo kind");
}

AiAnalysisType parseAiAnalysisType(std::string_view text) {
  return parseEnum(text, {AiAnalysisType::BeforeHazard, AiAnalysisType::AfterComparison,
                          AiAnalysisType::ReportSummary}, "AI analysis type");
}

AiReviewDecision parseAiReviewDecision(std::string_view text) {
  return parseEnum(text, {AiReviewDecision::Pending, AiReviewDecision::Accepted, AiReviewDecision::Edited,
                          AiReviewDecision::Rejected}, "AI review decision");
}

} // namespace safelog::storage::detail
