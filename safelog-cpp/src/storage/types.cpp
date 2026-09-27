#include "safelog/contracts/types.hpp"

namespace safelog {

std::string to_string(UserRole v) {
  switch (v) { case UserRole::Inspector: return "inspector"; case UserRole::Assignee: return "assignee"; case UserRole::Manager: return "manager"; }
  return "unknown";
}
std::string to_string(FindingStatus v) {
  switch (v) { case FindingStatus::Open: return "open"; case FindingStatus::InProgress: return "in_progress"; case FindingStatus::PendingReview: return "pending_review"; case FindingStatus::Verified: return "verified"; }
  return "unknown";
}
std::string to_string(PhotoKind v) {
  switch (v) { case PhotoKind::Before: return "before"; case PhotoKind::After: return "after"; }
  return "unknown";
}

} // namespace safelog
