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
std::string to_string(AiAnalysisType v) {
  switch (v) { case AiAnalysisType::BeforeHazard: return "before_hazard"; case AiAnalysisType::AfterComparison: return "after_comparison"; case AiAnalysisType::ReportSummary: return "report_summary"; }
  return "unknown";
}
std::string to_string(AiReviewDecision v) {
  switch (v) { case AiReviewDecision::Pending: return "pending"; case AiReviewDecision::Accepted: return "accepted"; case AiReviewDecision::Edited: return "edited"; case AiReviewDecision::Rejected: return "rejected"; }
  return "unknown";
}

} // namespace safelog
