#pragma once
#include "safelog/contracts/ports.hpp"
#include "safelog/contracts/errors.hpp"
#include "safelog/ai/ai_analyzer.hpp"
#include "safelog/workflow/workflow_service.hpp"
#include "adapters/application_support.hpp"
#include <algorithm>
#include <cmath>
namespace safelog::qtapp {
inline bool canViewFinding(const IRepository& repository, const Finding& finding, const Id& userId) {
  const auto user = repository.findProfile(userId);
  if (!user) return false;
  if (user->role == UserRole::Manager) return true;
  if (user->role == UserRole::Assignee) return finding.assigneeId == userId;
  const auto inspection = repository.findInspection(finding.inspectionId);
  return inspection && inspection->inspectorId == userId;
}
inline std::optional<Photo> latestPhoto(const IRepository& repository, const Id& findingId, PhotoKind kind) {
  std::optional<Photo> latest;
  for (const auto& photo : repository.photosForFinding(findingId))
    if (photo.kind == kind && (!latest || photo.capturedAt > latest->capturedAt ||
        (photo.capturedAt == latest->capturedAt && photo.id > latest->id))) latest = photo;
  return latest;
}
inline std::string latestAction(const IRepository& repository, const Id& findingId) {
  std::optional<ActionLog> latest;
  for (const auto& log : repository.logsForFinding(findingId))
    if (log.action == "action_submitted" && (!latest || log.createdAt > latest->createdAt ||
        (log.createdAt == latest->createdAt && log.id > latest->id))) latest = log;
  return latest ? latest->note : "";
}
inline AiAnalysis storeComparison(IRepository& repository, IClock& clock, IIdGenerator& ids,
                                  const Id& findingId, const ai::ActionAssessment& result, const Id& actorId) {
  const auto finding = repository.findFinding(findingId);
  if (!finding) throw NotFoundError("Finding not found");
  if (!canViewFinding(repository, *finding, actorId)) throw ValidationError("PermissionDenied: comparison actor cannot view finding");
  if (finding->status != FindingStatus::PendingReview) throw TransitionError("Comparison needs pending review");
  if (!std::isfinite(result.confidence) || result.confidence < 0 || result.confidence > 1 ||
      result.assessment.empty() || result.modelName.empty() || result.promptVersion.empty())
    throw ValidationError("Invalid comparison result");
  AiAnalysis analysis{ids.next("ai"), findingId, AiAnalysisType::AfterComparison, result.modelName,
    result.promptVersion, std::nullopt, result.likelyResolved ? "likely_resolved" : "remaining_risk",
    result.confidence, result.rawJson, AiReviewDecision::Pending, std::nullopt, clock.now()};
  repository.saveAiAnalysis(analysis);
  // The mock's raw JSON is intentionally minimal; retain its typed assessment without changing contracts.
  std::string note = result.assessment;
  for (const auto& risk : result.remainingRisks) note += "\n" + risk;
  repository.saveActionLog({ids.next("log"), findingId, actorId, "ai_comparison_suggestion", note, clock.now()});
  return analysis;
}
inline Finding finalizeReview(IRepository& repository, workflow::WorkflowService& workflow,
                              ai::AiSafetyService& reviews, const Id& findingId, const Id& inspectorId,
                              const Id& analysisId, std::optional<AiReviewDecision> decision,
                              const std::string& note, bool photosConfirmed) {
  if (!canReviewFinding(repository, findingId, inspectorId))
    throw ValidationError("PermissionDenied: only the original inspector can verify");
  const auto finding = repository.findFinding(findingId).value();
  if (finding.status != FindingStatus::PendingReview) throw TransitionError("Only pending review can be verified");
  if (!photosConfirmed || note.find_first_not_of(" \t\r\n") == std::string::npos)
    throw ValidationError("Confirm both photos and enter the final human judgment");
  const auto after = latestPhoto(repository, findingId, PhotoKind::After);
  if (!latestPhoto(repository, findingId, PhotoKind::Before) || !after)
    throw ValidationError("MissingActionData: both photos are required");
  if (decision && *decision != AiReviewDecision::Accepted && *decision != AiReviewDecision::Rejected)
    throw ValidationError("Invalid AI comparison review decision");
  if (analysisId.empty() && decision) throw ValidationError("No AI result; choose direct human review");
  if (!analysisId.empty()) {
    const auto analysis = repository.findAiAnalysis(analysisId);
    if (!analysis || analysis->subjectId != findingId || analysis->type != AiAnalysisType::AfterComparison ||
        analysis->decision != AiReviewDecision::Pending || analysis->createdAt < after->capturedAt)
      throw ValidationError("Comparison result is missing, reviewed, or stale");
    reviews.reviewAnalysis(analysisId, inspectorId, decision.value_or(AiReviewDecision::Rejected));
  }
  return workflow.verify(findingId, inspectorId,
    std::string("사람 최종 판단 [") + (decision ? to_string(*decision) : "manual") + "]: " + note);
}

}
