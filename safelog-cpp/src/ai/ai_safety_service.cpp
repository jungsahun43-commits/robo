#include "safelog/ai/ai_safety_service.hpp"
#include "safelog/contracts/errors.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace safelog::ai {
namespace {
const Photo* findPhoto(const std::vector<Photo>& photos, PhotoKind kind) {
  const auto it = std::find_if(photos.begin(), photos.end(), [kind](const Photo& p) { return p.kind == kind; });
  return it == photos.end() ? nullptr : &*it;
}
std::string latestActionNote(const std::vector<ActionLog>& logs) {
  for (auto it = logs.rbegin(); it != logs.rend(); ++it)
    if (it->action == "action_submitted") return it->note;
  return {};
}
}

AiSafetyService::AiSafetyService(IRepository& r, IClock& c, IIdGenerator& i,
                                 IAiSafetyAnalyzer& a)
  : repository_(r), clock_(c), ids_(i), analyzer_(a) {}

AiAnalysis AiSafetyService::analyzeFinding(const Id& findingId) {
  const auto finding = repository_.findFinding(findingId);
  if (!finding) throw NotFoundError("Finding not found");
  const auto photos = repository_.photosForFinding(findingId);
  const auto* before = findPhoto(photos, PhotoKind::Before);
  if (!before) throw ValidationError("Before photo is required for AI analysis");
  const auto result = analyzer_.analyzeHazard(before->storagePath, finding->description);
  if (result.riskLevel < 1 || result.riskLevel > 5 || !std::isfinite(result.confidence) ||
      result.confidence < 0.0 || result.confidence > 1.0 ||
      result.modelName.empty() || result.promptVersion.empty())
    throw ValidationError("AI analyzer returned values outside the contract");
  AiAnalysis analysis{ids_.next("ai"), findingId, AiAnalysisType::BeforeHazard,
    result.modelName, result.promptVersion, result.riskLevel, result.category, result.confidence,
    result.rawJson, AiReviewDecision::Pending, std::nullopt, clock_.now()};
  repository_.saveAiAnalysis(analysis);
  return analysis;
}

Finding AiSafetyService::reviewHazard(const Id& analysisId, const Id& reviewerId,
                                     const HazardReview& review) {
  auto analysis = repository_.findAiAnalysis(analysisId);
  if (!analysis || analysis->type != AiAnalysisType::BeforeHazard)
    throw NotFoundError("Hazard analysis not found");
  if (analysis->decision != AiReviewDecision::Pending)
    throw ValidationError("AI analysis has already been reviewed");
  auto finding = repository_.findFinding(analysis->subjectId);
  if (!finding) throw NotFoundError("Finding not found");
  const auto inspection = repository_.findInspection(finding->inspectionId);
  if (!inspection || inspection->inspectorId != reviewerId)
    throw ValidationError("Only the inspection owner can review AI suggestions");
  if (review.decision == AiReviewDecision::Pending) throw ValidationError("A final review decision is required");
  if ((review.decision == AiReviewDecision::Accepted || review.decision == AiReviewDecision::Edited) &&
      (review.description.empty() || review.actionOpinion.empty()))
    throw ValidationError("Reviewed description and action opinion are required");
  analysis->decision = review.decision;
  analysis->reviewedBy = reviewerId;
  repository_.saveAiAnalysis(*analysis);
  if (review.decision == AiReviewDecision::Accepted || review.decision == AiReviewDecision::Edited) {
    finding->description = review.description;
    finding->actionOpinion = review.actionOpinion;
    finding->hazardCategory = analysis->category;
    finding->riskLevel = analysis->riskLevel;
    repository_.saveFinding(*finding);
  }
  repository_.saveActionLog({ids_.next("log"), finding->id, reviewerId,
    "ai_hazard_" + to_string(review.decision), analysis->id, clock_.now()});
  return *finding;
}

AiAnalysis AiSafetyService::compareAction(const Id& findingId) {
  const auto finding = repository_.findFinding(findingId);
  if (!finding) throw NotFoundError("Finding not found");
  if (finding->status != FindingStatus::PendingReview)
    throw ValidationError("Action comparison requires pending review status");
  const auto photos = repository_.photosForFinding(findingId);
  const auto* before = findPhoto(photos, PhotoKind::Before);
  const auto* after = findPhoto(photos, PhotoKind::After);
  if (!before || !after) throw ValidationError("Both before and after photos are required");
  const auto result = analyzer_.compareBeforeAfter(before->storagePath, after->storagePath,
                                                   latestActionNote(repository_.logsForFinding(findingId)));
  if (result.modelName.empty() || result.promptVersion.empty() || result.assessment.empty() ||
      !std::isfinite(result.confidence) || result.confidence < 0.0 || result.confidence > 1.0)
    throw ValidationError("AI comparison returned values outside the contract");
  AiAnalysis analysis{ids_.next("ai"), findingId, AiAnalysisType::AfterComparison,
    result.modelName, result.promptVersion, std::nullopt,
    result.likelyResolved ? "likely_resolved" : "remaining_risk", result.confidence,
    result.rawJson, AiReviewDecision::Pending, std::nullopt, clock_.now()};
  repository_.saveAiAnalysis(analysis);
  return analysis;
}

AiAnalysis AiSafetyService::summarizeInspection(const Id& inspectionId) {
  const auto inspection = repository_.findInspection(inspectionId);
  if (!inspection) throw NotFoundError("Inspection not found");
  std::ostringstream context;
  for (const auto& finding : repository_.findingsForInspection(inspectionId)) {
    context << finding.location << " | " << finding.description << " | "
            << finding.actionOpinion << " | " << to_string(finding.status) << '\n';
  }
  if (context.str().empty()) throw ValidationError("Inspection has no findings to summarize");
  const auto result = analyzer_.summarize(context.str());
  if (result.modelName.empty() || result.promptVersion.empty() || result.summary.empty())
    throw ValidationError("AI summary returned values outside the contract");
  AiAnalysis analysis{ids_.next("ai"), inspectionId, AiAnalysisType::ReportSummary,
    result.modelName, result.promptVersion, std::nullopt, "inspection_summary", 1.0,
    result.rawJson, AiReviewDecision::Pending, std::nullopt, clock_.now()};
  repository_.saveAiAnalysis(analysis);
  return analysis;
}

AiAnalysis AiSafetyService::reviewAnalysis(const Id& analysisId, const Id& reviewerId,
                                           AiReviewDecision decision) {
  auto analysis = repository_.findAiAnalysis(analysisId);
  if (!analysis) throw NotFoundError("AI analysis not found");
  if (decision == AiReviewDecision::Pending) throw ValidationError("A final decision is required");
  analysis->decision = decision;
  analysis->reviewedBy = reviewerId;
  repository_.saveAiAnalysis(*analysis);
  return *analysis;
}

} // namespace safelog::ai
