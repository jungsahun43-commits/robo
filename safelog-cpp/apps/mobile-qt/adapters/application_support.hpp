#pragma once

#include "safelog/ai/ai_safety_service.hpp"
#include "safelog/contracts/errors.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

namespace safelog::qtapp {

struct AuthSession {
  Profile user;
  Site site;
};

// TODO(role 2): replace this development adapter with real authentication.
class MockAuthAdapter {
public:
  explicit MockAuthAdapter(const IRepository& repository) : repository_(repository) {}

  std::optional<AuthSession> login(const Id& userId, const std::string& password) const {
    const auto user = repository_.findProfile(userId);
    const auto site = repository_.findSite("site-1");
    if (password != "demo1234" || !user || !site) return std::nullopt;
    return AuthSession{*user, *site};
  }

private:
  const IRepository& repository_;
};

// Used only on the controller/event-loop thread. Workers capture a token, never this object.
// Logout, manual fallback, cancellation and deadlines invalidate all previous tokens.
class RequestEpoch {
public:
  std::uint64_t invalidate() { return ++generation_; }
  bool accepts(std::uint64_t token) const { return token == generation_; }
private:
  std::uint64_t generation_{0};
};

inline bool canCapture(const Profile& user) { return user.role == UserRole::Inspector; }

inline bool canReviewFinding(const IRepository& repository, const Id& findingId, const Id& userId) {
  const auto user = repository.findProfile(userId);
  const auto finding = repository.findFinding(findingId);
  if (!user || !canCapture(*user) || !finding) return false;
  const auto inspection = repository.findInspection(finding->inspectionId);
  return inspection && inspection->inspectorId == userId;
}

inline bool validHazard(const ai::HazardSuggestion& result) {
  return result.riskLevel >= 1 && result.riskLevel <= 5 && std::isfinite(result.confidence) &&
    result.confidence >= 0 && result.confidence <= 1 && !result.category.empty() &&
    !result.suggestedDescription.empty() && !result.suggestedAction.empty() && !result.modelName.empty() &&
    !result.promptVersion.empty();
}

inline AiReviewDecision hazardDecision(const ai::HazardSuggestion& suggestion,
                                       const std::string& description, const std::string& action) {
  return description == suggestion.suggestedDescription && action == suggestion.suggestedAction
    ? AiReviewDecision::Accepted : AiReviewDecision::Edited;
}

// Application integration service. It changes human-authored content, never workflow status.
inline void saveManualReview(IRepository& repository, IClock& clock, IIdGenerator& ids,
                             ai::AiSafetyService& reviews, const Id& findingId, const Id& userId,
                             const Id& analysisId, const std::string& description, const std::string& action) {
  if (!canReviewFinding(repository, findingId, userId))
    throw ValidationError("Only the inspection owner can save a manual review");
  if (description.find_first_not_of(" \t\r\n") == std::string::npos ||
      action.find_first_not_of(" \t\r\n") == std::string::npos)
    throw ValidationError("Description and action are required");
  auto finding = repository.findFinding(findingId).value();
  if (!analysisId.empty()) {
    const auto analysis = repository.findAiAnalysis(analysisId);
    if (!analysis || analysis->subjectId != findingId)
      throw ValidationError("Analysis does not belong to this finding");
    reviews.reviewHazard(analysisId, userId, {AiReviewDecision::Rejected, {}, {}});
  }
  finding.description = description;
  finding.actionOpinion = action;
  repository.saveFinding(finding);
  repository.saveActionLog({ids.next("log"), findingId, userId, "manual_hazard_review", "AI 없이 수동 입력", clock.now()});
}

} // namespace safelog::qtapp
