#pragma once

#include "safelog/ai/ai_analyzer.hpp"
#include "safelog/contracts/ports.hpp"

namespace safelog::ai {

struct HazardReview {
  AiReviewDecision decision{AiReviewDecision::Accepted};
  std::string description;
  std::string actionOpinion;
};

class AiSafetyService {
public:
  AiSafetyService(IRepository& repository, IClock& clock, IIdGenerator& ids,
                  IAiSafetyAnalyzer& analyzer);

  AiAnalysis analyzeFinding(const Id& findingId);
  Finding reviewHazard(const Id& analysisId, const Id& reviewerId, const HazardReview& review);
  AiAnalysis compareAction(const Id& findingId);
  AiAnalysis summarizeInspection(const Id& inspectionId);
  AiAnalysis reviewAnalysis(const Id& analysisId, const Id& reviewerId,
                            AiReviewDecision decision);

private:
  IRepository& repository_;
  IClock& clock_;
  IIdGenerator& ids_;
  IAiSafetyAnalyzer& analyzer_;
};

} // namespace safelog::ai
