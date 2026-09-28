#pragma once

#include "safelog/ai/ai_analyzer.hpp"

namespace safelog::ai {

class MockAiSafetyAnalyzer final : public IAiSafetyAnalyzer {
public:
  HazardSuggestion analyzeHazard(const std::string& imagePath,
                                 const std::string& userMemo) override;
  ActionAssessment compareBeforeAfter(const std::string& beforeImagePath,
                                      const std::string& afterImagePath,
                                      const std::string& actionNote) override;
  ReportSummary summarize(const std::string& inspectionContext) override;
};

} // namespace safelog::ai
