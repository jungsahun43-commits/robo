#pragma once

#include <string>
#include <vector>

namespace safelog::ai {

struct HazardSuggestion {
  std::string category;
  int riskLevel{1};
  std::vector<std::string> detectedHazards;
  std::string suggestedDescription;
  std::string suggestedAction;
  double confidence{0.0};
  std::string rawJson;
  std::string modelName;
  std::string promptVersion;
};

struct ActionAssessment {
  bool likelyResolved{false};
  std::vector<std::string> remainingRisks;
  std::string assessment;
  double confidence{0.0};
  std::string rawJson;
  std::string modelName;
  std::string promptVersion;
};

struct ReportSummary {
  std::string summary;
  std::vector<std::string> keyRisks;
  std::string rawJson;
  std::string modelName;
  std::string promptVersion;
};

class IAiSafetyAnalyzer {
public:
  virtual ~IAiSafetyAnalyzer() = default;
  virtual HazardSuggestion analyzeHazard(const std::string& imagePath,
                                         const std::string& userMemo) = 0;
  virtual ActionAssessment compareBeforeAfter(const std::string& beforeImagePath,
                                              const std::string& afterImagePath,
                                              const std::string& actionNote) = 0;
  virtual ReportSummary summarize(const std::string& inspectionContext) = 0;
  // Synchronous providers may use this hook to abort an in-flight network request.
  // Local providers can keep the default no-op implementation.
  virtual void cancel() {}
};

} // namespace safelog::ai
