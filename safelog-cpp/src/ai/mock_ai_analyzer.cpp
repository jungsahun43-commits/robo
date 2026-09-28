#include "safelog/ai/mock_ai_analyzer.hpp"
#include "safelog/contracts/errors.hpp"

#include <filesystem>

namespace safelog::ai {

HazardSuggestion MockAiSafetyAnalyzer::analyzeHazard(const std::string& imagePath,
                                                     const std::string&) {
  if (!std::filesystem::exists(imagePath)) throw ValidationError("AI input image does not exist");
  return {"통로 적치물", 4, {"보행 통로에 자재 적치", "걸림 및 넘어짐 가능성"},
    "2층 가공라인 통로에 자재가 적치되어 보행 중 걸려 넘어질 위험이 있습니다.",
    "자재를 지정 보관구역으로 이동하고 안전 통로 폭을 확보하세요.", 0.87,
    R"({"hazardCategory":"통로 적치물","riskLevel":4,"confidence":0.87})", "mock-vlm-1"};
}

ActionAssessment MockAiSafetyAnalyzer::compareBeforeAfter(const std::string& before,
                                                          const std::string& after,
                                                          const std::string&) {
  if (!std::filesystem::exists(before) || !std::filesystem::exists(after))
    throw ValidationError("AI comparison images do not exist");
  return {true, {"통로 가장자리 추가 확인 필요"},
    "주요 적치물이 제거되어 통로가 확보된 것으로 보입니다. 점검자의 현장 확인이 필요합니다.",
    0.81, R"({"likelyResolved":true,"confidence":0.81})", "mock-vlm-1"};
}

ReportSummary MockAiSafetyAnalyzer::summarize(const std::string& context) {
  if (context.empty()) throw ValidationError("AI summary context is empty");
  return {"통로 적치 위험이 발견되어 자재 이동 조치를 수행하고 점검자가 확인했습니다.",
    {"통로 적치", "넘어짐 위험"}, R"({"summary":"통로 적치 위험 개선"})", "mock-vlm-1"};
}

} // namespace safelog::ai
