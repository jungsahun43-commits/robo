#include "safelog/capture/capture_service.hpp"
#include "safelog/ai/ai_safety_service.hpp"
#include "safelog/ai/mock_ai_analyzer.hpp"
#include "safelog/reporting/report_service.hpp"
#include "safelog/storage/in_memory_repository.hpp"
#include "safelog/storage/system_services.hpp"
#include "safelog/workflow/workflow_service.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
  using namespace safelog;
  namespace fs = std::filesystem;
  storage::InMemoryRepository repository;
  storage::SystemClock clock;
  storage::SequentialIdGenerator ids;
  const auto runtime = fs::current_path() / "safelog-demo-data";
  fs::create_directories(runtime);
  const auto before = runtime / "before.jpg";
  const auto after = runtime / "after.jpg";
  std::ofstream(before, std::ios::binary) << "demo-before";
  std::ofstream(after, std::ios::binary) << "demo-after";
  storage::LocalPhotoStore photos(runtime / "photos");

  repository.saveSite({"site-1", "세이프 금속 가공공장", "서울시 가상구 산업로 10"});
  repository.saveProfile({"inspector-1", "김안전", UserRole::Inspector});
  repository.saveProfile({"assignee-1", "이조치", UserRole::Assignee});
  repository.saveProfile({"manager-1", "박관리", UserRole::Manager});

  capture::CaptureService capture(repository, clock, ids, photos);
  workflow::WorkflowService workflow(repository, clock, ids, photos);
  ai::MockAiSafetyAnalyzer mockAi;
  ai::AiSafetyService aiService(repository, clock, ids, mockAi);
  const auto inspection = capture.startInspection("site-1", "inspector-1");
  const auto bundle = capture.addFinding({inspection.id, "inspector-1", "2층 가공라인 통로",
    "통로에 자재가 적치되어 이동 중 걸려 넘어질 위험이 있음",
    "자재를 지정 보관구역으로 이동하고 통로 폭을 확보할 것", before.string()});
  const auto hazardAnalysis = aiService.analyzeFinding(bundle.finding.id);
  aiService.reviewHazard(hazardAnalysis.id, "inspector-1", {AiReviewDecision::Accepted,
    "2층 가공라인 통로에 자재가 적치되어 보행 중 걸려 넘어질 위험이 있습니다.",
    "자재를 지정 보관구역으로 이동하고 안전 통로 폭을 확보하세요."});
  workflow.assign(bundle.finding.id, "manager-1", "assignee-1");
  workflow.beginWork(bundle.finding.id, "assignee-1");
  workflow.submitAction(bundle.finding.id, "assignee-1", "자재 이동 및 통로 정리 완료", after.string());
  const auto comparison = aiService.compareAction(bundle.finding.id);
  aiService.reviewAnalysis(comparison.id, "inspector-1", AiReviewDecision::Accepted);
  workflow.verify(bundle.finding.id, "inspector-1", "현장 확인 완료");
  const auto summary = aiService.summarizeInspection(inspection.id);
  aiService.reviewAnalysis(summary.id, "inspector-1", AiReviewDecision::Accepted);

  reporting::ReportService reports(repository);
  reporting::HtmlRenderer renderer;
  const auto output = runtime / "inspection-report.html";
  renderer.writeFile(reports.build(inspection.id), output.string());
  std::cout << "SafeLog scenario complete\nReport: " << output << '\n';
  return 0;
}
