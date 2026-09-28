#include "safelog/capture/capture_service.hpp"
#include "safelog/ai/ai_safety_service.hpp"
#include "safelog/ai/mock_ai_analyzer.hpp"
#include "safelog/contracts/errors.hpp"
#include "safelog/reporting/report_service.hpp"
#include "safelog/storage/in_memory_repository.hpp"
#include "safelog/storage/system_services.hpp"
#include "safelog/workflow/workflow_service.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>

int main() {
  using namespace safelog;
  const auto root = std::filesystem::temp_directory_path() / "safelog-tests";
  std::filesystem::create_directories(root);
  const auto image = root / "photo.jpg";
  std::ofstream(image) << "test";
  storage::InMemoryRepository repo;
  storage::SystemClock clock;
  storage::SequentialIdGenerator ids;
  storage::LocalPhotoStore photoStore(root / "stored");
  repo.saveSite({"s", "공장", "주소"});
  repo.saveProfile({"i", "점검자", UserRole::Inspector});
  repo.saveProfile({"a", "담당자", UserRole::Assignee});
  repo.saveProfile({"m", "관리자", UserRole::Manager});
  capture::CaptureService capture(repo, clock, ids, photoStore);
  workflow::WorkflowService flow(repo, clock, ids, photoStore);
  ai::MockAiSafetyAnalyzer mockAi;
  ai::AiSafetyService aiService(repo, clock, ids, mockAi);
  const auto inspection = capture.startInspection("s", "i");
  const auto finding = capture.addFinding({inspection.id, "i", "통로", "적치물", "이동", image.string()}).finding;
  assert(finding.status == FindingStatus::Open);
  const auto hazard = aiService.analyzeFinding(finding.id);
  assert(hazard.type == AiAnalysisType::BeforeHazard);
  const auto aiReviewedFinding = aiService.reviewHazard(hazard.id, "i", {AiReviewDecision::Accepted,
    "통로 적치로 인한 넘어짐 위험", "자재 이동 및 통로 확보"});
  assert(aiReviewedFinding.riskLevel && *aiReviewedFinding.riskLevel == 4);
  flow.assign(finding.id, "m", "a");
  assert(flow.beginWork(finding.id, "a").status == FindingStatus::InProgress);
  assert(flow.submitAction(finding.id, "a", "이동 완료", image.string()).status == FindingStatus::PendingReview);
  const auto comparison = aiService.compareAction(finding.id);
  assert(comparison.type == AiAnalysisType::AfterComparison);
  aiService.reviewAnalysis(comparison.id, "i", AiReviewDecision::Accepted);
  assert(flow.verify(finding.id, "i").status == FindingStatus::Verified);
  const auto summary = aiService.summarizeInspection(inspection.id);
  assert(summary.type == AiAnalysisType::ReportSummary);
  const auto report = reporting::ReportService(repo).build(inspection.id);
  assert(report.findings.size() == 1);
  assert(report.findings.front().photos.size() == 2);
  assert(report.findings.front().aiAnalyses.size() == 2);
  return 0;
}
