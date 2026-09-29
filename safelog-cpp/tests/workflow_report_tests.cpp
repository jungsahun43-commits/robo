#include "adapters/workflow_support.hpp"
#include "adapters/sharing_adapter.hpp"
#include "safelog/ai/mock_ai_analyzer.hpp"
#include "safelog/capture/capture_service.hpp"
#include "safelog/storage/in_memory_repository.hpp"
#include "safelog/storage/system_services.hpp"
#include "safelog/reporting/report_service.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace safelog;
using namespace safelog::qtapp;
namespace {
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
  bool rejected = false;
  try { action(); } catch (const std::exception&) { rejected = true; }
  check(rejected, "Expected failure");
}
struct Fixture {
  std::filesystem::path root = std::filesystem::temp_directory_path() /
    ("safelog-workflow-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  storage::InMemoryRepository repo;
  storage::SystemClock clock;
  storage::SequentialIdGenerator ids;
  storage::LocalPhotoStore photos{root / "photos"};
  capture::CaptureService capture{repo, clock, ids, photos};
  workflow::WorkflowService workflow{repo, clock, ids, photos};
  ai::MockAiSafetyAnalyzer analyzer;
  ai::AiSafetyService ai{repo, clock, ids, analyzer};
  reporting::ReportService reports{repo};
  Finding finding;
  Fixture() {
    repo.saveSite({"site-1", "세이프 금속 가공공장", "가상 사업장"});
    repo.saveProfile({"i", "김안전", UserRole::Inspector});
    repo.saveProfile({"other", "다른 점검자", UserRole::Inspector});
    repo.saveProfile({"m", "박관리", UserRole::Manager});
    repo.saveProfile({"a", "이조치", UserRole::Assignee});
    repo.saveProfile({"a2", "다른 담당자", UserRole::Assignee});
    // Mock file fixture: real image decoding belongs to the optional Qt suite.
    std::ofstream(root / "before.jpg", std::ios::binary) << "before";
    std::ofstream(root / "after.jpg", std::ios::binary) << "after";
    const auto inspection = capture.startInspection("site-1", "i");
    finding = capture.addFinding({inspection.id, "i", "2층 통로", "적치물", "이동 필요", (root / "before.jpg").string()}).finding;
  }
  ~Fixture() { std::error_code ignored; std::filesystem::remove_all(root, ignored); }
  void progress() { workflow.assign(finding.id, "m", "a"); workflow.beginWork(finding.id, "a"); }
  void pending() { progress(); workflow.submitAction(finding.id, "a", "통로 확보", (root / "after.jpg").string()); }
  AiAnalysis compare() {
    const auto before = latestPhoto(repo, finding.id, PhotoKind::Before).value();
    const auto after = latestPhoto(repo, finding.id, PhotoKind::After).value();
    return storeComparison(repo, clock, ids, finding.id,
      analyzer.compareBeforeAfter(before.storagePath, after.storagePath, latestAction(repo, finding.id)), "a");
  }
};
void transitionsAndRoles() {
  Fixture f;
  rejects([&] { f.workflow.assign(f.finding.id, "i", "a"); });
  rejects([&] { f.workflow.assign(f.finding.id, "m", "missing"); });
  rejects([&] { f.workflow.beginWork(f.finding.id, "a"); });
  rejects([&] { f.workflow.verify(f.finding.id, "i"); });
  f.workflow.assign(f.finding.id, "m", "a");
  check(f.repo.findFinding(f.finding.id)->status == FindingStatus::Open, "Assignment changed status");
  rejects([&] { f.workflow.beginWork(f.finding.id, "a2"); });
  f.workflow.beginWork(f.finding.id, "a");
  rejects([&] { f.workflow.verify(f.finding.id, "i"); });
  rejects([&] { f.workflow.submitAction(f.finding.id, "a2", "action", (f.root / "after.jpg").string()); });
  rejects([&] { f.workflow.submitAction(f.finding.id, "a", "  ", (f.root / "after.jpg").string()); });
  rejects([&] { f.workflow.submitAction(f.finding.id, "a", "action", ""); });
  rejects([&] { f.workflow.submitAction(f.finding.id, "a", "action", (f.root / "missing.jpg").string()); });
  check(f.repo.findFinding(f.finding.id)->status == FindingStatus::InProgress, "Failed submission changed status");
  f.workflow.submitAction(f.finding.id, "a", "action", (f.root / "after.jpg").string());
  for (const auto& id : {"m", "a", "other"}) rejects([&] { f.workflow.verify(f.finding.id, id); });
  f.workflow.verify(f.finding.id, "i", "직접 확인");
  check(f.repo.findFinding(f.finding.id)->status == FindingStatus::Verified, "Owner verify failed");
  rejects([&] { f.workflow.verify(f.finding.id, "i"); });
  rejects([&] { f.workflow.requestChanges(f.finding.id, "i", "more"); });
}
void fullMockAndReport() {
  Fixture f;
  const auto h = f.ai.analyzeFinding(f.finding.id);
  f.ai.reviewHazard(h.id, "i", {AiReviewDecision::Accepted, "통로 위험", "자재 이동"});
  f.pending();
  const auto comparison = f.compare();
  check(f.repo.findFinding(f.finding.id)->status == FindingStatus::PendingReview, "AI automatically verified");
  rejects([&] { finalizeReview(f.repo, f.workflow, f.ai, f.finding.id, "i", comparison.id, AiReviewDecision::Accepted, "확인", false); });
  check(f.repo.findAiAnalysis(comparison.id)->decision == AiReviewDecision::Pending, "Unconfirmed AI review saved");
  finalizeReview(f.repo, f.workflow, f.ai, f.finding.id, "i", comparison.id, AiReviewDecision::Accepted, "장애물 없음 확인", true);
  const auto data = f.reports.build(f.finding.inspectionId);
  check(data.findings.front().finding.status == FindingStatus::Verified, "Report not verified");
  const auto path = f.root / "report.html";
  reporting::HtmlRenderer{}.writeFile(data, path.string());
  std::ifstream input(path); const std::string html{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  for (const auto& term : {"SafeLog AI", "김안전", "이조치", "hazard-v1", "comparison-v1", "0.87", "accepted", "verified", "장애물 없음 확인", "data:image/", "after_comparison", "before_hazard"})
    check(html.find(term) != std::string::npos, term);
  check(html.find("YmVmb3Jl") != std::string::npos && html.find("YWZ0ZXI=") != std::string::npos, "Photo base64 encoding failed");
  check(html.find(f.root.string()) == std::string::npos, "Export depends on private photo paths");
}
void comparisonFailureAndManual() {
  Fixture f; f.pending();
  rejects([&] { f.analyzer.compareBeforeAfter("missing", "missing", "note"); });
  rejects([&] { finalizeReview(f.repo, f.workflow, f.ai, f.finding.id, "i", "", AiReviewDecision::Accepted, "확인", true); });
  rejects([&] { finalizeReview(f.repo, f.workflow, f.ai, f.finding.id, "m", "", std::nullopt, "확인", true); });
  rejects([&] { finalizeReview(f.repo, f.workflow, f.ai, f.finding.id, "i", "", std::nullopt, "  ", true); });
  finalizeReview(f.repo, f.workflow, f.ai, f.finding.id, "i", "", std::nullopt, "AI 없이 직접 확인", true);
  const auto data = f.reports.build(f.finding.inspectionId);
  check(data.findings.front().finding.status == FindingStatus::Verified, "Manual verify failed");
  check(data.findings.front().aiAnalyses.empty(), "Manual path fabricated AI output");
  check(reporting::HtmlRenderer{}.render(data).find("AI 없이 직접 확인") != std::string::npos, "Manual audit missing");
}
void rejectedComparison() {
  Fixture f; f.pending(); const auto comparison = f.compare();
  finalizeReview(f.repo, f.workflow, f.ai, f.finding.id, "i", comparison.id, AiReviewDecision::Rejected, "현장 직접 확인 결과 안전", true);
  check(f.repo.findAiAnalysis(comparison.id)->decision == AiReviewDecision::Rejected, "Rejection missing");
}
void reworkAndLatestPhoto() {
  Fixture f; f.pending(); const auto old = f.compare();
  rejects([&] { f.workflow.requestChanges(f.finding.id, "m", "more"); });
  rejects([&] { f.workflow.requestChanges(f.finding.id, "i", " "); });
  f.workflow.requestChanges(f.finding.id, "i", "가장자리 추가 정리");
  check(f.repo.findFinding(f.finding.id)->status == FindingStatus::InProgress, "Rework transition failed");
  std::ofstream(f.root / "new-after.jpg") << "new-after";
  f.workflow.submitAction(f.finding.id, "a", "추가 정리 완료", (f.root / "new-after.jpg").string());
  check(latestPhoto(f.repo, f.finding.id, PhotoKind::After)->storagePath.find("new-after.jpg") != std::string::npos, "Old photo selected");
  check(latestAction(f.repo, f.finding.id) == "추가 정리 완료", "Old action selected");
  rejects([&] { finalizeReview(f.repo, f.workflow, f.ai, f.finding.id, "i", old.id, AiReviewDecision::Accepted, "확인", true); });
  const auto current = f.compare();
  finalizeReview(f.repo, f.workflow, f.ai, f.finding.id, "i", current.id, AiReviewDecision::Accepted, "재확인", true);
}
void missingAndFailedReports() {
  Fixture f;
  rejects([&] { f.reports.build("missing"); });
  const auto empty = f.capture.startInspection("site-1", "i");
  rejects([&] { f.reports.build(empty.id); });
  auto data = f.reports.build(f.finding.inspectionId);
  rejects([&] { reporting::HtmlRenderer{}.writeFile(data, (f.root / "missing" / "report.html").string()); });
  data.findings.front().finding.description = "<script>alert(1)</script> & text";
  auto html = reporting::HtmlRenderer{}.render(data);
  check(html.find("<script>") == std::string::npos && html.find("&lt;script&gt;") != std::string::npos, "HTML was not escaped");
  data.findings.front().photos.front().storagePath = "missing";
  rejects([&] { reporting::HtmlRenderer{}.render(data); });
}
void sharingAndVisibility() {
  Fixture f; MockSharingAdapter share;
  check(share.share("missing", "text/html") == ShareResult::ShareFailed, "Missing report shared");
  const auto path = f.root / "report.html";
  reporting::HtmlRenderer{}.writeFile(f.reports.build(f.finding.inspectionId), path.string());
  for (const auto result : {ShareResult::ShareSucceeded, ShareResult::ShareCancelled, ShareResult::ShareFailed}) {
    share.nextResult = result; check(share.share(path.string(), "text/html") == result, "Wrong sharing state");
  }
  check(canViewFinding(f.repo, f.finding, "i") && canViewFinding(f.repo, f.finding, "m"), "Owner/manager hidden");
  check(!canViewFinding(f.repo, f.finding, "other") && !canViewFinding(f.repo, f.finding, "a"), "Record exposed before assignment");
  const auto assigned = f.workflow.assign(f.finding.id, "m", "a");
  check(canViewFinding(f.repo, assigned, "a") && !canViewFinding(f.repo, assigned, "a2"), "Wrong assignee access");
}
}
int main() {
  struct Test { const char* name; void (*run)(); };
  const Test tests[] = {{"transitions and roles", transitionsAndRoles}, {"full mock and HTML report", fullMockAndReport},
    {"comparison failure and manual verify", comparisonFailureAndManual}, {"rejected comparison", rejectedComparison},
    {"rework and latest photo", reworkAndLatestPhoto}, {"report failures and escaping", missingAndFailedReports},
    {"sharing outcomes and visibility", sharingAndVisibility}};
  for (const auto& t : tests) {
    try { t.run(); std::cout << "PASS: " << t.name << '\n'; }
    catch (const std::exception& e) { std::cerr << "FAIL: " << t.name << ": " << e.what() << '\n'; return 1; }
  }
}
