#include "adapters/application_support.hpp"
#include "safelog/ai/mock_ai_analyzer.hpp"
#include "safelog/capture/capture_service.hpp"
#include "safelog/storage/in_memory_repository.hpp"
#include "safelog/storage/system_services.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace safelog;
using namespace safelog::qtapp;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template<class Function> void requireFailure(Function action) {
  bool failed = false;
  try { action(); } catch (const std::exception&) { failed = true; }
  require(failed, "Expected an exception");
}
struct Fixture {
  std::filesystem::path root = std::filesystem::temp_directory_path() /
    ("safelog-app-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  storage::InMemoryRepository repo;
  storage::SystemClock clock;
  storage::SequentialIdGenerator ids;
  storage::LocalPhotoStore photos{root / "photos"};
  capture::CaptureService capture{repo, clock, ids, photos};
  ai::MockAiSafetyAnalyzer analyzer;
  ai::AiSafetyService ai{repo, clock, ids, analyzer};
  MockAuthAdapter auth{repo};
  Finding finding;

  Fixture() {
    repo.saveSite({"site-1", "세이프 금속 가공공장", ""});
    repo.saveProfile({"inspector-1", "김안전", UserRole::Inspector});
    repo.saveProfile({"inspector-2", "다른 점검자", UserRole::Inspector});
    repo.saveProfile({"manager-1", "박관리", UserRole::Manager});
    repo.saveProfile({"assignee-1", "이조치", UserRole::Assignee});
    // Core mock checks existence; Qt image decoding is tested separately in controller_tests.
    std::ofstream(root / "before.jpg") << "mock-image";
    const auto inspection = capture.startInspection("site-1", "inspector-1");
    finding = capture.addFinding({inspection.id, "inspector-1", "2층 가공라인 통로",
      "통로에 자재가 적치되어 있음", "이동 필요", (root / "before.jpg").string()}).finding;
  }
  ~Fixture() { std::error_code ignored; std::filesystem::remove_all(root, ignored); }
};

void login() {
  Fixture f;
  require(!f.auth.login("inspector-1", "wrong"), "Wrong password accepted");
  require(!f.auth.login("unknown", "demo1234"), "Unknown account accepted");
  for (const auto& id : {"inspector-1", "manager-1", "assignee-1"}) {
    const auto session = f.auth.login(id, "demo1234");
    require(session && session->user.id == id, "Mock login failed");
    require(session->site.id == "site-1" && session->site.name == "세이프 금속 가공공장", "Wrong site");
  }
  storage::InMemoryRepository empty;
  require(!MockAuthAdapter(empty).login("inspector-1", "demo1234"), "Missing site/profile accepted");
}

void permissions() {
  Fixture f;
  require(canCapture(f.auth.login("inspector-1", "demo1234")->user), "Inspector cannot capture");
  for (const auto& id : {"manager-1", "assignee-1"})
    require(!canCapture(f.auth.login(id, "demo1234")->user), "Wrong role can capture");
  require(canReviewFinding(f.repo, f.finding.id, "inspector-1"), "Owner denied");
  for (const auto& id : {"manager-1", "assignee-1", "inspector-2", "missing"}) {
    require(!canReviewFinding(f.repo, f.finding.id, id), "Unauthorized review allowed");
    requireFailure([&] { saveManualReview(f.repo, f.clock, f.ids, f.ai, f.finding.id, id, "", "changed", "changed"); });
  }
  require(f.repo.findFinding(f.finding.id)->description == f.finding.description, "Unauthorized write occurred");
}

void mockAndHumanReview() {
  for (const auto decision : {AiReviewDecision::Accepted, AiReviewDecision::Edited, AiReviewDecision::Rejected}) {
    Fixture f;
    const auto suggestion = f.analyzer.analyzeHazard((f.root / "before.jpg").string(), "메모");
    require(validHazard(suggestion), "Invalid mock hazard");
    require(suggestion.riskLevel == 4 && suggestion.confidence == 0.87, "Mock values changed");
    const auto analysis = f.ai.analyzeFinding(f.finding.id);
    require(analysis.decision == AiReviewDecision::Pending, "AI automatically reviewed");
    require(f.repo.findFinding(f.finding.id)->description == f.finding.description, "AI automatically overwrote human input");
    const auto description = decision == AiReviewDecision::Edited ? "사람이 수정한 위험 설명" : suggestion.suggestedDescription;
    require(hazardDecision(suggestion, description, suggestion.suggestedAction) ==
      (decision == AiReviewDecision::Edited ? AiReviewDecision::Edited : AiReviewDecision::Accepted), "Wrong edit detection");
    const auto reviewed = f.ai.reviewHazard(analysis.id, "inspector-1", {decision, description, suggestion.suggestedAction});
    const auto stored = f.repo.findAiAnalysis(analysis.id).value();
    require(stored.decision == decision && stored.reviewedBy == "inspector-1", "Human audit missing");
    require(reviewed.status == FindingStatus::Open, "AI review finalized workflow");
    require(reviewed.description == (decision == AiReviewDecision::Rejected ? f.finding.description : description), "Wrong reviewed description");
    requireFailure([&] { f.ai.reviewHazard(analysis.id, "inspector-1", {decision, description, suggestion.suggestedAction}); });
  }
}

void invalidResults() {
  Fixture f;
  const auto valid = f.analyzer.analyzeHazard((f.root / "before.jpg").string(), "");
  for (const auto risk : {0, 6}) { auto h = valid; h.riskLevel = risk; require(!validHazard(h), "Invalid risk accepted"); }
  for (const auto confidence : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    auto h = valid; h.confidence = confidence; require(!validHazard(h), "Invalid confidence accepted");
  }
  auto h = valid; h.suggestedDescription.clear(); require(!validHazard(h), "Missing description accepted");
  h = valid; h.suggestedAction.clear(); require(!validHazard(h), "Missing action accepted");
  h = valid; h.category.clear(); require(!validHazard(h), "Missing category accepted");
  h = valid; h.modelName.clear(); require(!validHazard(h), "Missing model accepted");
}

void failureAndManualFallback() {
  Fixture f;
  requireFailure([&] { f.analyzer.analyzeHazard((f.root / "missing.jpg").string(), "memo"); });
  requireFailure([&] { f.analyzer.compareBeforeAfter((f.root / "before.jpg").string(), (f.root / "missing.jpg").string(), "action"); });
  require(f.repo.analysesForSubject(f.finding.id).empty(), "Failed AI recorded a success");
  requireFailure([&] { saveManualReview(f.repo, f.clock, f.ids, f.ai, f.finding.id, "inspector-1", "", "  ", "action"); });
  saveManualReview(f.repo, f.clock, f.ids, f.ai, f.finding.id, "inspector-1", "", "수동 위험", "수동 조치");
  const auto finding = f.repo.findFinding(f.finding.id).value();
  require(finding.description == "수동 위험" && finding.actionOpinion == "수동 조치", "Manual content missing");
  require(finding.status == FindingStatus::Open, "Manual review finalized workflow");
  const auto logs = f.repo.logsForFinding(finding.id);
  require(std::any_of(logs.begin(), logs.end(), [](const auto& log) {
    return log.action == "manual_hazard_review" && log.actorId == "inspector-1";
  }), "Manual audit missing");
}

void successfulAiToManual() {
  Fixture f;
  const auto analysis = f.ai.analyzeFinding(f.finding.id);
  saveManualReview(f.repo, f.clock, f.ids, f.ai, f.finding.id, "inspector-1", analysis.id, "수동 위험", "수동 조치");
  require(f.repo.findAiAnalysis(analysis.id)->decision == AiReviewDecision::Rejected, "Manual fallback did not reject AI");
  require(f.repo.findFinding(f.finding.id)->description == "수동 위험", "Manual fallback lost content");
}

void retryUsesSameFinding() {
  Fixture f;
  requireFailure([&] { f.analyzer.analyzeHazard("/not-a-photo", "memo"); });
  const auto analysis = f.ai.analyzeFinding(f.finding.id);
  require(analysis.subjectId == f.finding.id, "Retry changed finding");
  require(f.repo.findingsForInspection(f.finding.inspectionId).size() == 1, "Retry duplicated finding");
}

void lateResultIsIgnored() {
  // Deterministic worker completion after each GUI-side invalidation; no timing assumptions.
  for (const auto* reason : {"manual fallback", "logout", "timeout", "cancel"}) {
    (void)reason;
    RequestEpoch requests;
    const auto token = requests.invalidate();
    std::promise<void> release;
    auto worker = std::async(std::launch::async, [token, gate = release.get_future()]() mutable {
      gate.wait(); return token;
    });
    requests.invalidate();
    release.set_value();
    require(!requests.accepts(worker.get()), "Late canceled response accepted");
    const auto retry = requests.invalidate();
    require(requests.accepts(retry), "Current request rejected");
    require(!requests.accepts(token), "Old request revived after retry");
  }
}
}

int main() {
  struct Test { const char* name; void (*run)(); };
  const Test tests[] = {{"login", login}, {"permissions", permissions},
    {"mock and human review", mockAndHumanReview}, {"invalid results", invalidResults},
    {"AI failure and manual fallback", failureAndManualFallback}, {"success to manual", successfulAiToManual},
    {"retry uses same finding", retryUsesSameFinding}, {"late result ignored", lateResultIsIgnored}};
  for (const auto& test : tests) {
    try { test.run(); std::cout << "PASS: " << test.name << '\n'; }
    catch (const std::exception& error) { std::cerr << "FAIL: " << test.name << ": " << error.what() << '\n'; return 1; }
  }
  return 0;
}
