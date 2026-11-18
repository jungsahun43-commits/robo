// SqliteRepository tests. Each test uses a fresh database file under the temp directory.
// Uses its own CHECK macro (not assert) so checks still run in Release builds.

#include "safelog/ai/ai_safety_service.hpp"
#include "safelog/ai/mock_ai_analyzer.hpp"
#include "safelog/capture/capture_service.hpp"
#include "safelog/contracts/errors.hpp"
#include "safelog/reporting/report_service.hpp"
#include "safelog/storage/sqlite_repository.hpp"
#include "safelog/storage/system_services.hpp"
#include "safelog/storage/uuid_id_generator.hpp"
#include "safelog/workflow/workflow_service.hpp"
#include "sqlite_codec.hpp"

#include <sqlite3.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <vector>

using namespace safelog;
using namespace std::chrono;
namespace fs = std::filesystem;
using storage::SqliteRepository;

namespace {

int failures = 0;

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::fprintf(stderr, "  %s:%d CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                                  \
    }                                                                              \
  } while (0)

template <typename E, typename F>
bool throwsAs(F&& f) {
  try { f(); } catch (const E&) { return true; } catch (...) { return false; }
  return false;
}
#define CHECK_THROWS(ErrorType, expr) CHECK(throwsAs<ErrorType>([&] { expr; }))

fs::path freshDir(const std::string& name) {
  const auto dir = fs::temp_directory_path() / "safelog-sqlite-tests" / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

TimePoint at(int hour, int minute, int second, long long ticks = 0) {
  return time_point_cast<TimePoint::duration>(sys_days{2026y / November / 16} + hours{hour} + minutes{minute} +
                                              seconds{second}) + TimePoint::duration{ticks};
}

// --- field-by-field equality (contracts define no operator==) ---------------
bool same(const Site& a, const Site& b) { return a.id == b.id && a.name == b.name && a.address == b.address; }
bool same(const Profile& a, const Profile& b) {
  return a.id == b.id && a.displayName == b.displayName && a.role == b.role;
}
bool same(const Inspection& a, const Inspection& b) {
  return a.id == b.id && a.siteId == b.siteId && a.inspectorId == b.inspectorId && a.inspectedAt == b.inspectedAt;
}
bool same(const Finding& a, const Finding& b) {
  return a.id == b.id && a.inspectionId == b.inspectionId && a.location == b.location &&
         a.description == b.description && a.actionOpinion == b.actionOpinion && a.assigneeId == b.assigneeId &&
         a.dueAt == b.dueAt && a.status == b.status && a.createdAt == b.createdAt &&
         a.hazardCategory == b.hazardCategory && a.riskLevel == b.riskLevel;
}
bool same(const Photo& a, const Photo& b) {
  return a.id == b.id && a.findingId == b.findingId && a.kind == b.kind && a.storagePath == b.storagePath &&
         a.capturedAt == b.capturedAt;
}
bool same(const ActionLog& a, const ActionLog& b) {
  return a.id == b.id && a.findingId == b.findingId && a.actorId == b.actorId && a.action == b.action &&
         a.note == b.note && a.createdAt == b.createdAt;
}
bool same(const AiAnalysis& a, const AiAnalysis& b) {
  return a.id == b.id && a.subjectId == b.subjectId && a.type == b.type && a.modelName == b.modelName &&
         a.promptVersion == b.promptVersion && a.riskLevel == b.riskLevel && a.category == b.category &&
         a.confidence == b.confidence && a.resultJson == b.resultJson && a.decision == b.decision &&
         a.reviewedBy == b.reviewedBy && a.createdAt == b.createdAt;
}
template <typename T>
bool same(const std::optional<T>& a, const T& b) { return a && same(*a, b); }

// --- shared fixture data -----------------------------------------------------
const Site kSite{"site-1", "세이프 금속 가공공장", "서울시 가상구 산업로 10"};
const Profile kInspector{"inspector-1", "김안전", UserRole::Inspector};
const Profile kAssignee{"assignee-1", "이조치", UserRole::Assignee};
const Profile kManager{"manager-1", "박관리", UserRole::Manager};
const Inspection kInspection{"inspection-1", "site-1", "inspector-1", at(9, 0, 0)};

void saveBase(IRepository& repo) {
  repo.saveSite(kSite);
  repo.saveProfile(kInspector);
  repo.saveProfile(kAssignee);
  repo.saveProfile(kManager);
  repo.saveInspection(kInspection);
}

Finding fullFinding() {
  return {"finding-1", "inspection-1", "2층 가공라인 통로", "통로 자재 적치", "자재를 지정 구역으로 이동",
          Id{"assignee-1"}, at(18, 0, 0), FindingStatus::InProgress, at(9, 10, 0, 1234567),
          std::string{"넘어짐"}, 4};
}

Finding bareFinding() {
  return {"finding-2", "inspection-1", "1층 입구", "소화기 위치 불명확", "표지 부착", std::nullopt, std::nullopt,
          FindingStatus::Open, at(9, 20, 0), std::nullopt, std::nullopt};
}

AiAnalysis analysis(const Id& id, const Id& subject, AiAnalysisType type, TimePoint createdAt) {
  return {id, subject, type, "mock-vision-1", "hazard-v1", 3, "trip_hazard", 0.82,
          R"({"risk":3,"labels":["통로","적치"]})", AiReviewDecision::Pending, std::nullopt, createdAt};
}

// --- tests -------------------------------------------------------------------

void testCodec() {
  const auto t = time_point_cast<TimePoint::duration>(sys_days{2026y / November / 16} + 9h + 30min + 5s + 123ms);
  CHECK(storage::detail::formatTime(t) == "2026-11-16T09:30:05.123000000Z");
  CHECK(storage::detail::parseTime("2026-11-16T09:30:05.123000000Z") == t);
  CHECK(storage::detail::parseTime("2026-11-16T09:30:05.123Z") == t);
  const auto now = system_clock::now();
  CHECK(storage::detail::parseTime(storage::detail::formatTime(now)) == now);
  const auto old = time_point_cast<TimePoint::duration>(sys_days{1969y / December / 31} + 23h + 59min);
  CHECK(storage::detail::parseTime(storage::detail::formatTime(old)) == old);
  CHECK_THROWS(std::runtime_error, storage::detail::parseTime("2026-13-01T00:00:00Z"));
  CHECK_THROWS(std::runtime_error, storage::detail::parseTime("2026-11-16 09:30:05"));
  CHECK(storage::detail::parseFindingStatus("pending_review") == FindingStatus::PendingReview);
  CHECK_THROWS(std::runtime_error, storage::detail::parseUserRole("admin"));
}

// Required: save every object, close the DB, reopen and read back identical values.
void testRoundTripAfterReopen() {
  const auto db = freshDir("roundtrip") / fs::path(u8"한글 경로") / "safelog.db";
  const auto f1 = fullFinding();
  const auto f2 = bareFinding();
  const Photo before{"photo-1", "finding-1", PhotoKind::Before, "C:/data/photos/finding-1/before-a.jpg", at(9, 11, 0)};
  const Photo after{"photo-2", "finding-1", PhotoKind::After, "C:/data/photos/finding-1/after-a.jpg", at(15, 0, 0)};
  const ActionLog log{"log-1", "finding-1", "inspector-1", "created", "현장 확인", at(9, 12, 0)};
  auto reviewed = analysis("ai-1", "finding-1", AiAnalysisType::BeforeHazard, at(9, 13, 0));
  reviewed.decision = AiReviewDecision::Edited;
  reviewed.reviewedBy = "inspector-1";
  auto summary = analysis("ai-2", "inspection-1", AiAnalysisType::ReportSummary, at(16, 0, 0));
  summary.riskLevel = std::nullopt;
  summary.confidence = 1.0;
  {
    SqliteRepository repo(db);
    CHECK(repo.schemaVersion() == SqliteRepository::kSchemaVersion);
    saveBase(repo);
    repo.saveFinding(f1);
    repo.saveFinding(f2);
    repo.savePhoto(before);
    repo.savePhoto(after);
    repo.saveActionLog(log);
    repo.saveAiAnalysis(reviewed);
    repo.saveAiAnalysis(summary);
  }
  CHECK(fs::exists(db));
  SqliteRepository repo(db);
  CHECK(same(repo.findSite("site-1"), kSite));
  CHECK(same(repo.findProfile("inspector-1"), kInspector));
  CHECK(same(repo.findProfile("assignee-1"), kAssignee));
  CHECK(same(repo.findProfile("manager-1"), kManager));
  CHECK(same(repo.findInspection("inspection-1"), kInspection));
  CHECK(same(repo.findFinding("finding-1"), f1));
  CHECK(same(repo.findFinding("finding-2"), f2));
  CHECK(same(repo.findAiAnalysis("ai-1"), reviewed));
  CHECK(same(repo.findAiAnalysis("ai-2"), summary));

  const auto photos = repo.photosForFinding("finding-1");
  CHECK(photos.size() == 2 && same(photos[0], before) && same(photos[1], after));
  const auto logs = repo.logsForFinding("finding-1");
  CHECK(logs.size() == 1 && same(logs[0], log));
  const auto findings = repo.findingsForInspection("inspection-1");
  CHECK(findings.size() == 2 && same(findings[0], f1) && same(findings[1], f2));
  const auto assigned = repo.findingsAssignedTo("assignee-1");
  CHECK(assigned.size() == 1 && same(assigned[0], f1));

  CHECK(!repo.findSite("missing"));
  CHECK(!repo.findFinding("missing"));
  CHECK(!repo.findAiAnalysis("missing"));
  CHECK(repo.photosForFinding("finding-2").empty());
  CHECK(repo.findingsAssignedTo("manager-1").empty());
}

// Required: analysesForSubject returns AI history of every type for that subject only.
void testAnalysesForSubject() {
  const auto db = freshDir("analyses") / "safelog.db";
  {
    SqliteRepository repo(db);
    saveBase(repo);
    repo.saveFinding(fullFinding());
    repo.saveFinding(bareFinding());
    repo.saveAiAnalysis(analysis("ai-c", "finding-1", AiAnalysisType::AfterComparison, at(15, 0, 0)));
    repo.saveAiAnalysis(analysis("ai-a", "finding-1", AiAnalysisType::BeforeHazard, at(9, 0, 0)));
    repo.saveAiAnalysis(analysis("ai-b", "finding-1", AiAnalysisType::BeforeHazard, at(10, 0, 0)));
    repo.saveAiAnalysis(analysis("ai-other", "finding-2", AiAnalysisType::BeforeHazard, at(9, 30, 0)));
    repo.saveAiAnalysis(analysis("ai-sum", "inspection-1", AiAnalysisType::ReportSummary, at(16, 0, 0)));

    // The reviewer's decision is stored by re-saving the same id.
    auto reviewed = *repo.findAiAnalysis("ai-a");
    reviewed.decision = AiReviewDecision::Rejected;
    reviewed.reviewedBy = "inspector-1";
    repo.saveAiAnalysis(reviewed);
  }
  SqliteRepository repo(db);
  const auto history = repo.analysesForSubject("finding-1");
  CHECK(history.size() == 3);
  if (history.size() == 3) {
    CHECK(history[0].id == "ai-a" && history[0].type == AiAnalysisType::BeforeHazard);
    CHECK(history[0].decision == AiReviewDecision::Rejected && history[0].reviewedBy == Id{"inspector-1"});
    CHECK(history[1].id == "ai-b" && history[1].decision == AiReviewDecision::Pending && !history[1].reviewedBy);
    CHECK(history[2].id == "ai-c" && history[2].type == AiAnalysisType::AfterComparison);
  }
  const auto summary = repo.analysesForSubject("inspection-1");
  CHECK(summary.size() == 1 && summary[0].type == AiAnalysisType::ReportSummary);
  CHECK(repo.analysesForSubject("finding-2").size() == 1);
  CHECK(repo.analysesForSubject("nobody").empty());
}

// Required: confidence 0..1 and risk level 1..5 are enforced by the schema.
void testRangeConstraints() {
  SqliteRepository repo(freshDir("ranges") / "safelog.db");
  saveBase(repo);
  repo.saveFinding(bareFinding());
  auto make = [](const Id& id, double confidence, std::optional<int> risk) {
    auto a = analysis(id, "finding-2", AiAnalysisType::BeforeHazard, at(10, 0, 0));
    a.confidence = confidence;
    a.riskLevel = risk;
    return a;
  };
  CHECK_THROWS(ValidationError, repo.saveAiAnalysis(make("bad-1", 1.5, 3)));
  CHECK_THROWS(ValidationError, repo.saveAiAnalysis(make("bad-2", -0.1, 3)));
  CHECK_THROWS(ValidationError, repo.saveAiAnalysis(make("bad-3", 0.5, 0)));
  CHECK_THROWS(ValidationError, repo.saveAiAnalysis(make("bad-4", 0.5, 6)));
  CHECK(!repo.findAiAnalysis("bad-1") && !repo.findAiAnalysis("bad-4"));

  repo.saveAiAnalysis(make("ok-1", 0.0, 1));
  repo.saveAiAnalysis(make("ok-2", 1.0, 5));
  repo.saveAiAnalysis(make("ok-3", 0.5, std::nullopt));
  CHECK(repo.analysesForSubject("finding-2").size() == 3);

  auto finding = bareFinding();
  finding.riskLevel = 6;
  CHECK_THROWS(ValidationError, repo.saveFinding(finding));
  finding.riskLevel = 5;
  repo.saveFinding(finding);
  CHECK(repo.findFinding("finding-2")->riskLevel == 5);
}

// Required: invalid foreign keys and missing photo files are rejected.
void testForeignKeysAndMissingPhotos() {
  const auto dir = freshDir("foreign-keys");
  SqliteRepository repo(dir / "safelog.db");
  saveBase(repo);
  CHECK_THROWS(ValidationError, repo.saveInspection({"inspection-x", "no-site", "inspector-1", at(9, 0, 0)}));
  CHECK_THROWS(ValidationError, repo.saveInspection({"inspection-y", "site-1", "no-profile", at(9, 0, 0)}));
  CHECK(!repo.findInspection("inspection-x") && !repo.findInspection("inspection-y"));

  auto orphan = bareFinding();
  orphan.inspectionId = "no-inspection";
  CHECK_THROWS(ValidationError, repo.saveFinding(orphan));
  auto badAssignee = bareFinding();
  badAssignee.assigneeId = "no-profile";
  CHECK_THROWS(ValidationError, repo.saveFinding(badAssignee));

  CHECK_THROWS(ValidationError, repo.savePhoto({"photo-x", "no-finding", PhotoKind::Before, "C:/x.jpg", at(9, 0, 0)}));
  CHECK_THROWS(ValidationError, repo.saveActionLog({"log-x", "no-finding", "inspector-1", "created", "", at(9, 0, 0)}));
  repo.saveFinding(bareFinding());
  CHECK_THROWS(ValidationError, repo.saveActionLog({"log-y", "finding-2", "no-profile", "created", "", at(9, 0, 0)}));
  auto badReviewer = analysis("ai-x", "finding-2", AiAnalysisType::BeforeHazard, at(9, 0, 0));
  badReviewer.reviewedBy = "no-profile";
  CHECK_THROWS(ValidationError, repo.saveAiAnalysis(badReviewer));

  // A photo file that does not exist is rejected before anything reaches the DB.
  storage::LocalPhotoStore store(dir / "photos");
  CHECK_THROWS(ValidationError, store.importPhoto((dir / "missing.jpg").string(), "finding-2", PhotoKind::Before));
  CHECK(repo.photosForFinding("finding-2").empty());
}

// Re-saving a finding must not cascade-delete its photos and logs (INSERT OR REPLACE would).
void testUpsertKeepsChildren() {
  const auto db = freshDir("upsert") / "safelog.db";
  {
    SqliteRepository repo(db);
    saveBase(repo);
    auto finding = fullFinding();
    repo.saveFinding(finding);
    repo.savePhoto({"photo-1", "finding-1", PhotoKind::Before, "C:/p1.jpg", at(9, 11, 0)});
    repo.saveActionLog({"log-1", "finding-1", "inspector-1", "created", "", at(9, 12, 0)});
    for (const auto status : {FindingStatus::PendingReview, FindingStatus::Verified}) {
      finding.status = status;
      repo.saveFinding(finding);
    }
    repo.saveProfile({"inspector-1", "김안전(수정)", UserRole::Inspector});
  }
  SqliteRepository repo(db);
  CHECK(repo.findFinding("finding-1")->status == FindingStatus::Verified);
  CHECK(repo.photosForFinding("finding-1").size() == 1);
  CHECK(repo.logsForFinding("finding-1").size() == 1);
  CHECK(repo.findProfile("inspector-1")->displayName == "김안전(수정)");
  CHECK(repo.findInspection("inspection-1").has_value());
}

// Lists come back ordered by time, then id.
void testListOrdering() {
  SqliteRepository repo(freshDir("ordering") / "safelog.db");
  saveBase(repo);
  repo.saveFinding(fullFinding());
  repo.saveActionLog({"log-c", "finding-1", "inspector-1", "third", "", at(11, 0, 0)});
  repo.saveActionLog({"log-b", "finding-1", "inspector-1", "first-b", "", at(10, 0, 0)});
  repo.saveActionLog({"log-a", "finding-1", "inspector-1", "first-a", "", at(10, 0, 0)});
  const auto logs = repo.logsForFinding("finding-1");
  CHECK(logs.size() == 3 && logs[0].id == "log-a" && logs[1].id == "log-b" && logs[2].id == "log-c");
}

// A database created before schema v1 gains the findings hazard columns.
void testMigrationFromVersion0() {
  const auto db = freshDir("migration") / "safelog.db";
  {
    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(db.string().c_str(), &raw) == SQLITE_OK);
    const char* oldSchema =
      "CREATE TABLE sites (id TEXT PRIMARY KEY, name TEXT NOT NULL, address TEXT NOT NULL);"
      "CREATE TABLE profiles (id TEXT PRIMARY KEY, display_name TEXT NOT NULL, role TEXT NOT NULL);"
      "CREATE TABLE inspections (id TEXT PRIMARY KEY, site_id TEXT NOT NULL REFERENCES sites(id),"
      "  inspector_id TEXT NOT NULL REFERENCES profiles(id), inspected_at TEXT NOT NULL);"
      "CREATE TABLE findings (id TEXT PRIMARY KEY, inspection_id TEXT NOT NULL REFERENCES inspections(id),"
      "  location TEXT NOT NULL, description TEXT NOT NULL, action_opinion TEXT NOT NULL,"
      "  assignee_id TEXT REFERENCES profiles(id), due_at TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL);"
      "INSERT INTO sites VALUES ('site-1', 'factory', 'addr');"
      "INSERT INTO profiles VALUES ('inspector-1', 'kim', 'inspector');"
      "INSERT INTO inspections VALUES ('inspection-1', 'site-1', 'inspector-1', '2026-11-16T09:00:00.000000000Z');"
      "INSERT INTO findings VALUES ('finding-old', 'inspection-1', 'loc', 'desc', 'opinion', NULL, NULL, 'open',"
      "  '2026-11-16T09:10:00.000000000Z');";
    CHECK(sqlite3_exec(raw, oldSchema, nullptr, nullptr, nullptr) == SQLITE_OK);
    sqlite3_close(raw);
  }
  {
    SqliteRepository repo(db);
    CHECK(repo.schemaVersion() == SqliteRepository::kSchemaVersion);
    auto old = repo.findFinding("finding-old");
    CHECK(old && !old->hazardCategory && !old->riskLevel);
    if (old) {
      old->hazardCategory = "넘어짐";
      old->riskLevel = 3;
      repo.saveFinding(*old);
    }
  }
  SqliteRepository repo(db);
  const auto migrated = repo.findFinding("finding-old");
  CHECK(migrated && migrated->hazardCategory == std::string{"넘어짐"} && migrated->riskLevel == 3);
}

void testRejectsNewerSchema() {
  const auto db = freshDir("newer") / "safelog.db";
  { SqliteRepository repo(db); }
  {
    sqlite3* raw = nullptr;
    sqlite3_open(db.string().c_str(), &raw);
    sqlite3_exec(raw, "PRAGMA user_version = 99", nullptr, nullptr, nullptr);
    sqlite3_close(raw);
  }
  CHECK_THROWS(std::runtime_error, SqliteRepository repo(db));
}

// The real services run unchanged on SqliteRepository, and everything survives a restart.
void testServicesSurviveRestart() {
  const auto dir = freshDir("services");
  const auto db = dir / "safelog.db";
  const auto image = dir / "photo.jpg";
  std::ofstream(image) << "test";
  Id inspectionId, findingId, hazardId;
  {
    SqliteRepository repo(db);
    storage::SystemClock clock;
    storage::SequentialIdGenerator ids;
    storage::LocalPhotoStore photoStore(dir / "stored");
    repo.saveSite({"s", "공장", "주소"});
    repo.saveProfile({"i", "점검자", UserRole::Inspector});
    repo.saveProfile({"a", "담당자", UserRole::Assignee});
    repo.saveProfile({"m", "관리자", UserRole::Manager});
    capture::CaptureService capture(repo, clock, ids, photoStore);
    workflow::WorkflowService flow(repo, clock, ids, photoStore);
    ai::MockAiSafetyAnalyzer mockAi;
    ai::AiSafetyService aiService(repo, clock, ids, mockAi);
    inspectionId = capture.startInspection("s", "i").id;
    findingId = capture.addFinding({inspectionId, "i", "통로", "적치물", "이동", image.string()}).finding.id;
    hazardId = aiService.analyzeFinding(findingId).id;
    aiService.reviewHazard(hazardId, "i", {AiReviewDecision::Edited, "통로 적치로 인한 넘어짐 위험", "자재 이동"});
    flow.assign(findingId, "m", "a");
    flow.beginWork(findingId, "a");
    flow.submitAction(findingId, "a", "이동 완료", image.string());
    const auto comparison = aiService.compareAction(findingId);
    aiService.reviewAnalysis(comparison.id, "i", AiReviewDecision::Accepted);
    flow.verify(findingId, "i");
    aiService.summarizeInspection(inspectionId);
  }
  SqliteRepository repo(db);
  const auto finding = repo.findFinding(findingId);
  CHECK(finding && finding->status == FindingStatus::Verified);
  CHECK(finding && finding->riskLevel == 4 && finding->hazardCategory.has_value());
  CHECK(finding && finding->description == "통로 적치로 인한 넘어짐 위험");
  const auto hazard = repo.findAiAnalysis(hazardId);
  CHECK(hazard && hazard->decision == AiReviewDecision::Edited && hazard->reviewedBy == Id{"i"});
  const auto photos = repo.photosForFinding(findingId);
  CHECK(photos.size() == 2);
  for (const auto& photo : photos) CHECK(fs::path(photo.storagePath).is_absolute() && fs::exists(photo.storagePath));
  CHECK(repo.logsForFinding(findingId).size() >= 5);
  CHECK(repo.analysesForSubject(inspectionId).size() == 1);
  const auto report = reporting::ReportService(repo).build(inspectionId);
  CHECK(report.findings.size() == 1);
  CHECK(!report.findings.empty() && report.findings.front().photos.size() == 2);
  CHECK(!report.findings.empty() && report.findings.front().aiAnalyses.size() == 2);
}

// seedDemoData creates the demo site and three users once and never overwrites them.
void testSeedDemoData() {
  const auto db = freshDir("seed") / "safelog.db";
  {
    SqliteRepository repo(db);
    repo.seedDemoData();
    repo.seedDemoData();
    CHECK(same(repo.findSite("site-1"), kSite));
    CHECK(same(repo.findProfile("inspector-1"), kInspector));
    CHECK(same(repo.findProfile("assignee-1"), kAssignee));
    CHECK(same(repo.findProfile("manager-1"), kManager));
    repo.saveProfile({"manager-1", "박관리(변경)", UserRole::Manager});
  }
  SqliteRepository repo(db);
  repo.seedDemoData();  // app restart
  CHECK(repo.findProfile("manager-1")->displayName == "박관리(변경)");

  sqlite3* raw = nullptr;
  sqlite3_open(db.string().c_str(), &raw);
  sqlite3_stmt* count = nullptr;
  sqlite3_prepare_v2(raw, "SELECT (SELECT COUNT(*) FROM sites), (SELECT COUNT(*) FROM profiles)", -1, &count, nullptr);
  CHECK(sqlite3_step(count) == SQLITE_ROW);
  CHECK(sqlite3_column_int(count, 0) == 1 && sqlite3_column_int(count, 1) == 3);
  sqlite3_finalize(count);
  sqlite3_close(raw);
}

// UUID ids are well-formed and do not repeat, including across generator instances (app restarts).
void testUuidIdGenerator() {
  storage::UuidIdGenerator first;
  storage::UuidIdGenerator second;
  std::set<Id> seen;
  for (int i = 0; i < 5000; ++i) {
    seen.insert(first.next("finding"));
    seen.insert(second.next("finding"));
  }
  CHECK(seen.size() == 10000);
  const auto id = first.next("photo");
  // "photo-" + 8-4-4-4-12
  CHECK(id.size() == 6 + 36);
  CHECK(id.rfind("photo-", 0) == 0);
  CHECK(id[6 + 8] == '-' && id[6 + 13] == '-' && id[6 + 18] == '-' && id[6 + 23] == '-');
  CHECK(id[6 + 14] == '4');
  CHECK(std::string("89ab").find(id[6 + 19]) != std::string::npos);
}

} // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
    {"codec", testCodec},
    {"round trip after reopen", testRoundTripAfterReopen},
    {"analysesForSubject", testAnalysesForSubject},
    {"range constraints", testRangeConstraints},
    {"foreign keys and missing photos", testForeignKeysAndMissingPhotos},
    {"upsert keeps children", testUpsertKeepsChildren},
    {"list ordering", testListOrdering},
    {"migration from v0", testMigrationFromVersion0},
    {"rejects newer schema", testRejectsNewerSchema},
    {"services survive restart", testServicesSurviveRestart},
    {"seed demo data", testSeedDemoData},
    {"uuid id generator", testUuidIdGenerator},
  };
  for (const auto& [name, run] : tests) {
    const int before = failures;
    try {
      run();
    } catch (const std::exception& error) {
      std::fprintf(stderr, "  unexpected exception: %s\n", error.what());
      ++failures;
    }
    std::printf("[%s] %s\n", failures == before ? "PASS" : "FAIL", name);
  }
  std::printf("%s (%d failed checks)\n", failures == 0 ? "ALL PASSED" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
