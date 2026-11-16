#include "safelog/storage/sqlite_repository.hpp"

#include "safelog/contracts/errors.hpp"
#include "safelog/storage/schema_sql.hpp"
#include "sqlite_codec.hpp"

#include <sqlite3.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace safelog::storage {
namespace {

[[noreturn]] void throwDbError(sqlite3* db, int rc, std::string_view context) {
  std::string message = "SQLite error in ";
  message += context;
  message += ": ";
  message += db ? sqlite3_errmsg(db) : sqlite3_errstr(rc);
  if ((rc & 0xff) == SQLITE_CONSTRAINT) throw ValidationError(message);
  throw std::runtime_error(message);
}

// RAII wrapper around one prepared statement. All SQL goes through
// sqlite3_prepare_v2 + sqlite3_bind_*; values are never spliced into SQL text.
class Statement {
public:
  Statement(sqlite3* db, std::string_view sql) : db_(db), sql_(sql) {
    const int rc = sqlite3_prepare_v2(db_, sql.data(), static_cast<int>(sql.size()), &stmt_, nullptr);
    if (rc != SQLITE_OK) throwDbError(db_, rc, sql_);
  }
  ~Statement() { sqlite3_finalize(stmt_); }
  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;

  // Returns true while a result row is available, false when done.
  bool step() {
    const int rc = sqlite3_step(stmt_);
    if (rc == SQLITE_ROW) return true;
    if (rc == SQLITE_DONE) return false;
    throwDbError(db_, rc, sql_);
  }
  // Runs the statement to completion, discarding any rows.
  void run() { while (step()) {} }

  void bindText(int index, std::string_view value) {
    check(sqlite3_bind_text(stmt_, index, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT));
  }

  // Named parameters (":id") keep SQL and C++ field lists easy to compare.
  void bind(const char* name, std::string_view value) { bindText(parameter(name), value); }
  void bind(const char* name, int value) { check(sqlite3_bind_int(stmt_, parameter(name), value)); }
  void bind(const char* name, double value) { check(sqlite3_bind_double(stmt_, parameter(name), value)); }
  void bind(const char* name, TimePoint value) { bind(name, detail::formatTime(value)); }
  template <typename T>
  void bind(const char* name, const std::optional<T>& value) {
    if (value) bind(name, *value);
    else check(sqlite3_bind_null(stmt_, parameter(name)));
  }

  int columnInt(int index) const { return sqlite3_column_int(stmt_, index); }
  double columnDouble(int index) const { return sqlite3_column_double(stmt_, index); }
  bool isNull(int index) const { return sqlite3_column_type(stmt_, index) == SQLITE_NULL; }
  std::string columnText(int index) const {
    const auto* text = sqlite3_column_text(stmt_, index);  // must precede column_bytes
    const int size = sqlite3_column_bytes(stmt_, index);
    return text ? std::string(reinterpret_cast<const char*>(text), static_cast<std::size_t>(size)) : std::string();
  }
  TimePoint columnTime(int index) const { return detail::parseTime(columnText(index)); }
  std::optional<std::string> optionalText(int index) const {
    return isNull(index) ? std::nullopt : std::optional<std::string>{columnText(index)};
  }
  std::optional<int> optionalInt(int index) const {
    return isNull(index) ? std::nullopt : std::optional<int>{columnInt(index)};
  }
  std::optional<TimePoint> optionalTime(int index) const {
    return isNull(index) ? std::nullopt : std::optional<TimePoint>{columnTime(index)};
  }

private:
  int parameter(const char* name) const {
    const int index = sqlite3_bind_parameter_index(stmt_, name);
    if (index == 0) throw std::logic_error(std::string("Unknown SQL parameter ") + name + " in: " + std::string(sql_));
    return index;
  }
  void check(int rc) { if (rc != SQLITE_OK) throwDbError(db_, rc, sql_); }

  sqlite3* db_;
  std::string_view sql_;
  sqlite3_stmt* stmt_ = nullptr;
};

// Executes a multi-statement script one prepared statement at a time.
void runScript(sqlite3* db, std::string_view script) {
  const char* cursor = script.data();
  const char* const end = script.data() + script.size();
  while (cursor < end) {
    sqlite3_stmt* stmt = nullptr;
    const char* tail = nullptr;
    int rc = sqlite3_prepare_v2(db, cursor, static_cast<int>(end - cursor), &stmt, &tail);
    if (rc != SQLITE_OK) throwDbError(db, rc, "schema script");
    if (stmt) {  // null for trailing whitespace or comments
      while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {}
      sqlite3_finalize(stmt);
      if (rc != SQLITE_DONE) throwDbError(db, rc, "schema script");
    }
    cursor = tail;
  }
}

int readUserVersion(sqlite3* db) {
  Statement q(db, "PRAGMA user_version");
  return q.step() ? q.columnInt(0) : 0;
}

bool columnExists(sqlite3* db, std::string_view table, std::string_view column);

} // namespace

struct SqliteRepository::Impl {
  sqlite3* db = nullptr;
  mutable std::mutex mutex;

  ~Impl() { if (db) sqlite3_close_v2(db); }

  void open(const std::filesystem::path& path);
  void applySchema();
};

void SqliteRepository::Impl::open(const std::filesystem::path& path) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  // sqlite3_open_v2 expects UTF-8; path::string() would use the ANSI code page on Windows.
  const auto u8 = path.u8string();
  const std::string utf8Path(u8.begin(), u8.end());
  const int rc = sqlite3_open_v2(utf8Path.c_str(), &db,
                                 SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr);
  if (rc != SQLITE_OK) throwDbError(db, rc, "open " + utf8Path);
  sqlite3_extended_result_codes(db, 1);
  sqlite3_busy_timeout(db, 5000);

  Statement(db, "PRAGMA foreign_keys = ON").run();
  Statement check(db, "PRAGMA foreign_keys");
  if (!check.step() || check.columnInt(0) != 1)
    throw std::runtime_error("SQLite foreign key enforcement could not be enabled");
  Statement(db, "PRAGMA journal_mode = WAL").run();
}

void SqliteRepository::Impl::applySchema() {
  const int version = readUserVersion(db);
  if (version > kSchemaVersion)
    throw std::runtime_error("Database schema version " + std::to_string(version) +
                             " is newer than this app supports (" + std::to_string(kSchemaVersion) + ")");
  Statement(db, "BEGIN IMMEDIATE").run();
  try {
    // v0 -> v1: findings gained hazard_category and risk_level.
    if (version < 1 && columnExists(db, "findings", "id") && !columnExists(db, "findings", "hazard_category")) {
      Statement(db, "ALTER TABLE findings ADD COLUMN hazard_category TEXT").run();
      Statement(db, "ALTER TABLE findings ADD COLUMN risk_level INTEGER "
                    "CHECK (risk_level IS NULL OR risk_level BETWEEN 1 AND 5)").run();
    }
    runScript(db, detail::kSchemaSql);  // CREATE ... IF NOT EXISTS, then PRAGMA user_version
    Statement(db, "COMMIT").run();
  } catch (...) {
    sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
    throw;
  }
}

namespace {
bool columnExists(sqlite3* db, std::string_view table, std::string_view column) {
  Statement q(db, "SELECT 1 FROM pragma_table_info(?1) WHERE name = ?2");
  q.bindText(1, table);
  q.bindText(2, column);
  return q.step();
}
} // namespace

SqliteRepository::SqliteRepository(const std::filesystem::path& databasePath)
  : impl_(std::make_unique<Impl>()) {
  impl_->open(databasePath);
  impl_->applySchema();
}

SqliteRepository::~SqliteRepository() = default;

int SqliteRepository::schemaVersion() const {
  std::scoped_lock lock(impl_->mutex);
  return readUserVersion(impl_->db);
}

void SqliteRepository::seedDemoData() {
  const Site site{"site-1", "세이프 금속 가공공장", "서울시 가상구 산업로 10"};
  const Profile profiles[] = {
    {"inspector-1", "김안전", UserRole::Inspector},
    {"assignee-1", "이조치", UserRole::Assignee},
    {"manager-1", "박관리", UserRole::Manager},
  };
  std::scoped_lock lock(impl_->mutex);
  sqlite3* db = impl_->db;
  Statement(db, "BEGIN IMMEDIATE").run();
  try {
    Statement s(db, "INSERT INTO sites (id, name, address) VALUES (:id, :name, :address) "
                    "ON CONFLICT(id) DO NOTHING");
    s.bind(":id", site.id);
    s.bind(":name", site.name);
    s.bind(":address", site.address);
    s.run();
    for (const auto& p : profiles) {
      Statement q(db, "INSERT INTO profiles (id, display_name, role) VALUES (:id, :display_name, :role) "
                      "ON CONFLICT(id) DO NOTHING");
      q.bind(":id", p.id);
      q.bind(":display_name", p.displayName);
      q.bind(":role", to_string(p.role));
      q.run();
    }
    Statement(db, "COMMIT").run();
  } catch (...) {
    sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
    throw;
  }
}

// ---------------------------------------------------------------------------
// Column lists and row mappers. Column order in each SELECT must match readX().

namespace {

constexpr const char* kSiteColumns = "id, name, address";
constexpr const char* kProfileColumns = "id, display_name, role";
constexpr const char* kInspectionColumns = "id, site_id, inspector_id, inspected_at";
constexpr const char* kFindingColumns =
  "id, inspection_id, location, description, action_opinion, assignee_id, due_at, status, created_at, "
  "hazard_category, risk_level";
constexpr const char* kPhotoColumns = "id, finding_id, kind, storage_path, captured_at";
constexpr const char* kActionLogColumns = "id, finding_id, actor_id, action, note, created_at";
constexpr const char* kAiAnalysisColumns =
  "id, subject_id, analysis_type, model_name, prompt_version, risk_level, category, confidence, result_json, "
  "decision, reviewed_by, created_at";

Site readSite(const Statement& q) { return {q.columnText(0), q.columnText(1), q.columnText(2)}; }

Profile readProfile(const Statement& q) {
  return {q.columnText(0), q.columnText(1), detail::parseUserRole(q.columnText(2))};
}

Inspection readInspection(const Statement& q) {
  return {q.columnText(0), q.columnText(1), q.columnText(2), q.columnTime(3)};
}

Finding readFinding(const Statement& q) {
  Finding f;
  f.id = q.columnText(0);
  f.inspectionId = q.columnText(1);
  f.location = q.columnText(2);
  f.description = q.columnText(3);
  f.actionOpinion = q.columnText(4);
  f.assigneeId = q.optionalText(5);
  f.dueAt = q.optionalTime(6);
  f.status = detail::parseFindingStatus(q.columnText(7));
  f.createdAt = q.columnTime(8);
  f.hazardCategory = q.optionalText(9);
  f.riskLevel = q.optionalInt(10);
  return f;
}

Photo readPhoto(const Statement& q) {
  return {q.columnText(0), q.columnText(1), detail::parsePhotoKind(q.columnText(2)), q.columnText(3),
          q.columnTime(4)};
}

ActionLog readActionLog(const Statement& q) {
  return {q.columnText(0), q.columnText(1), q.columnText(2), q.columnText(3), q.columnText(4),
          q.columnTime(5)};
}

AiAnalysis readAiAnalysis(const Statement& q) {
  AiAnalysis a;
  a.id = q.columnText(0);
  a.subjectId = q.columnText(1);
  a.type = detail::parseAiAnalysisType(q.columnText(2));
  a.modelName = q.columnText(3);
  a.promptVersion = q.columnText(4);
  a.riskLevel = q.optionalInt(5);
  a.category = q.columnText(6);
  a.confidence = q.columnDouble(7);
  a.resultJson = q.columnText(8);
  a.decision = detail::parseAiReviewDecision(q.columnText(9));
  a.reviewedBy = q.optionalText(10);
  a.createdAt = q.columnTime(11);
  return a;
}

std::string selectSql(const char* columns, const char* table, const char* where, const char* orderBy = nullptr) {
  std::string sql = std::string("SELECT ") + columns + " FROM " + table + " WHERE " + where;
  if (orderBy) sql += std::string(" ORDER BY ") + orderBy;
  return sql;
}

template <typename T, typename Reader>
std::optional<T> queryOne(sqlite3* db, const std::string& sql, const Id& id, Reader read) {
  Statement q(db, sql);
  q.bind(":id", id);
  if (!q.step()) return std::nullopt;
  return read(q);
}

template <typename T, typename Reader>
std::vector<T> queryMany(sqlite3* db, const std::string& sql, const Id& key, Reader read) {
  Statement q(db, sql);
  q.bind(":key", key);
  std::vector<T> out;
  while (q.step()) out.push_back(read(q));
  return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Saves: INSERT ... ON CONFLICT(id) DO UPDATE. Never INSERT OR REPLACE, which
// deletes the old row first and would cascade-delete a finding's photos/logs.

void SqliteRepository::saveSite(const Site& v) {
  std::scoped_lock lock(impl_->mutex);
  Statement q(impl_->db,
    "INSERT INTO sites (id, name, address) VALUES (:id, :name, :address) "
    "ON CONFLICT(id) DO UPDATE SET name = excluded.name, address = excluded.address");
  q.bind(":id", v.id);
  q.bind(":name", v.name);
  q.bind(":address", v.address);
  q.run();
}

void SqliteRepository::saveProfile(const Profile& v) {
  std::scoped_lock lock(impl_->mutex);
  Statement q(impl_->db,
    "INSERT INTO profiles (id, display_name, role) VALUES (:id, :display_name, :role) "
    "ON CONFLICT(id) DO UPDATE SET display_name = excluded.display_name, role = excluded.role");
  q.bind(":id", v.id);
  q.bind(":display_name", v.displayName);
  q.bind(":role", to_string(v.role));
  q.run();
}

void SqliteRepository::saveInspection(const Inspection& v) {
  std::scoped_lock lock(impl_->mutex);
  Statement q(impl_->db,
    "INSERT INTO inspections (id, site_id, inspector_id, inspected_at) "
    "VALUES (:id, :site_id, :inspector_id, :inspected_at) "
    "ON CONFLICT(id) DO UPDATE SET site_id = excluded.site_id, inspector_id = excluded.inspector_id, "
    "inspected_at = excluded.inspected_at");
  q.bind(":id", v.id);
  q.bind(":site_id", v.siteId);
  q.bind(":inspector_id", v.inspectorId);
  q.bind(":inspected_at", v.inspectedAt);
  q.run();
}

void SqliteRepository::saveFinding(const Finding& v) {
  std::scoped_lock lock(impl_->mutex);
  Statement q(impl_->db,
    "INSERT INTO findings (id, inspection_id, location, description, action_opinion, assignee_id, due_at, "
    "status, created_at, hazard_category, risk_level) "
    "VALUES (:id, :inspection_id, :location, :description, :action_opinion, :assignee_id, :due_at, "
    ":status, :created_at, :hazard_category, :risk_level) "
    "ON CONFLICT(id) DO UPDATE SET inspection_id = excluded.inspection_id, location = excluded.location, "
    "description = excluded.description, action_opinion = excluded.action_opinion, "
    "assignee_id = excluded.assignee_id, due_at = excluded.due_at, status = excluded.status, "
    "created_at = excluded.created_at, hazard_category = excluded.hazard_category, "
    "risk_level = excluded.risk_level");
  q.bind(":id", v.id);
  q.bind(":inspection_id", v.inspectionId);
  q.bind(":location", v.location);
  q.bind(":description", v.description);
  q.bind(":action_opinion", v.actionOpinion);
  q.bind(":assignee_id", v.assigneeId);
  q.bind(":due_at", v.dueAt);
  q.bind(":status", to_string(v.status));
  q.bind(":created_at", v.createdAt);
  q.bind(":hazard_category", v.hazardCategory);
  q.bind(":risk_level", v.riskLevel);
  q.run();
}

void SqliteRepository::savePhoto(const Photo& v) {
  std::scoped_lock lock(impl_->mutex);
  Statement q(impl_->db,
    "INSERT INTO photos (id, finding_id, kind, storage_path, captured_at) "
    "VALUES (:id, :finding_id, :kind, :storage_path, :captured_at) "
    "ON CONFLICT(id) DO UPDATE SET finding_id = excluded.finding_id, kind = excluded.kind, "
    "storage_path = excluded.storage_path, captured_at = excluded.captured_at");
  q.bind(":id", v.id);
  q.bind(":finding_id", v.findingId);
  q.bind(":kind", to_string(v.kind));
  q.bind(":storage_path", v.storagePath);
  q.bind(":captured_at", v.capturedAt);
  q.run();
}

void SqliteRepository::saveActionLog(const ActionLog& v) {
  std::scoped_lock lock(impl_->mutex);
  Statement q(impl_->db,
    "INSERT INTO action_logs (id, finding_id, actor_id, action, note, created_at) "
    "VALUES (:id, :finding_id, :actor_id, :action, :note, :created_at) "
    "ON CONFLICT(id) DO UPDATE SET finding_id = excluded.finding_id, actor_id = excluded.actor_id, "
    "action = excluded.action, note = excluded.note, created_at = excluded.created_at");
  q.bind(":id", v.id);
  q.bind(":finding_id", v.findingId);
  q.bind(":actor_id", v.actorId);
  q.bind(":action", v.action);
  q.bind(":note", v.note);
  q.bind(":created_at", v.createdAt);
  q.run();
}

void SqliteRepository::saveAiAnalysis(const AiAnalysis& v) {
  std::scoped_lock lock(impl_->mutex);
  Statement q(impl_->db,
    "INSERT INTO ai_analyses (id, subject_id, analysis_type, model_name, prompt_version, risk_level, category, "
    "confidence, result_json, decision, reviewed_by, created_at) "
    "VALUES (:id, :subject_id, :analysis_type, :model_name, :prompt_version, :risk_level, :category, "
    ":confidence, :result_json, :decision, :reviewed_by, :created_at) "
    "ON CONFLICT(id) DO UPDATE SET subject_id = excluded.subject_id, analysis_type = excluded.analysis_type, "
    "model_name = excluded.model_name, prompt_version = excluded.prompt_version, "
    "risk_level = excluded.risk_level, category = excluded.category, confidence = excluded.confidence, "
    "result_json = excluded.result_json, decision = excluded.decision, reviewed_by = excluded.reviewed_by, "
    "created_at = excluded.created_at");
  q.bind(":id", v.id);
  q.bind(":subject_id", v.subjectId);
  q.bind(":analysis_type", to_string(v.type));
  q.bind(":model_name", v.modelName);
  q.bind(":prompt_version", v.promptVersion);
  q.bind(":risk_level", v.riskLevel);
  q.bind(":category", v.category);
  q.bind(":confidence", v.confidence);
  q.bind(":result_json", v.resultJson);
  q.bind(":decision", to_string(v.decision));
  q.bind(":reviewed_by", v.reviewedBy);
  q.bind(":created_at", v.createdAt);
  q.run();
}

// ---------------------------------------------------------------------------
// Lookups by id

std::optional<Site> SqliteRepository::findSite(const Id& id) const {
  std::scoped_lock lock(impl_->mutex);
  return queryOne<Site>(impl_->db, selectSql(kSiteColumns, "sites", "id = :id"), id, readSite);
}

std::optional<Profile> SqliteRepository::findProfile(const Id& id) const {
  std::scoped_lock lock(impl_->mutex);
  return queryOne<Profile>(impl_->db, selectSql(kProfileColumns, "profiles", "id = :id"), id, readProfile);
}

std::optional<Inspection> SqliteRepository::findInspection(const Id& id) const {
  std::scoped_lock lock(impl_->mutex);
  return queryOne<Inspection>(impl_->db, selectSql(kInspectionColumns, "inspections", "id = :id"), id,
                              readInspection);
}

std::optional<Finding> SqliteRepository::findFinding(const Id& id) const {
  std::scoped_lock lock(impl_->mutex);
  return queryOne<Finding>(impl_->db, selectSql(kFindingColumns, "findings", "id = :id"), id, readFinding);
}

std::optional<AiAnalysis> SqliteRepository::findAiAnalysis(const Id& id) const {
  std::scoped_lock lock(impl_->mutex);
  return queryOne<AiAnalysis>(impl_->db, selectSql(kAiAnalysisColumns, "ai_analyses", "id = :id"), id,
                              readAiAnalysis);
}

// ---------------------------------------------------------------------------
// Lists: ordered by time, then id (InMemoryRepository's order is unspecified).

std::vector<Photo> SqliteRepository::photosForFinding(const Id& findingId) const {
  std::scoped_lock lock(impl_->mutex);
  return queryMany<Photo>(impl_->db,
    selectSql(kPhotoColumns, "photos", "finding_id = :key", "captured_at, id"), findingId, readPhoto);
}

std::vector<ActionLog> SqliteRepository::logsForFinding(const Id& findingId) const {
  std::scoped_lock lock(impl_->mutex);
  return queryMany<ActionLog>(impl_->db,
    selectSql(kActionLogColumns, "action_logs", "finding_id = :key", "created_at, id"), findingId,
    readActionLog);
}

std::vector<Finding> SqliteRepository::findingsForInspection(const Id& inspectionId) const {
  std::scoped_lock lock(impl_->mutex);
  return queryMany<Finding>(impl_->db,
    selectSql(kFindingColumns, "findings", "inspection_id = :key", "created_at, id"), inspectionId,
    readFinding);
}

std::vector<Finding> SqliteRepository::findingsAssignedTo(const Id& assigneeId) const {
  std::scoped_lock lock(impl_->mutex);
  return queryMany<Finding>(impl_->db,
    selectSql(kFindingColumns, "findings", "assignee_id = :key", "created_at, id"), assigneeId, readFinding);
}

std::vector<AiAnalysis> SqliteRepository::analysesForSubject(const Id& subjectId) const {
  std::scoped_lock lock(impl_->mutex);
  return queryMany<AiAnalysis>(impl_->db,
    selectSql(kAiAnalysisColumns, "ai_analyses", "subject_id = :key", "created_at, id"), subjectId,
    readAiAnalysis);
}

} // namespace safelog::storage
