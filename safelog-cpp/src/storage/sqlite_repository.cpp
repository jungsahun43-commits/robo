#include "safelog/storage/sqlite_repository.hpp"

#include "safelog/contracts/errors.hpp"
#include "safelog/storage/schema_sql.hpp"

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

  int columnInt(int index) const { return sqlite3_column_int(stmt_, index); }

private:
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

namespace {
[[noreturn]] void notImplemented(const char* name) {
  throw std::logic_error(std::string("SqliteRepository::") + name + " is not implemented yet");
}
} // namespace

void SqliteRepository::saveSite(const Site&) { notImplemented("saveSite"); }
void SqliteRepository::saveProfile(const Profile&) { notImplemented("saveProfile"); }
void SqliteRepository::saveInspection(const Inspection&) { notImplemented("saveInspection"); }
void SqliteRepository::saveFinding(const Finding&) { notImplemented("saveFinding"); }
void SqliteRepository::savePhoto(const Photo&) { notImplemented("savePhoto"); }
void SqliteRepository::saveActionLog(const ActionLog&) { notImplemented("saveActionLog"); }
void SqliteRepository::saveAiAnalysis(const AiAnalysis&) { notImplemented("saveAiAnalysis"); }

std::optional<Site> SqliteRepository::findSite(const Id&) const { notImplemented("findSite"); }
std::optional<Profile> SqliteRepository::findProfile(const Id&) const { notImplemented("findProfile"); }
std::optional<Inspection> SqliteRepository::findInspection(const Id&) const { notImplemented("findInspection"); }
std::optional<Finding> SqliteRepository::findFinding(const Id&) const { notImplemented("findFinding"); }
std::optional<AiAnalysis> SqliteRepository::findAiAnalysis(const Id&) const { notImplemented("findAiAnalysis"); }
std::vector<Photo> SqliteRepository::photosForFinding(const Id&) const { notImplemented("photosForFinding"); }
std::vector<ActionLog> SqliteRepository::logsForFinding(const Id&) const { notImplemented("logsForFinding"); }
std::vector<Finding> SqliteRepository::findingsForInspection(const Id&) const { notImplemented("findingsForInspection"); }
std::vector<Finding> SqliteRepository::findingsAssignedTo(const Id&) const { notImplemented("findingsAssignedTo"); }
std::vector<AiAnalysis> SqliteRepository::analysesForSubject(const Id&) const { notImplemented("analysesForSubject"); }

} // namespace safelog::storage
