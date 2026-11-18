// AuthService tests. Behavioural tests run against both credential stores
// (in-memory and SQLite); persistence and migration tests run on SQLite only.
// Uses its own CHECK macro (not assert) so checks still run in Release builds.

#include "safelog/auth/auth_service.hpp"
#include "safelog/contracts/errors.hpp"
#include "safelog/storage/in_memory_credential_store.hpp"
#include "safelog/storage/in_memory_repository.hpp"
#include "safelog/storage/sqlite_repository.hpp"
#include "safelog/storage/uuid_id_generator.hpp"

#include <sqlite3.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace safelog;
using namespace safelog::auth;
using namespace std::chrono;
namespace fs = std::filesystem;

namespace {

int failures = 0;

#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "  %s:%d CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
      ++failures;                                                                     \
    }                                                                                 \
  } while (0)

template <typename E, typename F>
bool throwsAs(F&& f) {
  try { f(); } catch (const E&) { return true; } catch (...) { return false; }
  return false;
}
#define CHECK_THROWS(ErrorType, expr) CHECK(throwsAs<ErrorType>([&] { expr; }))

// Returns the AuthenticationError message, or "<no AuthenticationError>".
template <typename F>
std::string authFailure(F&& f) {
  try { f(); } catch (const AuthenticationError& e) { return e.what(); } catch (...) {}
  return "<no AuthenticationError>";
}
#define CHECK_LOGIN_FAILS(expr) CHECK(authFailure([&] { expr; }) == kLoginFailedMessage)

fs::path freshDir(const std::string& name) {
  const auto dir = fs::temp_directory_path() / "safelog-auth-tests" / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

std::string readFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int countRows(const fs::path& db, const char* table) {
  sqlite3* raw = nullptr;
  sqlite3_open(db.string().c_str(), &raw);
  sqlite3_stmt* q = nullptr;
  sqlite3_prepare_v2(raw, (std::string("SELECT COUNT(*) FROM ") + table).c_str(), -1, &q, nullptr);
  const int n = sqlite3_step(q) == SQLITE_ROW ? sqlite3_column_int(q, 0) : -1;
  sqlite3_finalize(q);
  sqlite3_close(raw);
  return n;
}

void execRaw(const fs::path& db, const char* sql) {
  sqlite3* raw = nullptr;
  CHECK(sqlite3_open(db.string().c_str(), &raw) == SQLITE_OK);
  char* error = nullptr;
  if (sqlite3_exec(raw, sql, nullptr, nullptr, &error) != SQLITE_OK) {
    std::fprintf(stderr, "  raw SQL failed: %s\n", error ? error : "?");
    sqlite3_free(error);
    ++failures;
  }
  sqlite3_close(raw);
}

class FakeClock final : public IClock {
public:
  TimePoint now() const override { return now_; }
  void advance(system_clock::duration d) { now_ += d; }
private:
  TimePoint now_ = time_point_cast<TimePoint::duration>(sys_days{2026y / November / 18} + 9h);
};

// Cheap Argon2 settings keep the suite fast; defaults are covered in testDefaultHasher.
const Argon2Params kFastParams{8, 1, 1};

const DemoAccountPasswords kDemoPasswords{"inspector-pass", "manager-pass", "assignee-pass"};

// --- backends ------------------------------------------------------------------

struct Backend {
  virtual ~Backend() = default;
  virtual IRepository& repo() = 0;
  virtual ICredentialStore& store() = 0;
  virtual void reopen() {}
};

struct MemoryBackend final : Backend {
  storage::InMemoryRepository repository;
  storage::InMemoryCredentialStore credentials{repository};
  IRepository& repo() override { return repository; }
  ICredentialStore& store() override { return credentials; }
};

struct SqliteBackend final : Backend {
  fs::path path;
  std::unique_ptr<storage::SqliteRepository> db;
  explicit SqliteBackend(fs::path p) : path(std::move(p)), db(std::make_unique<storage::SqliteRepository>(path)) {}
  IRepository& repo() override { return *db; }
  ICredentialStore& store() override { return *db; }
  void reopen() override {
    db.reset();
    db = std::make_unique<storage::SqliteRepository>(path);
  }
};

// Everything a test needs; service() is rebuilt after reopen() because it holds references.
struct Env {
  std::unique_ptr<Backend> backend;
  FakeClock clock;
  storage::UuidIdGenerator ids;
  PasswordHasher hasher{kFastParams};

  AuthService service() { return AuthService(backend->store(), backend->repo(), clock, ids, hasher); }
  CredentialRecord credential(const Id& profileId) { return *backend->store().findCredentialByProfileId(profileId); }

  // The three demo people (as storage seedDemoData creates them) with demo logins.
  void addDemoPeople() {
    backend->repo().saveProfile({"inspector-1", "김안전", UserRole::Inspector});
    backend->repo().saveProfile({"manager-1", "박관리", UserRole::Manager});
    backend->repo().saveProfile({"assignee-1", "이조치", UserRole::Assignee});
    service().seedDemoAccounts(kDemoPasswords);
  }
};

using EnvFactory = std::function<std::unique_ptr<Env>(const std::string& testName)>;

std::unique_ptr<Env> memoryEnv(const std::string&) {
  auto env = std::make_unique<Env>();
  env->backend = std::make_unique<MemoryBackend>();
  return env;
}

std::unique_ptr<Env> sqliteEnv(const std::string& testName) {
  auto env = std::make_unique<Env>();
  env->backend = std::make_unique<SqliteBackend>(freshDir(testName) / "safelog.db");
  return env;
}

// --- tests on both backends -----------------------------------------------------

void testLoginSuccessAndFailure(Env& env) {
  env.addDemoPeople();
  auto auth = env.service();
  const auto ok = auth.login("inspector1", "inspector-pass");
  CHECK(ok.profile.id == "inspector-1" && ok.profile.role == UserRole::Inspector);
  CHECK(!ok.mustChangePassword);
  CHECK(auth.login("  MANAGER1 ", "manager-pass").profile.id == "manager-1");  // trimmed, case-insensitive

  // Wrong password, unknown id, malformed id and over-long input all fail identically.
  CHECK_LOGIN_FAILS(auth.login("inspector1", "wrong-password"));
  CHECK_LOGIN_FAILS(auth.login("nobody", "inspector-pass"));
  CHECK_LOGIN_FAILS(auth.login("x", "inspector-pass"));
  CHECK_LOGIN_FAILS(auth.login("inspector1", ""));
  CHECK_LOGIN_FAILS(auth.login("inspector1", std::string(129, 'a')));
  CHECK_LOGIN_FAILS(auth.login("inspector1", "Inspector-pass"));  // passwords are case-sensitive
}

void testNoPlaintextAndSalt(Env& env) {
  env.addDemoPeople();
  auto auth = env.service();
  const std::string password = "Same-Temp-Pass-123";
  const auto a = auth.createAccount("manager-1", "최점검", UserRole::Inspector, "choi", password);
  const auto b = auth.createAccount("manager-1", "정담당", UserRole::Assignee, "jung", password);
  const auto ca = env.credential(a.id);
  const auto cb = env.credential(b.id);
  CHECK(ca.salt.size() == 16 && cb.salt.size() == 16);
  CHECK(ca.salt != cb.salt);
  CHECK(ca.passwordHash != cb.passwordHash);
  CHECK(ca.passwordHash.rfind("$argon2id$v=19$m=8,t=1,p=1$", 0) == 0);
  CHECK(ca.passwordHash.find(password) == std::string::npos);
  CHECK(auth.login("choi", password).profile.id == a.id);
  CHECK(auth.login("jung", password).profile.id == b.id);
}

void testLockAndUnlock(Env& env) {
  env.addDemoPeople();
  auto auth = env.service();
  for (int i = 0; i < AuthService::kMaxFailedAttempts - 1; ++i) CHECK_LOGIN_FAILS(auth.login("assignee1", "wrong-pass"));
  CHECK(env.credential("assignee-1").failedCount == 4);
  CHECK(!env.credential("assignee-1").lockedUntil);

  CHECK_LOGIN_FAILS(auth.login("assignee1", "wrong-pass"));  // 5th failure locks
  const auto locked = env.credential("assignee-1");
  CHECK(locked.lockedUntil && *locked.lockedUntil == env.clock.now() + AuthService::kLockDuration);
  CHECK(locked.failedCount == 0);

  CHECK_LOGIN_FAILS(auth.login("assignee1", "assignee-pass"));  // correct password refused while locked
  CHECK_LOGIN_FAILS(auth.login("assignee1", "wrong-pass"));     // does not extend or count
  CHECK(env.credential("assignee-1").lockedUntil == locked.lockedUntil);
  CHECK(env.credential("assignee-1").failedCount == 0);
  CHECK(auth.login("manager1", "manager-pass").profile.id == "manager-1");  // other accounts unaffected

  env.clock.advance(59s);
  CHECK_LOGIN_FAILS(auth.login("assignee1", "assignee-pass"));
  env.clock.advance(2s);
  CHECK(auth.login("assignee1", "assignee-pass").profile.id == "assignee-1");
  CHECK(!env.credential("assignee-1").lockedUntil);

  // A success resets the streak: 4 failures, success, 4 failures -> still not locked.
  for (int i = 0; i < 4; ++i) CHECK_LOGIN_FAILS(auth.login("assignee1", "wrong-pass"));
  CHECK(auth.login("assignee1", "assignee-pass").profile.id == "assignee-1");
  CHECK(env.credential("assignee-1").failedCount == 0);
  for (int i = 0; i < 4; ++i) CHECK_LOGIN_FAILS(auth.login("assignee1", "wrong-pass"));
  CHECK(auth.login("assignee1", "assignee-pass").profile.id == "assignee-1");
}

void testManagerOnlyAccountCreation(Env& env) {
  env.addDemoPeople();
  auto auth = env.service();
  CHECK_THROWS(PermissionError, auth.createAccount("inspector-1", "홍길동", UserRole::Inspector, "hong", "temp-pass-1"));
  CHECK_THROWS(PermissionError, auth.createAccount("assignee-1", "홍길동", UserRole::Inspector, "hong", "temp-pass-1"));
  CHECK_THROWS(PermissionError, auth.createAccount("no-such-person", "홍길동", UserRole::Inspector, "hong", "temp-pass-1"));
  CHECK(!env.backend->store().findCredentialByLoginId("hong"));

  const auto created = auth.createAccount("manager-1", "홍길동", UserRole::Assignee, "Hong.GD", "temp-pass-1");
  const auto stored = env.backend->repo().findProfile(created.id);
  CHECK(stored && stored->displayName == "홍길동" && stored->role == UserRole::Assignee);
  CHECK(env.credential(created.id).loginId == "hong.gd");
  const auto first = auth.login("hong.gd", "temp-pass-1");
  CHECK(first.profile.id == created.id && first.mustChangePassword);

  // Input validation
  CHECK_THROWS(ValidationError, auth.createAccount("manager-1", "", UserRole::Inspector, "empty-name", "temp-pass-1"));
  CHECK_THROWS(ValidationError, auth.createAccount("manager-1", "가", UserRole::Inspector, "bad id!", "temp-pass-1"));
  CHECK_THROWS(ValidationError, auth.createAccount("manager-1", "가", UserRole::Inspector, "ab", "temp-pass-1"));
  CHECK_THROWS(ValidationError, auth.createAccount("manager-1", "가", UserRole::Inspector, "short", "1234567"));
  CHECK_THROWS(ValidationError, auth.createAccount("manager-1", "가", UserRole::Inspector, "long", std::string(129, 'a')));
  auth.createAccount("manager-1", "가", UserRole::Inspector, "korean-pw", "가나다라마바사아");  // 8 chars, 24 bytes
  CHECK(auth.login("korean-pw", "가나다라마바사아").mustChangePassword);
}

void testDuplicateLoginId(Env& env) {
  env.addDemoPeople();
  auto auth = env.service();
  CHECK_THROWS(ValidationError, auth.createAccount("manager-1", "중복", UserRole::Inspector, "inspector1", "temp-pass-1"));
  CHECK_THROWS(ValidationError, auth.createAccount("manager-1", "중복", UserRole::Inspector, " Inspector1 ", "temp-pass-1"));

  // The store itself rejects duplicates atomically: no orphan profile is left behind.
  auto record = env.credential("inspector-1");
  record.profileId = "orphan-candidate";
  record.loginId = "INSPECTOR1";
  CHECK_THROWS(ValidationError, env.backend->store().createAccount({"orphan-candidate", "고아", UserRole::Inspector}, record));
  CHECK(!env.backend->repo().findProfile("orphan-candidate"));

  // A second login for the same profile is rejected too.
  record = env.credential("inspector-1");
  record.loginId = "another-id";
  CHECK_THROWS(ValidationError, env.backend->store().addCredential(record));
  CHECK(!env.backend->store().findCredentialByLoginId("another-id"));
}

void testChangePassword(Env& env) {
  env.addDemoPeople();
  auto auth = env.service();
  const auto person = auth.createAccount("manager-1", "변경자", UserRole::Inspector, "changer", "temp-pass-1");
  CHECK(auth.login("changer", "temp-pass-1").mustChangePassword);

  CHECK_LOGIN_FAILS(auth.changePassword(person.id, "wrong-current", "new-pass-123"));
  CHECK_THROWS(ValidationError, auth.changePassword(person.id, "temp-pass-1", "short"));
  CHECK(auth.login("changer", "temp-pass-1").mustChangePassword);  // unchanged so far

  const auto before = env.credential(person.id);
  auth.changePassword(person.id, "temp-pass-1", "new-pass-123");
  const auto after = env.credential(person.id);
  CHECK(after.salt != before.salt && after.passwordHash != before.passwordHash);
  CHECK(!after.mustChange);
  CHECK_LOGIN_FAILS(auth.login("changer", "temp-pass-1"));
  const auto result = auth.login("changer", "new-pass-123");
  CHECK(result.profile.id == person.id && !result.mustChangePassword);

  // Wrong current passwords count toward the lock.
  for (int i = 0; i < AuthService::kMaxFailedAttempts; ++i)
    CHECK_LOGIN_FAILS(auth.changePassword(person.id, "wrong-current", "another-pass-1"));
  CHECK_LOGIN_FAILS(auth.login("changer", "new-pass-123"));
  CHECK_LOGIN_FAILS(auth.changePassword("no-such-person", "x", "another-pass-1"));
}

void testSeedDemoAccounts(Env& env) {
  auto& repo = env.backend->repo();
  auto& store = env.backend->store();

  // Weak passwords are rejected before anything is written.
  CHECK_THROWS(ValidationError, env.service().seedDemoAccounts({"inspector-pass", "short", "assignee-pass"}));
  CHECK(!store.findCredentialByProfileId("inspector-1") && !repo.findProfile("inspector-1"));

  // Existing people (as storage::seedDemoData creates them) get logins; nobody is duplicated or renamed.
  repo.saveProfile({"inspector-1", "김안전", UserRole::Inspector});
  repo.saveProfile({"manager-1", "박관리(수정됨)", UserRole::Manager});
  env.service().seedDemoAccounts(kDemoPasswords);
  CHECK(repo.findProfile("manager-1")->displayName == "박관리(수정됨)");
  // assignee-1 did not exist, so it was created with the demo name.
  const auto assignee = repo.findProfile("assignee-1");
  CHECK(assignee && assignee->displayName == "이조치" && assignee->role == UserRole::Assignee);

  auto auth = env.service();
  for (const auto& [login, password, profileId] :
       {std::tuple{"inspector1", "inspector-pass", "inspector-1"}, std::tuple{"manager1", "manager-pass", "manager-1"},
        std::tuple{"assignee1", "assignee-pass", "assignee-1"}}) {
    const auto result = auth.login(login, password);
    CHECK(result.profile.id == profileId && !result.mustChangePassword);
  }

  // Seeding again (next app start) neither fails nor overwrites passwords.
  env.service().seedDemoAccounts({"other-pass-1", "other-pass-2", "other-pass-3"});
  CHECK(auth.login("manager1", "manager-pass").profile.id == "manager-1");
  CHECK_LOGIN_FAILS(auth.login("manager1", "other-pass-2"));
}

// --- SQLite-only tests ------------------------------------------------------------

// Demo seed in both orders never produces a second person.
void testSeedNoDuplicatePeopleSqlite() {
  const auto db = freshDir("seed-order") / "safelog.db";
  FakeClock clock;
  storage::UuidIdGenerator ids;
  PasswordHasher hasher{kFastParams};
  {
    storage::SqliteRepository repo(db);
    repo.seedDemoData();
    AuthService(repo, repo, clock, ids, hasher).seedDemoAccounts(kDemoPasswords);
    repo.seedDemoData();
    AuthService(repo, repo, clock, ids, hasher).seedDemoAccounts(kDemoPasswords);
  }
  CHECK(countRows(db, "profiles") == 3);
  CHECK(countRows(db, "credentials") == 3);

  const auto reversed = freshDir("seed-order-reversed") / "safelog.db";
  {
    storage::SqliteRepository repo(reversed);
    AuthService(repo, repo, clock, ids, hasher).seedDemoAccounts(kDemoPasswords);  // creates the people
    repo.seedDemoData();                                                           // must not add more
  }
  CHECK(countRows(reversed, "profiles") == 3);
  CHECK(countRows(reversed, "credentials") == 3);
}

void testPlaintextNeverWrittenToDisk() {
  const auto dir = freshDir("plaintext");
  const std::string password = "Unique-Plaintext-Marker-42";
  {
    storage::SqliteRepository repo(dir / "safelog.db");
    FakeClock clock;
    storage::UuidIdGenerator ids;
    PasswordHasher hasher{kFastParams};
    repo.seedDemoData();
    AuthService auth(repo, repo, clock, ids, hasher);
    auth.seedDemoAccounts(kDemoPasswords);
    const auto p = auth.createAccount("manager-1", "디스크", UserRole::Inspector, "disk", password);
    auth.changePassword(p.id, password, password + "-2");
    CHECK_LOGIN_FAILS(auth.login("disk", password));
  }
  for (const auto& entry : fs::directory_iterator(dir)) {
    const auto bytes = readFile(entry.path());
    CHECK(bytes.find(password) == std::string::npos);
    CHECK(bytes.find("inspector-pass") == std::string::npos);
  }
}

void testStateSurvivesRestart() {
  auto env = sqliteEnv("restart");
  env->addDemoPeople();
  const auto person = env->service().createAccount("manager-1", "재시작", UserRole::Assignee, "restart", "temp-pass-1");
  CHECK_LOGIN_FAILS(env->service().login("restart", "wrong-pass"));
  CHECK_LOGIN_FAILS(env->service().login("restart", "wrong-pass"));

  env->backend->reopen();
  auto record = env->credential(person.id);
  CHECK(record.failedCount == 2 && record.mustChange && record.loginId == "restart");
  for (int i = 0; i < 3; ++i) CHECK_LOGIN_FAILS(env->service().login("restart", "wrong-pass"));

  env->backend->reopen();
  CHECK(env->credential(person.id).lockedUntil.has_value());
  CHECK_LOGIN_FAILS(env->service().login("restart", "temp-pass-1"));  // still locked after restart
  env->clock.advance(61s);
  CHECK(env->service().login("restart", "temp-pass-1").mustChangePassword);
  env->service().changePassword(person.id, "temp-pass-1", "final-pass-1");

  env->backend->reopen();
  const auto result = env->service().login("restart", "final-pass-1");
  CHECK(result.profile.id == person.id && result.profile.displayName == "재시작" && !result.mustChangePassword);
  CHECK(env->service().login("manager1", "manager-pass").profile.id == "manager-1");
}

void testMigrationV1ToV2() {
  const auto db = freshDir("migration-v1") / "safelog.db";
  {
    storage::SqliteRepository repo(db);
    repo.seedDemoData();
    repo.saveInspection({"inspection-1", "site-1", "inspector-1", system_clock::now()});
  }
  // Turn it into a v1 database: v1 is v2 without the credentials table.
  execRaw(db, "DROP TABLE credentials; PRAGMA user_version = 1;");
  CHECK(countRows(db, "profiles") == 3);

  storage::SqliteRepository repo(db);
  CHECK(repo.schemaVersion() == 2);
  CHECK(repo.findInspection("inspection-1").has_value());
  CHECK(repo.findProfile("manager-1")->displayName == "박관리");
  FakeClock clock;
  storage::UuidIdGenerator ids;
  PasswordHasher hasher{kFastParams};
  AuthService auth(repo, repo, clock, ids, hasher);
  auth.seedDemoAccounts(kDemoPasswords);
  CHECK(auth.login("manager1", "manager-pass").profile.id == "manager-1");
}

void testMigrationV0ToV2() {
  const auto db = freshDir("migration-v0") / "safelog.db";
  execRaw(db,
    "CREATE TABLE sites (id TEXT PRIMARY KEY, name TEXT NOT NULL, address TEXT NOT NULL);"
    "CREATE TABLE profiles (id TEXT PRIMARY KEY, display_name TEXT NOT NULL, role TEXT NOT NULL);"
    "CREATE TABLE inspections (id TEXT PRIMARY KEY, site_id TEXT NOT NULL REFERENCES sites(id),"
    "  inspector_id TEXT NOT NULL REFERENCES profiles(id), inspected_at TEXT NOT NULL);"
    "CREATE TABLE findings (id TEXT PRIMARY KEY, inspection_id TEXT NOT NULL REFERENCES inspections(id),"
    "  location TEXT NOT NULL, description TEXT NOT NULL, action_opinion TEXT NOT NULL,"
    "  assignee_id TEXT REFERENCES profiles(id), due_at TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL);"
    "INSERT INTO profiles VALUES ('manager-1', 'old manager', 'manager');");
  storage::SqliteRepository repo(db);
  CHECK(repo.schemaVersion() == 2);
  FakeClock clock;
  storage::UuidIdGenerator ids;
  PasswordHasher hasher{kFastParams};
  AuthService auth(repo, repo, clock, ids, hasher);
  auth.seedDemoAccounts(kDemoPasswords);
  const auto result = auth.login("manager1", "manager-pass");
  CHECK(result.profile.id == "manager-1" && result.profile.displayName == "old manager");
}

// --- hasher ---------------------------------------------------------------------------

void testDefaultHasher() {
  const PasswordHasher hasher;  // production parameters
  const auto hashed = hasher.hash("correct horse battery");
  CHECK(hashed.encoded.rfind("$argon2id$v=19$m=19456,t=2,p=1$", 0) == 0);
  CHECK(hashed.salt.size() == PasswordHasher::kSaltSize);
  CHECK(hasher.verify("correct horse battery", hashed.salt, hashed.encoded));
  CHECK(!hasher.verify("correct horse batterY", hashed.salt, hashed.encoded));

  // A hash made with other parameters still verifies (parameters are read from the string).
  const PasswordHasher fast{kFastParams};
  const auto fastHashed = fast.hash("pw-12345678");
  CHECK(hasher.verify("pw-12345678", fastHashed.salt, fastHashed.encoded));

  // Corrupted stored values are rejected without throwing.
  CHECK(!hasher.verify("pw-12345678", fastHashed.salt, "garbage"));
  CHECK(!hasher.verify("pw-12345678", fastHashed.salt, "$argon2id$v=19$m=99999999,t=1,p=1$AAAA"));
  CHECK(!hasher.verify("pw-12345678", std::vector<std::uint8_t>(8), fastHashed.encoded));
  auto truncated = fastHashed.encoded;
  truncated.pop_back();
  CHECK(!hasher.verify("pw-12345678", fastHashed.salt, truncated));
}

} // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void(Env&)>>> sharedTests = {
    {"login success and failure", testLoginSuccessAndFailure},
    {"no plaintext, unique salt", testNoPlaintextAndSalt},
    {"lock and unlock", testLockAndUnlock},
    {"manager-only account creation", testManagerOnlyAccountCreation},
    {"duplicate login id", testDuplicateLoginId},
    {"change password", testChangePassword},
    {"seed demo accounts", testSeedDemoAccounts},
  };
  const std::vector<std::pair<const char*, std::function<void()>>> standaloneTests = {
    {"sqlite: seed creates no duplicate people", testSeedNoDuplicatePeopleSqlite},
    {"sqlite: plaintext never written to disk", testPlaintextNeverWrittenToDisk},
    {"sqlite: state survives restart", testStateSurvivesRestart},
    {"sqlite: migration v1 -> v2", testMigrationV1ToV2},
    {"sqlite: migration v0 -> v2", testMigrationV0ToV2},
    {"default Argon2id hasher", testDefaultHasher},
  };

  const auto report = [](const std::string& name, const std::function<void()>& run) {
    const int before = failures;
    try {
      run();
    } catch (const std::exception& error) {
      std::fprintf(stderr, "  unexpected exception: %s\n", error.what());
      ++failures;
    }
    std::printf("[%s] %s\n", failures == before ? "PASS" : "FAIL", name.c_str());
  };

  for (const auto& [backendName, factory] :
       {std::pair<const char*, EnvFactory>{"memory", memoryEnv}, std::pair<const char*, EnvFactory>{"sqlite", sqliteEnv}}) {
    int index = 0;
    for (const auto& [name, test] : sharedTests) {
      const std::string label = std::string(backendName) + ": " + name;
      report(label, [&, i = index++] {
        auto env = factory(std::string(backendName) + "-" + std::to_string(i));
        test(*env);
      });
    }
  }
  for (const auto& [name, test] : standaloneTests) report(name, test);

  std::printf("%s (%d failed checks)\n", failures == 0 ? "ALL PASSED" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
