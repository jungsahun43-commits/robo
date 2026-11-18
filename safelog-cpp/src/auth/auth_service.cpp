#include "safelog/auth/auth_service.hpp"

#include "safelog/contracts/errors.hpp"

#include <array>

namespace safelog::auth {
namespace {

struct DemoAccount {
  const char* profileId;
  const char* displayName;
  UserRole role;
  const char* loginId;
  const std::string DemoAccountPasswords::*password;
};

// Same profiles as storage::SqliteRepository::seedDemoData().
const std::array<DemoAccount, 3> kDemoAccounts{{
  {"inspector-1", "김안전", UserRole::Inspector, "inspector1", &DemoAccountPasswords::inspector},
  {"manager-1", "박관리", UserRole::Manager, "manager1", &DemoAccountPasswords::manager},
  {"assignee-1", "이조치", UserRole::Assignee, "assignee1", &DemoAccountPasswords::assignee},
}};

std::size_t countUtf8Chars(std::string_view text) {
  std::size_t count = 0;
  for (const char c : text)
    if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) ++count;  // skip continuation bytes
  return count;
}

} // namespace

AuthService::AuthService(ICredentialStore& credentials, IRepository& repository, IClock& clock, IIdGenerator& ids,
                         const PasswordHasher& hasher)
  : credentials_(credentials), repository_(repository), clock_(clock), ids_(ids), hasher_(hasher) {}

std::string AuthService::normalizeLoginId(std::string_view loginId) {
  const auto isSpace = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  while (!loginId.empty() && isSpace(loginId.front())) loginId.remove_prefix(1);
  while (!loginId.empty() && isSpace(loginId.back())) loginId.remove_suffix(1);
  if (loginId.size() < 3 || loginId.size() > 32) return {};
  std::string out;
  out.reserve(loginId.size());
  for (char c : loginId) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    const bool allowed = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    if (!allowed) return {};
    out += c;
  }
  return out;
}

void AuthService::checkPasswordPolicy(std::string_view password) {
  if (password.size() > kMaxPasswordBytes)
    throw ValidationError("비밀번호는 " + std::to_string(kMaxPasswordBytes) + "바이트 이하여야 합니다");
  if (countUtf8Chars(password) < kMinPasswordChars)
    throw ValidationError("비밀번호는 " + std::to_string(kMinPasswordChars) + "자 이상이어야 합니다");
}

CredentialRecord AuthService::newCredential(const Id& profileId, const std::string& loginId,
                                            std::string_view password, bool mustChange) const {
  auto hashed = hasher_.hash(password);
  const auto now = clock_.now();
  CredentialRecord record;
  record.profileId = profileId;
  record.loginId = loginId;
  record.passwordHash = std::move(hashed.encoded);
  record.salt = std::move(hashed.salt);
  record.mustChange = mustChange;
  record.createdAt = now;
  record.updatedAt = now;
  return record;
}

bool AuthService::isLocked(const CredentialRecord& record) const {
  return record.lockedUntil && clock_.now() < *record.lockedUntil;
}

void AuthService::recordFailure(CredentialRecord& record) {
  const auto now = clock_.now();
  if (record.lockedUntil && now >= *record.lockedUntil) record.lockedUntil.reset();  // expired lock
  if (++record.failedCount >= kMaxFailedAttempts) {
    record.lockedUntil = now + kLockDuration;
    record.failedCount = 0;
  }
  record.updatedAt = now;
  credentials_.updateCredential(record);
}

LoginResult AuthService::login(const std::string& loginId, const std::string& password) {
  std::scoped_lock lock(mutex_);
  // Over-long input is still hashed (truncated) so the failure costs the same time.
  const std::string_view attempt = std::string_view(password).substr(0, kMaxPasswordBytes);
  const auto normalized = normalizeLoginId(loginId);
  auto record = normalized.empty() ? std::nullopt : credentials_.findCredentialByLoginId(normalized);

  if (!record || isLocked(*record) || password.size() > kMaxPasswordBytes) {
    hasher_.burn(attempt);
    throw AuthenticationError();
  }
  if (!hasher_.verify(attempt, record->salt, record->passwordHash)) {
    recordFailure(*record);
    throw AuthenticationError();
  }
  const auto profile = repository_.findProfile(record->profileId);
  if (!profile) throw AuthenticationError();  // cannot happen with the profiles foreign key

  if (record->failedCount != 0 || record->lockedUntil) {
    record->failedCount = 0;
    record->lockedUntil.reset();
    record->updatedAt = clock_.now();
    credentials_.updateCredential(*record);
  }
  return {*profile, record->mustChange};
}

void AuthService::changePassword(const Id& profileId, const std::string& currentPassword,
                                 const std::string& newPassword) {
  checkPasswordPolicy(newPassword);
  std::scoped_lock lock(mutex_);
  const std::string_view attempt = std::string_view(currentPassword).substr(0, kMaxPasswordBytes);
  auto record = credentials_.findCredentialByProfileId(profileId);
  if (!record || isLocked(*record) || currentPassword.size() > kMaxPasswordBytes) {
    hasher_.burn(attempt);
    throw AuthenticationError();
  }
  if (!hasher_.verify(attempt, record->salt, record->passwordHash)) {
    recordFailure(*record);
    throw AuthenticationError();
  }
  auto hashed = hasher_.hash(newPassword);
  record->passwordHash = std::move(hashed.encoded);
  record->salt = std::move(hashed.salt);
  record->mustChange = false;
  record->failedCount = 0;
  record->lockedUntil.reset();
  record->updatedAt = clock_.now();
  credentials_.updateCredential(*record);
}

Profile AuthService::createAccount(const Id& managerId, const std::string& displayName, UserRole role,
                                   const std::string& loginId, const std::string& temporaryPassword) {
  const auto manager = repository_.findProfile(managerId);
  if (!manager || manager->role != UserRole::Manager)
    throw PermissionError("관리자만 계정을 만들 수 있습니다");
  if (displayName.empty()) throw ValidationError("이름을 입력해야 합니다");
  const auto normalized = normalizeLoginId(loginId);
  if (normalized.empty()) throw ValidationError("아이디는 3~32자의 영문 소문자, 숫자, '.', '_', '-'만 쓸 수 있습니다");
  checkPasswordPolicy(temporaryPassword);

  std::scoped_lock lock(mutex_);
  if (credentials_.findCredentialByLoginId(normalized)) throw ValidationError("이미 사용 중인 아이디입니다");
  const Profile profile{ids_.next("profile"), displayName, role};
  // The store re-checks uniqueness and writes profile + credential atomically.
  credentials_.createAccount(profile, newCredential(profile.id, normalized, temporaryPassword, true));
  return profile;
}

void AuthService::seedDemoAccounts(const DemoAccountPasswords& passwords) {
  for (const auto& account : kDemoAccounts) checkPasswordPolicy(passwords.*account.password);
  std::scoped_lock lock(mutex_);
  for (const auto& account : kDemoAccounts) {
    if (credentials_.findCredentialByProfileId(account.profileId)) continue;  // keep existing password
    if (credentials_.findCredentialByLoginId(account.loginId))
      throw ValidationError(std::string("시연 계정 아이디가 다른 사용자에게 연결되어 있습니다: ") + account.loginId);
    auto credential = newCredential(account.profileId, account.loginId, passwords.*account.password, false);
    if (repository_.findProfile(account.profileId)) {
      credentials_.addCredential(credential);  // link to the existing person
    } else {
      credentials_.createAccount({account.profileId, account.displayName, account.role}, credential);
    }
  }
}

} // namespace safelog::auth
