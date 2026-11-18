#pragma once

#include "safelog/auth/auth_errors.hpp"
#include "safelog/auth/credential_store.hpp"
#include "safelog/auth/password_hasher.hpp"
#include "safelog/contracts/ports.hpp"

#include <chrono>
#include <mutex>
#include <string>
#include <string_view>

namespace safelog::auth {

struct LoginResult {
  Profile profile;
  bool mustChangePassword{false};
};

// Passwords for the three demo accounts. Supplied by the caller; never hard-coded.
struct DemoAccountPasswords {
  std::string inspector;  // login "inspector1" -> profile "inspector-1"
  std::string manager;    // login "manager1"   -> profile "manager-1"
  std::string assignee;   // login "assignee1"  -> profile "assignee-1"
};

// Local (serverless) login for Manager / Inspector / Assignee profiles.
//
// - Every authentication failure (unknown id, wrong password, locked account)
//   throws AuthenticationError with the same message and costs one Argon2 run.
// - kMaxFailedAttempts consecutive failures lock the account for kLockDuration.
// - Passwords: at least kMinPasswordChars characters, at most kMaxPasswordBytes bytes.
// - Login ids are trimmed and lower-cased; 3-32 chars of [a-z0-9._-].
class AuthService {
public:
  static constexpr int kMaxFailedAttempts = 5;
  static constexpr std::chrono::seconds kLockDuration{60};
  static constexpr std::size_t kMinPasswordChars = 8;
  static constexpr std::size_t kMaxPasswordBytes = 128;

  AuthService(ICredentialStore& credentials, IRepository& repository, IClock& clock, IIdGenerator& ids,
              const PasswordHasher& hasher);

  // Throws AuthenticationError on any failure.
  LoginResult login(const std::string& loginId, const std::string& password);

  // Requires the current password (a wrong one counts toward the lock).
  // Clears must_change. Throws AuthenticationError or ValidationError (policy).
  void changePassword(const Id& profileId, const std::string& currentPassword, const std::string& newPassword);

  // Registers a new person: creates the profile and its login atomically, with
  // must_change = 1. Only an existing Manager may call it (PermissionError otherwise).
  // Throws ValidationError for a bad/duplicate login id, empty name or weak password.
  Profile createAccount(const Id& managerId, const std::string& displayName, UserRole role,
                        const std::string& loginId, const std::string& temporaryPassword);

  // Links logins inspector1 / manager1 / assignee1 (must_change = 0) to the demo
  // profiles inspector-1 / manager-1 / assignee-1, creating a profile only if it
  // does not exist. Accounts that already have a login are left untouched, so it
  // is safe to call on every start.
  void seedDemoAccounts(const DemoAccountPasswords& passwords);

  // Trimmed, ASCII lower-cased login id, or "" if it does not match the rules.
  static std::string normalizeLoginId(std::string_view loginId);
  // Throws ValidationError if the password violates the policy.
  static void checkPasswordPolicy(std::string_view password);

private:
  CredentialRecord newCredential(const Id& profileId, const std::string& loginId, std::string_view password,
                                 bool mustChange) const;
  bool isLocked(const CredentialRecord& record) const;
  void recordFailure(CredentialRecord& record);

  ICredentialStore& credentials_;
  IRepository& repository_;
  IClock& clock_;
  IIdGenerator& ids_;
  const PasswordHasher& hasher_;
  std::mutex mutex_;  // serializes read-modify-write of failed_count / locked_until
};

} // namespace safelog::auth
