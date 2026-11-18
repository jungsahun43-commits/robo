#pragma once

#include "safelog/contracts/types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace safelog::auth {

// One login per profile. The password itself is never stored.
struct CredentialRecord {
  Id profileId;
  std::string loginId;            // normalized: trimmed, ASCII lower case
  std::string passwordHash;       // "$argon2id$v=19$m=...,t=...,p=...$<base64>"
  std::vector<std::uint8_t> salt; // 16 random bytes
  bool mustChange{true};
  int failedCount{0};
  std::optional<TimePoint> lockedUntil;
  TimePoint createdAt;
  TimePoint updatedAt;
};

// Persistence port for credentials, implemented by storage::SqliteRepository
// and storage::InMemoryCredentialStore.
// Every method throws ValidationError for duplicate login ids, missing profiles
// or other constraint violations.
class ICredentialStore {
public:
  virtual ~ICredentialStore() = default;

  virtual std::optional<CredentialRecord> findCredentialByLoginId(const std::string& loginId) const = 0;
  virtual std::optional<CredentialRecord> findCredentialByProfileId(const Id& profileId) const = 0;

  // Inserts a new profile and its credential atomically: either both are
  // stored or neither is. Fails if the profile id or login id already exists.
  virtual void createAccount(const Profile& profile, const CredentialRecord& credential) = 0;

  // Adds a credential to an existing profile. Fails if the profile is missing,
  // already has a credential, or the login id is taken.
  virtual void addCredential(const CredentialRecord& credential) = 0;

  // Replaces the stored credential of credential.profileId. Throws NotFoundError if none exists.
  virtual void updateCredential(const CredentialRecord& credential) = 0;
};

} // namespace safelog::auth
