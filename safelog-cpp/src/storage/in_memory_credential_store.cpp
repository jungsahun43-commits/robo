#include "safelog/storage/in_memory_credential_store.hpp"

#include "safelog/contracts/errors.hpp"

#include <algorithm>

namespace safelog::storage {
namespace {

bool equalsIgnoreAsciiCase(const std::string& a, const std::string& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
    const auto lower = [](char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; };
    return lower(x) == lower(y);
  });
}

} // namespace

InMemoryCredentialStore::InMemoryCredentialStore(IRepository& profiles) : profiles_(profiles) {}

std::optional<auth::CredentialRecord> InMemoryCredentialStore::findCredentialByLoginId(
    const std::string& loginId) const {
  std::scoped_lock lock(mutex_);
  for (const auto& [_, record] : byProfile_)
    if (equalsIgnoreAsciiCase(record.loginId, loginId)) return record;
  return std::nullopt;
}

std::optional<auth::CredentialRecord> InMemoryCredentialStore::findCredentialByProfileId(const Id& profileId) const {
  std::scoped_lock lock(mutex_);
  const auto it = byProfile_.find(profileId);
  return it == byProfile_.end() ? std::nullopt : std::optional<auth::CredentialRecord>{it->second};
}

void InMemoryCredentialStore::checkInsertable(const auth::CredentialRecord& credential) const {
  if (credential.salt.size() != 16) throw ValidationError("Salt must be 16 bytes");
  if (byProfile_.count(credential.profileId)) throw ValidationError("Profile already has a credential");
  for (const auto& [_, record] : byProfile_)
    if (equalsIgnoreAsciiCase(record.loginId, credential.loginId)) throw ValidationError("Login id already exists");
}

void InMemoryCredentialStore::createAccount(const Profile& profile, const auth::CredentialRecord& credential) {
  if (credential.profileId != profile.id) throw ValidationError("Credential does not belong to the new profile");
  std::scoped_lock lock(mutex_);
  // Validate everything before writing so a failure leaves no profile behind.
  if (profiles_.findProfile(profile.id)) throw ValidationError("Profile already exists");
  checkInsertable(credential);
  profiles_.saveProfile(profile);
  byProfile_[credential.profileId] = credential;
}

void InMemoryCredentialStore::addCredential(const auth::CredentialRecord& credential) {
  std::scoped_lock lock(mutex_);
  if (!profiles_.findProfile(credential.profileId)) throw ValidationError("Profile does not exist");
  checkInsertable(credential);
  byProfile_[credential.profileId] = credential;
}

void InMemoryCredentialStore::updateCredential(const auth::CredentialRecord& credential) {
  std::scoped_lock lock(mutex_);
  const auto it = byProfile_.find(credential.profileId);
  if (it == byProfile_.end()) throw NotFoundError("No credential for profile " + credential.profileId);
  for (const auto& [id, record] : byProfile_)
    if (id != credential.profileId && equalsIgnoreAsciiCase(record.loginId, credential.loginId))
      throw ValidationError("Login id already exists");
  it->second = credential;
}

} // namespace safelog::storage
