#pragma once

#include "safelog/auth/credential_store.hpp"
#include "safelog/contracts/ports.hpp"

#include <mutex>
#include <unordered_map>

namespace safelog::storage {

// In-memory ICredentialStore for tests and the demo, paired with an IRepository
// (e.g. InMemoryRepository) that holds the profiles. Mirrors the SQLite rules:
// unique login id ignoring ASCII case, credential requires an existing profile.
class InMemoryCredentialStore final : public auth::ICredentialStore {
public:
  explicit InMemoryCredentialStore(IRepository& profiles);

  std::optional<auth::CredentialRecord> findCredentialByLoginId(const std::string& loginId) const override;
  std::optional<auth::CredentialRecord> findCredentialByProfileId(const Id& profileId) const override;
  void createAccount(const Profile& profile, const auth::CredentialRecord& credential) override;
  void addCredential(const auth::CredentialRecord& credential) override;
  void updateCredential(const auth::CredentialRecord& credential) override;

private:
  void checkInsertable(const auth::CredentialRecord& credential) const;

  IRepository& profiles_;
  mutable std::mutex mutex_;
  std::unordered_map<Id, auth::CredentialRecord> byProfile_;
};

} // namespace safelog::storage
