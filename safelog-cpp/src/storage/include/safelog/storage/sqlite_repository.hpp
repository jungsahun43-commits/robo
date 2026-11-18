#pragma once

#include "safelog/auth/credential_store.hpp"
#include "safelog/contracts/ports.hpp"

#include <filesystem>
#include <memory>

namespace safelog::storage {

// IRepository backed by a SQLite file. Drop-in replacement for InMemoryRepository.
//
// - save* is an upsert (INSERT ... ON CONFLICT(id) DO UPDATE), so services may
//   save the same id repeatedly without cascading deletes of photos/logs.
// - Foreign key, CHECK and NOT NULL violations throw ValidationError; other
//   database failures throw std::runtime_error.
// - List queries are ordered by time, then id.
// - Also stores login credentials (auth::ICredentialStore) in the same database.
class SqliteRepository final : public IRepository, public auth::ICredentialStore {
public:
  static constexpr int kSchemaVersion = 2;

  // Opens (or creates) the database file, enables foreign keys and applies
  // database/schema.sql, migrating older schema versions first.
  explicit SqliteRepository(const std::filesystem::path& databasePath);
  ~SqliteRepository() override;

  SqliteRepository(const SqliteRepository&) = delete;
  SqliteRepository& operator=(const SqliteRepository&) = delete;

  void saveSite(const Site& value) override;
  void saveProfile(const Profile& value) override;
  void saveInspection(const Inspection& value) override;
  void saveFinding(const Finding& value) override;
  void savePhoto(const Photo& value) override;
  void saveActionLog(const ActionLog& value) override;
  void saveAiAnalysis(const AiAnalysis& value) override;

  std::optional<Site> findSite(const Id& id) const override;
  std::optional<Profile> findProfile(const Id& id) const override;
  std::optional<Inspection> findInspection(const Id& id) const override;
  std::optional<Finding> findFinding(const Id& id) const override;
  std::optional<AiAnalysis> findAiAnalysis(const Id& id) const override;
  std::vector<Photo> photosForFinding(const Id& findingId) const override;
  std::vector<ActionLog> logsForFinding(const Id& findingId) const override;
  std::vector<Finding> findingsForInspection(const Id& inspectionId) const override;
  std::vector<Finding> findingsAssignedTo(const Id& assigneeId) const override;
  std::vector<AiAnalysis> analysesForSubject(const Id& subjectId) const override;

  // auth::ICredentialStore
  std::optional<auth::CredentialRecord> findCredentialByLoginId(const std::string& loginId) const override;
  std::optional<auth::CredentialRecord> findCredentialByProfileId(const Id& profileId) const override;
  void createAccount(const Profile& profile, const auth::CredentialRecord& credential) override;
  void addCredential(const auth::CredentialRecord& credential) override;
  void updateCredential(const auth::CredentialRecord& credential) override;

  // PRAGMA user_version of the open database.
  int schemaVersion() const;

  // Inserts the demo site "site-1" and profiles "inspector-1", "assignee-1",
  // "manager-1" if missing. Safe to call on every start: existing rows,
  // including ones the user edited, are left untouched.
  void seedDemoData();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace safelog::storage
