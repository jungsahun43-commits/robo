#pragma once

#include "safelog/contracts/ports.hpp"

#include <mutex>
#include <unordered_map>

namespace safelog::storage {

class InMemoryRepository final : public IRepository {
public:
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

private:
  mutable std::mutex mutex_;
  std::unordered_map<Id, Site> sites_;
  std::unordered_map<Id, Profile> profiles_;
  std::unordered_map<Id, Inspection> inspections_;
  std::unordered_map<Id, Finding> findings_;
  std::unordered_map<Id, Photo> photos_;
  std::unordered_map<Id, ActionLog> logs_;
  std::unordered_map<Id, AiAnalysis> analyses_;
};

} // namespace safelog::storage
