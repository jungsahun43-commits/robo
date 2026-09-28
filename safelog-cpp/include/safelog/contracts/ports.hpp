#pragma once

#include "safelog/contracts/types.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace safelog {

class IRepository {
public:
  virtual ~IRepository() = default;

  virtual void saveSite(const Site& value) = 0;
  virtual void saveProfile(const Profile& value) = 0;
  virtual void saveInspection(const Inspection& value) = 0;
  virtual void saveFinding(const Finding& value) = 0;
  virtual void savePhoto(const Photo& value) = 0;
  virtual void saveActionLog(const ActionLog& value) = 0;
  virtual void saveAiAnalysis(const AiAnalysis& value) = 0;

  virtual std::optional<Site> findSite(const Id& id) const = 0;
  virtual std::optional<Profile> findProfile(const Id& id) const = 0;
  virtual std::optional<Inspection> findInspection(const Id& id) const = 0;
  virtual std::optional<Finding> findFinding(const Id& id) const = 0;
  virtual std::optional<AiAnalysis> findAiAnalysis(const Id& id) const = 0;
  virtual std::vector<Photo> photosForFinding(const Id& findingId) const = 0;
  virtual std::vector<ActionLog> logsForFinding(const Id& findingId) const = 0;
  virtual std::vector<Finding> findingsForInspection(const Id& inspectionId) const = 0;
  virtual std::vector<Finding> findingsAssignedTo(const Id& assigneeId) const = 0;
  virtual std::vector<AiAnalysis> analysesForSubject(const Id& subjectId) const = 0;
};

class IClock {
public:
  virtual ~IClock() = default;
  virtual TimePoint now() const = 0;
};

class IIdGenerator {
public:
  virtual ~IIdGenerator() = default;
  virtual Id next(const std::string& prefix) = 0;
};

class IPhotoStore {
public:
  virtual ~IPhotoStore() = default;
  virtual std::string importPhoto(const std::string& localPath,
                                  const Id& findingId,
                                  PhotoKind kind) = 0;
};

} // namespace safelog
