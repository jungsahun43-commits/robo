#include "safelog/storage/in_memory_repository.hpp"

namespace safelog::storage {

void InMemoryRepository::saveSite(const Site& v) { std::scoped_lock l(mutex_); sites_[v.id] = v; }
void InMemoryRepository::saveProfile(const Profile& v) { std::scoped_lock l(mutex_); profiles_[v.id] = v; }
void InMemoryRepository::saveInspection(const Inspection& v) { std::scoped_lock l(mutex_); inspections_[v.id] = v; }
void InMemoryRepository::saveFinding(const Finding& v) { std::scoped_lock l(mutex_); findings_[v.id] = v; }
void InMemoryRepository::savePhoto(const Photo& v) { std::scoped_lock l(mutex_); photos_[v.id] = v; }
void InMemoryRepository::saveActionLog(const ActionLog& v) { std::scoped_lock l(mutex_); logs_[v.id] = v; }

template <typename T>
std::optional<T> copyById(const std::unordered_map<Id, T>& values, const Id& id) {
  const auto it = values.find(id);
  return it == values.end() ? std::nullopt : std::optional<T>{it->second};
}

std::optional<Site> InMemoryRepository::findSite(const Id& id) const { std::scoped_lock l(mutex_); return copyById(sites_, id); }
std::optional<Profile> InMemoryRepository::findProfile(const Id& id) const { std::scoped_lock l(mutex_); return copyById(profiles_, id); }
std::optional<Inspection> InMemoryRepository::findInspection(const Id& id) const { std::scoped_lock l(mutex_); return copyById(inspections_, id); }
std::optional<Finding> InMemoryRepository::findFinding(const Id& id) const { std::scoped_lock l(mutex_); return copyById(findings_, id); }

std::vector<Photo> InMemoryRepository::photosForFinding(const Id& findingId) const {
  std::scoped_lock l(mutex_); std::vector<Photo> out;
  for (const auto& [_, value] : photos_) if (value.findingId == findingId) out.push_back(value);
  return out;
}

std::vector<ActionLog> InMemoryRepository::logsForFinding(const Id& findingId) const {
  std::scoped_lock l(mutex_); std::vector<ActionLog> out;
  for (const auto& [_, value] : logs_) if (value.findingId == findingId) out.push_back(value);
  return out;
}

std::vector<Finding> InMemoryRepository::findingsForInspection(const Id& inspectionId) const {
  std::scoped_lock l(mutex_); std::vector<Finding> out;
  for (const auto& [_, value] : findings_) if (value.inspectionId == inspectionId) out.push_back(value);
  return out;
}

std::vector<Finding> InMemoryRepository::findingsAssignedTo(const Id& assigneeId) const {
  std::scoped_lock l(mutex_); std::vector<Finding> out;
  for (const auto& [_, value] : findings_)
    if (value.assigneeId && *value.assigneeId == assigneeId) out.push_back(value);
  return out;
}

} // namespace safelog::storage
