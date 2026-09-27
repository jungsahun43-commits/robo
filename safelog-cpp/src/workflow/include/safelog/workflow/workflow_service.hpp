#pragma once

#include "safelog/contracts/ports.hpp"

namespace safelog::workflow {

class WorkflowService {
public:
  WorkflowService(IRepository& repository, IClock& clock, IIdGenerator& ids, IPhotoStore& photos);
  Finding assign(const Id& findingId, const Id& managerId, const Id& assigneeId,
                 std::optional<TimePoint> dueAt = std::nullopt);
  Finding beginWork(const Id& findingId, const Id& assigneeId);
  Finding submitAction(const Id& findingId, const Id& assigneeId,
                       const std::string& note, const std::string& afterPhotoLocalPath);
  Finding verify(const Id& findingId, const Id& inspectorId, const std::string& note = "");

private:
  Finding requireFinding(const Id& id) const;
  void log(const Finding& finding, const Id& actorId, const std::string& action, const std::string& note);
  IRepository& repository_;
  IClock& clock_;
  IIdGenerator& ids_;
  IPhotoStore& photos_;
};

} // namespace safelog::workflow
