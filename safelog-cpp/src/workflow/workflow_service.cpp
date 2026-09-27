#include "safelog/workflow/workflow_service.hpp"
#include "safelog/contracts/errors.hpp"

namespace safelog::workflow {

WorkflowService::WorkflowService(IRepository& r, IClock& c, IIdGenerator& i, IPhotoStore& p)
  : repository_(r), clock_(c), ids_(i), photos_(p) {}

Finding WorkflowService::requireFinding(const Id& id) const {
  auto finding = repository_.findFinding(id);
  if (!finding) throw NotFoundError("Finding not found");
  return *finding;
}

void WorkflowService::log(const Finding& f, const Id& actor, const std::string& action, const std::string& note) {
  repository_.saveActionLog({ids_.next("log"), f.id, actor, action, note, clock_.now()});
}

Finding WorkflowService::assign(const Id& findingId, const Id& managerId, const Id& assigneeId,
                                std::optional<TimePoint> dueAt) {
  auto f = requireFinding(findingId);
  if (f.status != FindingStatus::Open) throw TransitionError("Only open findings can be assigned");
  const auto assignee = repository_.findProfile(assigneeId);
  if (!assignee || assignee->role != UserRole::Assignee) throw ValidationError("Valid assignee is required");
  f.assigneeId = assigneeId; f.dueAt = dueAt;
  repository_.saveFinding(f); log(f, managerId, "assigned", assigneeId);
  return f;
}

Finding WorkflowService::beginWork(const Id& findingId, const Id& assigneeId) {
  auto f = requireFinding(findingId);
  if (f.status != FindingStatus::Open || !f.assigneeId || *f.assigneeId != assigneeId)
    throw TransitionError("Assigned user can begin an open finding");
  f.status = FindingStatus::InProgress;
  repository_.saveFinding(f); log(f, assigneeId, "work_started", "");
  return f;
}

Finding WorkflowService::submitAction(const Id& findingId, const Id& assigneeId,
                                      const std::string& note, const std::string& afterPhotoLocalPath) {
  auto f = requireFinding(findingId);
  if (f.status != FindingStatus::InProgress || !f.assigneeId || *f.assigneeId != assigneeId)
    throw TransitionError("Assigned user can submit an in-progress finding");
  if (note.empty() || afterPhotoLocalPath.empty()) throw ValidationError("Action note and after photo are required");
  const auto path = photos_.importPhoto(afterPhotoLocalPath, f.id, PhotoKind::After);
  repository_.savePhoto({ids_.next("photo"), f.id, PhotoKind::After, path, clock_.now()});
  f.status = FindingStatus::PendingReview;
  repository_.saveFinding(f); log(f, assigneeId, "action_submitted", note);
  return f;
}

Finding WorkflowService::verify(const Id& findingId, const Id& inspectorId, const std::string& note) {
  auto f = requireFinding(findingId);
  if (f.status != FindingStatus::PendingReview) throw TransitionError("Only pending review can be verified");
  const auto inspection = repository_.findInspection(f.inspectionId);
  if (!inspection || inspection->inspectorId != inspectorId)
    throw ValidationError("Only the inspection owner can verify this finding");
  f.status = FindingStatus::Verified;
  repository_.saveFinding(f); log(f, inspectorId, "verified", note);
  return f;
}

} // namespace safelog::workflow
