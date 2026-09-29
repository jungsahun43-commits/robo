#include "safelog/workflow/workflow_service.hpp"
#include "safelog/contracts/errors.hpp"

namespace safelog::workflow {
namespace {
void requireRole(const IRepository& repository, const Id& userId, UserRole role) {
  const auto user = repository.findProfile(userId);
  if (!user || user->role != role) throw ValidationError("PermissionDenied: user role is not permitted");
}
bool blank(const std::string& value) { return value.find_first_not_of(" \t\r\n") == std::string::npos; }
}


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
  requireRole(repository_, managerId, UserRole::Manager);
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
  requireRole(repository_, assigneeId, UserRole::Assignee);
  if (!f.assigneeId) throw ValidationError("AssigneeMissing: assign a user first");
  if (f.status != FindingStatus::Open || *f.assigneeId != assigneeId)
    throw TransitionError("Assigned user can begin an open finding");
  f.status = FindingStatus::InProgress;
  repository_.saveFinding(f); log(f, assigneeId, "work_started", "");
  return f;
}

Finding WorkflowService::submitAction(const Id& findingId, const Id& assigneeId,
                                      const std::string& note, const std::string& afterPhotoLocalPath) {
  auto f = requireFinding(findingId);
  requireRole(repository_, assigneeId, UserRole::Assignee);
  if (!f.assigneeId) throw ValidationError("AssigneeMissing: assign a user first");
  if (f.status != FindingStatus::InProgress || *f.assigneeId != assigneeId)
    throw TransitionError("Assigned user can submit an in-progress finding");
  if (blank(note) || blank(afterPhotoLocalPath)) throw ValidationError("MissingActionData: action note and after photo are required");
  const auto path = photos_.importPhoto(afterPhotoLocalPath, f.id, PhotoKind::After);
  repository_.savePhoto({ids_.next("photo"), f.id, PhotoKind::After, path, clock_.now()});
  f.status = FindingStatus::PendingReview;
  repository_.saveFinding(f); log(f, assigneeId, "action_submitted", note);
  return f;
}

Finding WorkflowService::verify(const Id& findingId, const Id& inspectorId, const std::string& note) {
  auto f = requireFinding(findingId);
  requireRole(repository_, inspectorId, UserRole::Inspector);
  if (f.status != FindingStatus::PendingReview) throw TransitionError("Only pending review can be verified");
  const auto inspection = repository_.findInspection(f.inspectionId);
  if (!inspection || inspection->inspectorId != inspectorId)
    throw ValidationError("Only the inspection owner can verify this finding");
  const auto photos = repository_.photosForFinding(findingId);
  bool before = false, after = false;
  for (const auto& photo : photos) {
    before |= photo.kind == PhotoKind::Before && !photo.storagePath.empty();
    after |= photo.kind == PhotoKind::After && !photo.storagePath.empty();
  }
  if (!before || !after) throw ValidationError("MissingActionData: both photos are required");
  f.status = FindingStatus::Verified;
  repository_.saveFinding(f); log(f, inspectorId, "verified", note);
  return f;
}

Finding WorkflowService::requestChanges(const Id& findingId, const Id& inspectorId, const std::string& note) {
  requireRole(repository_, inspectorId, UserRole::Inspector);
  auto f = requireFinding(findingId);
  if (f.status != FindingStatus::PendingReview) throw TransitionError("Only pending review can request changes");
  const auto inspection = repository_.findInspection(f.inspectionId);
  if (!inspection || inspection->inspectorId != inspectorId)
    throw ValidationError("PermissionDenied: only the inspection owner can request changes");
  if (blank(note)) throw ValidationError("MissingActionData: additional action instructions are required");
  f.status = FindingStatus::InProgress;
  repository_.saveFinding(f);
  log(f, inspectorId, "changes_requested", note);
  return f;
}
} // namespace safelog::workflow
