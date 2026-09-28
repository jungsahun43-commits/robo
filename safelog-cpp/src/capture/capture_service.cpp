#include "safelog/capture/capture_service.hpp"
#include "safelog/contracts/errors.hpp"

namespace safelog::capture {

CaptureService::CaptureService(IRepository& r, IClock& c, IIdGenerator& i, IPhotoStore& p)
  : repository_(r), clock_(c), ids_(i), photos_(p) {}

Inspection CaptureService::startInspection(const Id& siteId, const Id& inspectorId) {
  if (!repository_.findSite(siteId)) throw NotFoundError("Site not found");
  const auto profile = repository_.findProfile(inspectorId);
  if (!profile || profile->role != UserRole::Inspector) throw ValidationError("Inspector profile is required");
  Inspection value{ids_.next("inspection"), siteId, inspectorId, clock_.now()};
  repository_.saveInspection(value);
  return value;
}

FindingBundle CaptureService::addFinding(const NewFindingInput& input) {
  if (!repository_.findInspection(input.inspectionId)) throw NotFoundError("Inspection not found");
  if (input.location.empty() || input.description.empty() || input.actionOpinion.empty())
    throw ValidationError("Location, description and action opinion are required");
  if (input.beforePhotoLocalPath.empty()) throw ValidationError("At least one before photo is required");

  const auto now = clock_.now();
  Finding finding{ids_.next("finding"), input.inspectionId, input.location,
                  input.description, input.actionOpinion, std::nullopt,
                  std::nullopt, FindingStatus::Open, now, std::nullopt, std::nullopt};
  repository_.saveFinding(finding);
  const auto storedPath = photos_.importPhoto(input.beforePhotoLocalPath, finding.id, PhotoKind::Before);
  Photo photo{ids_.next("photo"), finding.id, PhotoKind::Before, storedPath, now};
  repository_.savePhoto(photo);
  ActionLog log{ids_.next("log"), finding.id, input.actorId, "finding_created", "", now};
  repository_.saveActionLog(log);
  return {finding, {photo}, {log}, {}};
}

} // namespace safelog::capture
