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
  if (input.location.empty() || input.description.empty())
      throw ValidationError("Location and description are required");
  if (input.beforePhotoLocalPath.empty()) throw ValidationError("At least one before photo is required");

  const auto now = clock_.now();
  Finding finding{ids_.next("finding"), input.inspectionId, input.location,
                  input.description, input.actionOpinion, std::nullopt,
                  std::nullopt, FindingStatus::Open, now, std::nullopt, std::nullopt};
  const auto storedPath = photos_.importPhoto(input.beforePhotoLocalPath, finding.id, PhotoKind::Before);
  repository_.saveFinding(finding);
  Photo photo{ids_.next("photo"), finding.id, PhotoKind::Before, storedPath, now};
  repository_.savePhoto(photo);
  ActionLog log{ids_.next("log"), finding.id, input.actorId, "finding_created", "", now};
  repository_.saveActionLog(log);
  return {finding, {photo}, {log}, {}};
}

Finding CaptureService::saveManualFinding(
    const Id& findingId,
    const Id& actorId,
    const std::string& description,
    const std::string& actionOpinion)
{
    // 빈 문자열과 공백만 입력한 경우를 모두 거부한다.
    if (description.find_first_not_of(" \t\r\n\f\v") == std::string::npos
        || actionOpinion.find_first_not_of(" \t\r\n\f\v") == std::string::npos) {
        throw ValidationError("Description and action opinion are required");
    }

    auto finding = repository_.findFinding(findingId);
    if (!finding)
        throw NotFoundError("Finding not found");

    // 해당 점검을 생성한 점검자만 내용을 저장할 수 있다.
    const auto inspection =
        repository_.findInspection(finding->inspectionId);

    if (!inspection || inspection->inspectorId != actorId) {
        throw ValidationError(
            "Only the inspection owner can save manual input");
    }

    if (finding->status != FindingStatus::Open) {
        throw ValidationError(
            "Manual input requires an open finding");
    }

    // 이미 검토를 확정한 기록을 실패 후 수동 입력으로 덮어쓰지 않는다.
    for (const auto& analysis :
         repository_.analysesForSubject(findingId)) {
        if (analysis.type == AiAnalysisType::BeforeHazard
            && analysis.decision != AiReviewDecision::Pending) {
            throw ValidationError(
                "The AI suggestion has already been reviewed");
        }
    }

    // 수정 전 내용은 이력에 남긴다.
    const std::string previousText =
        "Previous description: " + finding->description
        + "\nPrevious action opinion: " + finding->actionOpinion;

    finding->description = description;
    finding->actionOpinion = actionOpinion;

    repository_.saveFinding(*finding);

    repository_.saveActionLog({
        ids_.next("log"),
        finding->id,
        actorId,
        "finding_manual_saved",
        previousText,
        clock_.now()
    });

    return *finding;
}

} // namespace safelog::capture
