#include "safelog/reporting/report_service.hpp"
#include "safelog/contracts/errors.hpp"
#include <algorithm>

namespace safelog::reporting {

ReportData ReportService::build(const Id& inspectionId) const {
  const auto inspection = repository_.findInspection(inspectionId);
  if (!inspection) throw NotFoundError("Inspection not found");
  const auto site = repository_.findSite(inspection->siteId);
  const auto inspector = repository_.findProfile(inspection->inspectorId);
  if (!site || !inspector) throw NotFoundError("Report relation is missing");
  std::vector<FindingBundle> bundles;
  for (const auto& finding : repository_.findingsForInspection(inspectionId)) {
    bundles.push_back({finding, repository_.photosForFinding(finding.id),
      repository_.logsForFinding(finding.id), repository_.analysesForSubject(finding.id)});
  }
  if (bundles.empty()) throw ValidationError("Report needs at least one finding");
  std::vector<Profile> participants{*inspector};
  for (auto& bundle : bundles) {
    std::sort(bundle.actionLogs.begin(), bundle.actionLogs.end(), [](const auto& a, const auto& b) { return a.createdAt < b.createdAt; });
    std::sort(bundle.aiAnalyses.begin(), bundle.aiAnalyses.end(), [](const auto& a, const auto& b) { return a.createdAt < b.createdAt; });
    std::sort(bundle.photos.begin(), bundle.photos.end(), [](const auto& a, const auto& b) { return a.capturedAt < b.capturedAt; });
    if (bundle.finding.assigneeId) {
      const auto assignee = repository_.findProfile(*bundle.finding.assigneeId);
      if (assignee) participants.push_back(*assignee);
    }
  }
  return {*site, *inspection, *inspector, std::move(bundles), std::move(participants)};
}

} // namespace safelog::reporting
