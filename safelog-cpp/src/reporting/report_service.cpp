#include "safelog/reporting/report_service.hpp"
#include "safelog/contracts/errors.hpp"

namespace safelog::reporting {

ReportData ReportService::build(const Id& inspectionId) const {
  const auto inspection = repository_.findInspection(inspectionId);
  if (!inspection) throw NotFoundError("Inspection not found");
  const auto site = repository_.findSite(inspection->siteId);
  const auto inspector = repository_.findProfile(inspection->inspectorId);
  if (!site || !inspector) throw NotFoundError("Report relation is missing");
  std::vector<FindingBundle> bundles;
  for (const auto& finding : repository_.findingsForInspection(inspectionId)) {
    bundles.push_back({finding, repository_.photosForFinding(finding.id), repository_.logsForFinding(finding.id)});
  }
  if (bundles.empty()) throw ValidationError("Report needs at least one finding");
  return {*site, *inspection, *inspector, std::move(bundles)};
}

} // namespace safelog::reporting
