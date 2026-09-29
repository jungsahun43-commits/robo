#pragma once

#include "safelog/contracts/ports.hpp"

namespace safelog::reporting {

struct ReportData {
  Site site;
  Inspection inspection;
  Profile inspector;
  std::vector<FindingBundle> findings;
  std::vector<Profile> participants;
};

class ReportService {
public:
  explicit ReportService(const IRepository& repository) : repository_(repository) {}
  ReportData build(const Id& inspectionId) const;
private:
  const IRepository& repository_;
};

class HtmlRenderer {
public:
  std::string render(const ReportData& report) const;
  void writeFile(const ReportData& report, const std::string& outputPath) const;
};

} // namespace safelog::reporting
