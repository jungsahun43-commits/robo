#include "safelog/reporting/report_service.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace safelog::reporting {
namespace {
std::string escape(std::string value) {
  std::string out;
  for (char c : value) {
    switch (c) { case '&': out += "&amp;"; break; case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break; case '"': out += "&quot;"; break; default: out += c; }
  }
  return out;
}
std::string date(TimePoint value) {
  const std::time_t raw = std::chrono::system_clock::to_time_t(value);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &raw);
#else
  localtime_r(&raw, &tm);
#endif
  std::ostringstream out; out << std::put_time(&tm, "%Y-%m-%d %H:%M"); return out.str();
}
}

std::string HtmlRenderer::render(const ReportData& r) const {
  std::ostringstream h;
  h << "<!doctype html><html lang=\"ko\"><meta charset=\"utf-8\"><style>"
       "body{font-family:sans-serif;margin:36px;color:#17212b}h1{border-bottom:3px solid #167d65;padding-bottom:12px}"
       ".meta{background:#eef7f4;padding:16px}.card{page-break-inside:avoid;border:1px solid #ccd7d3;margin:18px 0;padding:16px}"
       ".photos{display:flex;gap:12px}.photos figure{width:48%;margin:0}.photos img{width:100%;max-height:300px;object-fit:contain}"
       "table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccd7d3;padding:8px;text-align:left}</style>"
       "<body><h1>현장 안전점검 및 조치 보고서</h1><div class=\"meta\"><b>사업장:</b> " << escape(r.site.name)
    << "<br><b>주소:</b> " << escape(r.site.address) << "<br><b>점검자:</b> " << escape(r.inspector.displayName)
    << "<br><b>점검일시:</b> " << date(r.inspection.inspectedAt) << "</div>";
  for (std::size_t i = 0; i < r.findings.size(); ++i) {
    const auto& b = r.findings[i];
    h << "<section class=\"card\"><h2>발견 사항 " << (i + 1) << "</h2><table>"
      << "<tr><th>장소</th><td>" << escape(b.finding.location) << "</td></tr>"
      << "<tr><th>발견 내용</th><td>" << escape(b.finding.description) << "</td></tr>"
      << "<tr><th>조치 의견</th><td>" << escape(b.finding.actionOpinion) << "</td></tr>"
      << "<tr><th>상태</th><td>" << to_string(b.finding.status) << "</td></tr></table><div class=\"photos\">";
    for (const auto& p : b.photos)
      h << "<figure><img src=\"" << escape(p.storagePath) << "\"><figcaption>" << to_string(p.kind) << " / " << date(p.capturedAt) << "</figcaption></figure>";
    h << "</div><h3>처리 이력</h3><ul>";
    for (const auto& log : b.actionLogs)
      h << "<li>" << date(log.createdAt) << " · " << escape(log.action) << " · " << escape(log.note) << "</li>";
    h << "</ul></section>";
  }
  h << "</body></html>";
  return h.str();
}

void HtmlRenderer::writeFile(const ReportData& report, const std::string& outputPath) const {
  std::ofstream out(outputPath, std::ios::binary);
  out << render(report);
}

} // namespace safelog::reporting
