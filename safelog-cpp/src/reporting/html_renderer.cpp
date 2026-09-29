#include "safelog/reporting/report_service.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace safelog::reporting {
namespace {
std::string escape(const std::string& value) {
  std::string out;
  for (char c : value) {
    switch (c) {
      case '&': out += "&amp;"; break; case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break; case '"': out += "&quot;"; break;
      default: out += c;
    }
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
std::string embeddedPhoto(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("ReportGenerationFailed: photo cannot be read");
  if (std::filesystem::file_size(path) > 25 * 1024 * 1024)
    throw std::runtime_error("ReportGenerationFailed: photo exceeds 25 MiB");
  const std::string bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  if (input.bad()) throw std::runtime_error("ReportGenerationFailed: photo read failed");
  const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string encoded;
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const auto a = static_cast<unsigned char>(bytes[i]);
    const auto b = i + 1 < bytes.size() ? static_cast<unsigned char>(bytes[i + 1]) : 0;
    const auto c = i + 2 < bytes.size() ? static_cast<unsigned char>(bytes[i + 2]) : 0;
    encoded += alphabet[a >> 2]; encoded += alphabet[((a & 3) << 4) | (b >> 4)];
    encoded += i + 1 < bytes.size() ? alphabet[((b & 15) << 2) | (c >> 6)] : '=';
    encoded += i + 2 < bytes.size() ? alphabet[c & 63] : '=';
  }
  const bool png = bytes.size() >= 8 && static_cast<unsigned char>(bytes[0]) == 137 && bytes.substr(1, 3) == "PNG";
  return std::string("data:image/") + (png ? "png" : "jpeg") + ";base64," + encoded;
}
std::string person(const ReportData& report, const Id& id) {
  for (const auto& p : report.participants) if (p.id == id) return p.displayName + " (" + id + ")";
  return id;
}
}

std::string HtmlRenderer::render(const ReportData& r) const {
  std::ostringstream h;
  h << "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><title>SafeLog AI 안전점검 보고서</title><style>"
       "body{font-family:sans-serif;margin:36px;color:#17212b}h1{border-bottom:3px solid #167d65;padding-bottom:12px}"
       ".meta,.human{background:#eef7f4;padding:16px}.ai{background:#eef4ff;padding:16px}"
       ".card{page-break-inside:avoid;border:1px solid #ccd7d3;margin:18px 0;padding:16px}"
       ".photos{display:flex;flex-wrap:wrap;gap:12px}.photos figure{width:46%;margin:0}.photos img{width:100%;max-height:300px;object-fit:contain}"
       "pre{white-space:pre-wrap;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}"
       "td,th{border:1px solid #ccd7d3;padding:8px;text-align:left}</style></head>"
       "<body><h1>SafeLog AI 안전점검 보고서</h1><div class=\"meta\"><b>사업장:</b> " << escape(r.site.name)
    << "<br><b>주소:</b> " << escape(r.site.address) << "<br><b>점검자:</b> " << escape(r.inspector.displayName)
    << "<br><b>점검일시:</b> " << date(r.inspection.inspectedAt) << "</div>";
  for (std::size_t i = 0; i < r.findings.size(); ++i) {
    const auto& b = r.findings[i];
    h << "<section class=\"card\"><h2>발견 사항 " << (i + 1) << "</h2><table>"
      << "<tr><th>장소</th><td>" << escape(b.finding.location) << "</td></tr>"
      << "<tr><th>발견 내용</th><td>" << escape(b.finding.description) << "</td></tr>"
      << "<tr><th>개선 의견</th><td>" << escape(b.finding.actionOpinion) << "</td></tr>"
      << "<tr><th>AI 위험 분류</th><td>" << escape(b.finding.hazardCategory.value_or("수동 입력 또는 검토 전")) << "</td></tr>"
      << "<tr><th>위험 등급</th><td>" << (b.finding.riskLevel ? std::to_string(*b.finding.riskLevel) : "미지정") << " / 5</td></tr>"
      << "<tr><th>담당자</th><td>" << escape(b.finding.assigneeId ? person(r, *b.finding.assigneeId) : "미지정") << "</td></tr>"
      << "<tr><th>Finding 최종 상태</th><td>" << to_string(b.finding.status) << "</td></tr></table><div class=\"photos\">";
    for (const auto& p : b.photos)
      h << "<figure><img src=\"" << embeddedPhoto(p.storagePath) << "\"><figcaption>" << to_string(p.kind) << " / " << date(p.capturedAt) << "</figcaption></figure>";
    h << "</div><div class=\"human\"><h3>사람의 조치 및 최종 판단</h3><ul>";
    for (const auto& log : b.actionLogs) {
      if ((log.action == "ai_hazard_suggestion" || log.action == "ai_comparison_suggestion")) continue;
      h << "<li>" << date(log.createdAt) << " · " << escape(log.action) << " · "
        << escape(person(r, log.actorId)) << "<pre>" << escape(log.note) << "</pre></li>";
    }
    h << "</ul></div><div class=\"ai\"><h3>AI 분석 이력 · 참고 제안</h3>"
         "<p>AI 분석 결과는 참고 제안입니다. 최종 안전 판단과 확인은 사람이 수행합니다.</p><ul>";
    for (const auto& analysis : b.aiAnalyses)
      h << "<li>" << to_string(analysis.type) << " · " << escape(analysis.modelName)
        << " · 프롬프트 " << escape(analysis.promptVersion) << " · 분류 " << escape(analysis.category)
        << " · 신뢰도 " << analysis.confidence << " · 사용자 검토 " << to_string(analysis.decision)
        << " · 검토자 " << escape(analysis.reviewedBy.value_or("미검토"))
        << "<pre>" << escape(analysis.resultJson) << "</pre></li>";
    for (const auto& log : b.actionLogs)
      if ((log.action == "ai_hazard_suggestion" || log.action == "ai_comparison_suggestion")) h << "<li>" << date(log.createdAt) << "<pre>" << escape(log.note) << "</pre></li>";
    h << "</ul></div></section>";
  }
  h << "</body></html>";
  return h.str();
}

void HtmlRenderer::writeFile(const ReportData& report, const std::string& outputPath) const {
  const auto html = render(report);
  std::ofstream out(outputPath, std::ios::binary);
  if (!out) throw std::runtime_error("ReportGenerationFailed: output cannot be opened");
  out << html; out.flush();
  if (!out) throw std::runtime_error("ReportGenerationFailed: output write failed");
}
} // namespace safelog::reporting
