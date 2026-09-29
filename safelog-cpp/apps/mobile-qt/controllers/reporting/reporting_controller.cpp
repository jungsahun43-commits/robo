#include "reporting_controller.hpp"
#include "safelog/contracts/errors.hpp"
#include <QDateTime>
#include <QDir>
#include <QStandardPaths>
#include <QUuid>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrentRun>

namespace safelog::qtapp {
namespace {
QString text(const std::string& value) { return QString::fromStdString(value); }
QString date(TimePoint value) {
  return QDateTime::fromMSecsSinceEpoch(std::chrono::duration_cast<std::chrono::milliseconds>(value.time_since_epoch()).count()).toString(Qt::ISODate);
}
QVariantMap view(const reporting::ReportData& data) {
  QVariantList findings;
  for (const auto& b : data.findings) {
    QVariantList photos, analyses, logs;
    for (const auto& p : b.photos) photos << QVariantMap{{"path", text(p.storagePath)}, {"kind", text(to_string(p.kind))}};
    for (const auto& a : b.aiAnalyses) analyses << QVariantMap{{"type", text(to_string(a.type))},
      {"model", text(a.modelName)}, {"prompt", text(a.promptVersion)}, {"confidence", a.confidence},
      {"decision", text(to_string(a.decision))}, {"reviewer", text(a.reviewedBy.value_or("미검토"))}, {"json", text(a.resultJson)}};
    for (const auto& l : b.actionLogs) logs << QVariantMap{{"isAi", l.action == "ai_hazard_suggestion" || l.action == "ai_comparison_suggestion"}, {"action", text(l.action)}, {"actor", text(l.actorId)}, {"note", text(l.note)}, {"at", date(l.createdAt)}};
    QString assignee = text(b.finding.assigneeId.value_or("미지정"));
    for (const auto& p : data.participants) if (b.finding.assigneeId == p.id) assignee = text(p.displayName + " (" + p.id + ")");
    findings << QVariantMap{{"location", text(b.finding.location)}, {"description", text(b.finding.description)},
      {"riskLevel", b.finding.riskLevel.value_or(0)}, {"actionOpinion", text(b.finding.actionOpinion)},
      {"assignee", assignee}, {"status", text(to_string(b.finding.status))}, {"photos", photos}, {"analyses", analyses}, {"logs", logs}};
  }
  return {{"site", text(data.site.name)}, {"address", text(data.site.address)}, {"inspector", text(data.inspector.displayName)},
    {"inspectedAt", date(data.inspection.inspectedAt)}, {"findings", findings}};
}
struct Generated { QString path, error; };
}
ReportingController::ReportingController(IRepository& r, reporting::ReportService& service, AuthController& auth,
    SharingAdapter& sharing, QObject* parent)
  : QObject(parent), repository_(r), service_(service), auth_(auth), sharing_(sharing) {
  connect(&auth_, &AuthController::sessionChanged, this, &ReportingController::reset);
}
void ReportingController::reset() {
  requests_.invalidate(); report_.clear(); state_ = "Idle"; error_.clear(); outputPath_.clear(); shareState_.clear(); emit changed();
}
void ReportingController::generate(QString inspectionId) {
  reset();
  try {
    const auto session = auth_.session();
    const auto inspection = repository_.findInspection(inspectionId.toStdString());
    if (!inspection) throw NotFoundError("MissingReportData: 점검 데이터가 없습니다.");
    if (session.value("role").toString() != "Manager" &&
        !(session.value("role").toString() == "Inspector" && inspection->inspectorId == session.value("userId").toString().toStdString()))
      throw ValidationError("PermissionDenied: 보고서 접근 권한이 없습니다.");
    auto data = service_.build(inspection->id);
    report_ = view(data); state_ = "Loading"; emit changed();
    const auto request = requests_.invalidate();
    const auto folder = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/reports";
    if (!QDir().mkpath(folder)) throw std::runtime_error("ReportGenerationFailed: 보고서 폴더를 만들 수 없습니다.");
    const auto path = folder + "/safelog-" + QUuid::createUuid().toString(QUuid::WithoutBraces) + ".html";
    auto* watcher = new QFutureWatcher<Generated>(this);
    connect(watcher, &QFutureWatcher<Generated>::finished, this, [this, watcher, request] {
      const auto result = watcher->result(); watcher->deleteLater();
      if (!requests_.accepts(request)) return;
      if (!result.error.isEmpty()) { error_ = result.error; state_ = "Failure"; emit changed(); emit generationFailed(error_); }
      else { outputPath_ = result.path; state_ = "Success"; emit changed(); emit generationSucceeded(); }
    });
    watcher->setFuture(QtConcurrent::run([data = std::move(data), path] {
      Generated result;
      try { reporting::HtmlRenderer{}.writeFile(data, path.toStdString()); result.path = path; }
      catch (const std::exception& e) { result.error = QString::fromUtf8(e.what()); }
      catch (...) { result.error = "ReportGenerationFailed"; }
      return result;
    }));
  } catch (const std::exception& e) { error_ = QString::fromUtf8(e.what()); state_ = "Failure"; emit changed(); emit generationFailed(error_); }
}
void ReportingController::share(QString outcome) {
  try {
    error_.clear();
    if (state_ != "Success" || outputPath_.isEmpty()) throw ValidationError("먼저 보고서를 생성하세요.");
    if (auto* mock = dynamic_cast<MockSharingAdapter*>(&sharing_)) {
      mock->nextResult = outcome == "ShareCancelled" ? ShareResult::ShareCancelled :
                         outcome == "ShareFailed" ? ShareResult::ShareFailed : ShareResult::ShareSucceeded;
    }
    const auto result = sharing_.share(outputPath_.toStdString(), "text/html");
    shareState_ = result == ShareResult::ShareSucceeded ? "ShareSucceeded" :
                  result == ShareResult::ShareCancelled ? "ShareCancelled" : "ShareFailed";
    if (result == ShareResult::ShareFailed) error_ = "보고서 공유에 실패했습니다.";
  } catch (const std::exception& e) { error_ = QString::fromUtf8(e.what()); shareState_ = "ShareFailed"; }
  emit changed(); emit shareFinished(shareState_);
}
}
