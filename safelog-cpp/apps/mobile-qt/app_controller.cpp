#include "app_controller.hpp"
#include "adapters/ai/http_ai_safety_analyzer.hpp"
#include "safelog/ai/mock_ai_analyzer.hpp"
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <QUuid>
#include <algorithm>
namespace safelog::qtapp {
namespace {
std::shared_ptr<ai::IAiSafetyAnalyzer> makeDefaultAnalyzer() {
  const QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
  const QString baseUrl = environment.value("SAFELOG_AI_BASE_URL").trimmed();
  if (baseUrl.isEmpty()) return std::make_shared<ai::MockAiSafetyAnalyzer>();

  bool timeoutOk = false;
  const int configuredTimeout = environment.value("SAFELOG_AI_TIMEOUT_MS").toInt(&timeoutOk);
  HttpAiSafetyAnalyzerConfig config;
  config.baseUrl = baseUrl.toStdString();
  if (timeoutOk && configuredTimeout > 0) config.timeoutMs = configuredTimeout;
  return std::make_shared<HttpAiSafetyAnalyzer>(std::move(config), makeQtJsonHttpClient());
}
}

AppController::AppController(QObject* parent, std::shared_ptr<ai::IAiSafetyAnalyzer> analyzer)
  : QObject(parent),
    // Memory IDs restart each process; isolate photo directories to avoid overwriting earlier files.
    photoStore_((QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/photos/" +
      QUuid::createUuid().toString(QUuid::WithoutBraces)).toStdString()),
    // Integration seam: inject role-3's IAiSafetyAnalyzer implementation here when available.
    analyzer_(analyzer ? std::move(analyzer) : makeDefaultAnalyzer()),
    capture_(repository_, clock_, ids_, photoStore_), workflow_(repository_, clock_, ids_, photoStore_),
    reviews_(repository_, clock_, ids_, *analyzer_), auth_(repository_), ai_(analyzer_),
    captureController_(repository_, clock_, ids_, capture_, reviews_, auth_, ai_), reports_(repository_),
    workflowController_(repository_, clock_, ids_, workflow_, reviews_, auth_, ai_, inspectionIds_),
    reportingController_(repository_, reports_, auth_, sharing_) {
  connect(&captureController_, &CaptureController::changed, this, [this] {
    const auto id = captureController_.draft().value("inspectionId").toString().toStdString();
    if (!id.empty() && std::find(inspectionIds_.begin(), inspectionIds_.end(), id) == inspectionIds_.end())
      inspectionIds_.push_back(id);
    workflowController_.refresh();
    emit dashboardChanged();
  });
  connect(&auth_, &AuthController::sessionChanged, this, &AppController::dashboardChanged);
  connect(&workflowController_, &WorkflowController::changed, this, &AppController::dashboardChanged);
  repository_.saveSite({"site-1", "세이프 금속 가공공장", "서울시 가상구 산업로 10"});
  repository_.saveProfile({"inspector-1", "김안전", UserRole::Inspector});
  repository_.saveProfile({"manager-1", "박관리", UserRole::Manager});
  repository_.saveProfile({"assignee-1", "이조치", UserRole::Assignee});
}
QVariantList AppController::dashboard() const {
  int open = 0, progress = 0, pending = 0, complete = 0;
  const auto user = auth_.session().value("userId").toString().toStdString();
  const auto role = auth_.session().value("role").toString();
  if (user.empty()) return {0, 0, 0, 0};
  for (const auto& id : inspectionIds_) {
    const auto inspection = repository_.findInspection(id);
    if (!inspection || (role == "Inspector" && inspection->inspectorId != user)) continue;
    const auto findings = repository_.findingsForInspection(id);
    bool allVerified = !findings.empty();
    bool visible = false;
    for (const auto& f : findings) {
      if (role == "Assignee" && f.assigneeId != user) continue;
      visible = true;
      open += f.status == FindingStatus::Open;
      progress += f.status == FindingStatus::InProgress;
      pending += f.status == FindingStatus::PendingReview;
      allVerified = allVerified && f.status == FindingStatus::Verified;
    }
    complete += visible && allVerified;
  }
  return {open, progress, pending, complete};
}
}
