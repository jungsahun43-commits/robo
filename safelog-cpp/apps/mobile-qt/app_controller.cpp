#include "app_controller.hpp"

#include <QDir>
#include <QFile>
#include <QStandardPaths>

namespace safelog::qtapp {

AppController::AppController(QObject* parent)
  : QObject(parent),
    photoStore_((QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/photos").toStdString()),
    capture_(repository_, clock_, ids_, photoStore_),
    workflow_(repository_, clock_, ids_, photoStore_) {
  repository_.saveSite({"site-1", "세이프 금속 가공공장", "서울시 가상구 산업로 10"});
  repository_.saveProfile({"inspector-1", "김안전", UserRole::Inspector});
  repository_.saveProfile({"assignee-1", "이조치", UserRole::Assignee});
  repository_.saveProfile({"manager-1", "박관리", UserRole::Manager});
}

void AppController::setStatus(QString message) {
  if (statusMessage_ == message) return;
  statusMessage_ = std::move(message);
  emit statusMessageChanged();
}

void AppController::runDemoScenario() {
  try {
    const QString root = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(root);
    const QString photoPath = root + "/sample.jpg";
    QFile sample(photoPath);
    if (sample.open(QIODevice::WriteOnly)) sample.write("replace-with-camera-image");
    const auto inspection = capture_.startInspection("site-1", "inspector-1");
    const auto finding = capture_.addFinding({inspection.id, "inspector-1", "2층 가공라인 통로",
      "통로 자재 적치", "자재를 지정 구역으로 이동", photoPath.toStdString()}).finding;
    workflow_.assign(finding.id, "manager-1", "assignee-1");
    workflow_.beginWork(finding.id, "assignee-1");
    workflow_.submitAction(finding.id, "assignee-1", "통로 정리 완료", photoPath.toStdString());
    workflow_.verify(finding.id, "inspector-1");
    setStatus(QStringLiteral("전체 흐름 완료: 보고서 생성 가능"));
  } catch (const std::exception& error) {
    setStatus(QString::fromUtf8(error.what()));
  }
}

} // namespace safelog::qtapp
