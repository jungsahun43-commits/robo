#include "capture_controller.hpp"
#include "adapters/workflow_support.hpp"
#include <QUrl>
#include <QImageReader>
#include <QImage>
#include <QPainter>
#include <QStandardPaths>
#include <QDir>
#include <QUuid>

namespace safelog::qtapp {
CaptureController::CaptureController(IRepository& r, IClock& c, IIdGenerator& i,
    capture::CaptureService& capture, ai::AiSafetyService& reviews, AuthController& auth,
    AiController& ai, QObject* parent)
  : QObject(parent), repository_(r), clock_(c), ids_(i), capture_(capture), reviews_(reviews), auth_(auth), ai_(ai) {
  connect(&ai_, &AiController::analysisSucceeded, this, [this] {
    if (!authorized() || saved_) return;
    try {
      const auto& h = ai_.hazard();
      AiAnalysis analysis{ids_.next("ai"), findingId_, AiAnalysisType::BeforeHazard,
        h.modelName, "hazard-v1", h.riskLevel, h.category, h.confidence, h.rawJson,
        AiReviewDecision::Pending, std::nullopt, clock_.now()};
      // Preserve the provider's original JSON, and the typed suggestions separately in the UI.
      repository_.saveAiAnalysis(analysis);
      repository_.saveActionLog({ids_.next("log"), findingId_, auth_.session().value("userId").toString().toStdString(), "ai_hazard_suggestion",
        h.category + "\n" + h.suggestedDescription + "\n" + h.suggestedAction, clock_.now()});
      analysisId_ = analysis.id;
      error_.clear(); emit changed();
    } catch (const std::exception& e) { error_ = QString::fromUtf8(e.what()); emit changed(); }
  });
  connect(&auth_, &AuthController::sessionChanged, this, &CaptureController::reset);
}
bool CaptureController::authorized() const {
  if (auth_.session().value("role").toString() != "Inspector" || findingId_.empty()) return false;
  return canReviewFinding(repository_, findingId_, auth_.session().value("userId").toString().toStdString());
}
void CaptureController::reset() {
  ai_.reset(); draft_.clear(); findingId_.clear(); analysisId_.clear(); error_.clear(); saved_ = false; emit changed();
}
bool CaptureController::createFinding(QString photoPath, QString location, QString memo, QString action) {
  const auto profile = repository_.findProfile(auth_.session().value("userId").toString().toStdString());
  if (!profile || !canCapture(*profile) || ai_.loading()) return false;
  error_.clear();
  const QUrl url(photoPath);
  if (url.isLocalFile()) photoPath = url.toLocalFile();
  location = location.trimmed(); memo = memo.trimmed(); action = action.trimmed();
  if (location.isEmpty() || memo.isEmpty() || action.isEmpty() || !QImageReader(photoPath).canRead()) {
    error_ = "장소, 메모, 수동 권장 조치와 읽을 수 있는 로컬 사진을 입력하세요."; emit changed(); return false;
  }
  try {
    reset();
    const auto session = auth_.session();
    const auto user = session.value("userId").toString().toStdString();
    const auto inspection = capture_.startInspection(session.value("siteId").toString().toStdString(), user);
    const auto bundle = capture_.addFinding({inspection.id, user, location.toStdString(), memo.toStdString(),
      action.toStdString(), photoPath.toStdString()});
    findingId_ = bundle.finding.id;
    draft_ = {{"inspectionId", QString::fromStdString(inspection.id)}, {"findingId", QString::fromStdString(findingId_)},
      {"userId", QString::fromStdString(user)}, {"location", location}, {"memo", memo}, {"action", action},
      {"photoPath", QString::fromStdString(bundle.photos.front().storagePath)}};
    emit changed(); retry(); return true;
  } catch (const std::exception& e) { error_ = QString::fromUtf8(e.what()); emit changed(); return false; }
}
void CaptureController::retry() {
  if (!authorized() || saved_ || ai_.loading()) return;
  analysisId_.clear();
  ai_.analyzeHazard(draft_.value("photoPath").toString(), draft_.value("memo").toString());
}
bool CaptureController::saveReview(QString decision, QString description, QString action) {
  if (!authorized() || saved_ || ai_.loading()) return false;
  description = description.trimmed(); action = action.trimmed();
  if (description.isEmpty() || action.isEmpty()) {
    error_ = "위험 설명과 권장 조치를 입력하세요."; emit changed(); return false;
  }
  try {
    const auto user = auth_.session().value("userId").toString().toStdString();
    if (decision == "Accepted" || decision == "Edited" || decision == "Rejected") {
      if (analysisId_.empty()) throw std::runtime_error("저장된 AI 분석이 없습니다. 수동 입력을 이용하세요.");
      const auto actualDecision = decision == "Rejected" ? AiReviewDecision::Rejected :
        hazardDecision(ai_.hazard(), description.toStdString(), action.toStdString());
      reviews_.reviewHazard(analysisId_, user, {actualDecision, description.toStdString(), action.toStdString()});
    } else if (decision == "Manual" && ai_.state() == "ManualFallback") {
      saveManualReview(repository_, clock_, ids_, reviews_, findingId_, user, analysisId_,
        description.toStdString(), action.toStdString());
    } else throw std::runtime_error("올바른 검토 방법을 선택하세요.");
    saved_ = true; error_.clear(); emit changed(); emit saved(); return true;
  } catch (const std::exception& e) { error_ = QString::fromUtf8(e.what()); emit changed(); return false; }
}
QString CaptureController::demoPhoto(bool after) {
  const auto directory = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
  QDir().mkpath(directory);
  const auto path = directory + "/demo-" + QUuid::createUuid().toString(QUuid::WithoutBraces) + ".png";
  QImage image(640, 480, QImage::Format_RGB32); image.fill(QColor("#e1e8ec"));
  QPainter painter(&image);
  painter.fillRect(220, 0, 200, 480, QColor("#f5d65c"));
  if (!after) painter.fillRect(260, 220, 180, 120, QColor("#9c7957"));
  painter.setPen(Qt::black); painter.drawText(20, 35, "SAFELOG MOCK PHOTO - NOT A REAL SITE"); painter.end();
  if (!image.save(path)) { error_ = "시연 사진을 만들 수 없습니다."; emit changed(); return {}; }
  return QUrl::fromLocalFile(path).toString();
}
bool CaptureController::resumeFinding(QString findingId) {
  reset();
  const auto user = auth_.session().value("userId").toString().toStdString();
  const auto finding = repository_.findFinding(findingId.toStdString());
  if (!finding || !canReviewFinding(repository_, finding->id, user) || finding->status != FindingStatus::Open) {
    error_ = "PermissionDenied: 본인의 Open 기록만 위험 검토할 수 있습니다."; emit changed(); return false;
  }
  const auto before = latestPhoto(repository_, finding->id, PhotoKind::Before);
  if (!before) { error_ = "조치 전 사진이 없습니다."; emit changed(); return false; }
  findingId_ = finding->id;
  draft_ = {{"findingId", findingId}, {"inspectionId", QString::fromStdString(finding->inspectionId)},
    {"userId", QString::fromStdString(user)}, {"location", QString::fromStdString(finding->location)},
    {"memo", QString::fromStdString(finding->description)}, {"action", QString::fromStdString(finding->actionOpinion)},
    {"photoPath", QString::fromStdString(before->storagePath)}};
  emit changed(); retry(); return true;
}

}
