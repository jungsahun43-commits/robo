#include "workflow_controller.hpp"
#include <QUrl>
#include <QImageReader>
namespace safelog::qtapp {
WorkflowController::WorkflowController(IRepository& r, IClock& c, IIdGenerator& i,
    workflow::WorkflowService& workflow, ai::AiSafetyService& reviews, AuthController& auth,
    AiController& ai, const std::vector<Id>& inspections, QObject* parent)
  : QObject(parent), repository_(r), clock_(c), ids_(i), workflow_(workflow), reviews_(reviews),
    auth_(auth), ai_(ai), inspectionIds_(inspections) {
  connect(&auth_, &AuthController::sessionChanged, this, &WorkflowController::reset);
  connect(&ai_, &AiController::comparisonSucceeded, this, [this] {
    if (comparingId_.empty() || comparingId_ != selectedId_) return;
    try {
      const auto finding = requireSelected();
      analysisId_ = storeComparison(repository_, clock_, ids_, finding.id, ai_.assessment(), userId()).id;
      comparingId_.clear(); error_.clear(); emit changed();
    } catch (const std::exception& e) { fail(e); }
  });
}
Id WorkflowController::userId() const { return auth_.session().value("userId").toString().toStdString(); }
bool WorkflowController::visible(const Finding& f) const { return canViewFinding(repository_, f, userId()); }
Finding WorkflowController::requireSelected() const {
  const auto f = repository_.findFinding(selectedId_);
  if (!f) throw NotFoundError("선택한 위험 기록이 없습니다.");
  if (!visible(*f)) throw ValidationError("PermissionDenied: 이 기록을 볼 권한이 없습니다.");
  return *f;
}
bool WorkflowController::fail(const std::exception& e) {
  error_ = QString::fromUtf8(e.what()); emit changed(); return false;
}
QVariantMap WorkflowController::asMap(const Finding& f) const {
  const auto assignee = f.assigneeId ? repository_.findProfile(*f.assigneeId) : std::nullopt;
  const auto before = latestPhoto(repository_, f.id, PhotoKind::Before);
  const auto after = latestPhoto(repository_, f.id, PhotoKind::After);
  return {{"findingId", QString::fromStdString(f.id)}, {"inspectionId", QString::fromStdString(f.inspectionId)},
    {"location", QString::fromStdString(f.location)}, {"description", QString::fromStdString(f.description)},
    {"riskLevel", f.riskLevel.value_or(0)}, {"status", QString::fromStdString(to_string(f.status))},
    {"assigneeId", QString::fromStdString(f.assigneeId.value_or(""))},
    {"assigneeName", assignee ? QString::fromStdString(assignee->displayName) : QStringLiteral("담당자가 지정되지 않았습니다.")},
    {"beforePhotoPath", before ? QString::fromStdString(before->storagePath) : QString()},
    {"afterPhotoPath", after ? QString::fromStdString(after->storagePath) : QString()},
    {"actionNote", QString::fromStdString(latestAction(repository_, f.id))}};
}
QVariantList WorkflowController::findings() const {
  QVariantList list;
  for (const auto& id : inspectionIds_)
    for (const auto& f : repository_.findingsForInspection(id)) if (visible(f)) list << asMap(f);
  return list;
}
QVariantMap WorkflowController::selected() const {
  const auto f = repository_.findFinding(selectedId_);
  return f && visible(*f) ? asMap(*f) : QVariantMap{};
}
void WorkflowController::reset() {
  ai_.reset(); selectedId_.clear(); comparingId_.clear(); analysisId_.clear(); error_.clear(); emit changed();
}
bool WorkflowController::selectFinding(QString id) {
  reset(); selectedId_ = id.toStdString();
  try { requireSelected(); emit changed(); return true; } catch (const std::exception& e) { selectedId_.clear(); return fail(e); }
}
bool WorkflowController::assign(QString assigneeId) {
  try { workflow_.assign(requireSelected().id, userId(), assigneeId.toStdString()); error_.clear(); emit changed(); return true; }
  catch (const std::exception& e) { return fail(e); }
}
bool WorkflowController::beginWork() {
  try { workflow_.beginWork(requireSelected().id, userId()); error_.clear(); emit changed(); return true; }
  catch (const std::exception& e) { return fail(e); }
}
bool WorkflowController::submitAction(QString note, QString photo, QString extraMemo) {
  try {
    note = note.trimmed();
    const QUrl url(photo); if (url.isLocalFile()) photo = url.toLocalFile();
    if (note.isEmpty() || !QImageReader(photo).canRead())
      throw ValidationError("MissingActionData: 조치 내용과 읽을 수 있는 조치 후 사진이 필요합니다.");
    if (!extraMemo.trimmed().isEmpty()) note += "\n추가 메모: " + extraMemo.trimmed();
    workflow_.submitAction(requireSelected().id, userId(), note.toStdString(), photo.toStdString());
    error_.clear(); emit changed(); emit actionSubmitted(); compare(); return true;
  } catch (const std::exception& e) { return fail(e); }
}
void WorkflowController::compare() {
  try {
    const auto f = requireSelected();
    if (f.status != FindingStatus::PendingReview) throw TransitionError("PendingReview 상태에서 비교할 수 있습니다.");
    const auto before = latestPhoto(repository_, f.id, PhotoKind::Before);
    const auto after = latestPhoto(repository_, f.id, PhotoKind::After);
    if (!before || !after) throw ValidationError("MissingActionData: 전후 사진이 필요합니다.");
    ai_.reset(); analysisId_.clear(); comparingId_ = f.id; error_.clear();
    ai_.compareBeforeAfter(QString::fromStdString(before->storagePath), QString::fromStdString(after->storagePath),
      QString::fromStdString(latestAction(repository_, f.id)));
    emit changed();
  } catch (const std::exception& e) { fail(e); }
}
void WorkflowController::directReview() { comparingId_.clear(); ai_.manualFallback(); emit changed(); }
bool WorkflowController::verify(QString decision, QString note, bool photosConfirmed) {
  try {
    const auto f = requireSelected();
    if (ai_.loading()) throw ValidationError("AI 완료를 기다리거나 AI 없이 직접 확인을 선택하세요.");
    if (decision != "Accepted" && decision != "Rejected" && decision != "Manual") throw ValidationError("AI 검토 방법을 선택하세요.");
    const std::optional<AiReviewDecision> review = decision == "Manual" ? std::nullopt :
      std::optional<AiReviewDecision>{decision == "Accepted" ? AiReviewDecision::Accepted : AiReviewDecision::Rejected};
    finalizeReview(repository_, workflow_, reviews_, f.id, userId(), analysisId_, review,
      note.trimmed().toStdString(), photosConfirmed);
    error_.clear(); emit changed(); emit verified(); return true;
  } catch (const std::exception& e) { return fail(e); }
}
bool WorkflowController::requestChanges(QString note) {
  try {
    workflow_.requestChanges(requireSelected().id, userId(), note.trimmed().toStdString());
    ai_.reset(); analysisId_.clear(); comparingId_.clear(); error_.clear(); emit changed(); return true;
  } catch (const std::exception& e) { return fail(e); }
}
}
