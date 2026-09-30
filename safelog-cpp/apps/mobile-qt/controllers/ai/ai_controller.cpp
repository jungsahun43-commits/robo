#include "ai_controller.hpp"
#include "adapters/ai/http_ai_safety_analyzer.hpp"
#include <QtConcurrent/QtConcurrentRun>
#include <QFutureWatcher>
#include <QJsonDocument>
#include <QJsonObject>
#include <cmath>
#include <QStringList>

namespace safelog::qtapp {
namespace {
struct Reply {
  ai::HazardSuggestion hazard;
  ai::ActionAssessment comparison;
  QString state, error;
};
void validate(double confidence, const std::string& json) {
  if (!std::isfinite(confidence) || confidence < 0 || confidence > 1 ||
      !QJsonDocument::fromJson(QByteArray::fromStdString(json)).isObject())
    throw AiRequestError("InvalidJson", "AI 응답 형식 또는 신뢰도가 올바르지 않습니다.");
}
}
AiController::AiController(std::shared_ptr<ai::IAiSafetyAnalyzer> analyzer, QObject* parent, int timeoutMs)
  : QObject(parent), analyzer_(std::move(analyzer)) {
  timeout_.setSingleShot(true);
  timeout_.setInterval(qMax(1, timeoutMs));
  connect(&timeout_, &QTimer::timeout, this, [this] {
    requests_.invalidate();
    analyzer_->cancel();
    changeState("Timeout", "AI 응답 시간이 초과되었습니다.");
    if (comparison_) emit comparisonFailed(state_, error_);
    else emit analysisFailed(state_, error_);
  });
}
void AiController::changeState(QString state, QString error) {
  const bool wasLoading = loading();
  state_ = std::move(state); error_ = std::move(error);
  emit stateChanged();
  if (wasLoading != loading()) emit loadingChanged();
}
void AiController::reset() {
  requests_.invalidate(); analyzer_->cancel(); timeout_.stop(); result_.clear(); hazard_ = {}; assessment_ = {};
  changeState("Idle");
}
void AiController::manualFallback() { reset(); changeState("ManualFallback"); }
void AiController::analyzeHazard(const QString& photo, const QString& memo) { run(false, photo, {}, memo); }
void AiController::compareBeforeAfter(const QString& before, const QString& after, const QString& memo) {
  run(true, before, after, memo);
}
void AiController::run(bool comparison, QString before, QString after, QString memo) {
  if (loading()) return;
  const auto request = requests_.invalidate();
  comparison_ = comparison; result_.clear(); hazard_ = {}; assessment_ = {};
  changeState("Loading");
  if (comparison) emit comparisonStarted(); else emit analysisStarted();
  timeout_.start();
  auto* watcher = new QFutureWatcher<Reply>(this);
  connect(watcher, &QFutureWatcher<Reply>::finished, this, [this, watcher, request, comparison] {
    const Reply reply = watcher->result(); watcher->deleteLater();
    if (!requests_.accepts(request)) return; // Logout, cancellation, timeout, or a newer request.
    timeout_.stop();
    if (!reply.error.isEmpty()) {
      changeState(reply.state, reply.error);
      if (comparison) emit comparisonFailed(state_, error_); else emit analysisFailed(state_, error_);
      return;
    }
    if (comparison) {
      assessment_ = reply.comparison;
      QStringList risks;
      for (const auto& risk : reply.comparison.remainingRisks) risks << QString::fromStdString(risk);
      result_ = {{"likelyResolved", reply.comparison.likelyResolved}, {"remainingRisks", risks},
        {"assessment", QString::fromStdString(reply.comparison.assessment)}, {"confidence", reply.comparison.confidence}};
    } else {
      hazard_ = reply.hazard;
      QStringList hazards;
      for (const auto& hazard : hazard_.detectedHazards) hazards << QString::fromStdString(hazard);
      result_ = {{"category", QString::fromStdString(hazard_.category)}, {"riskLevel", hazard_.riskLevel},
        {"description", QString::fromStdString(hazard_.suggestedDescription)},
        {"action", QString::fromStdString(hazard_.suggestedAction)}, {"confidence", hazard_.confidence},
        {"detectedHazards", hazards}, {"modelName", QString::fromStdString(hazard_.modelName)}};
    }
    changeState("Success");
    if (comparison) emit comparisonSucceeded(); else emit analysisSucceeded();
  });
  watcher->setFuture(QtConcurrent::run([provider = analyzer_, mutex = providerMutex_, comparison,
      before = before.toStdString(), after = after.toStdString(), memo = memo.toStdString(), failure = demoFailure_] {
    Reply reply;
    try {
      if (!failure.isEmpty()) throw AiRequestError(failure, "개발용 AI 실패 시연입니다.");
      std::lock_guard lock(*mutex);
      if (comparison) {
        reply.comparison = provider->compareBeforeAfter(before, after, memo);
        validate(reply.comparison.confidence, reply.comparison.rawJson);
        if (reply.comparison.assessment.empty() || reply.comparison.modelName.empty() ||
            reply.comparison.promptVersion.empty())
          throw AiRequestError("InvalidJson", "AI 비교 응답의 필수 정보가 없습니다.");
      } else {
        reply.hazard = provider->analyzeHazard(before, memo);
        validate(reply.hazard.confidence, reply.hazard.rawJson);
        const auto& h = reply.hazard;
        if (!validHazard(h))
          throw AiRequestError("InvalidJson", "AI 위험 분석 필수 필드가 올바르지 않습니다.");
      }
    } catch (const AiProviderError& e) {
      reply.state = QString::fromStdString(aiProviderState(e.code));
      reply.error = QString::fromUtf8(e.what());
    } catch (const AiRequestError& e) { reply.state = e.state; reply.error = QString::fromUtf8(e.what());
    } catch (const std::exception& e) { reply.state = "Failure"; reply.error = QString::fromUtf8(e.what());
    } catch (...) { reply.state = "Failure"; reply.error = "AI 분석 중 알 수 없는 오류가 발생했습니다."; }
    return reply;
  }));
}
}
