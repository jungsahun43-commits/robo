#pragma once
#include "safelog/ai/ai_analyzer.hpp"
#include "adapters/application_support.hpp"
#include <QObject>
#include <QVariantMap>
#include <QTimer>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace safelog::qtapp {
// Providers throw AiRequestError to distinguish transport/schema failures.
class AiRequestError : public std::runtime_error {
public:
  AiRequestError(QString state, const char* message)
    : std::runtime_error(message), state(std::move(state)) {}
  QString state;
};
class AiController final : public QObject {
  Q_OBJECT
  Q_PROPERTY(QString state READ state NOTIFY stateChanged)
  Q_PROPERTY(bool loading READ loading NOTIFY loadingChanged)
  Q_PROPERTY(QString error READ error NOTIFY stateChanged)
  Q_PROPERTY(QVariantMap result READ result NOTIFY stateChanged)
  Q_PROPERTY(QString demoFailure READ demoFailure WRITE setDemoFailure NOTIFY stateChanged)
public:
  explicit AiController(std::shared_ptr<ai::IAiSafetyAnalyzer> analyzer, QObject* parent = nullptr,
    int timeoutMs = 30000);
  QString state() const { return state_; }
  bool loading() const { return state_ == "Loading"; }
  QString error() const { return error_; }
  QVariantMap result() const { return result_; }
  QString demoFailure() const { return demoFailure_; }
  void setDemoFailure(const QString& value) { demoFailure_ = value; emit stateChanged(); }
  const ai::ActionAssessment& assessment() const { return assessment_; }
  const ai::HazardSuggestion& hazard() const { return hazard_; }
  Q_INVOKABLE void analyzeHazard(const QString& photoPath, const QString& memo);
  Q_INVOKABLE void compareBeforeAfter(const QString& before, const QString& after, const QString& memo);
  Q_INVOKABLE void reset();
  Q_INVOKABLE void manualFallback();
 signals:
  void analysisStarted();
  void analysisSucceeded();
  void analysisFailed(QString state, QString message);
  void comparisonStarted();
  void comparisonSucceeded();
  void comparisonFailed(QString state, QString message);
  void loadingChanged();
  void stateChanged();
private:
  void run(bool comparison, QString before, QString after, QString memo);
  void changeState(QString state, QString error = {});
  std::shared_ptr<ai::IAiSafetyAnalyzer> analyzer_;
  // Serializes providers that are not reentrant, even after UI cancellation.
  std::shared_ptr<std::mutex> providerMutex_{std::make_shared<std::mutex>()};
  QTimer timeout_;
  RequestEpoch requests_;
  bool comparison_{false};
  QString state_{"Idle"}, error_, demoFailure_;
  QVariantMap result_;
  ai::HazardSuggestion hazard_;
  ai::ActionAssessment assessment_;
};
}
