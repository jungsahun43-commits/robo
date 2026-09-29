#pragma once
#include "controllers/auth/auth_controller.hpp"
#include "controllers/ai/ai_controller.hpp"
#include "safelog/workflow/workflow_service.hpp"
#include "safelog/ai/ai_safety_service.hpp"
#include "adapters/workflow_support.hpp"
#include <QVariantList>
namespace safelog::qtapp {
class WorkflowController final : public QObject {
  Q_OBJECT
  Q_PROPERTY(QVariantList findings READ findings NOTIFY changed)
  Q_PROPERTY(QVariantMap selected READ selected NOTIFY changed)
  Q_PROPERTY(QString error READ error NOTIFY changed)
public:
  WorkflowController(IRepository&, IClock&, IIdGenerator&, workflow::WorkflowService&,
    ai::AiSafetyService&, AuthController&, AiController&, const std::vector<Id>&, QObject* parent = nullptr);
  QVariantList findings() const;
  QVariantMap selected() const;
  QString error() const { return error_; }
  Q_INVOKABLE bool selectFinding(QString id);
  Q_INVOKABLE bool assign(QString assigneeId);
  Q_INVOKABLE bool beginWork();
  Q_INVOKABLE bool submitAction(QString note, QString photo, QString extraMemo);
  Q_INVOKABLE void compare();
  Q_INVOKABLE void directReview();
  Q_INVOKABLE bool verify(QString decision, QString note, bool photosConfirmed);
  Q_INVOKABLE bool requestChanges(QString note);
  Q_INVOKABLE void reset();
  void refresh() { emit changed(); }
  bool visible(const Finding& finding) const;
signals:
  void changed();
  void actionSubmitted();
  void verified();
private:
  QVariantMap asMap(const Finding&) const;
  Finding requireSelected() const;
  Id userId() const;
  bool fail(const std::exception&);
  IRepository& repository_;
  IClock& clock_;
  IIdGenerator& ids_;
  workflow::WorkflowService& workflow_;
  ai::AiSafetyService& reviews_;
  AuthController& auth_;
  AiController& ai_;
  const std::vector<Id>& inspectionIds_;
  Id selectedId_, analysisId_, comparingId_;
  QString error_;
};
}
