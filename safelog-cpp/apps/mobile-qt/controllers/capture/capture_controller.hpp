#pragma once
#include "controllers/auth/auth_controller.hpp"
#include "controllers/ai/ai_controller.hpp"
#include "safelog/capture/capture_service.hpp"
#include "safelog/ai/ai_safety_service.hpp"
#include <QObject>
#include <QVariantMap>
namespace safelog::qtapp {
// Temporary role-1 integration controller: all repository access stays on the GUI thread.
class CaptureController final : public QObject {
  Q_OBJECT
  Q_PROPERTY(QVariantMap draft READ draft NOTIFY changed)
  Q_PROPERTY(QString error READ error NOTIFY changed)
public:
  CaptureController(IRepository& repository, IClock& clock, IIdGenerator& ids,
    capture::CaptureService& capture, ai::AiSafetyService& reviews,
    AuthController& auth, AiController& ai, QObject* parent = nullptr);
  QVariantMap draft() const { return draft_; }
  QString error() const { return error_; }
  Q_INVOKABLE bool createFinding(QString photoPath, QString location, QString memo, QString action);
  Q_INVOKABLE void retry();
  Q_INVOKABLE bool saveReview(QString decision, QString description, QString action);
  Q_INVOKABLE void reset();
  Q_INVOKABLE QString demoPhoto(bool after = false);
  Q_INVOKABLE bool resumeFinding(QString findingId);
signals:
  void changed();
  void saved();
private:
  bool authorized() const;
  IRepository& repository_;
  IClock& clock_;
  IIdGenerator& ids_;
  capture::CaptureService& capture_;
  ai::AiSafetyService& reviews_;
  AuthController& auth_;
  AiController& ai_;
  QVariantMap draft_;
  QString error_;
  Id findingId_, analysisId_;
  bool saved_{false};
};
}
