#pragma once
#include "controllers/auth/auth_controller.hpp"
#include "safelog/reporting/report_service.hpp"
#include "adapters/sharing_adapter.hpp"
#include <QVariantMap>
#include <QVariantList>
namespace safelog::qtapp {
class ReportingController final : public QObject {
  Q_OBJECT
  Q_PROPERTY(QVariantMap report READ report NOTIFY changed)
  Q_PROPERTY(QString state READ state NOTIFY changed)
  Q_PROPERTY(QString error READ error NOTIFY changed)
  Q_PROPERTY(QString outputPath READ outputPath NOTIFY changed)
  Q_PROPERTY(QString shareState READ shareState NOTIFY changed)
public:
  ReportingController(IRepository&, reporting::ReportService&, AuthController&, SharingAdapter&, QObject* parent = nullptr);
  QVariantMap report() const { return report_; }
  QString state() const { return state_; }
  QString error() const { return error_; }
  QString outputPath() const { return outputPath_; }
  QString shareState() const { return shareState_; }
  Q_INVOKABLE void generate(QString inspectionId);
  Q_INVOKABLE void share(QString mockOutcome = "ShareSucceeded");
  Q_INVOKABLE void reset();
signals:
  void changed();
  void generationSucceeded();
  void generationFailed(QString message);
  void shareFinished(QString result);
private:
  IRepository& repository_;
  reporting::ReportService& service_;
  AuthController& auth_;
  SharingAdapter& sharing_;
  RequestEpoch requests_;
  QVariantMap report_;
  QString state_{"Idle"}, error_, outputPath_, shareState_;
};
}
