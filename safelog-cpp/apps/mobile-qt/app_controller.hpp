#pragma once

#include "safelog/capture/capture_service.hpp"
#include "safelog/reporting/report_service.hpp"
#include "safelog/storage/in_memory_repository.hpp"
#include "safelog/storage/system_services.hpp"
#include "safelog/workflow/workflow_service.hpp"

#include <QObject>
#include <QString>

namespace safelog::qtapp {

class AppController final : public QObject {
  Q_OBJECT
  Q_PROPERTY(QString statusMessage READ statusMessage NOTIFY statusMessageChanged)
public:
  explicit AppController(QObject* parent = nullptr);
  QString statusMessage() const { return statusMessage_; }

  Q_INVOKABLE void runDemoScenario();

signals:
  void statusMessageChanged();

private:
  void setStatus(QString message);
  storage::InMemoryRepository repository_;
  storage::SystemClock clock_;
  storage::SequentialIdGenerator ids_;
  storage::LocalPhotoStore photoStore_;
  capture::CaptureService capture_;
  workflow::WorkflowService workflow_;
  QString statusMessage_{QStringLiteral("준비됨")};
};

} // namespace safelog::qtapp
