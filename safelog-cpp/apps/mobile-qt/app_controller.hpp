#pragma once
#include "safelog/storage/in_memory_repository.hpp"
#include "safelog/storage/system_services.hpp"
#include "safelog/workflow/workflow_service.hpp"
#include "controllers/auth/auth_controller.hpp"
#include "controllers/ai/ai_controller.hpp"
#include "controllers/capture/capture_controller.hpp"
#include "controllers/workflow/workflow_controller.hpp"
#include "controllers/reporting/reporting_controller.hpp"

namespace safelog::qtapp {
// Composition root; feature logic remains in services/controllers.
class AppController final : public QObject {
  Q_OBJECT
  Q_PROPERTY(QVariantList dashboard READ dashboard NOTIFY dashboardChanged)
public:
  explicit AppController(QObject* parent = nullptr,
    std::shared_ptr<ai::IAiSafetyAnalyzer> analyzer = {});
  QVariantList dashboard() const;
signals:
  void dashboardChanged();
public:
  AuthController* auth() { return &auth_; }
  AiController* ai() { return &ai_; }
  WorkflowController* workflow() { return &workflowController_; }
  ReportingController* reporting() { return &reportingController_; }
  CaptureController* capture() { return &captureController_; }
private:
  std::vector<Id> inspectionIds_;
  storage::InMemoryRepository repository_;
  storage::SystemClock clock_;
  storage::SequentialIdGenerator ids_;
  storage::LocalPhotoStore photoStore_;
  std::shared_ptr<ai::IAiSafetyAnalyzer> analyzer_;
  capture::CaptureService capture_;
  workflow::WorkflowService workflow_;
  ai::AiSafetyService reviews_;
  AuthController auth_;
  AiController ai_;
  CaptureController captureController_;
  reporting::ReportService reports_;
  MockSharingAdapter sharing_;
  WorkflowController workflowController_;
  ReportingController reportingController_;
};
}
