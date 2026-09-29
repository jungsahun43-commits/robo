#include "app_controller.hpp"
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
int main(int argc, char* argv[]) {
  QGuiApplication app(argc, argv);
  QGuiApplication::setApplicationName("SafeLog");
  safelog::qtapp::AppController controller;
  QQmlApplicationEngine engine;
  engine.rootContext()->setContextProperty("appController", &controller);
  engine.rootContext()->setContextProperty("authController", controller.auth());
  engine.rootContext()->setContextProperty("aiController", controller.ai());
  engine.rootContext()->setContextProperty("captureController", controller.capture());
  engine.rootContext()->setContextProperty("workflowController", controller.workflow());
  engine.rootContext()->setContextProperty("reportingController", controller.reporting());
  engine.loadFromModule("SafeLog", "Main");
  if (engine.rootObjects().isEmpty()) return -1;
  return app.exec();
}
