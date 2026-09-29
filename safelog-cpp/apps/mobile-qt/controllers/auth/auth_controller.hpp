#pragma once
#include "adapters/application_support.hpp"
#include <QObject>
#include <QVariantMap>
namespace safelog::qtapp {
// TODO(role 2): replace mock password validation with the real authentication adapter.
class AuthController final : public QObject {
  Q_OBJECT
  Q_PROPERTY(QVariantMap session READ session NOTIFY sessionChanged)
  Q_PROPERTY(bool loading READ loading NOTIFY changed)
  Q_PROPERTY(QString error READ error NOTIFY changed)
public:
  explicit AuthController(IRepository& repository, QObject* parent = nullptr) : QObject(parent), adapter_(repository) {}
  QVariantMap session() const { return session_; }
  bool loading() const { return loading_; }
  QString error() const { return error_; }
  Q_INVOKABLE void login(QString userId, QString password);
  Q_INVOKABLE void logout();
signals:
  void changed();
  void sessionChanged();
private:
  MockAuthAdapter adapter_;
  QVariantMap session_;
  QString error_;
  bool loading_{false};
  RequestEpoch requests_;
};
}
