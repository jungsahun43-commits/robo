#include "auth_controller.hpp"
#include <QTimer>
namespace safelog::qtapp {
void AuthController::login(QString userId, QString password) {
  if (loading_ || !session_.isEmpty()) return;
  loading_ = true; error_.clear(); emit changed();
  const auto request = requests_.invalidate();
  QTimer::singleShot(150, this, [this, request, userId = userId.trimmed(), password] {
    if (!requests_.accepts(request)) return;
    const auto authenticated = adapter_.login(userId.toStdString(), password.toStdString());
    if (!authenticated) error_ = "사용자 ID 또는 비밀번호를 확인하세요.";
    else {
      const auto& profile = authenticated->user;
      const auto& site = authenticated->site;
      const QString role = profile.role == UserRole::Inspector ? "Inspector" :
                           profile.role == UserRole::Manager ? "Manager" : "Assignee";
      session_ = {{"userId", userId}, {"userName", QString::fromStdString(profile.displayName)},
        {"role", role}, {"siteId", QString::fromStdString(site.id)}, {"siteName", QString::fromStdString(site.name)}};
    }
    loading_ = false; emit changed();
    if (!session_.isEmpty()) emit sessionChanged();
  });
}
void AuthController::logout() {
  requests_.invalidate(); loading_ = false; error_.clear(); session_.clear(); emit changed(); emit sessionChanged();
}
}
