#pragma once

#include "safelog/contracts/types.hpp"

#include <array>

namespace safelog::auth {

// The three demo people, shared by storage::SqliteRepository::seedDemoData()
// (profiles) and AuthService::seedDemoAccounts() (logins). Edit them only here.
// Passwords are never part of this table; they are supplied at runtime.
struct DemoPerson {
  const char* profileId;
  const char* displayName;
  UserRole role;
  const char* loginId;
};

inline constexpr std::array<DemoPerson, 3> kDemoPeople{{
  {"inspector-1", "김안전", UserRole::Inspector, "inspector1"},
  {"manager-1", "박관리", UserRole::Manager, "manager1"},
  {"assignee-1", "이조치", UserRole::Assignee, "assignee1"},
}};

} // namespace safelog::auth
