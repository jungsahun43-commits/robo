#pragma once

#include <stdexcept>
#include <string>

namespace safelog::auth {

// Login or password check failed. The message is always kLoginFailedMessage so
// callers cannot tell an unknown id, a wrong password and a locked account apart.
class AuthenticationError : public std::runtime_error {
public:
  AuthenticationError();
};

// The acting user is not allowed to perform the operation (e.g. non-manager creating accounts).
class PermissionError : public std::runtime_error {
public:
  explicit PermissionError(const std::string& message) : std::runtime_error(message) {}
};

inline constexpr const char* kLoginFailedMessage = "아이디 또는 비밀번호가 올바르지 않습니다";

inline AuthenticationError::AuthenticationError() : std::runtime_error(kLoginFailedMessage) {}

} // namespace safelog::auth
