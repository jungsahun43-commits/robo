#pragma once

#include <stdexcept>
#include <string>

namespace safelog {

class ValidationError : public std::runtime_error {
public:
  explicit ValidationError(const std::string& message) : std::runtime_error(message) {}
};

class NotFoundError : public std::runtime_error {
public:
  explicit NotFoundError(const std::string& message) : std::runtime_error(message) {}
};

class TransitionError : public std::runtime_error {
public:
  explicit TransitionError(const std::string& message) : std::runtime_error(message) {}
};

} // namespace safelog
