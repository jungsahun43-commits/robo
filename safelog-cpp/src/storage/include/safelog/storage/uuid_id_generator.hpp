#pragma once

#include "safelog/contracts/ports.hpp"

#include <mutex>
#include <random>

namespace safelog::storage {

// Generates "<prefix>-<random UUID v4>", e.g. "finding-3f2b8c1e-9a4d-4c7e-b5f0-1d2e3f4a5b6c".
// Unlike SequentialIdGenerator, ids stay unique across app restarts, so new
// records never collide with (and overwrite) rows already in SqliteRepository.
class UuidIdGenerator final : public IIdGenerator {
public:
  UuidIdGenerator();
  Id next(const std::string& prefix) override;

private:
  std::mutex mutex_;
  std::mt19937_64 engine_;
};

} // namespace safelog::storage
