#pragma once

#include "safelog/contracts/ports.hpp"

#include <atomic>
#include <filesystem>

namespace safelog::storage {

class SystemClock final : public IClock {
public:
  TimePoint now() const override;
};

class SequentialIdGenerator final : public IIdGenerator {
public:
  Id next(const std::string& prefix) override;
private:
  std::atomic<unsigned long long> sequence_{1};
};

class LocalPhotoStore final : public IPhotoStore {
public:
  explicit LocalPhotoStore(std::filesystem::path root);
  std::string importPhoto(const std::string& localPath,
                          const Id& findingId,
                          PhotoKind kind) override;
private:
  std::filesystem::path root_;
};

} // namespace safelog::storage
