#include "safelog/storage/system_services.hpp"
#include "safelog/contracts/errors.hpp"

#include <fstream>

namespace safelog::storage {

TimePoint SystemClock::now() const { return std::chrono::system_clock::now(); }

Id SequentialIdGenerator::next(const std::string& prefix) {
  return prefix + "-" + std::to_string(sequence_.fetch_add(1));
}

LocalPhotoStore::LocalPhotoStore(std::filesystem::path root) : root_(std::move(root)) {
  std::filesystem::create_directories(root_);
}

std::string LocalPhotoStore::importPhoto(const std::string& localPath,
                                         const Id& findingId,
                                         PhotoKind kind) {
  const std::filesystem::path source(localPath);
  if (!std::filesystem::exists(source)) throw ValidationError("Photo file does not exist: " + localPath);
  const auto folder = root_ / findingId;
  std::filesystem::create_directories(folder);
  const auto destination = folder / (to_string(kind) + "-" + source.filename().string());
  std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing);
  return destination.generic_string();
}

} // namespace safelog::storage
