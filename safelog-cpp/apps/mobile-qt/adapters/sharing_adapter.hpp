#pragma once
#include <filesystem>
#include <string>
namespace safelog::qtapp {
enum class ShareResult { ShareSucceeded, ShareCancelled, ShareFailed };
class SharingAdapter {
public:
  virtual ~SharingAdapter() = default;
  virtual ShareResult share(const std::string& path, const std::string& mimeType) = 0;
};
// Development simulation only. Android integration replaces this with FileProvider + ACTION_SEND.
class MockSharingAdapter final : public SharingAdapter {
public:
  ShareResult nextResult{ShareResult::ShareSucceeded};
  ShareResult share(const std::string& path, const std::string& mimeType) override {
    if (!std::filesystem::is_regular_file(path) || mimeType != "text/html") return ShareResult::ShareFailed;
    return nextResult;
  }
};
}
