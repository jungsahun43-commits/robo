#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace safelog {

using Id = std::string;
using TimePoint = std::chrono::system_clock::time_point;

enum class UserRole { Inspector, Assignee, Manager };
enum class FindingStatus { Open, InProgress, PendingReview, Verified };
enum class PhotoKind { Before, After };

struct Site {
  Id id;
  std::string name;
  std::string address;
};

struct Profile {
  Id id;
  std::string displayName;
  UserRole role{UserRole::Assignee};
};

struct Inspection {
  Id id;
  Id siteId;
  Id inspectorId;
  TimePoint inspectedAt;
};

struct Finding {
  Id id;
  Id inspectionId;
  std::string location;
  std::string description;
  std::string actionOpinion;
  std::optional<Id> assigneeId;
  std::optional<TimePoint> dueAt;
  FindingStatus status{FindingStatus::Open};
  TimePoint createdAt;
};

struct Photo {
  Id id;
  Id findingId;
  PhotoKind kind{PhotoKind::Before};
  std::string storagePath;
  TimePoint capturedAt;
};

struct ActionLog {
  Id id;
  Id findingId;
  Id actorId;
  std::string action;
  std::string note;
  TimePoint createdAt;
};

struct FindingBundle {
  Finding finding;
  std::vector<Photo> photos;
  std::vector<ActionLog> actionLogs;
};

std::string to_string(UserRole value);
std::string to_string(FindingStatus value);
std::string to_string(PhotoKind value);

} // namespace safelog
