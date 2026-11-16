#include "safelog/storage/uuid_id_generator.hpp"

#include <array>
#include <chrono>
#include <cstdint>

namespace safelog::storage {

UuidIdGenerator::UuidIdGenerator() {
  std::random_device device;
  // Mix in the clock in case random_device is deterministic on some platform.
  const auto now = static_cast<std::uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
  std::seed_seq seed{device(), device(), device(), device(),
                     static_cast<unsigned>(now), static_cast<unsigned>(now >> 32)};
  engine_.seed(seed);
}

Id UuidIdGenerator::next(const std::string& prefix) {
  std::array<std::uint8_t, 16> bytes{};
  {
    std::scoped_lock lock(mutex_);
    const std::uint64_t hi = engine_();
    const std::uint64_t lo = engine_();
    for (int i = 0; i < 8; ++i) {
      bytes[i] = static_cast<std::uint8_t>(hi >> (56 - 8 * i));
      bytes[8 + i] = static_cast<std::uint8_t>(lo >> (56 - 8 * i));
    }
  }
  bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0F) | 0x40);  // version 4
  bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3F) | 0x80);  // RFC 4122 variant

  static constexpr char kHex[] = "0123456789abcdef";
  Id id = prefix;
  id += '-';
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) id += '-';
    id += kHex[bytes[i] >> 4];
    id += kHex[bytes[i] & 0x0F];
  }
  return id;
}

} // namespace safelog::storage
