#pragma once

#include <cstddef>
#include <cstdint>

namespace safelog::auth::detail {

// Fills `out` with bytes from the operating system's CSPRNG
// (BCryptGenRandom on Windows, /dev/urandom elsewhere). Throws std::runtime_error on failure.
void secureRandom(std::uint8_t* out, std::size_t size);

} // namespace safelog::auth::detail
