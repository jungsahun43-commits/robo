#include "secure_random.hpp"

#include <stdexcept>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <bcrypt.h>
#else
#  include <cstdio>
#endif

namespace safelog::auth::detail {

void secureRandom(std::uint8_t* out, std::size_t size) {
#if defined(_WIN32)
  // BCryptGenRandom takes a ULONG length; our requests are tiny (salts).
  const NTSTATUS status = BCryptGenRandom(nullptr, out, static_cast<ULONG>(size), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
  if (status < 0) throw std::runtime_error("BCryptGenRandom failed");
#else
  std::FILE* file = std::fopen("/dev/urandom", "rb");
  if (!file) throw std::runtime_error("Cannot open /dev/urandom");
  const std::size_t read = std::fread(out, 1, size, file);
  std::fclose(file);
  if (read != size) throw std::runtime_error("Short read from /dev/urandom");
#endif
}

} // namespace safelog::auth::detail
