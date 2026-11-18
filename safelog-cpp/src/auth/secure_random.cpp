#include "secure_random.hpp"

#include <stdexcept>

// Platform CSPRNG:
//   Windows          BCryptGenRandom
//   Android, Apple   arc4random_buf (getrandom needs Android API 28+ and does not exist on iOS/macOS)
//   other Linux      getrandom (glibc 2.25+ / musl)
//   anything else    /dev/urandom
#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <bcrypt.h>
#elif defined(__ANDROID__) || defined(__APPLE__)
#  include <stdlib.h>
#elif defined(__linux__)
#  include <cerrno>
#  include <sys/random.h>
#else
#  include <cstdio>
#endif

namespace safelog::auth::detail {

void secureRandom(std::uint8_t* out, std::size_t size) {
#if defined(_WIN32)
  // BCryptGenRandom takes a ULONG length; our requests are tiny (salts).
  const NTSTATUS status = BCryptGenRandom(nullptr, out, static_cast<ULONG>(size), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
  if (status < 0) throw std::runtime_error("BCryptGenRandom failed");
#elif defined(__ANDROID__) || defined(__APPLE__)
  arc4random_buf(out, size);  // cannot fail
#elif defined(__linux__)
  std::size_t filled = 0;
  while (filled < size) {
    const ssize_t n = getrandom(out + filled, size - filled, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("getrandom failed");
    }
    filled += static_cast<std::size_t>(n);
  }
#else
  std::FILE* file = std::fopen("/dev/urandom", "rb");
  if (!file) throw std::runtime_error("Cannot open /dev/urandom");
  const std::size_t read = std::fread(out, 1, size, file);
  std::fclose(file);
  if (read != size) throw std::runtime_error("Short read from /dev/urandom");
#endif
}

} // namespace safelog::auth::detail
