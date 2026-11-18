#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace safelog::auth {

struct Argon2Params {
  std::uint32_t memoryKiB{19456};  // 19 MiB (OWASP minimum for Argon2id)
  std::uint32_t iterations{2};
  std::uint32_t lanes{1};
};

// Argon2id password hashing via Monocypher.
class PasswordHasher {
public:
  static constexpr std::size_t kSaltSize = 16;
  static constexpr std::size_t kHashSize = 32;

  struct Hashed {
    std::string encoded;              // "$argon2id$v=19$m=19456,t=2,p=1$<base64 hash>"
    std::vector<std::uint8_t> salt;   // kSaltSize random bytes
  };

  explicit PasswordHasher(Argon2Params params = {});

  const Argon2Params& params() const { return params_; }

  // Hashes with a fresh random salt and this hasher's parameters.
  Hashed hash(std::string_view password) const;

  // Recomputes with the parameters recorded in `encoded` and compares in constant time.
  // Returns false (never throws) for malformed or out-of-range stored values.
  bool verify(std::string_view password, const std::vector<std::uint8_t>& salt, std::string_view encoded) const;

  // Spends the same work as hash() without a result. Used for unknown or locked
  // accounts so response time does not reveal whether a login id exists.
  void burn(std::string_view password) const;

private:
  Argon2Params params_;
};

} // namespace safelog::auth
