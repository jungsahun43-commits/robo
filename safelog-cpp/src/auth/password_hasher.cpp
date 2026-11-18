#include "safelog/auth/password_hasher.hpp"

#include "secure_random.hpp"

#include <monocypher.h>

#include <array>
#include <charconv>
#include <memory>
#include <optional>
#include <stdexcept>

namespace safelog::auth {
namespace {

constexpr char kPrefix[] = "$argon2id$v=19$";
constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Upper bounds for parameters read back from the database, so a corrupted or
// hostile row cannot make verify() allocate gigabytes or spin for minutes.
constexpr std::uint32_t kMaxMemoryKiB = 1u << 20;  // 1 GiB
constexpr std::uint32_t kMaxIterations = 64;
constexpr std::uint32_t kMaxLanes = 16;

bool validParams(const Argon2Params& p) {
  return p.lanes >= 1 && p.lanes <= kMaxLanes && p.iterations >= 1 && p.iterations <= kMaxIterations &&
         p.memoryKiB >= 8 * p.lanes && p.memoryKiB <= kMaxMemoryKiB;
}

using Digest = std::array<std::uint8_t, PasswordHasher::kHashSize>;

// Runs Argon2id; work area and inputs stay inside this function.
void argon2id(Digest& out, std::string_view password, const std::uint8_t* salt, std::size_t saltSize,
              const Argon2Params& p) {
  if (password.size() > UINT32_MAX || saltSize > UINT32_MAX) throw std::invalid_argument("Argon2 input too large");
  // One Argon2 block is 1024 bytes = 128 uint64_t; uint64_t keeps the work area aligned.
  const std::size_t words = static_cast<std::size_t>(p.memoryKiB) * 128;
  std::unique_ptr<std::uint64_t[]> work(new std::uint64_t[words]);
  crypto_argon2_config config{CRYPTO_ARGON2_ID, p.memoryKiB, p.iterations, p.lanes};
  crypto_argon2_inputs inputs{reinterpret_cast<const std::uint8_t*>(password.data()), salt,
                              static_cast<std::uint32_t>(password.size()), static_cast<std::uint32_t>(saltSize)};
  crypto_argon2(out.data(), static_cast<std::uint32_t>(out.size()), work.get(), config, inputs,
                crypto_argon2_no_extras);
  crypto_wipe(work.get(), words * sizeof(std::uint64_t));
}

std::string base64(const std::uint8_t* data, std::size_t size) {  // unpadded, as in the PHC string format
  std::string out;
  std::uint32_t buffer = 0;
  int bits = 0;
  for (std::size_t i = 0; i < size; ++i) {
    buffer = (buffer << 8) | data[i];
    bits += 8;
    while (bits >= 6) {
      bits -= 6;
      out += kAlphabet[(buffer >> bits) & 0x3F];
    }
  }
  if (bits > 0) out += kAlphabet[(buffer << (6 - bits)) & 0x3F];
  return out;
}

std::optional<Digest> decodeDigest(std::string_view text) {
  Digest out{};
  std::size_t n = 0;
  std::uint32_t buffer = 0;
  int bits = 0;
  for (const char c : text) {
    const char* pos = std::char_traits<char>::find(kAlphabet, 64, c);
    if (!pos) return std::nullopt;
    buffer = (buffer << 6) | static_cast<std::uint32_t>(pos - kAlphabet);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      if (n == out.size()) return std::nullopt;
      out[n++] = static_cast<std::uint8_t>((buffer >> bits) & 0xFF);
    }
  }
  if (n != out.size()) return std::nullopt;
  return out;
}

bool readNumber(std::string_view& text, std::string_view key, std::uint32_t& value) {
  if (text.substr(0, key.size()) != key) return false;
  text.remove_prefix(key.size());
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end == text.data()) return false;
  text.remove_prefix(static_cast<std::size_t>(end - text.data()));
  return true;
}

// Parses "$argon2id$v=19$m=<m>,t=<t>,p=<p>$<base64>".
bool parse(std::string_view encoded, Argon2Params& params, Digest& digest) {
  if (encoded.substr(0, sizeof kPrefix - 1) != kPrefix) return false;
  encoded.remove_prefix(sizeof kPrefix - 1);
  if (!readNumber(encoded, "m=", params.memoryKiB) || !readNumber(encoded, ",t=", params.iterations) ||
      !readNumber(encoded, ",p=", params.lanes) || encoded.empty() || encoded.front() != '$')
    return false;
  encoded.remove_prefix(1);
  const auto decoded = decodeDigest(encoded);
  if (!decoded) return false;
  digest = *decoded;
  return validParams(params);
}

} // namespace

PasswordHasher::PasswordHasher(Argon2Params params) : params_(params) {
  if (!validParams(params_)) throw std::invalid_argument("Invalid Argon2 parameters");
}

PasswordHasher::Hashed PasswordHasher::hash(std::string_view password) const {
  Hashed result;
  result.salt.resize(kSaltSize);
  detail::secureRandom(result.salt.data(), result.salt.size());
  Digest digest{};
  argon2id(digest, password, result.salt.data(), result.salt.size(), params_);
  result.encoded = std::string(kPrefix) + "m=" + std::to_string(params_.memoryKiB) +
                   ",t=" + std::to_string(params_.iterations) + ",p=" + std::to_string(params_.lanes) + "$" +
                   base64(digest.data(), digest.size());
  crypto_wipe(digest.data(), digest.size());
  return result;
}

bool PasswordHasher::verify(std::string_view password, const std::vector<std::uint8_t>& salt,
                            std::string_view encoded) const {
  Argon2Params stored;
  Digest expected{};
  if (salt.size() != kSaltSize || !parse(encoded, stored, expected)) {
    burn(password);  // keep timing uniform even for corrupted rows
    return false;
  }
  Digest actual{};
  argon2id(actual, password, salt.data(), salt.size(), stored);
  const bool same = crypto_verify32(actual.data(), expected.data()) == 0;  // constant time
  crypto_wipe(actual.data(), actual.size());
  return same;
}

void PasswordHasher::burn(std::string_view password) const {
  const std::array<std::uint8_t, kSaltSize> salt{};
  Digest ignored{};
  argon2id(ignored, password, salt.data(), salt.size(), params_);
  crypto_wipe(ignored.data(), ignored.size());
}

} // namespace safelog::auth
