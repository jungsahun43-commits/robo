import { scryptAsync } from '@noble/hashes/scrypt.js';

/** scrypt cost: N = 2^logN, block size r, parallelism p. */
export interface ScryptParams { logN: number; r: number; p: number }

/**
 * Default cost: N = 2^11, r = 8 (2 MiB).
 *
 * Measured on an iPhone in Expo Go (pure-JS scrypt, measureScrypt()):
 * logN 12 = 1823 ms, 13 = 3567 ms, 14 = 7514 ms, 15 = 13574 ms. Time doubles
 * per step, so logN 11 is about 0.9 s, the largest cost under the 1 s target.
 *
 * This is far weaker than the OWASP minimum for scrypt (N = 2^17, r = 8) and is
 * only acceptable for the local demo. A real service should hash with a native
 * crypto module (or on a server) and raise the cost.
 *
 * Stored hashes record their own parameters: older hashes still verify, and
 * AuthService re-hashes them with this default on the next successful login.
 */
export const DEFAULT_SCRYPT_PARAMS: ScryptParams = { logN: 11, r: 8, p: 1 };

export type RandomSource = (size: number) => Uint8Array;

export const SALT_SIZE = 16;
const KEY_SIZE = 32;
const PREFIX = '$scrypt$';
const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

/** Bounds for parameters read back from storage, so a corrupted row cannot request gigabytes. */
function validParams({ logN, r, p }: ScryptParams) {
  const int = (value: number, min: number, max: number) => Number.isInteger(value) && value >= min && value <= max;
  return int(logN, 1, 20) && int(r, 1, 32) && int(p, 1, 16) && 128 * r * 2 ** logN <= 2 ** 28;
}

export function utf8ByteLength(text: string): number {
  let bytes = 0;
  for (const char of text) {
    const code = char.codePointAt(0)!;
    bytes += code < 0x80 ? 1 : code < 0x800 ? 2 : code < 0x10000 ? 3 : 4;
  }
  return bytes;
}

function base64(bytes: Uint8Array): string {
  let out = '';
  let buffer = 0;
  let bits = 0;
  for (const byte of bytes) {
    buffer = (buffer << 8) | byte;
    bits += 8;
    while (bits >= 6) {
      bits -= 6;
      out += ALPHABET[(buffer >> bits) & 63];
    }
  }
  if (bits > 0) out += ALPHABET[(buffer << (6 - bits)) & 63];
  return out;
}

function fromBase64(text: string): Uint8Array | undefined {
  const out: number[] = [];
  let buffer = 0;
  let bits = 0;
  for (const char of text) {
    const value = ALPHABET.indexOf(char);
    if (value < 0) return undefined;
    buffer = ((buffer << 6) | value) & 0xffffff;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push((buffer >> bits) & 0xff);
    }
  }
  return Uint8Array.from(out);
}

/** Compares without an early exit, so timing does not reveal the first differing byte. */
export function constantTimeEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  let difference = 0;
  for (let i = 0; i < a.length; i++) difference |= a[i] ^ b[i];
  return difference === 0;
}

function parse(encoded: string): { params: ScryptParams; key: Uint8Array } | undefined {
  const match = /^\$scrypt\$ln=(\d+),r=(\d+),p=(\d+)\$([A-Za-z0-9+/]+)$/.exec(encoded);
  if (!match) return undefined;
  const params = { logN: Number(match[1]), r: Number(match[2]), p: Number(match[3]) };
  const key = fromBase64(match[4]);
  return validParams(params) && key?.length === KEY_SIZE ? { params, key } : undefined;
}

/** scrypt password hashing (@noble/hashes, async so the UI thread keeps responding). */
export class PasswordHasher {
  constructor(private readonly random: RandomSource, readonly params: ScryptParams = DEFAULT_SCRYPT_PARAMS) {
    if (!validParams(params)) throw new Error('Invalid scrypt parameters');
  }

  private derive(password: string, salt: Uint8Array, { logN, r, p }: ScryptParams) {
    return scryptAsync(password, salt, { N: 2 ** logN, r, p, dkLen: KEY_SIZE, asyncTick: 10 });
  }

  /** Hashes with a new random salt. */
  async hash(password: string): Promise<{ encoded: string; salt: Uint8Array }> {
    const salt = this.random(SALT_SIZE);
    if (salt.length !== SALT_SIZE) throw new Error('Random source returned the wrong salt size');
    const key = await this.derive(password, salt, this.params);
    const { logN, r, p } = this.params;
    return { encoded: `${PREFIX}ln=${logN},r=${r},p=${p}$${base64(key)}`, salt };
  }

  /** Recomputes with the stored parameters; false (never throws) for malformed stored values. */
  async verify(password: string, salt: Uint8Array, encoded: string): Promise<boolean> {
    const stored = parse(encoded);
    if (!stored || salt.length !== SALT_SIZE) {
      await this.burn(password);
      return false;
    }
    const key = await this.derive(password, salt, stored.params);
    return constantTimeEqual(key, stored.key);
  }

  /** True when `encoded` was made with other parameters than this hasher's (or cannot be read). */
  needsRehash(encoded: string): boolean {
    const stored = parse(encoded);
    if (!stored) return true;
    const { logN, r, p } = stored.params;
    return logN !== this.params.logN || r !== this.params.r || p !== this.params.p;
  }

  /** Same work as hash() without a result: keeps unknown-id and locked failures as slow as wrong passwords. */
  async burn(password: string): Promise<void> {
    await this.derive(password, new Uint8Array(SALT_SIZE), this.params);
  }
}
