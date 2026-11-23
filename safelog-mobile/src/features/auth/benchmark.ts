import { PasswordHasher, type RandomSource } from './password';

export interface ScryptTiming { logN: number; memoryMiB: number; averageMs: number; runs: number[] }

/**
 * Measures how long one password hash takes on this device for each scrypt
 * cost (N = 2^logN, r = 8, p = 1). Use it to pick DEFAULT_SCRYPT_PARAMS:
 * the largest logN whose averageMs stays under ~1000 ms.
 */
export async function measureScrypt(
  random: RandomSource,
  logNs: readonly number[] = [12, 13, 14, 15],
  rounds = 3,
): Promise<ScryptTiming[]> {
  const results: ScryptTiming[] = [];
  for (const logN of logNs) {
    const hasher = new PasswordHasher(random, { logN, r: 8, p: 1 });
    const runs: number[] = [];
    for (let i = 0; i < rounds; i++) {
      const start = Date.now();
      await hasher.hash('benchmark-password');
      runs.push(Date.now() - start);
    }
    results.push({
      logN,
      memoryMiB: (128 * 8 * 2 ** logN) / 2 ** 20,
      averageMs: Math.round(runs.reduce((sum, ms) => sum + ms, 0) / runs.length),
      runs,
    });
  }
  return results;
}
