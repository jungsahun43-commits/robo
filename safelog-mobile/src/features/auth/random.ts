import { getRandomValues, randomUUID } from 'expo-crypto';

/**
 * Cryptographically secure bytes from expo-crypto. getRandomValues is used
 * rather than getRandomBytes, whose docs say it falls back to Math.random in
 * development.
 */
export const secureRandom = (size: number) => getRandomValues(new Uint8Array(size));

export const newProfileId = () => 'profile-' + randomUUID();
