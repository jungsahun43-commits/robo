import type { Profile } from '@/contracts/types';

/** One login per profile. The password itself is never stored. */
export interface CredentialRecord {
  profileId: string;
  /** Normalized (trimmed, lower case). */
  loginId: string;
  /** "$scrypt$ln=..,r=..,p=..$<base64 key>" */
  passwordHash: string;
  /** 16 random bytes */
  salt: Uint8Array;
  mustChange: boolean;
  failedCount: number;
  lockedUntil?: string;
  createdAt: string;
  updatedAt: string;
}

/** Duplicate login id / profile, or a credential for a missing profile. */
export class AccountConflictError extends Error {
  constructor(message = '이미 사용 중인 아이디이거나 계정을 저장할 수 없습니다.') {
    super(message);
    this.name = 'AccountConflictError';
  }
}

/**
 * Storage for profiles and credentials used by AuthService.
 * Implemented by SqliteAccountStore (app) and MemoryAccountStore (web, tests).
 */
export interface ProfileDirectory {
  findProfile(id: string): Profile | undefined;
  listProfiles(): readonly Profile[];
}

export interface AccountStore extends ProfileDirectory {
  /** Case-insensitive. */
  findCredentialByLoginId(loginId: string): CredentialRecord | undefined;
  findCredentialByProfileId(profileId: string): CredentialRecord | undefined;
  /** Inserts a new profile and its credential atomically (both or neither). */
  createAccount(profile: Profile, credential: CredentialRecord): void;
  /** Adds a credential to an existing profile that has none. */
  addCredential(credential: CredentialRecord): void;
  /** Replaces the credential of credential.profileId. */
  updateCredential(credential: CredentialRecord): void;
}

/**
 * Wipes demo data in one step: every finding with its photos, logs, AI analyses
 * and inspections, and every person not in `keep` with their login. The people
 * in `keep` are restored and get exactly `credentials`. Afterwards the findings
 * snapshot is empty (subscribers notified) and managed photo files are deleted.
 */
export interface DemoDataReset {
  resetDemoData(keep: readonly Profile[], credentials: readonly CredentialRecord[]): void;
}

const copy = (record: CredentialRecord): CredentialRecord => ({ ...record, salt: Uint8Array.from(record.salt) });

/** In-memory AccountStore with the same rules as the SQLite one. */
export class MemoryAccountStore implements AccountStore {
  private readonly profiles = new Map<string, Profile>();
  private readonly credentials = new Map<string, CredentialRecord>();

  findProfile = (id: string) => {
    const profile = this.profiles.get(id);
    return profile && { ...profile };
  };

  listProfiles = (): readonly Profile[] => [...this.profiles.values()].map(profile => ({ ...profile }));

  findCredentialByLoginId = (loginId: string) => {
    const wanted = loginId.toLowerCase();
    const found = [...this.credentials.values()].find(record => record.loginId.toLowerCase() === wanted);
    return found && copy(found);
  };

  findCredentialByProfileId = (profileId: string) => {
    const found = this.credentials.get(profileId);
    return found && copy(found);
  };

  private checkInsertable(credential: CredentialRecord) {
    if (this.credentials.has(credential.profileId) || this.findCredentialByLoginId(credential.loginId)) {
      throw new AccountConflictError();
    }
  }

  createAccount(profile: Profile, credential: CredentialRecord) {
    if (profile.id !== credential.profileId || this.profiles.has(profile.id)) throw new AccountConflictError();
    this.checkInsertable(credential);
    this.profiles.set(profile.id, { ...profile });
    this.credentials.set(profile.id, copy(credential));
  }

  addCredential(credential: CredentialRecord) {
    if (!this.profiles.has(credential.profileId)) throw new AccountConflictError();
    this.checkInsertable(credential);
    this.credentials.set(credential.profileId, copy(credential));
  }

  updateCredential(credential: CredentialRecord) {
    if (!this.credentials.has(credential.profileId)) throw new AccountConflictError('계정을 찾을 수 없습니다.');
    this.credentials.set(credential.profileId, copy(credential));
  }

  /** Adds a profile without a login (as the SQLite profiles table can hold). */
  addProfile(profile: Profile) {
    this.profiles.set(profile.id, { ...profile });
  }

  /** Keeps only `keep` (restored as given) with exactly `credentials`. Used by the demo reset. */
  replaceAccounts(keep: readonly Profile[], credentials: readonly CredentialRecord[]) {
    this.profiles.clear();
    this.credentials.clear();
    keep.forEach(profile => this.profiles.set(profile.id, { ...profile }));
    credentials.forEach(credential => this.credentials.set(credential.profileId, copy(credential)));
  }
}
