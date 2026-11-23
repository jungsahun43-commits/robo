import type { Profile, Role } from '@/contracts/types';
import { AccountConflictError, type AccountStore, type CredentialRecord } from './account-store';
import { utf8ByteLength, type PasswordHasher } from './password';

/** The one message for every login failure (unknown id, wrong password, locked account). */
export const LOGIN_FAILED_MESSAGE = '사용자 ID 또는 비밀번호를 확인하세요.';

export class AuthenticationError extends Error {
  constructor() {
    super(LOGIN_FAILED_MESSAGE);
    this.name = 'AuthenticationError';
  }
}

export class PermissionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PermissionError';
  }
}

export class AccountValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AccountValidationError';
  }
}

export interface LoginResult { profile: Profile; mustChangePassword: boolean }

export interface AuthOptions {
  now?: () => Date;
  /** Profile ids for new accounts. */
  newProfileId?: () => string;
  /** Extra login names that resolve to another login id, e.g. { inspector1: 'inspector-1' }. */
  aliases?: Readonly<Record<string, string>>;
}

/**
 * Local login (no server) for Inspector / Manager / Assignee.
 * - Every failure throws AuthenticationError with LOGIN_FAILED_MESSAGE and costs one scrypt run.
 * - MAX_FAILED_ATTEMPTS consecutive failures (login or wrong current password) lock for LOCK_MS.
 * - Passwords: at least MIN_PASSWORD_CHARS characters, at most MAX_PASSWORD_BYTES bytes.
 * - Login ids are trimmed and lower-cased; 3-32 of [a-z0-9._-].
 * Operations run one at a time so failed-attempt counting cannot race.
 */
export class AuthService {
  static readonly MAX_FAILED_ATTEMPTS = 5;
  static readonly LOCK_MS = 60_000;
  static readonly MIN_PASSWORD_CHARS = 8;
  static readonly MAX_PASSWORD_BYTES = 128;

  private queue: Promise<unknown> = Promise.resolve();
  private readonly now: () => Date;
  private readonly newProfileId: () => string;
  private readonly aliases: Readonly<Record<string, string>>;

  constructor(private readonly store: AccountStore, private readonly hasher: PasswordHasher, options: AuthOptions = {}) {
    let sequence = 0;
    this.now = options.now ?? (() => new Date());
    this.newProfileId = options.newProfileId ?? (() => `profile-${Date.now().toString(36)}-${++sequence}`);
    this.aliases = options.aliases ?? {};
  }

  static normalizeLoginId(raw: string): string {
    const id = raw.trim().toLowerCase();
    return /^[a-z0-9._-]{3,32}$/.test(id) ? id : '';
  }

  static checkPasswordPolicy(password: string) {
    if (utf8ByteLength(password) > AuthService.MAX_PASSWORD_BYTES) {
      throw new AccountValidationError(`비밀번호는 ${AuthService.MAX_PASSWORD_BYTES}바이트 이하여야 합니다.`);
    }
    if ([...password].length < AuthService.MIN_PASSWORD_CHARS) {
      throw new AccountValidationError(`비밀번호는 ${AuthService.MIN_PASSWORD_CHARS}자 이상이어야 합니다.`);
    }
  }

  private serialize<T>(task: () => Promise<T>): Promise<T> {
    const run = this.queue.then(task, task);
    this.queue = run.catch(() => undefined);
    return run;
  }

  private isLocked(record: CredentialRecord) {
    return !!record.lockedUntil && Date.parse(record.lockedUntil) > this.now().getTime();
  }

  private findByLogin(loginId: string) {
    const id = AuthService.normalizeLoginId(loginId);
    if (!id) return undefined;
    return this.store.findCredentialByLoginId(id)
      ?? (this.aliases[id] ? this.store.findCredentialByLoginId(this.aliases[id]) : undefined);
  }

  private recordFailure(record: CredentialRecord) {
    const now = this.now();
    let { failedCount, lockedUntil } = record;
    if (lockedUntil && Date.parse(lockedUntil) <= now.getTime()) lockedUntil = undefined;  // expired lock
    failedCount += 1;
    if (failedCount >= AuthService.MAX_FAILED_ATTEMPTS) {
      lockedUntil = new Date(now.getTime() + AuthService.LOCK_MS).toISOString();
      failedCount = 0;
    }
    this.store.updateCredential({ ...record, failedCount, lockedUntil, updatedAt: now.toISOString() });
  }

  /** Checks the password of `record` (undefined = unknown account), counting failures. */
  private async authenticate(record: CredentialRecord | undefined, password: string): Promise<CredentialRecord> {
    if (!record || this.isLocked(record) || utf8ByteLength(password) > AuthService.MAX_PASSWORD_BYTES) {
      await this.hasher.burn(password.slice(0, AuthService.MAX_PASSWORD_BYTES));
      throw new AuthenticationError();
    }
    if (!(await this.hasher.verify(password, record.salt, record.passwordHash))) {
      this.recordFailure(record);
      throw new AuthenticationError();
    }
    return record;
  }

  private async newCredential(profileId: string, loginId: string, password: string, mustChange: boolean) {
    const { encoded, salt } = await this.hasher.hash(password);
    const at = this.now().toISOString();
    return { profileId, loginId, passwordHash: encoded, salt, mustChange, failedCount: 0, createdAt: at, updatedAt: at };
  }

  login(loginId: string, password: string): Promise<LoginResult> {
    return this.serialize(async () => {
      const record = await this.authenticate(this.findByLogin(loginId), password);
      const profile = this.store.findProfile(record.profileId);
      if (!profile) throw new AuthenticationError();
      // Hashes made with older cost settings are upgraded now, while the password is known.
      const upgrade = this.hasher.needsRehash(record.passwordHash) ? await this.hasher.hash(password) : undefined;
      if (upgrade || record.failedCount !== 0 || record.lockedUntil) {
        this.store.updateCredential({
          ...record,
          ...(upgrade && { passwordHash: upgrade.encoded, salt: upgrade.salt }),
          failedCount: 0,
          lockedUntil: undefined,
          updatedAt: this.now().toISOString(),
        });
      }
      return { profile, mustChangePassword: record.mustChange };
    });
  }

  changePassword(profileId: string, currentPassword: string, newPassword: string): Promise<void> {
    AuthService.checkPasswordPolicy(newPassword);
    return this.serialize(async () => {
      const record = await this.authenticate(this.store.findCredentialByProfileId(profileId), currentPassword);
      const { encoded, salt } = await this.hasher.hash(newPassword);
      this.store.updateCredential({
        ...record, passwordHash: encoded, salt, mustChange: false, failedCount: 0, lockedUntil: undefined,
        updatedAt: this.now().toISOString(),
      });
    });
  }

  /** Registers a new person (profile + login, must change password). Managers only. */
  createAccount(managerId: string, name: string, role: Role, loginId: string, temporaryPassword: string): Promise<Profile> {
    return this.serialize(async () => {
      if (this.store.findProfile(managerId)?.role !== 'Manager') {
        throw new PermissionError('관리자만 계정을 만들 수 있습니다.');
      }
      const displayName = name.trim();
      if (!displayName) throw new AccountValidationError('이름을 입력하세요.');
      const id = AuthService.normalizeLoginId(loginId);
      if (!id) throw new AccountValidationError('아이디는 3~32자의 영문 소문자, 숫자, ".", "_", "-"만 쓸 수 있습니다.');
      if (this.aliases[id] || this.store.findCredentialByLoginId(id)) {
        throw new AccountValidationError('이미 사용 중인 아이디입니다.');
      }
      AuthService.checkPasswordPolicy(temporaryPassword);
      const profile: Profile = { id: this.newProfileId(), name: displayName, role };
      try {
        this.store.createAccount(profile, await this.newCredential(profile.id, id, temporaryPassword, true));
      } catch (error) {
        if (error instanceof AccountConflictError) throw new AccountValidationError('이미 사용 중인 아이디입니다.');
        throw error;
      }
      return profile;
    });
  }

  /**
   * Gives each demo person a login equal to their profile id (must_change = 0).
   * Existing profiles are reused, missing ones created; people who already have
   * a login are left untouched, so this is safe on every start.
   */
  seedDemoAccounts(people: readonly Profile[], password: string): Promise<void> {
    AuthService.checkPasswordPolicy(password);
    return this.serialize(async () => {
      for (const person of people) {
        if (this.store.findCredentialByProfileId(person.id)) continue;
        const loginId = AuthService.normalizeLoginId(person.id);
        if (!loginId) throw new AccountValidationError(`시연 계정 아이디가 올바르지 않습니다: ${person.id}`);
        if (this.store.findCredentialByLoginId(loginId)) {
          throw new AccountValidationError(`시연 계정 아이디가 다른 사용자에게 연결되어 있습니다: ${loginId}`);
        }
        const credential = await this.newCredential(person.id, loginId, password, false);
        if (this.store.findProfile(person.id)) this.store.addCredential(credential);
        else this.store.createAccount(person, credential);
      }
    });
  }
}
