import type { Profile } from '@/contracts/types';
import { AccountConflictError, type AccountStore, type CredentialRecord } from '@/features/auth/account-store';
import { isConstraintError, type SqlDatabase } from './database';
import { nullable, optional, roleMap } from './mapping';

interface CredentialRow {
  profile_id: string; login_id: string; password_hash: string; salt: Uint8Array;
  must_change: number; failed_count: number; locked_until: string | null; created_at: string; updated_at: string;
}

const COLUMNS =
  'profile_id, login_id, password_hash, salt, must_change, failed_count, locked_until, created_at, updated_at';

function toRecord(row: CredentialRow): CredentialRecord {
  const record: CredentialRecord = {
    profileId: row.profile_id,
    loginId: row.login_id,
    passwordHash: row.password_hash,
    salt: Uint8Array.from(row.salt),
    mustChange: row.must_change === 1,
    failedCount: row.failed_count,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
  const lockedUntil = optional(row.locked_until);
  return lockedUntil ? { ...record, lockedUntil } : record;
}

const values = (c: CredentialRecord) => [
  c.loginId, c.passwordHash, c.salt, c.mustChange ? 1 : 0, c.failedCount, nullable(c.lockedUntil), c.createdAt,
  c.updatedAt, c.profileId,
];

/** AccountStore on the profiles and credentials tables. Constraint violations become AccountConflictError. */
export class SqliteAccountStore implements AccountStore {
  constructor(private readonly db: SqlDatabase) {}

  private guard<T>(work: () => T): T {
    try {
      return work();
    } catch (error) {
      if (isConstraintError(error)) throw new AccountConflictError();
      throw error;
    }
  }

  findProfile(id: string): Profile | undefined {
    const row = this.db.get<{ id: string; display_name: string; role: string }>(
      'SELECT id, display_name, role FROM profiles WHERE id = ?', [id]);
    return row && { id: row.id, name: row.display_name, role: roleMap.toApp(row.role) };
  }

  findCredentialByLoginId(loginId: string) {
    const row = this.db.get<CredentialRow>(`SELECT ${COLUMNS} FROM credentials WHERE login_id = ?`, [loginId]);
    return row && toRecord(row);  // login_id is COLLATE NOCASE
  }

  findCredentialByProfileId(profileId: string) {
    const row = this.db.get<CredentialRow>(`SELECT ${COLUMNS} FROM credentials WHERE profile_id = ?`, [profileId]);
    return row && toRecord(row);
  }

  private insertCredential(c: CredentialRecord) {
    this.db.run(
      `INSERT INTO credentials (login_id, password_hash, salt, must_change, failed_count, locked_until, created_at,
         updated_at, profile_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`, values(c));
  }

  createAccount(profile: Profile, credential: CredentialRecord) {
    if (profile.id !== credential.profileId) throw new AccountConflictError();
    this.guard(() => this.db.transaction(() => {
      this.db.run('INSERT INTO profiles (id, display_name, role) VALUES (?, ?, ?)',
        [profile.id, profile.name, roleMap.toDb(profile.role)]);
      this.insertCredential(credential);
    }));
  }

  addCredential(credential: CredentialRecord) {
    this.guard(() => this.insertCredential(credential));
  }

  updateCredential(credential: CredentialRecord) {
    const { changes } = this.guard(() => this.db.run(
      `UPDATE credentials SET login_id = ?, password_hash = ?, salt = ?, must_change = ?, failed_count = ?,
         locked_until = ?, created_at = ?, updated_at = ? WHERE profile_id = ?`, values(credential)));
    if (changes === 0) throw new AccountConflictError('계정을 찾을 수 없습니다.');
  }
}
