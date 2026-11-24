import type { Profile } from '@/contracts/types';
import type { CredentialRecord, DemoDataReset } from '@/features/auth/account-store';
import type { SqlDatabase } from './database';
import type { SqliteFindingRepository } from './finding-repository';
import { nullable, roleMap } from './mapping';

/**
 * Demo reset on SQLite. All row changes run in one transaction; photo files are
 * deleted only after it commits (a failed reset leaves rows and files intact).
 */
export class SqliteDemoReset implements DemoDataReset {
  constructor(private readonly db: SqlDatabase, private readonly findings: SqliteFindingRepository) {}

  resetDemoData(keep: readonly Profile[], credentials: readonly CredentialRecord[]) {
    const keepIds = JSON.stringify(keep.map(person => person.id));
    this.db.transaction(() => {
      // Children first; inspections and people are referenced by findings, logs and analyses.
      this.db.run('DELETE FROM ai_analyses');
      this.db.run('DELETE FROM action_logs');
      this.db.run('DELETE FROM photos');
      this.db.run('DELETE FROM findings');
      this.db.run('DELETE FROM inspections');
      this.db.run('DELETE FROM credentials WHERE profile_id NOT IN (SELECT value FROM json_each(?))', [keepIds]);
      this.db.run('DELETE FROM profiles WHERE id NOT IN (SELECT value FROM json_each(?))', [keepIds]);
      for (const person of keep) {
        this.db.run(
          `INSERT INTO profiles (id, display_name, role) VALUES (?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET display_name = excluded.display_name, role = excluded.role`,
          [person.id, person.name, roleMap.toDb(person.role)]);
      }
      for (const c of credentials) {
        this.db.run(
          `INSERT INTO credentials (profile_id, login_id, password_hash, salt, must_change, failed_count, locked_until,
             created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(profile_id) DO UPDATE SET login_id = excluded.login_id, password_hash = excluded.password_hash,
             salt = excluded.salt, must_change = excluded.must_change, failed_count = excluded.failed_count,
             locked_until = excluded.locked_until, created_at = excluded.created_at, updated_at = excluded.updated_at`,
          [c.profileId, c.loginId, c.passwordHash, c.salt, c.mustChange ? 1 : 0, c.failedCount,
            nullable(c.lockedUntil), c.createdAt, c.updatedAt]);
      }
    });
    this.findings.reload();             // snapshot becomes empty; screens update
    this.findings.removeOrphanPhotos(); // nothing is referenced any more, so every photo file goes
  }
}
