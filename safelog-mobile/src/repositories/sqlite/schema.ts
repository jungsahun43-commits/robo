import type { SqlDatabase } from './database';

/**
 * Mobile schema, derived from safelog-cpp/database/schema.sql (v2) with the
 * columns the mobile model needs (photo ordinal and MIME type, log sequence,
 * AI assessment/action/submission count). Enum values are snake_case.
 *
 * MIGRATIONS[i] upgrades user_version i to i + 1. Append only; never edit a
 * released migration.
 */
const MIGRATIONS: readonly string[] = [
  // v1: findings and their history
  `
  CREATE TABLE sites (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT NOT NULL
  );
  CREATE TABLE profiles (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('inspector', 'assignee', 'manager'))
  );
  CREATE TABLE inspections (
    id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL REFERENCES sites(id),
    inspector_id TEXT NOT NULL REFERENCES profiles(id),
    inspected_at TEXT NOT NULL
  );
  CREATE TABLE findings (
    id TEXT PRIMARY KEY,
    inspection_id TEXT NOT NULL REFERENCES inspections(id),
    location TEXT NOT NULL,
    description TEXT NOT NULL,
    action TEXT NOT NULL,
    assignee_id TEXT REFERENCES profiles(id),
    due_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('open', 'in_progress', 'pending_review', 'verified')),
    created_at TEXT NOT NULL,
    category TEXT,
    risk INTEGER CHECK (risk IS NULL OR risk BETWEEN 1 AND 5)
  );
  CREATE TABLE photos (
    id TEXT PRIMARY KEY,
    finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('before', 'after')),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    storage_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    UNIQUE (finding_id, kind, ordinal)
  );
  CREATE TABLE action_logs (
    id TEXT PRIMARY KEY,
    finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL CHECK (seq >= 0),
    actor_id TEXT NOT NULL REFERENCES profiles(id),
    action TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE (finding_id, seq)
  );
  CREATE TABLE ai_analyses (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL CHECK (seq >= 0),
    analysis_type TEXT NOT NULL CHECK (analysis_type IN ('before_hazard', 'after_comparison')),
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    risk_level INTEGER CHECK (risk_level IS NULL OR risk_level BETWEEN 1 AND 5),
    category TEXT,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    assessment TEXT NOT NULL,
    action TEXT NOT NULL,
    result_json TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('pending', 'accepted', 'edited', 'rejected')),
    reviewed_by TEXT REFERENCES profiles(id),
    submission_count INTEGER,
    created_at TEXT NOT NULL
  );
  CREATE INDEX idx_findings_inspection ON findings(inspection_id);
  CREATE INDEX idx_findings_assignee_status ON findings(assignee_id, status);
  CREATE INDEX idx_photos_finding ON photos(finding_id, kind, ordinal);
  CREATE INDEX idx_logs_finding ON action_logs(finding_id, seq);
  CREATE INDEX idx_ai_subject ON ai_analyses(subject_id, seq);
  `,
  // v2: local login credentials (same shape as safelog-cpp feature/auth)
  `
  CREATE TABLE credentials (
    profile_id TEXT PRIMARY KEY REFERENCES profiles(id) ON DELETE CASCADE,
    login_id TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    salt BLOB NOT NULL CHECK (length(salt) = 16),
    must_change INTEGER NOT NULL CHECK (must_change IN (0, 1)),
    failed_count INTEGER NOT NULL DEFAULT 0 CHECK (failed_count >= 0),
    locked_until TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );
  `,
];

export const SCHEMA_VERSION = MIGRATIONS.length;

/** Exposed for tests that build an older database. */
export const migrationsUpTo = (version: number) => MIGRATIONS.slice(0, version);

export function userVersion(db: SqlDatabase): number {
  return db.get<{ user_version: number }>('PRAGMA user_version')?.user_version ?? 0;
}

/**
 * Enables foreign keys (outside any transaction, where the PRAGMA has no
 * effect), then applies pending migrations, each in its own transaction.
 */
export function migrate(db: SqlDatabase): void {
  db.exec('PRAGMA foreign_keys = ON');
  if (db.get<{ foreign_keys: number }>('PRAGMA foreign_keys')?.foreign_keys !== 1) {
    throw new Error('SQLite foreign key enforcement could not be enabled');
  }
  const current = userVersion(db);
  if (current > SCHEMA_VERSION) {
    throw new Error(`Database schema v${current} is newer than this app (v${SCHEMA_VERSION})`);
  }
  for (let version = current; version < SCHEMA_VERSION; version++) {
    db.transaction(() => {
      db.exec(MIGRATIONS[version]);
      db.exec(`PRAGMA user_version = ${version + 1}`);
    });
  }
}
