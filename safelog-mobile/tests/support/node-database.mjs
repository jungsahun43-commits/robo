// node:sqlite adapter implementing the SqlDatabase port from
// src/repositories/sqlite/database.ts, so tests run the app's SQL unchanged.
// JavaScript (not TypeScript) because the project has no @types/node.
import { DatabaseSync } from 'node:sqlite';

/** @returns {import('../../src/repositories/sqlite/database').SqlDatabase} */
export function openNodeDatabase(path) {
  const db = new DatabaseSync(path);
  db.exec('PRAGMA journal_mode = WAL');
  const plain = row => (row ? { ...row } : undefined);
  return {
    exec: sql => db.exec(sql),
    run: (sql, params = []) => ({ changes: Number(db.prepare(sql).run(...params).changes) }),
    get: (sql, params = []) => plain(db.prepare(sql).get(...params)),
    all: (sql, params = []) => db.prepare(sql).all(...params).map(plain),
    transaction(work) {
      db.exec('BEGIN IMMEDIATE');
      try {
        const result = work();
        db.exec('COMMIT');
        return result;
      } catch (error) {
        db.exec('ROLLBACK');
        throw error;
      }
    },
    close: () => db.close(),
  };
}
