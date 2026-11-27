/**
 * Minimal synchronous SQLite port. The app uses expo-sqlite (expo-database.ts);
 * Node tests use node:sqlite (tests/support/node-database.mjs) with the same SQL.
 * Only positional `?` parameters are used so both drivers bind identically.
 */
export type SqlValue = string | number | null | Uint8Array;

export interface SqlDatabase {
  /** Runs one or more statements without parameters (DDL, PRAGMA). */
  exec(sql: string): void;
  run(sql: string, params?: readonly SqlValue[]): { changes: number };
  get<T extends object>(sql: string, params?: readonly SqlValue[]): T | undefined;
  all<T extends object>(sql: string, params?: readonly SqlValue[]): T[];
  /** Runs work in one transaction; rolls back and rethrows if it throws. Do not nest. */
  transaction<T>(work: () => T): T;
  close(): void;
}

/** True for FOREIGN KEY / UNIQUE / CHECK / NOT NULL violations from either driver. */
export function isConstraintError(error: unknown): boolean {
  return error instanceof Error && /constraint failed/i.test(error.message);
}
