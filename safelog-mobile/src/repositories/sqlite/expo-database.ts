import { openDatabaseSync } from 'expo-sqlite';
import type { SqlDatabase, SqlValue } from './database';

/**
 * expo-sqlite adapter (synchronous API, SDK 58). Uses withTransactionSync on the
 * same connection: withExclusiveTransactionAsync opens a new connection where
 * PRAGMA foreign_keys would not apply.
 */
export function openExpoDatabase(name: string): SqlDatabase {
  const db = openDatabaseSync(name);
  db.execSync('PRAGMA journal_mode = WAL');
  return {
    exec: sql => db.execSync(sql),
    run: (sql, params = []) => ({ changes: db.runSync(sql, [...params] as SqlValue[]).changes }),
    get: <T extends object>(sql: string, params: readonly SqlValue[] = []) =>
      db.getFirstSync<T>(sql, [...params]) ?? undefined,
    all: <T extends object>(sql: string, params: readonly SqlValue[] = []) => db.getAllSync<T>(sql, [...params]),
    transaction<T>(work: () => T): T {
      let result: { value: T } | undefined;
      db.withTransactionSync(() => { result = { value: work() }; });
      return result!.value;
    },
    close: () => db.closeSync(),
  };
}
