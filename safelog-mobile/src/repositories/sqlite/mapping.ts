import type { Analysis, Decision, Role, Status } from '@/contracts/types';

/** Two-way map between the app's PascalCase enums and the schema's snake_case values. */
function enumMap<App extends string, Db extends string>(pairs: readonly (readonly [App, Db])[], name: string) {
  const toDb = new Map<App, Db>(pairs);
  const toApp = new Map<string, App>(pairs.map(([app, db]) => [db, app]));
  return {
    toDb(value: App): Db {
      const mapped = toDb.get(value);
      if (!mapped) throw new Error(`Unknown ${name}: ${value}`);
      return mapped;
    },
    toApp(value: string): App {
      const mapped = toApp.get(value);
      if (!mapped) throw new Error(`Unknown ${name} in database: ${value}`);
      return mapped;
    },
  };
}

export const statusMap = enumMap<Status, string>([
  ['Open', 'open'], ['InProgress', 'in_progress'], ['PendingReview', 'pending_review'], ['Verified', 'verified'],
], 'status');

export const roleMap = enumMap<Role, string>([
  ['Inspector', 'inspector'], ['Assignee', 'assignee'], ['Manager', 'manager'],
], 'role');

export const decisionMap = enumMap<Decision, string>([
  ['Pending', 'pending'], ['Accepted', 'accepted'], ['Edited', 'edited'], ['Rejected', 'rejected'],
], 'decision');

export const analysisTypeMap = enumMap<Analysis['type'], string>([
  ['BeforeHazard', 'before_hazard'], ['AfterComparison', 'after_comparison'],
], 'analysis type');

/** Converts SQL NULL to an absent optional field. */
export const optional = <T>(value: T | null | undefined): T | undefined => (value === null ? undefined : value);

/** Converts an absent optional field to SQL NULL. */
export const nullable = <T>(value: T | undefined): T | null => (value === undefined ? null : value);

/** Preserve absent booleans separately from measured false. */
export const booleanToDb = (value: boolean | undefined): number | null => value === undefined ? null : Number(value);

export function booleanFromDb(value: number | null): boolean | undefined {
  if (value === null) return undefined;
  if (value !== 0 && value !== 1) throw new Error('Invalid boolean in database');
  return value === 1;
}

/** Drops keys whose value is undefined, so restored objects have no `key: undefined` entries. */
export function compact<T extends object>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== undefined)) as T;
}
