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

/** Drops keys whose value is undefined, so restored objects have no `key: undefined` entries. */
export function compact<T extends object>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== undefined)) as T;
}
