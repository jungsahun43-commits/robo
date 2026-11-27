import type { Profile } from '@/contracts/types';

export const site = { id: 'site-1', name: '세이프 금속 가공공장', address: '서울시 가상구 산업로 10' };
export const profiles: Profile[] = [
  { id: 'inspector-1', name: '김안전', role: 'Inspector' },
  { id: 'manager-1', name: '박관리', role: 'Manager' },
  { id: 'assignee-1', name: '이조치', role: 'Assignee' },
];

/** When true, the three demo people above get logins (login id = profile id) at start-up. */
export const DEMO_MODE = true;
/** Public demo password, shown on the login screen. Only its scrypt hash is stored in the database. */
export const DEMO_PASSWORD = 'demo1234';
/** Hyphen-less login names also accepted for the demo accounts, e.g. inspector1 -> inspector-1. */
export const demoAliases: Readonly<Record<string, string>> = Object.fromEntries(
  profiles.map(person => [person.id.replace('-', ''), person.id]),
);

/**
 * Kept only for tests/workflow.test.mjs, which builds its actors with it.
 * The app signs in through AuthService (see session.tsx), not this function.
 */
export function demoLogin(id: string, password: string) {
  const login = id.trim();
  const profile = profiles.find(person => person.id === login || person.id.replace('-', '') === login);
  if (password !== DEMO_PASSWORD || !profile) throw new Error('사용자 ID 또는 비밀번호를 확인하세요.');
  return profile;
}
