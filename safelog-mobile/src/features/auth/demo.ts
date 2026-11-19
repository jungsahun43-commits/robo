import type { Profile } from '@/contracts/types';

export const site = { id: 'site-1', name: '세이프 금속 가공공장', address: '서울시 가상구 산업로 10' };
export const profiles: Profile[] = [
  { id: 'inspector-1', name: '김안전', role: 'Inspector' },
  { id: 'manager-1', name: '박관리', role: 'Manager' },
  { id: 'assignee-1', name: '이조치', role: 'Assignee' },
];

export function demoLogin(id: string, password: string) {
  const login = id.trim();
  const profile = profiles.find(person => person.id === login || person.id.replace('-', '') === login);
  if (password !== 'demo1234' || !profile) throw new Error('사용자 ID 또는 비밀번호를 확인하세요.');
  return profile;
}
