import type { Role } from '@/contracts/types';

export const roleLabels = {
  Inspector: '점검자',
  Manager: '관리자',
  Assignee: '조치 담당자',
} satisfies Record<Role, string>;
