export type Role = 'Inspector' | 'Manager' | 'Assignee';
export type Status = 'Open' | 'InProgress' | 'PendingReview' | 'Verified';
export type Decision = 'Pending' | 'Accepted' | 'Edited' | 'Rejected';

export interface Profile { id: string; name: string; role: Role }
export interface Photo { uri: string; mimeType: string }
export interface Draft { photo: Photo; location: string; description: string; action: string }
export interface Suggestion {
  model: string;
  prompt: string;
  confidence: number;
  assessment: string;
  action: string;
  category?: string;
  risk?: number;
}
export interface Analysis extends Suggestion {
  id: string;
  type: 'BeforeHazard' | 'AfterComparison';
  at: string;
  decision: Decision;
  reviewer?: string;
  submissionCount?: number;
}
export interface ActionLog { actor: string; action: string; note: string; at: string }
export interface Finding extends Draft {
  id: string;
  inspectionId: string;
  inspectorId: string;
  siteId: string;
  createdAt: string;
  status: Status;
  assigneeId?: string;
  dueAt?: string;
  after: Photo[];
  logs: ActionLog[];
  analyses: Analysis[];
  category?: string;
  risk?: number;
}
