import type { InspectionTask } from './inspection';

export type Role = 'Inspector' | 'Manager' | 'Assignee';
export type Status = 'Open' | 'InProgress' | 'PendingReview' | 'Verified';
export type Decision = 'Pending' | 'Accepted' | 'Edited' | 'Rejected';
export type { InspectionTask } from './inspection';
export type AiMode = 'mock' | 'fixed-cpu' | 'manual';

export interface Profile { id: string; name: string; role: Role }
export interface Photo { uri: string; mimeType: string; width?: number; height?: number; fileSize?: number }
export interface Draft {
  photo: Photo; location: string; description: string; action: string;
  aiMode?: AiMode;
  inspectionTask?: InspectionTask;
}
export interface Suggestion {
  model: string;
  prompt: string;
  confidence?: number;
  assessment: string;
  action: string;
  category?: string;
  risk?: number;
  source?: AiMode;
  inspectionTask?: InspectionTask;
  candidateDetected?: boolean;
  modelSha256?: string;
  inputSha256?: string;
  rawJson?: string;
  anomalyScore?: number;
  threshold?: number;
  likelyResolved?: boolean;
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
