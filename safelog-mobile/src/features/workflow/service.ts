import type { FindingRepository } from '@/contracts/interfaces';
import type { Analysis, Decision, Draft, Finding, Photo, Profile, Suggestion } from '@/contracts/types';
import { profiles, site } from '@/features/auth/demo';

const now = () => new Date().toISOString();
let sequence = 0;
export const nextId = (prefix: string) => prefix + '-' + Date.now().toString(36) + '-' + (++sequence);
type Command =
  | { kind: 'assign'; assigneeId: string; dueAt?: string }
  | { kind: 'start' }
  | { kind: 'submit'; note: string; photo: Photo }
  | { kind: 'verify'; note: string; confirmed: boolean; analysisId?: string; decision?: 'Accepted' | 'Rejected' }
  | { kind: 'changes'; note: string }
  | { kind: 'suggest'; type: Analysis['type']; suggestion: Suggestion }
  | { kind: 'hazard'; analysisId?: string; decision: Exclude<Decision, 'Pending'>; description: string; action: string; manual?: boolean };

export const ownsFinding = (actor: Profile, finding: Finding) =>
  actor.role === 'Inspector' && actor.id === finding.inspectorId;

export function visibleTo(actor: Profile, finding: Finding) {
  if (actor.role === 'Manager') return true;
  if (actor.role === 'Inspector') return ownsFinding(actor, finding);
  return actor.id === finding.assigneeId;
}

export class WorkflowService {
  constructor(private readonly repository: FindingRepository) {}

  create(actor: Profile, draft: Draft) {
    if (actor.role !== 'Inspector') throw new Error('점검자만 등록할 수 있습니다.');
    const location = draft.location.trim();
    const description = draft.description.trim();
    const action = draft.action.trim();
    if (!draft.photo.uri || !location || !description || !action) {
      throw new Error('사진과 장소, 발견 메모, 권장 조치를 모두 입력하세요.');
    }
    const finding: Finding = {
      ...draft, location, description, action,
      id: nextId('finding'), inspectionId: nextId('inspection'), inspectorId: actor.id,
      siteId: site.id, createdAt: now(), status: 'Open', after: [], analyses: [],
      logs: [{ actor: actor.id, action: 'finding_created', note: description, at: now() }],
    };
    this.repository.save(finding);
    return finding;
  }

  execute(actor: Profile, id: string, command: Command) {
    const finding = this.repository.snapshot().find(item => item.id === id);
    if (!finding || !visibleTo(actor, finding)) throw new Error('기록 접근 권한이 없습니다.');
    const assigned = actor.role === 'Assignee' && actor.id === finding.assigneeId;
    const owner = ownsFinding(actor, finding);
    const update = (patch: Partial<Finding>, action: string, note = '') => {
      const updated = {
        ...finding, ...patch,
        logs: [...finding.logs, { actor: actor.id, action, note, at: now() }],
      };
      this.repository.save(updated);
      return updated;
    };
    const require = (valid: boolean) => {
      if (!valid) throw new Error('현재 역할 또는 작업 상태에서 수행할 수 없습니다.');
    };
    const reviewed = (analysisId: string | undefined, type: Analysis['type'], decision: Decision) => {
      if (!analysisId) return finding.analyses;
      const analysis = finding.analyses.find(item => item.id === analysisId && item.type === type && item.decision === 'Pending');
      if (!analysis || (type === 'AfterComparison' && analysis.submissionCount !== finding.after.length)) {
        throw new Error('현재 제출에 대한 AI 제안을 찾을 수 없습니다.');
      }
      return finding.analyses.map(item => item.id === analysisId ? { ...item, decision, reviewer: actor.id } : item);
    };

    switch (command.kind) {
      case 'assign': {
        require(actor.role === 'Manager' && finding.status === 'Open');
        require(profiles.some(person => person.id === command.assigneeId && person.role === 'Assignee'));
        return update({ assigneeId: command.assigneeId, dueAt: command.dueAt }, 'assigned', command.assigneeId);
      }
      case 'start':
        require(assigned && finding.status === 'Open');
        return update({ status: 'InProgress' }, 'work_started');
      case 'submit':
        require(assigned && finding.status === 'InProgress');
        if (!command.photo.uri || !command.note.trim()) throw new Error('조치 내용과 조치 후 사진이 필요합니다.');
        return update({ status: 'PendingReview', after: [...finding.after, command.photo] }, 'action_submitted', command.note.trim());
      case 'changes':
        require(owner && finding.status === 'PendingReview');
        if (!command.note.trim()) throw new Error('추가 조치 요청 사유를 입력하세요.');
        return update({ status: 'InProgress' }, 'changes_requested', command.note.trim());
      case 'verify': {
        require(owner && finding.status === 'PendingReview');
        if (!command.confirmed || !finding.photo.uri || !finding.after.at(-1)?.uri) {
          throw new Error('전후 사진과 조치 내용을 직접 확인하세요.');
        }
        const note = command.note.trim();
        if (!note) throw new Error('사람의 최종 판단 내용을 입력하세요.');
        const decision = command.analysisId ? command.decision ?? 'Rejected' : 'Manual';
        const analyses = reviewed(command.analysisId, 'AfterComparison', decision === 'Manual' ? 'Rejected' : decision);
        return update({ status: 'Verified', analyses }, 'verified', '사람 최종 판단 [' + decision + ']: ' + note);
      }
      case 'suggest': {
        require(command.type === 'BeforeHazard'
          ? owner && finding.status === 'Open'
          : finding.status === 'PendingReview' && finding.after.length > 0);
        const analysis: Analysis = {
          ...command.suggestion, id: nextId('analysis'), type: command.type, at: now(), decision: 'Pending',
          submissionCount: command.type === 'AfterComparison' ? finding.after.length : undefined,
        };
        this.repository.save({ ...finding, analyses: [...finding.analyses, analysis] });
        return analysis;
      }
      case 'hazard': {
        require(owner && finding.status === 'Open');
        const description = command.description.trim();
        const action = command.action.trim();
        if (!description || !action) throw new Error('위험 설명과 권장 조치를 입력하세요.');
        const analyses = reviewed(command.analysisId, 'BeforeHazard', command.decision);
        const accepted = command.decision !== 'Rejected' && analyses.find(item => item.id === command.analysisId);
        return update({
          description, action, analyses,
          category: accepted ? accepted.category : finding.category,
          risk: accepted ? accepted.risk : finding.risk,
        }, 'hazard_reviewed', command.manual === true ? 'Manual' : command.decision);
      }
    }
  }
}
