import type { FindingRepository } from '@/contracts/interfaces';
import type { Analysis, Decision, Draft, Finding, Photo, Profile, Suggestion } from '@/contracts/types';
import { profiles, site } from '@/features/auth/demo';
import { isFixedTask, isInspectionTask } from '@/contracts/inspection';
import type { ProfileDirectory } from '@/features/auth/account-store';

const now = () => new Date().toISOString();
let sequence = 0;
export const nextId = (prefix: string) => prefix + '-' + Date.now().toString(36) + '-' + (++sequence);

const sha256 = /^[a-f0-9]{64}$/i;

type Command =
  | { kind: 'assign'; assigneeId: string; dueAt?: string }
  | { kind: 'start' }
  | { kind: 'submit'; note: string; photo: Photo }
  | { kind: 'verify'; note: string; confirmed: boolean; analysisId?: string; decision?: 'Accepted' | 'Rejected' }
  | { kind: 'changes'; note: string }
  | { kind: 'suggest'; type: Analysis['type']; suggestion: Suggestion;
      expectedSubmissionCount?: number; expectedAfterUri?: string }
  | { kind: 'hazard'; analysisId?: string; decision: Exclude<Decision, 'Pending'>; description: string; action: string };

export const ownsFinding = (actor: Profile, finding: Finding) =>
  actor.role === 'Inspector' && actor.id === finding.inspectorId;

export function visibleTo(actor: Profile, finding: Finding) {
  if (actor.role === 'Manager') return true;
  if (actor.role === 'Inspector') return ownsFinding(actor, finding);
  return actor.id === finding.assigneeId;
}

function validateSuggestion(suggestion: Suggestion, type: Analysis['type'], finding: Finding) {
  if (!suggestion || typeof suggestion !== 'object' || Array.isArray(suggestion)
    || [suggestion.model, suggestion.prompt, suggestion.assessment, suggestion.action]
      .some(value => typeof value !== 'string' || !value.trim())) {
    throw new Error('AI 제안의 모델, 버전, 설명과 권장 조치를 확인하세요.');
  }
  if (suggestion.confidence !== undefined && (typeof suggestion.confidence !== 'number'
    || !Number.isFinite(suggestion.confidence) || suggestion.confidence < 0 || suggestion.confidence > 1)) {
    throw new Error('AI 참고 점수는 0에서 1 사이의 유한한 수여야 합니다.');
  }
  if (suggestion.risk !== undefined && (!Number.isInteger(suggestion.risk)
    || suggestion.risk < 1 || suggestion.risk > 5)) {
    throw new Error('위험 등급은 1에서 5 사이의 정수여야 합니다.');
  }
  if ((suggestion.category !== undefined && (typeof suggestion.category !== 'string' || !suggestion.category.trim()))
    || (suggestion.candidateDetected !== undefined && typeof suggestion.candidateDetected !== 'boolean')
    || (suggestion.likelyResolved !== undefined && typeof suggestion.likelyResolved !== 'boolean')
    || [suggestion.anomalyScore, suggestion.threshold]
      .some(value => value !== undefined && (typeof value !== 'number' || !Number.isFinite(value)))) {
    throw new Error('AI 판정 값이 올바르지 않습니다.');
  }

  const mode = finding.aiMode ?? 'mock';
  const source = suggestion.source ?? 'mock';
  if ((source !== 'mock' && source !== 'fixed-cpu') || source !== mode) {
    throw new Error('등록한 AI 모드와 다른 제안을 저장할 수 없습니다.');
  }
  if (source !== 'fixed-cpu') return;
  if (!finding.inspectionTask || suggestion.inspectionTask !== finding.inspectionTask) {
    throw new Error('등록한 점검 분야와 AI 분석 분야가 다릅니다.');
  }
  if (typeof suggestion.modelSha256 !== 'string' || !sha256.test(suggestion.modelSha256)
    || (type === 'BeforeHazard' && (typeof suggestion.inputSha256 !== 'string' || !sha256.test(suggestion.inputSha256)))
    || (suggestion.inputSha256 !== undefined
      && (typeof suggestion.inputSha256 !== 'string' || !sha256.test(suggestion.inputSha256)))) {
    throw new Error('AI 모델과 입력 사진의 식별값이 필요합니다.');
  }
  try {
    if (typeof suggestion.rawJson !== 'string' || !suggestion.rawJson.trim()) throw new Error();
    const raw: unknown = JSON.parse(suggestion.rawJson);
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error();
  } catch {
    throw new Error('AI 원응답을 올바른 JSON 객체로 보관해야 합니다.');
  }
}

export class WorkflowService {
  constructor(private readonly repository: FindingRepository, private readonly people: ProfileDirectory = {
    findProfile: id => profiles.find(profile => profile.id === id),
    listProfiles: () => profiles,
  }) {}

  private currentActor(actor: Profile): Profile {
    const current = this.people.findProfile(actor.id);
    if (!current || current.role !== actor.role) throw new Error('현재 계정과 역할을 확인하세요.');
    return current;
  }

  create(actor: Profile, draft: Draft, initialSuggestion?: Suggestion) {
    actor = this.currentActor(actor);
    if (actor.role !== 'Inspector') throw new Error('점검자만 등록할 수 있습니다.');
    const aiMode = draft.aiMode ?? 'mock';
    if ((aiMode !== 'mock' && aiMode !== 'fixed-cpu' && aiMode !== 'manual')
      || (draft.inspectionTask !== undefined && !isInspectionTask(draft.inspectionTask))
      || (aiMode === 'fixed-cpu' && !isFixedTask(draft.inspectionTask))
      || (aiMode === 'manual' && draft.inspectionTask !== 'manual_other')) {
      throw new Error('AI 모드와 점검 분야를 올바르게 선택하세요.');
    }
    const location = draft.location.trim();
    const description = draft.description.trim();
    const action = draft.action.trim();
    if (!draft.photo.uri || !location || !description || !action) {
      throw new Error('사진과 장소, 발견 메모, 권장 조치를 모두 입력하세요.');
    }
    const finding: Finding = {
      ...draft, aiMode, location, description, action,
      id: nextId('finding'), inspectionId: nextId('inspection'), inspectorId: actor.id,
      siteId: site.id, createdAt: now(), status: 'Open', after: [], analyses: [],
      logs: [{ actor: actor.id, action: 'finding_created', note: description, at: now() }],
    };
    if (initialSuggestion) {
      validateSuggestion(initialSuggestion, 'BeforeHazard', finding);
      finding.analyses.push({
        ...initialSuggestion, source: initialSuggestion.source ?? 'mock',
        id: nextId('analysis'), type: 'BeforeHazard', at: now(), decision: 'Pending',
      });
    }
    this.repository.save(finding);
    return finding;
  }

  execute(actor: Profile, id: string, command: Command) {
    actor = this.currentActor(actor);
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
        require(this.people.findProfile(command.assigneeId)?.role === 'Assignee');
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
        require(command.type === 'BeforeHazard' || command.type === 'AfterComparison');
        require(command.type === 'BeforeHazard'
          ? owner && finding.status === 'Open'
          : finding.status === 'PendingReview' && finding.after.length > 0);
        validateSuggestion(command.suggestion, command.type, finding);
        if (command.type === 'AfterComparison') {
          if ((finding.aiMode === 'fixed-cpu'
            && (command.expectedSubmissionCount === undefined || command.expectedAfterUri === undefined))
            || (command.expectedSubmissionCount !== undefined
              && (!Number.isInteger(command.expectedSubmissionCount) || command.expectedSubmissionCount !== finding.after.length))
            || (command.expectedAfterUri !== undefined && command.expectedAfterUri !== finding.after.at(-1)?.uri)) {
            throw new Error('분석을 시작한 조치 제출과 현재 제출이 다릅니다. 다시 비교하세요.');
          }
        }
        const analysis: Analysis = {
          ...command.suggestion, source: command.suggestion.source ?? 'mock',
          id: nextId('analysis'), type: command.type, at: now(), decision: 'Pending',
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
        }, 'hazard_reviewed', command.decision);
      }
    }
  }
}
