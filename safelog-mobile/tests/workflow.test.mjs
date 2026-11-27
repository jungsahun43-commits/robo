import assert from 'node:assert/strict';
import test from 'node:test';
import { mockAnalyzer } from '../src/features/ai/mock.ts';
import { demoLogin } from '../src/features/auth/demo.ts';
import { WorkflowService, visibleTo } from '../src/features/workflow/service.ts';
import { MemoryFindingRepository } from '../src/repositories/memory.ts';
import { fixedTaskIds } from '../src/contracts/inspection.ts';

test('registers every fixed field while rejecting unknown or unconnected AI categories', () => {
  const service = new WorkflowService(new MemoryFindingRepository());
  const actor = demoLogin('inspector-1', 'demo1234');
  for (const inspectionTask of fixedTaskIds) {
    const finding = service.create(actor, { photo: { uri: 'file:///protocol.jpg', mimeType: 'image/jpeg' },
      location: '시험', description: '프로토콜', action: '직접 확인', aiMode: 'fixed-cpu', inspectionTask });
    assert.equal(finding.inspectionTask, inspectionTask);
  }
  for (const inspectionTask of ['unknown_field', 'manual_other']) {
    assert.throws(() => service.create(actor, { photo: { uri: 'file:///protocol.jpg', mimeType: 'image/jpeg' },
      location: '시험', description: '프로토콜', action: '직접 확인', aiMode: 'fixed-cpu', inspectionTask }));
  }
});

test('unconnected fields use human review without inventing AI results', () => {
  const repo = new MemoryFindingRepository();
  const service = new WorkflowService(repo);
  const owner = demoLogin('inspector-1', 'demo1234');
  const manager = demoLogin('manager-1', 'demo1234');
  const worker = demoLogin('assignee-1', 'demo1234');
  const { id } = service.create(owner, { photo: { uri: 'file:///protocol.jpg', mimeType: 'image/jpeg' },
    location: '기타 설비', description: '직접 관찰', action: '직접 점검', aiMode: 'manual', inspectionTask: 'manual_other' });
  assert.throws(() => service.execute(owner, id, { kind: 'suggest', type: 'BeforeHazard', suggestion: {
    source: 'mock', model: 'mock', prompt: 'test', assessment: '가짜', action: '가짜',
  } }));
  service.execute(owner, id, { kind: 'hazard', decision: 'Rejected', description: '직접 관찰', action: '직접 점검' });
  service.execute(manager, id, { kind: 'assign', assigneeId: worker.id });
  service.execute(worker, id, { kind: 'start' });
  service.execute(worker, id, { kind: 'submit', note: '직접 조치', photo: { uri: 'file:///after.jpg', mimeType: 'image/jpeg' } });
  const finding = service.execute(owner, id, { kind: 'verify', note: '직접 최종 확인', confirmed: true });
  assert.equal(finding.status, 'Verified');
  assert.equal(finding.analyses.length, 0);
  assert.equal(finding.aiMode, 'manual');
});

const login = id => demoLogin(id, 'demo1234');
const inspector = login('inspector-1');
const manager = login('manager1');
const assignee = login('assignee-1');
const photo = { uri: 'file:///photo.jpg', mimeType: 'image/jpeg' };
const draft = { photo, location: '2층', description: '통로 적치', action: '정리' };

test('checks login credentials and capture validation', () => {
  assert.throws(() => demoLogin('manager-1', 'wrong'));
  assert.equal(manager.role, 'Manager');
  const service = new WorkflowService(new MemoryFindingRepository());
  assert.throws(() => service.create(manager, draft));
  assert.throws(() => service.create(inspector, { ...draft, location: ' ' }));
  assert.throws(() => service.create(inspector, { ...draft, photo: { ...photo, uri: '' } }));
});

test('assigns without changing Open, checks ownership and allows rework before human verification', () => {
  const repo = new MemoryFindingRepository();
  const service = new WorkflowService(repo);
  const { id } = service.create(inspector, draft);
  const current = () => repo.snapshot()[0];
  const execute = (actor, command) => service.execute(actor, id, command);

  assert.equal(visibleTo(assignee, current()), false);
  assert.throws(() => execute(inspector, { kind: 'assign', assigneeId: assignee.id }));
  execute(manager, { kind: 'assign', assigneeId: assignee.id });
  assert.equal(current().status, 'Open');
  assert.equal(visibleTo(assignee, current()), true);
  assert.throws(() => execute(manager, { kind: 'start' }));
  execute(assignee, { kind: 'start' });
  assert.equal(current().status, 'InProgress');
  assert.throws(() => execute(assignee, { kind: 'submit', note: ' ', photo }));
  execute(assignee, { kind: 'submit', note: '정리 완료', photo });
  assert.equal(current().status, 'PendingReview');
  assert.throws(() => execute(assignee, { kind: 'verify', note: '', confirmed: true }));
  assert.throws(() => execute({ ...inspector, id: 'other' }, { kind: 'verify', note: '', confirmed: true }));
  assert.throws(() => execute(inspector, { kind: 'verify', note: '직접 확인', confirmed: false }));
  assert.throws(() => execute(inspector, { kind: 'verify', note: ' ', confirmed: true }));
  const stale = execute(assignee, { kind: 'suggest', type: 'AfterComparison', suggestion: {
    model: 'mock', prompt: 'test', confidence: 0.8, assessment: '참고 제안', action: '직접 확인',
  } });
  assert.equal(current().status, 'PendingReview');
  assert.throws(() => execute(inspector, { kind: 'changes', note: ' ' }));
  execute(inspector, { kind: 'changes', note: '통로 너비 확인' });
  assert.equal(current().status, 'InProgress');
  execute(assignee, { kind: 'submit', note: '추가 정리 완료', photo });
  assert.throws(() => execute(inspector, { kind: 'verify', note: '직접 확인', confirmed: true, analysisId: stale.id }));
  execute(inspector, { kind: 'verify', note: '직접 확인', confirmed: true });
  assert.equal(current().status, 'Verified');
  assert.equal(current().after.length, 2);
  assert.deepEqual(current().logs.map(log => log.action), [
    'finding_created', 'assigned', 'work_started', 'action_submitted', 'changes_requested', 'action_submitted', 'verified',
  ]);
});

test('stores AI separately and handles failure, invalid results and cancellation', async () => {
  const signal = new AbortController();
  const repo = new MemoryFindingRepository();
  const service = new WorkflowService(repo);
  const finding = service.create(inspector, draft);
  const suggestion = await mockAnalyzer('정상').hazard(photo, signal.signal);
  const analysis = service.execute(inspector, finding.id, { kind: 'suggest', type: 'BeforeHazard', suggestion });
  assert.equal(repo.snapshot()[0].description, draft.description);
  service.execute(inspector, finding.id, { kind: 'hazard', analysisId: analysis.id,
    decision: 'Rejected', description: draft.description, action: draft.action });
  assert.equal(repo.snapshot()[0].analyses[0].decision, 'Rejected');
  assert.equal(repo.snapshot()[0].status, 'Open');
  const cancelled = mockAnalyzer('정상').compare(photo, photo, signal.signal);
  signal.abort();
  await assert.rejects(cancelled, /취소/);
  await assert.rejects(mockAnalyzer('분석 실패').hazard(photo, new AbortController().signal), /오류/);
  await assert.rejects(mockAnalyzer('읽을 수 없는 결과').hazard(photo, new AbortController().signal), /읽을 수 없습니다/);
});

const fixedSuggestion = overrides => ({
  model: 'fixed-cpu-test', prompt: 'safelog-hazard-v1', source: 'fixed-cpu',
  inspectionTask: 'conveyor_foreign', modelSha256: '5'.repeat(64), inputSha256: '7'.repeat(64),
  rawJson: JSON.stringify({ candidateDetected: true, modelSha256: '5'.repeat(64) }),
  candidateDetected: true, assessment: '이물질 후보가 있습니다.', action: '사진과 현장을 직접 확인하세요.',
  ...overrides,
});

test('pins the AI mode and task and refuses results from another mode or task', () => {
  const repo = new MemoryFindingRepository();
  const service = new WorkflowService(repo);
  for (const settings of [
    { aiMode: 'unknown' }, { aiMode: 'fixed-cpu' },
    { aiMode: 'fixed-cpu', inspectionTask: 'unknown' },
    { aiMode: 'mock', inspectionTask: 'unknown' },
  ]) {
    assert.throws(() => service.create(inspector, { ...draft, ...settings }));
    assert.equal(repo.snapshot().length, 0);
  }

  const finding = service.create(inspector, { ...draft, aiMode: 'fixed-cpu', inspectionTask: 'conveyor_foreign' });
  assert.equal(finding.aiMode, 'fixed-cpu');
  assert.equal(finding.inspectionTask, 'conveyor_foreign');
  const snapshot = repo.snapshot();
  for (const suggestion of [
    { model: 'mock', prompt: 'demo-v1', confidence: 0.8, assessment: '시연', action: '직접 확인' },
    fixedSuggestion({ source: 'mock' }),
    fixedSuggestion({ inspectionTask: 'facility_surface' }),
  ]) {
    assert.throws(() => service.execute(inspector, finding.id, { kind: 'suggest', type: 'BeforeHazard', suggestion }));
    assert.equal(repo.snapshot(), snapshot);
  }
  const mockFinding = service.create(inspector, draft);
  assert.equal(mockFinding.aiMode, 'mock');
  assert.throws(() => service.execute(inspector, mockFinding.id, {
    kind: 'suggest', type: 'BeforeHazard', suggestion: fixedSuggestion(),
  }));
});

test('preserves fixed model metadata and human decisions without finalizing a finding', () => {
  const repo = new MemoryFindingRepository();
  const service = new WorkflowService(repo);
  const finding = service.create(inspector, { ...draft, aiMode: 'fixed-cpu', inspectionTask: 'conveyor_foreign' });
  const suggestion = fixedSuggestion({ candidateDetected: false, assessment: '후보가 검출되지 않았습니다.',
    rawJson: JSON.stringify({ candidateDetected: false, modelSha256: '5'.repeat(64) }) });
  const analysis = service.execute(inspector, finding.id, { kind: 'suggest', type: 'BeforeHazard', suggestion });
  assert.equal(analysis.confidence, undefined);
  for (const key of ['source', 'inspectionTask', 'modelSha256', 'inputSha256', 'rawJson', 'candidateDetected']) {
    assert.equal(repo.snapshot()[0].analyses[0][key], suggestion[key]);
  }
  assert.equal(repo.snapshot()[0].status, 'Open');
  service.execute(inspector, finding.id, {
    kind: 'hazard', analysisId: analysis.id, decision: 'Accepted', description: analysis.assessment, action: analysis.action,
  });
  assert.equal(repo.snapshot()[0].analyses[0].reviewer, inspector.id);
  assert.equal(repo.snapshot()[0].analyses[0].rawJson, suggestion.rawJson);
  assert.equal(repo.snapshot()[0].status, 'Open');
});

test('rejects malformed AI values atomically before storing an analysis', () => {
  const repo = new MemoryFindingRepository();
  const service = new WorkflowService(repo);
  const finding = service.create(inspector, { ...draft, aiMode: 'fixed-cpu', inspectionTask: 'conveyor_foreign' });
  const snapshot = repo.snapshot();
  const invalid = [
    { model: ' ' }, { prompt: '' }, { assessment: '' }, { action: ' ' },
    { confidence: NaN }, { confidence: Infinity }, { confidence: -0.1 }, { confidence: 1.1 }, { confidence: '0.5' },
    { risk: 0 }, { risk: 6 }, { risk: 1.5 }, { category: ' ' },
    { candidateDetected: 'false' }, { likelyResolved: 'true' }, { anomalyScore: Infinity }, { threshold: NaN },
    { modelSha256: '' }, { modelSha256: 'z'.repeat(64) }, { inputSha256: undefined }, { inputSha256: 'short' },
    { rawJson: undefined }, { rawJson: '' }, { rawJson: '{' }, { rawJson: 'null' }, { rawJson: '[]' },
  ];
  for (const patch of invalid) {
    assert.throws(() => service.execute(inspector, finding.id, {
      kind: 'suggest', type: 'BeforeHazard', suggestion: fixedSuggestion(patch),
    }));
    assert.equal(repo.snapshot(), snapshot);
  }
  assert.throws(() => service.execute(inspector, finding.id, {
    kind: 'suggest', type: 'Unknown', suggestion: fixedSuggestion(),
  }));
  assert.equal(repo.snapshot(), snapshot);
});

test('does not attach a late comparison response to a newer action submission', () => {
  const repo = new MemoryFindingRepository();
  const service = new WorkflowService(repo);
  const finding = service.create(inspector, { ...draft, aiMode: 'fixed-cpu', inspectionTask: 'conveyor_foreign' });
  const execute = (actor, command) => service.execute(actor, finding.id, command);
  execute(manager, { kind: 'assign', assigneeId: assignee.id });
  execute(assignee, { kind: 'start' });
  const firstPhoto = { ...photo, uri: 'file:///first-after.jpg' };
  execute(assignee, { kind: 'submit', note: '첫 조치', photo: firstPhoto });
  const request = {
    kind: 'suggest', type: 'AfterComparison', expectedSubmissionCount: 1, expectedAfterUri: firstPhoto.uri,
    suggestion: fixedSuggestion({ prompt: 'safelog-compare-v1', inputSha256: undefined, likelyResolved: true }),
  };
  execute(inspector, { kind: 'changes', note: '추가 조치 필요' });
  const secondPhoto = { ...photo, uri: 'file:///second-after.jpg' };
  execute(assignee, { kind: 'submit', note: '추가 조치 완료', photo: secondPhoto });
  const snapshot = repo.snapshot();
  assert.throws(() => execute(inspector, request), /제출/);
  assert.throws(() => execute(inspector, { ...request, expectedSubmissionCount: 2 }), /제출/);
  assert.throws(() => execute(inspector, { ...request, expectedSubmissionCount: undefined, expectedAfterUri: undefined }), /제출/);
  assert.equal(repo.snapshot(), snapshot);
  const analysis = execute(inspector, { ...request, expectedSubmissionCount: 2, expectedAfterUri: secondPhoto.uri });
  assert.equal(analysis.submissionCount, 2);
  assert.equal(analysis.inputSha256, undefined);
  assert.equal(repo.snapshot()[0].status, 'PendingReview');
  execute(inspector, { kind: 'verify', note: '사진과 현장 직접 확인', confirmed: true, analysisId: analysis.id, decision: 'Accepted' });
  assert.equal(repo.snapshot()[0].status, 'Verified');
  assert.equal(repo.snapshot()[0].analyses[0].decision, 'Accepted');
});
