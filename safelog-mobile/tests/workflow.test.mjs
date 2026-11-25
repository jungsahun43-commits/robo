import assert from 'node:assert/strict';
import test from 'node:test';
import { mockAnalyzer } from '../src/features/ai/mock.ts';
import { checklistWithAi, latestInspectionCategory, preliminarySheetHtml } from '../src/features/reporting/checklist.ts';

test('puts the detected photo category in the preliminary sheet and respects review', () => {
  const analysis = { type: 'BeforeHazard', category: 'robustad_pcb', model: 'pcb-v1', decision: 'Pending' };
  const category = latestInspectionCategory([analysis]);
  assert.deepEqual(category, { id: 'robustad_pcb', label: 'RobustAD PCB', mock: false });
  const checklist = checklistWithAi([analysis], {});
  assert.deepEqual(checklist.robustad_pcb, { applicable: true, suggestedByAi: true });
  const html = preliminarySheetHtml([['점검일자', '2026-11-25'], ['점검 대상 (Part)', category.label]], checklist);
  assert.match(html, /점검 대상 \(Part\)<\/th><td[^>]*>RobustAD PCB/);
  assert.deepEqual(checklistWithAi([analysis], { robustad_pcb: { applicable: false } }).robustad_pcb, { applicable: false });
  assert.equal(latestInspectionCategory([{ ...analysis, decision: 'Rejected' }]), undefined);
});
import { demoLogin } from '../src/features/auth/demo.ts';
import { WorkflowService, visibleTo } from '../src/features/workflow/service.ts';
import { MemoryFindingRepository } from '../src/repositories/memory.ts';

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
