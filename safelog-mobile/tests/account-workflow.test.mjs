import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { AuthService } from '../src/features/auth/auth-service.ts';
import { MemoryAccountStore } from '../src/features/auth/account-store.ts';
import { PasswordHasher } from '../src/features/auth/password.ts';
import { profiles, site, DEMO_PASSWORD } from '../src/features/auth/demo.ts';
import { WorkflowService } from '../src/features/workflow/service.ts';
import { MemoryFindingRepository } from '../src/repositories/memory.ts';
import { SqliteAccountStore } from '../src/repositories/sqlite/account-store.ts';
import { SqliteFindingRepository, ensureSite } from '../src/repositories/sqlite/finding-repository.ts';
import { migrate } from '../src/repositories/sqlite/schema.ts';
import { reportFor } from '../src/features/reporting/report-data.ts';
import { openNodeDatabase } from './support/node-database.mjs';

const hasher = () => new PasswordHasher(size => crypto.getRandomValues(new Uint8Array(size)), { logN: 6, r: 8, p: 1 });
const draft = { photo: { uri: 'file:///synthetic-before.jpg', mimeType: 'image/jpeg' },
  location: '통합 검사', description: '합성 테스트 기록', action: '사람 검토', aiMode: 'fixed-cpu', inspectionTask: 'robustad_pcb' };

test('new SQLite accounts can own and complete an AI finding; identities and decisions survive restart', async t => {
  const folder = mkdtempSync(join(tmpdir(), 'safelog-account-workflow-'));
  t.after(() => rmSync(folder, { recursive: true, force: true }));
  const path = join(folder, 'test.db');
  const db = openNodeDatabase(path);
  migrate(db);
  ensureSite(db, site);
  const accounts = new SqliteAccountStore(db);
  let id = 0;
  const auth = new AuthService(accounts, hasher(), { newProfileId: () => `new-profile-${++id}` });
  await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);
  const manager = profiles.find(profile => profile.role === 'Manager');
  const inspector = await auth.createAccount(manager.id, '새 점검자', 'Inspector', 'new-inspector', 'temporary123');
  const assignee = await auth.createAccount(manager.id, '새 조치자', 'Assignee', 'new-assignee', 'temporary123');
  await auth.changePassword(inspector.id, 'temporary123', 'inspector-updated');
  await auth.changePassword(assignee.id, 'temporary123', 'assignee-updated');
  assert.equal((await auth.login('new-assignee', 'assignee-updated')).mustChangePassword, false);

  const repo = new SqliteFindingRepository(db);
  const workflow = new WorkflowService(repo, accounts);
  const finding = workflow.create(inspector, draft);
  const suggestion = { source: 'fixed-cpu', inspectionTask: 'robustad_pcb', model: 'fixture-pcb',
    prompt: 'pcb-fixture', assessment: '외관 후보 참고', action: '직접 확인',
    modelSha256: 'a'.repeat(64), inputSha256: 'b'.repeat(64), rawJson: '{ "candidateDetected": false }',
    candidateDetected: false, anomalyScore: 0, threshold: 0.3 };
  const analysis = workflow.execute(inspector, finding.id, { kind: 'suggest', type: 'BeforeHazard', suggestion });
  workflow.execute(inspector, finding.id, { kind: 'hazard', analysisId: analysis.id, decision: 'Edited',
    description: '점검자가 확인한 내용', action: '조치 후 직접 확인' });
  workflow.execute(manager, finding.id, { kind: 'assign', assigneeId: assignee.id });
  workflow.execute(assignee, finding.id, { kind: 'start' });
  workflow.execute(assignee, finding.id, { kind: 'submit', note: '테스트 조치',
    photo: { uri: 'file:///synthetic-after.jpg', mimeType: 'image/jpeg' } });
  workflow.execute(inspector, finding.id, { kind: 'verify', confirmed: true, note: 'PCB는 자동 비교 없이 직접 확인' });
  db.close();

  const reopened = openNodeDatabase(path);
  try {
    migrate(reopened);
    const restoredAccounts = new SqliteAccountStore(reopened);
    const restored = new SqliteFindingRepository(reopened).snapshot()[0];
    assert.equal(restored.status, 'Verified');
    assert.equal(restored.assigneeId, assignee.id);
    assert.equal(restored.inspectorId, inspector.id);
    assert.equal(restored.analyses[0].rawJson, suggestion.rawJson);
    assert.equal(restored.analyses[0].candidateDetected, false);
    assert.equal(restored.analyses[0].confidence, undefined);
    assert.equal(restored.analyses[0].decision, 'Edited');
    assert.equal(restored.analyses[0].reviewer, inspector.id);
    const report = reportFor(restored, restoredAccounts.listProfiles());
    assert(report.header.some(line => line.includes('새 점검자')));
    assert(report.record.some(line => line.includes('새 조치자')));
    assert(report.ai[0].text.includes('새 점검자'));
    assert(report.logs.some(line => line.includes('새 조치자')));
    assert(restoredAccounts.listProfiles().every(person => !('passwordHash' in person) && !('salt' in person)));
  } finally { reopened.close(); }
});

test('workflow validates current account roles rather than trusting a forged or unknown profile', async () => {
  const accounts = new MemoryAccountStore();
  const auth = new AuthService(accounts, hasher());
  await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);
  const repo = new MemoryFindingRepository();
  const workflow = new WorkflowService(repo, accounts);
  const inspector = profiles.find(profile => profile.role === 'Inspector');
  const manager = profiles.find(profile => profile.role === 'Manager');
  const finding = workflow.create(inspector, draft);
  assert.throws(() => workflow.execute({ ...inspector, role: 'Manager' }, finding.id,
    { kind: 'assign', assigneeId: 'assignee-1' }), /계정과 역할/);
  assert.throws(() => workflow.execute(manager, finding.id, { kind: 'assign', assigneeId: 'missing-user' }), /역할 또는 작업 상태/);
  assert.throws(() => workflow.create({ ...inspector, id: 'missing-user' }, draft), /계정과 역할/);
  assert.equal(repo.snapshot()[0].assigneeId, undefined);
});
