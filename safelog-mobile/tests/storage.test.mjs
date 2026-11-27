// SqliteFindingRepository on node:sqlite: the same SQL as the app's expo-sqlite adapter.
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { profiles, site } from '../src/features/auth/demo.ts';
import { WorkflowService } from '../src/features/workflow/service.ts';
import { ensureSite, SqliteFindingRepository } from '../src/repositories/sqlite/finding-repository.ts';
import { migrate, migrationsUpTo, SCHEMA_VERSION, userVersion } from '../src/repositories/sqlite/schema.ts';
import { openNodeDatabase } from './support/node-database.mjs';

const [inspector, manager, assignee] = profiles;
const suggestion = { model: 'mock-vision', prompt: 'hazard-v1', confidence: 0.82, assessment: '통로 적치', action: '정리', category: '넘어짐', risk: 4 };
const comparison = { model: 'mock-vision', prompt: 'compare-v1', confidence: 0.7, assessment: '정리됨', action: '확인' };

/** Stand-in for AppPhotoStore: "copies" outside URIs into app://photos/. */
class FakePhotoStore {
  files = new Set();
  failNext = false;
  import(uri, findingId, name) {
    if (uri.startsWith('app://photos/')) return { uri, created: false };
    if (this.failNext) { this.failNext = false; throw new Error('copy failed'); }
    const copy = `app://photos/${findingId}/${name}-${this.files.size}.jpg`;
    this.files.add(copy);
    return { uri: copy, created: true };
  }
  remove(uri) { this.files.delete(uri); }
  listManaged() { return [...this.files]; }
}

function tempDb(t) {
  const dir = mkdtempSync(join(tmpdir(), 'safelog-storage-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return join(dir, 'safelog.db');
}

/** Opens (or reopens) the file the way the app does. */
function open(path, photos = new FakePhotoStore()) {
  const db = openNodeDatabase(path);
  migrate(db);
  ensureSite(db, site);
  for (const p of profiles) {
    db.run('INSERT INTO profiles (id, display_name, role) VALUES (?, ?, ?) ON CONFLICT(id) DO NOTHING',
      [p.id, p.name, p.role.toLowerCase()]);
  }
  return { db, photos, repo: new SqliteFindingRepository(db, photos) };
}

const plain = value => JSON.parse(JSON.stringify(value));
const count = (db, sql, params = []) => db.get(sql, params).n;

/** Runs the whole workflow (with AI review and rework) and returns the finding id. */
function runWorkflow(repo) {
  const service = new WorkflowService(repo);
  const photo = { uri: 'file:///picker/before.jpg', mimeType: 'image/jpeg' };
  const afterPhoto = { uri: 'file:///picker/after.png', mimeType: 'image/png' };
  const { id } = service.create(inspector, { photo, location: '2층 통로', description: '자재 적치', action: '정리' });
  const run = (actor, command) => service.execute(actor, id, command);
  const hazard = run(inspector, { kind: 'suggest', type: 'BeforeHazard', suggestion });
  run(inspector, { kind: 'hazard', analysisId: hazard.id, decision: 'Edited', description: '통로 적치 위험', action: '자재 이동' });
  run(manager, { kind: 'assign', assigneeId: assignee.id, dueAt: '2026-11-30T09:00:00.000Z' });
  run(assignee, { kind: 'start' });
  run(assignee, { kind: 'submit', note: '1차 정리', photo: afterPhoto });
  run(inspector, { kind: 'changes', note: '통로 폭 부족' });
  run(assignee, { kind: 'submit', note: '2차 정리', photo: afterPhoto });
  const compared = run(assignee, { kind: 'suggest', type: 'AfterComparison', suggestion: comparison });
  run(inspector, { kind: 'verify', note: '직접 확인', confirmed: true, analysisId: compared.id, decision: 'Accepted' });
  return id;
}

test('restores findings, photos, logs, AI analyses and review decisions after restart', t => {
  const path = tempDb(t);
  const first = open(path);
  const id = runWorkflow(first.repo);
  const before = plain(first.repo.snapshot());
  first.db.close();

  const second = open(path, first.photos);
  const restored = plain(second.repo.snapshot());
  assert.deepEqual(restored, before);

  const [finding] = restored;
  assert.equal(finding.id, id);
  assert.equal(finding.status, 'Verified');
  assert.equal(finding.assigneeId, assignee.id);
  assert.equal(finding.risk, 4);
  assert.equal(finding.category, '넘어짐');
  assert.match(finding.photo.uri, /^app:\/\/photos\//);                 // copied, only the path stored
  assert.deepEqual(finding.after.map(p => p.mimeType), ['image/png', 'image/png']);
  assert.deepEqual(finding.logs.map(l => l.action), [
    'finding_created', 'hazard_reviewed', 'assigned', 'work_started', 'action_submitted', 'changes_requested',
    'action_submitted', 'verified']);
  assert.deepEqual(finding.analyses.map(a => [a.type, a.decision, a.reviewer, a.submissionCount]),
    [['BeforeHazard', 'Edited', inspector.id, undefined], ['AfterComparison', 'Accepted', inspector.id, 2]]);
  second.db.close();
});

test('re-saving keeps rows in place (upsert, not INSERT OR REPLACE) and deletes removed children', t => {
  const { db, repo } = open(tempDb(t));
  const id = runWorkflow(repo);
  const rowid = () => db.get('SELECT rowid AS r FROM findings WHERE id = ?', [id]).r;
  const analysisRowids = () => db.all('SELECT rowid AS r FROM ai_analyses ORDER BY seq').map(row => row.r);
  const firstRowid = rowid();
  const firstAnalyses = analysisRowids();

  const finding = repo.snapshot()[0];
  for (const note of ['a', 'b', 'c']) repo.save({ ...finding, description: note });
  assert.equal(rowid(), firstRowid);
  assert.deepEqual(analysisRowids(), firstAnalyses);
  assert.equal(count(db, 'SELECT COUNT(*) AS n FROM photos WHERE finding_id = ?', [id]), 3);
  assert.equal(count(db, 'SELECT COUNT(*) AS n FROM action_logs WHERE finding_id = ?', [id]), finding.logs.length);

  repo.save({ ...finding, after: finding.after.slice(0, 1), analyses: finding.analyses.slice(0, 1), logs: finding.logs.slice(0, 2) });
  assert.equal(count(db, 'SELECT COUNT(*) AS n FROM photos WHERE finding_id = ?', [id]), 2);
  assert.equal(count(db, 'SELECT COUNT(*) AS n FROM ai_analyses WHERE subject_id = ?', [id]), 1);
  assert.equal(count(db, 'SELECT COUNT(*) AS n FROM action_logs WHERE finding_id = ?', [id]), 2);
  db.close();
});

test('a failed save rolls back the database, keeps the snapshot and deletes copied photos', t => {
  const { db, repo, photos } = open(tempDb(t));
  const service = new WorkflowService(repo);
  const created = service.create(inspector, { photo: { uri: 'file:///p.jpg', mimeType: 'image/jpeg' }, location: 'A', description: 'B', action: 'C' });
  const snapshot = repo.snapshot();
  const filesBefore = photos.listManaged();

  const broken = { ...created, assigneeId: 'no-such-person', after: [{ uri: 'file:///new.jpg', mimeType: 'image/jpeg' }] };
  assert.throws(() => repo.save(broken), /FOREIGN KEY constraint failed/);
  assert.equal(repo.snapshot(), snapshot);
  assert.deepEqual(photos.listManaged(), filesBefore);
  assert.equal(count(db, 'SELECT COUNT(*) AS n FROM photos'), 1);
  assert.equal(db.get('SELECT assignee_id AS a FROM findings').a, null);

  photos.failNext = true;  // copy fails before the transaction
  assert.throws(() => repo.save({ ...created, after: [{ uri: 'file:///x.jpg', mimeType: 'image/jpeg' }] }), /copy failed/);
  assert.equal(repo.snapshot(), snapshot);
  db.close();
});

test('foreign keys are enforced and orphan photo files are cleaned up', t => {
  const { db, repo, photos } = open(tempDb(t));
  assert.throws(() => db.run(
    "INSERT INTO photos (id, finding_id, kind, ordinal, storage_path, mime_type) VALUES ('x', 'missing', 'before', 0, 'p', 'image/jpeg')"),
  /FOREIGN KEY/);
  runWorkflow(repo);
  photos.files.add('app://photos/stale/before-0-99.jpg');
  assert.equal(repo.removeOrphanPhotos(), 1);
  assert.equal(photos.listManaged().length, 3);
  db.close();
});

test('migrates v1 to the current version keeping data, creates fresh databases and rejects newer versions', t => {
  const path = tempDb(t);
  const legacy = openNodeDatabase(path);
  legacy.exec('PRAGMA foreign_keys = ON');
  legacy.exec(migrationsUpTo(1).join('\n'));
  legacy.exec('PRAGMA user_version = 1');
  legacy.exec(`INSERT INTO sites VALUES ('site-1', 'f', 'a');
    INSERT INTO profiles VALUES ('inspector-1', '김안전', 'inspector');
    INSERT INTO inspections VALUES ('inspection-1', 'site-1', 'inspector-1', '2026-11-01T00:00:00.000Z');
    INSERT INTO findings (id, inspection_id, location, description, action, status, created_at)
      VALUES ('finding-1', 'inspection-1', '2층', '적치', '정리', 'open', '2026-11-01T00:00:00.000Z');
    INSERT INTO photos VALUES ('finding-1:before:0', 'finding-1', 'before', 0, 'app://photos/a.jpg', 'image/jpeg');`);
  assert.equal(count(legacy, "SELECT COUNT(*) AS n FROM sqlite_master WHERE name = 'credentials'"), 0);
  legacy.close();

  const { db, repo } = open(path);
  assert.equal(userVersion(db), SCHEMA_VERSION);
  assert.equal(SCHEMA_VERSION, 3);
  assert.equal(count(db, "SELECT COUNT(*) AS n FROM sqlite_master WHERE name = 'credentials'"), 1);
  assert.equal(repo.snapshot()[0].description, '적치');
  assert.equal(repo.snapshot()[0].photo.uri, 'app://photos/a.jpg');
  db.close();

  const fresh = open(tempDb(t));
  assert.equal(userVersion(fresh.db), 3);
  fresh.db.close();

  const newer = openNodeDatabase(tempDb(t));
  newer.exec('PRAGMA user_version = 99');
  assert.throws(() => migrate(newer), /newer than this app/);
  newer.close();
});
