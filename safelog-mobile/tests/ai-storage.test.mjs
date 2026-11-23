// The native SQLite adapter and this driver execute the same v3 SQL.
// These tests use temporary databases and a fake photo copier; no model runs.
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { profiles, site } from '../src/features/auth/demo.ts';
import { ensureSite, SqliteFindingRepository } from '../src/repositories/sqlite/finding-repository.ts';
import { migrate, migrationsUpTo, SCHEMA_VERSION, userVersion } from '../src/repositories/sqlite/schema.ts';
import { openNodeDatabase } from './support/node-database.mjs';

const at = '2026-11-23T01:02:03.456Z';
const raw = ' {\r\n  "candidateDetected": false, "score": 0, "note": "균열 확인", "unknown": [null, true]\r\n}\t';
const plain = value => JSON.parse(JSON.stringify(value));

function databasePath(t) {
  const directory = mkdtempSync(join(tmpdir(), 'safelog-ai-storage-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  return join(directory, 'safelog.db');
}

class PhotoFiles {
  files = new Set();
  sequence = 0;
  import(uri, findingId, name) {
    if (uri.startsWith('app://photos/')) return { uri, created: false };
    const copy = `app://photos/${findingId}/${name}-${++this.sequence}.png`;
    this.files.add(copy);
    return { uri: copy, created: true };
  }
  remove(uri) { this.files.delete(uri); }
  listManaged() { return [...this.files]; }
}

function open(path, photos = new PhotoFiles()) {
  const db = openNodeDatabase(path);
  migrate(db);
  ensureSite(db, site);
  for (const p of profiles) {
    db.run('INSERT INTO profiles (id, display_name, role) VALUES (?, ?, ?) ON CONFLICT(id) DO NOTHING',
      [p.id, p.name, p.role.toLowerCase()]);
  }
  return { db, photos, repo: new SqliteFindingRepository(db, photos) };
}

function finding() {
  return {
    id: 'finding-fixed', inspectionId: 'inspection-fixed', inspectorId: profiles[0].id,
    siteId: site.id, createdAt: at, status: 'Verified', aiMode: 'fixed-cpu', inspectionTask: 'facility_surface',
    photo: { uri: 'file:///picker/before.png', mimeType: 'image/png', width: 1920, height: 1080, fileSize: 0 },
    location: '2층', description: '점검 사진', action: '직접 확인', assigneeId: profiles[2].id,
    category: 'concrete_crack', risk: 3, dueAt: at,
    after: [{ uri: 'file:///picker/after.png', mimeType: 'image/png', width: 1024, height: 768, fileSize: 12345 }],
    logs: [{ actor: profiles[0].id, action: 'verified', note: '사람의 확인', at }],
    analyses: [
      {
        id: 'analysis-hazard', type: 'BeforeHazard', at, model: 'fixture-fixed', prompt: 'safelog-hazard-v1',
        confidence: 0, assessment: '후보 미검출', action: '사진과 현장 확인', category: 'concrete_crack', risk: 3,
        source: 'fixed-cpu', inspectionTask: 'facility_surface', candidateDetected: false,
        modelSha256: 'a'.repeat(64), inputSha256: 'b'.repeat(64), rawJson: raw,
        anomalyScore: 0, threshold: 0, likelyResolved: false, decision: 'Edited', reviewer: profiles[0].id,
      },
      {
        id: 'analysis-comparison', type: 'AfterComparison', at, model: 'fixture-fixed', prompt: 'safelog-comparison-v1',
        assessment: '전후 후보 변화', action: '사람의 최종 확인', source: 'fixed-cpu',
        inspectionTask: 'conveyor_foreign', candidateDetected: true, likelyResolved: true,
        modelSha256: 'c'.repeat(64), inputSha256: 'd'.repeat(64), rawJson: '\n{ "after": false }\n',
        decision: 'Accepted', reviewer: profiles[0].id, submissionCount: 1,
      },
    ],
  };
}

test('fixed AI provenance, exact raw responses, absent confidence and photo metadata survive restart', t => {
  const path = databasePath(t);
  const first = open(path);
  first.repo.save(finding());
  const before = plain(first.repo.snapshot());
  const sql = first.db.all(
    'SELECT id, confidence, candidate_detected, anomaly_score, threshold, likely_resolved, result_json, raw_json_present FROM ai_analyses ORDER BY seq');
  assert.deepEqual(sql.map(row => row.confidence), [0, null]);
  assert.deepEqual(sql.map(row => row.candidate_detected), [0, 1]);
  assert.deepEqual(sql.map(row => row.likely_resolved), [0, 1]);
  assert.equal(sql[0].anomaly_score, 0);
  assert.equal(sql[0].threshold, 0);
  assert.equal(sql[0].result_json, raw);
  assert.equal(sql[0].raw_json_present, 1);
  first.db.close();

  const second = open(path, first.photos);
  assert.deepEqual(plain(second.repo.snapshot()), before);
  const restored = second.repo.snapshot()[0];
  assert.equal(restored.aiMode, 'fixed-cpu');
  assert.equal(restored.inspectionTask, 'facility_surface');
  assert.equal(restored.photo.fileSize, 0);
  assert.equal(restored.analyses[0].rawJson, raw);
  assert.equal(Object.hasOwn(restored.analyses[1], 'confidence'), false);
  second.repo.save({ ...restored, description: '기존 기록 다시 저장' });
  assert.equal(second.db.get('SELECT result_json FROM ai_analyses WHERE id = ?', ['analysis-hazard']).result_json, raw);
  const withoutRaw = { ...restored, analyses: restored.analyses.map(({ rawJson, ...analysis }) => analysis) };
  second.repo.save(withoutRaw);
  assert.equal(second.db.get('SELECT raw_json_present FROM ai_analyses WHERE id = ?', ['analysis-hazard']).raw_json_present, 1);
  assert.equal(new SqliteFindingRepository(second.db, first.photos).snapshot()[0].analyses[0].rawJson, raw);
  assert.equal(second.repo.removeOrphanPhotos(), 0);
  second.db.close();
});

test('PCB anomaly score remains a score with SQL NULL confidence, and optional false/zero/raw empty values survive', t => {
  const path = databasePath(t);
  const first = open(path);
  const pcb = finding();
  pcb.inspectionTask = 'robustad_pcb';
  pcb.analyses = [{
    id: 'analysis-pcb', type: 'BeforeHazard', at, model: 'pcb-fixture', prompt: 'pcb-fixture-v1',
    assessment: '기판 외관 후보', action: '기판 직접 확인', source: 'fixed-cpu', inspectionTask: 'robustad_pcb',
    modelSha256: 'e'.repeat(64), inputSha256: 'f'.repeat(64), anomalyScore: 0.9825180768966675,
    threshold: 0.36685922741889954, candidateDetected: false, likelyResolved: false,
    rawJson: '', submissionCount: 0, decision: 'Rejected', reviewer: profiles[0].id,
  }];
  first.repo.save(pcb);
  const row = first.db.get('SELECT confidence, result_json, raw_json_present FROM ai_analyses');
  assert.equal(row.confidence, null);
  assert.equal(row.result_json, '');
  assert.equal(row.raw_json_present, 1);
  const before = plain(first.repo.snapshot());
  first.db.close();
  const second = open(path, first.photos);
  assert.deepEqual(plain(second.repo.snapshot()), before);
  const restored = second.repo.snapshot()[0].analyses[0];
  assert.equal(Object.hasOwn(restored, 'confidence'), false);
  assert.equal(restored.anomalyScore, pcb.analyses[0].anomalyScore);
  assert.equal(restored.rawJson, '');
  assert.equal(restored.submissionCount, 0);
  second.db.close();
});

const tables = ['sites', 'profiles', 'inspections', 'findings', 'photos', 'action_logs', 'ai_analyses', 'credentials'];
const originalRows = db => Object.fromEntries(tables.map(table => [table, db.all(`SELECT * FROM ${table} ORDER BY rowid`)]));
function assertOriginalRows(db, expected) {
  for (const [table, rows] of Object.entries(expected)) {
    if (!rows.length) continue;
    const columns = Object.keys(rows[0]).join(', ');
    assert.deepEqual(db.all(`SELECT ${columns} FROM ${table} ORDER BY rowid`), rows, table);
  }
}

function legacyV2(path) {
  const db = openNodeDatabase(path);
  db.exec('PRAGMA foreign_keys = ON');
  db.transaction(() => {
    for (const sql of migrationsUpTo(2)) db.exec(sql);
    db.exec('PRAGMA user_version = 2');
    db.run('INSERT INTO sites VALUES (?, ?, ?)', [site.id, site.name, site.address]);
    for (const profile of profiles) {
      db.run('INSERT INTO profiles VALUES (?, ?, ?)', [profile.id, profile.name, profile.role.toLowerCase()]);
    }
    db.run('INSERT INTO inspections VALUES (?, ?, ?, ?)', ['inspection-old', site.id, profiles[0].id, at]);
    db.run(`INSERT INTO findings
      (id, inspection_id, location, description, action, assignee_id, due_at, status, created_at, category, risk)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ['finding-old', 'inspection-old', '통로', '이전 기록', '정리', profiles[2].id, at, 'verified', at, 'trip_hazard', 4]);
    db.run('INSERT INTO photos VALUES (?, ?, ?, ?, ?, ?)',
      ['photo-old', 'finding-old', 'before', 0, 'app://photos/old.png', 'image/png']);
    db.run('INSERT INTO action_logs VALUES (?, ?, ?, ?, ?, ?, ?)',
      ['log-old', 'finding-old', 0, profiles[0].id, 'verified', '기존 판단', at]);
    db.run(`INSERT INTO ai_analyses
      (id, subject_id, seq, analysis_type, model_name, prompt_version, risk_level, category, confidence,
       assessment, action, result_json, decision, reviewed_by, submission_count, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ['analysis-old', 'finding-old', 0, 'after_comparison', 'legacy-model', 'legacy-v1', 4, 'trip_hazard',
      0.82, '기존 평가', '정리', raw, 'edited', profiles[0].id, 2, at]);
    db.run('INSERT INTO credentials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
      [profiles[0].id, 'inspector-old', 'unchanged-test-hash', new Uint8Array(16).fill(7), 1, 2, at, at, at]);
  });
  return db;
}

test('v2 to v3 preserves every existing column including raw JSON, credentials, roles and human review', t => {
  const path = databasePath(t);
  const db = legacyV2(path);
  const before = originalRows(db);
  migrate(db);
  assert.equal(userVersion(db), 3);
  assertOriginalRows(db, before);
  assert.equal(db.get('PRAGMA foreign_keys').foreign_keys, 1);
  const repo = new SqliteFindingRepository(db);
  const restored = repo.snapshot()[0];
  assert.equal(restored.analyses[0].confidence, 0.82);
  assert.equal(restored.analyses[0].decision, 'Edited');
  assert.equal(restored.analyses[0].reviewer, profiles[0].id);
  assert.equal(restored.analyses[0].submissionCount, 2);
  assert.equal(Object.hasOwn(restored.analyses[0], 'rawJson'), false); // legacy generated JSON is not new server evidence
  assert.equal(Object.hasOwn(restored, 'aiMode'), false);
  repo.save({ ...restored, description: '추가 설명만 변경' });
  assert.equal(db.get('SELECT id FROM photos').id, 'photo-old');
  assert.equal(db.get('SELECT id FROM action_logs').id, 'log-old');
  assert.equal(db.get('SELECT result_json FROM ai_analyses').result_json, raw);
  assert.equal(db.get('SELECT raw_json_present FROM ai_analyses').raw_json_present, 0);
  db.close();
  const again = openNodeDatabase(path);
  migrate(again); // v3 startup is idempotent
  assert.equal(new SqliteFindingRepository(again).snapshot()[0].description, '추가 설명만 변경');
  assert.equal(again.get('SELECT result_json FROM ai_analyses').result_json, raw);
  again.close();
});

test('an interrupted v3 rebuild rolls back DDL, raw responses and version before retry', t => {
  const db = legacyV2(databasePath(t));
  const before = originalRows(db);
  const failing = {
    ...db,
    exec(sql) {
      db.exec(sql);
      if (sql.includes('ALTER TABLE ai_analyses_v3 RENAME TO ai_analyses')) throw new Error('simulated migration failure');
    },
  };
  assert.throws(() => migrate(failing), /simulated migration failure/);
  assert.equal(userVersion(db), 2);
  assertOriginalRows(db, before);
  assert.equal(db.all('PRAGMA table_info(findings)').some(row => row.name === 'ai_mode'), false);
  assert.equal(db.get("SELECT COUNT(*) AS n FROM sqlite_master WHERE name = 'ai_analyses_v3'").n, 0);
  migrate(db);
  assert.equal(userVersion(db), 3);
  assertOriginalRows(db, before);
  db.close();
});

test('invalid AI confidence rolls back all metadata, copied photos, database rows and snapshot notifications', t => {
  const path = databasePath(t);
  const first = open(path);
  first.repo.save(finding());
  const before = first.repo.snapshot();
  const rows = originalRows(first.db);
  const files = first.photos.listManaged();
  let notifications = 0;
  first.repo.subscribe(() => notifications++);
  const changed = {
    ...before[0], aiMode: 'manual', inspectionTask: 'conveyor_foreign', description: 'must not commit',
    photo: { ...before[0].photo, width: 1 },
    after: [...before[0].after, { uri: 'file:///extra.png', mimeType: 'image/png', width: 0, height: 0, fileSize: 0 }],
    analyses: before[0].analyses.map((analysis, i) => i ? analysis : { ...analysis, confidence: 1.01, rawJson: 'must not commit' }),
  };
  assert.throws(() => first.repo.save(changed), /CHECK constraint failed/);
  assert.equal(first.repo.snapshot(), before);
  assert.equal(notifications, 0);
  assert.deepEqual(first.photos.listManaged(), files);
  assertOriginalRows(first.db, rows);
  first.db.close();
  const second = open(path, first.photos);
  assert.deepEqual(plain(second.repo.snapshot()), plain(before));
  second.db.close();
});

test('a newer schema is rejected without rewriting its existing evidence', t => {
  const db = legacyV2(databasePath(t));
  const before = originalRows(db);
  db.exec(`PRAGMA user_version = ${SCHEMA_VERSION + 1}`);
  assert.throws(() => migrate(db), /newer than this app/);
  assert.equal(userVersion(db), SCHEMA_VERSION + 1);
  assertOriginalRows(db, before);
  db.close();
});
