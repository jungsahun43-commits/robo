import type { FindingRepository } from '@/contracts/interfaces';
import type { ActionLog, Analysis, Finding, Photo } from '@/contracts/types';
import type { SqlDatabase } from './database';
import { analysisTypeMap, compact, decisionMap, nullable, optional, statusMap } from './mapping';
import { passthroughPhotoStore, type PhotoStore } from './photo-store';

interface FindingRow {
  id: string; inspection_id: string; site_id: string; inspector_id: string;
  location: string; description: string; action: string;
  assignee_id: string | null; due_at: string | null; status: string; created_at: string;
  category: string | null; risk: number | null;
}
interface PhotoRow { finding_id: string; kind: string; storage_path: string; mime_type: string }
interface LogRow { finding_id: string; actor_id: string; action: string; note: string; created_at: string }
interface AnalysisRow {
  id: string; subject_id: string; analysis_type: string; model_name: string; prompt_version: string;
  risk_level: number | null; category: string | null; confidence: number; assessment: string; action: string;
  decision: string; reviewed_by: string | null; submission_count: number | null; created_at: string;
}

const photoId = (findingId: string, kind: 'before' | 'after', ordinal: number) => `${findingId}:${kind}:${ordinal}`;
const logId = (findingId: string, seq: number) => `${findingId}:log:${seq}`;

function groupBy<T extends { finding_id: string }>(rows: T[]) {
  const groups = new Map<string, T[]>();
  for (const row of rows) groups.set(row.finding_id, [...(groups.get(row.finding_id) ?? []), row]);
  return groups;
}

function toPhoto(row: PhotoRow): Photo {
  return { uri: row.storage_path, mimeType: row.mime_type };
}

function toAnalysis(row: AnalysisRow): Analysis {
  return compact({
    id: row.id,
    type: analysisTypeMap.toApp(row.analysis_type),
    model: row.model_name,
    prompt: row.prompt_version,
    confidence: row.confidence,
    assessment: row.assessment,
    action: row.action,
    category: optional(row.category),
    risk: optional(row.risk_level),
    at: row.created_at,
    decision: decisionMap.toApp(row.decision),
    reviewer: optional(row.reviewed_by),
    submissionCount: optional(row.submission_count),
  });
}

/** Reads every finding with its photos, logs and AI analyses, newest first. */
export function loadFindings(db: SqlDatabase): Finding[] {
  const findings = db.all<FindingRow>(
    `SELECT f.*, i.site_id, i.inspector_id FROM findings f JOIN inspections i ON i.id = f.inspection_id
     ORDER BY f.created_at DESC, f.id DESC`);
  const photos = groupBy(db.all<PhotoRow>(
    'SELECT finding_id, kind, storage_path, mime_type FROM photos ORDER BY finding_id, kind, ordinal'));
  const logs = groupBy(db.all<LogRow>(
    'SELECT finding_id, actor_id, action, note, created_at FROM action_logs ORDER BY finding_id, seq'));
  const analyses = groupBy(db.all<AnalysisRow & { finding_id: string }>(
    'SELECT *, subject_id AS finding_id FROM ai_analyses ORDER BY subject_id, seq'));

  return findings.map(row => {
    const own = photos.get(row.id) ?? [];
    const before = own.find(photo => photo.kind === 'before');
    if (!before) throw new Error(`Finding ${row.id} has no before photo`);
    return compact<Finding>({
      id: row.id,
      inspectionId: row.inspection_id,
      inspectorId: row.inspector_id,
      siteId: row.site_id,
      createdAt: row.created_at,
      status: statusMap.toApp(row.status),
      photo: toPhoto(before),
      location: row.location,
      description: row.description,
      action: row.action,
      assigneeId: optional(row.assignee_id),
      dueAt: optional(row.due_at),
      category: optional(row.category),
      risk: optional(row.risk),
      after: own.filter(photo => photo.kind === 'after').map(toPhoto),
      logs: (logs.get(row.id) ?? []).map((log): ActionLog => ({
        actor: log.actor_id, action: log.action, note: log.note, at: log.created_at,
      })),
      analyses: (analyses.get(row.id) ?? []).map(toAnalysis),
    });
  });
}

/** Writes one finding document. Call inside a transaction. */
function writeFinding(db: SqlDatabase, f: Finding) {
  db.run(
    `INSERT INTO inspections (id, site_id, inspector_id, inspected_at) VALUES (?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET site_id = excluded.site_id, inspector_id = excluded.inspector_id`,
    [f.inspectionId, f.siteId, f.inspectorId, f.createdAt]);
  db.run(
    `INSERT INTO findings (id, inspection_id, location, description, action, assignee_id, due_at, status,
       created_at, category, risk)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET inspection_id = excluded.inspection_id, location = excluded.location,
       description = excluded.description, action = excluded.action, assignee_id = excluded.assignee_id,
       due_at = excluded.due_at, status = excluded.status, created_at = excluded.created_at,
       category = excluded.category, risk = excluded.risk`,
    [f.id, f.inspectionId, f.location, f.description, f.action, nullable(f.assigneeId), nullable(f.dueAt),
      statusMap.toDb(f.status), f.createdAt, nullable(f.category), nullable(f.risk)]);

  const photos: [string, 'before' | 'after', number, Photo][] = [
    [photoId(f.id, 'before', 0), 'before', 0, f.photo],
    ...f.after.map((photo, i): [string, 'before' | 'after', number, Photo] => [photoId(f.id, 'after', i), 'after', i, photo]),
  ];
  for (const [id, kind, ordinal, photo] of photos) {
    db.run(
      `INSERT INTO photos (id, finding_id, kind, ordinal, storage_path, mime_type) VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET storage_path = excluded.storage_path, mime_type = excluded.mime_type`,
      [id, f.id, kind, ordinal, photo.uri, photo.mimeType]);
  }

  f.logs.forEach((log, seq) => db.run(
    `INSERT INTO action_logs (id, finding_id, seq, actor_id, action, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET actor_id = excluded.actor_id, action = excluded.action, note = excluded.note,
       created_at = excluded.created_at`,
    [logId(f.id, seq), f.id, seq, log.actor, log.action, log.note, log.at]));

  f.analyses.forEach((a, seq) => db.run(
    `INSERT INTO ai_analyses (id, subject_id, seq, analysis_type, model_name, prompt_version, risk_level, category,
       confidence, assessment, action, result_json, decision, reviewed_by, submission_count, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET seq = excluded.seq, analysis_type = excluded.analysis_type,
       model_name = excluded.model_name, prompt_version = excluded.prompt_version, risk_level = excluded.risk_level,
       category = excluded.category, confidence = excluded.confidence, assessment = excluded.assessment,
       action = excluded.action, result_json = excluded.result_json, decision = excluded.decision,
       reviewed_by = excluded.reviewed_by, submission_count = excluded.submission_count,
       created_at = excluded.created_at`,
    [a.id, f.id, seq, analysisTypeMap.toDb(a.type), a.model, a.prompt, nullable(a.risk), nullable(a.category),
      a.confidence, a.assessment, a.action,
      JSON.stringify(compact({ model: a.model, prompt: a.prompt, confidence: a.confidence, assessment: a.assessment,
        action: a.action, category: a.category, risk: a.risk })),
      decisionMap.toDb(a.decision), nullable(a.reviewer), nullable(a.submissionCount), a.at]));

  // Children no longer in the document are deleted explicitly (never via INSERT OR REPLACE cascades).
  db.run('DELETE FROM photos WHERE finding_id = ? AND id NOT IN (SELECT value FROM json_each(?))',
    [f.id, JSON.stringify(photos.map(([id]) => id))]);
  db.run('DELETE FROM action_logs WHERE finding_id = ? AND seq >= ?', [f.id, f.logs.length]);
  db.run('DELETE FROM ai_analyses WHERE subject_id = ? AND id NOT IN (SELECT value FROM json_each(?))',
    [f.id, JSON.stringify(f.analyses.map(a => a.id))]);
}

/** Inserts the site findings refer to, if missing. */
export function ensureSite(db: SqlDatabase, site: { id: string; name: string; address: string }) {
  db.run('INSERT INTO sites (id, name, address) VALUES (?, ?, ?) ON CONFLICT(id) DO NOTHING',
    [site.id, site.name, site.address]);
}

/**
 * FindingRepository on SQLite. Keeps the synchronous snapshot/subscribe/save
 * contract by caching all findings in memory: the cache changes only after the
 * transaction commits, so a failed save leaves both DB and snapshot unchanged.
 */
export class SqliteFindingRepository implements FindingRepository {
  private findings: readonly Finding[];
  private readonly listeners = new Set<() => void>();

  constructor(private readonly db: SqlDatabase, private readonly photos: PhotoStore = passthroughPhotoStore) {
    this.findings = loadFindings(db);
  }

  snapshot = () => this.findings;

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  save(finding: Finding) {
    const created: string[] = [];
    const store = (photo: Photo, name: string): Photo => {
      const result = this.photos.import(photo.uri, finding.id, name);
      if (result.created) created.push(result.uri);
      return result.uri === photo.uri ? photo : { ...photo, uri: result.uri };
    };
    let stored: Finding;
    try {
      stored = { ...finding, photo: store(finding.photo, 'before-0'), after: finding.after.map((p, i) => store(p, `after-${i}`)) };
      this.db.transaction(() => writeFinding(this.db, stored));
    } catch (error) {
      created.forEach(uri => this.photos.remove(uri));  // compensate copies made for this failed save
      throw error;
    }
    const exists = this.findings.some(item => item.id === stored.id);
    this.findings = exists
      ? this.findings.map(item => (item.id === stored.id ? stored : item))
      : [stored, ...this.findings];
    this.listeners.forEach(listener => listener());
  }

  /** Deletes managed photo files that no finding references (e.g. after a crash). Returns how many. */
  removeOrphanPhotos(): number {
    const referenced = new Set(this.findings.flatMap(f => [f.photo.uri, ...f.after.map(p => p.uri)]));
    const orphans = this.photos.listManaged().filter(uri => !referenced.has(uri));
    orphans.forEach(uri => this.photos.remove(uri));
    return orphans.length;
  }
}
