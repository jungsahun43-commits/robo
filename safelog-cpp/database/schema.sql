PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sites (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  address TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('inspector', 'assignee', 'manager'))
);

CREATE TABLE IF NOT EXISTS inspections (
  id TEXT PRIMARY KEY,
  site_id TEXT NOT NULL REFERENCES sites(id),
  inspector_id TEXT NOT NULL REFERENCES profiles(id),
  inspected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS findings (
  id TEXT PRIMARY KEY,
  inspection_id TEXT NOT NULL REFERENCES inspections(id),
  location TEXT NOT NULL,
  description TEXT NOT NULL,
  action_opinion TEXT NOT NULL,
  assignee_id TEXT REFERENCES profiles(id),
  due_at TEXT,
  status TEXT NOT NULL CHECK (status IN ('open', 'in_progress', 'pending_review', 'verified')),
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS photos (
  id TEXT PRIMARY KEY,
  finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('before', 'after')),
  storage_path TEXT NOT NULL,
  captured_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_logs (
  id TEXT PRIMARY KEY,
  finding_id TEXT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
  actor_id TEXT NOT NULL REFERENCES profiles(id),
  action TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_findings_inspection ON findings(inspection_id);
CREATE INDEX IF NOT EXISTS idx_findings_assignee_status ON findings(assignee_id, status);
CREATE INDEX IF NOT EXISTS idx_photos_finding ON photos(finding_id);
CREATE INDEX IF NOT EXISTS idx_logs_finding ON action_logs(finding_id, created_at);
