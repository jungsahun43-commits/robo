// Demo data reset on SQLite (node:sqlite, fake photo files) and on the web memory stores.
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { AuthenticationError, AuthService, PermissionError } from '../src/features/auth/auth-service.ts';
import { DEMO_PASSWORD, demoAliases, profiles, site } from '../src/features/auth/demo.ts';
import { PasswordHasher } from '../src/features/auth/password.ts';
import { WorkflowService } from '../src/features/workflow/service.ts';
import { createAppStores as createWebStores } from '../src/repositories/app-stores.web.ts';
import { SqliteAccountStore } from '../src/repositories/sqlite/account-store.ts';
import { SqliteDemoReset } from '../src/repositories/sqlite/demo-reset.ts';
import { ensureSite, SqliteFindingRepository } from '../src/repositories/sqlite/finding-repository.ts';
import { migrate } from '../src/repositories/sqlite/schema.ts';
import { openNodeDatabase } from './support/node-database.mjs';

const random = size => globalThis.crypto.getRandomValues(new Uint8Array(size));
const FAST = { logN: 6, r: 8, p: 1 };
const DEMO_IDS = profiles.map(person => person.id).sort();

/** Stand-in for AppPhotoStore: "copies" picker URIs into app://photos/. */
class FakePhotoStore {
  files = new Set();
  import(uri, findingId, name) {
    if (uri.startsWith('app://photos/')) return { uri, created: false };
    const copy = `app://photos/${findingId}/${name}-${this.files.size}.jpg`;
    this.files.add(copy);
    return { uri: copy, created: true };
  }
  remove(uri) { this.files.delete(uri); }
  listManaged() { return [...this.files]; }
}

function openSqlite(path, photos) {
  const db = openNodeDatabase(path);
  migrate(db);
  ensureSite(db, site);
  const findings = new SqliteFindingRepository(db, photos);
  return { db, findings, accounts: new SqliteAccountStore(db), demo: new SqliteDemoReset(db, findings) };
}

const backends = {
  sqlite: () => {
    const dir = mkdtempSync(join(tmpdir(), 'safelog-reset-'));
    const path = join(dir, 'safelog.db');
    const photos = new FakePhotoStore();
    const stores = openSqlite(path, photos);
    return {
      ...stores, photos,
      reopen() { stores.db.close(); return openSqlite(path, photos); },
      count: table => stores.db.get(`SELECT COUNT(*) AS n FROM ${table}`).n,
      close() { try { stores.db.close(); } catch { /* already closed by reopen */ } rmSync(dir, { recursive: true, force: true }); },
    };
  },
  memory: () => ({ ...createWebStores(), close() {} }),
};

const auth = stores => new AuthService(stores.accounts, new PasswordHasher(random, FAST),
  { aliases: demoAliases, newProfileId: (() => { let n = 0; return () => `profile-${++n}`; })() });

const failsLogin = (service, id, password) =>
  assert.rejects(service.login(id, password), AuthenticationError);

/** Demo people, an extra assignee, findings with photos/logs/AI, a locked and a changed demo login. */
async function fillWithTestData(stores) {
  const service = auth(stores);
  await service.seedDemoAccounts(profiles, DEMO_PASSWORD);
  const extra = await service.createAccount('manager-1', '임시 담당', 'Assignee', 'temp-assignee', 'temp-pass-1');
  const [inspector, manager] = profiles;
  const workflow = new WorkflowService(stores.findings, stores.accounts);
  for (const location of ['1층', '2층']) {
    const { id } = workflow.create(inspector, {
      photo: { uri: `file:///picker/${location}.jpg`, mimeType: 'image/jpeg' }, location, description: '적치', action: '정리',
    });
    workflow.execute(manager, id, { kind: 'assign', assigneeId: extra.id });
    workflow.execute(inspector, id, { kind: 'suggest', type: 'BeforeHazard', suggestion: {
      model: 'mock', prompt: 'hazard-v1', confidence: 0.8, assessment: '위험', action: '정리', risk: 3 } });
  }
  await service.changePassword('inspector-1', DEMO_PASSWORD, 'changed-pass-1');
  for (let i = 0; i < 5; i++) await failsLogin(service, 'assignee-1', 'wrong-pass');  // locked now
  return { extra };
}

for (const [name, makeBackend] of Object.entries(backends)) {
  test(`[${name}] a manager's reset leaves no records, only the three demo accounts, and no photo files`, async t => {
    const stores = makeBackend();
    t.after(() => stores.close());
    const { extra } = await fillWithTestData(stores);
    assert.equal(stores.findings.snapshot().length, 2);
    if (stores.photos) assert.equal(stores.photos.listManaged().length, 2);

    let notified = 0;
    stores.findings.subscribe(() => { notified++; });
    await auth(stores).resetDemoData('manager-1', profiles, DEMO_PASSWORD, stores.demo);

    assert.deepEqual(stores.findings.snapshot(), []);  // screens see an empty list right away
    assert.ok(notified >= 1);
    assert.deepEqual(stores.accounts.listProfiles().map(p => p.id).sort(), DEMO_IDS);
    assert.equal(stores.accounts.findProfile(extra.id), undefined);
    assert.equal(stores.accounts.findCredentialByLoginId('temp-assignee'), undefined);
    if (stores.photos) assert.deepEqual(stores.photos.listManaged(), []);
    if (stores.count) {
      for (const table of ['findings', 'photos', 'action_logs', 'ai_analyses', 'inspections']) {
        assert.equal(stores.count(table), 0, table);
      }
      assert.equal(stores.count('profiles'), 3);
      assert.equal(stores.count('credentials'), 3);
      assert.equal(stores.count('sites'), 1);
    }

    // Demo logins: password back to demo1234, unlocked, no forced change.
    const service = auth(stores);
    for (const person of profiles) {
      const record = stores.accounts.findCredentialByProfileId(person.id);
      assert.equal(record.failedCount, 0);
      assert.equal(record.lockedUntil, undefined);
      assert.equal(record.mustChange, false);
      assert.deepEqual(await service.login(person.id, DEMO_PASSWORD), { profile: person, mustChangePassword: false });
    }
    await failsLogin(service, 'inspector-1', 'changed-pass-1');
    await failsLogin(service, 'temp-assignee', 'temp-pass-1');

    if (stores.reopen) {  // the reset is persisted
      const reopened = stores.reopen();
      assert.deepEqual(reopened.findings.snapshot(), []);
      assert.deepEqual(reopened.accounts.listProfiles().map(p => p.id).sort(), DEMO_IDS);
      reopened.db.close();
    }
  });

  test(`[${name}] only managers can reset; a refused reset changes nothing`, async t => {
    const stores = makeBackend();
    t.after(() => stores.close());
    const { extra } = await fillWithTestData(stores);
    for (const actor of ['inspector-1', 'assignee-1', extra.id, 'nobody']) {
      await assert.rejects(auth(stores).resetDemoData(actor, profiles, DEMO_PASSWORD, stores.demo), PermissionError);
    }
    assert.equal(stores.findings.snapshot().length, 2);
    assert.ok(stores.accounts.findProfile(extra.id));
    if (stores.photos) assert.equal(stores.photos.listManaged().length, 2);
    assert.ok(stores.accounts.findCredentialByProfileId('assignee-1').lockedUntil);
  });
}

test('[sqlite] a reset that fails inside the transaction keeps every row and photo file', async t => {
  const stores = backends.sqlite();
  t.after(() => stores.close());
  await fillWithTestData(stores);
  const broken = { resetDemoData: (keep, credentials) =>
    stores.demo.resetDemoData(keep, credentials.map(c => ({ ...c, salt: new Uint8Array(3) }))) };  // violates CHECK
  await assert.rejects(auth(stores).resetDemoData('manager-1', profiles, DEMO_PASSWORD, broken), /constraint/i);
  assert.equal(stores.findings.snapshot().length, 2);
  assert.equal(stores.count('findings'), 2);
  assert.equal(stores.count('profiles'), 4);
  assert.equal(stores.photos.listManaged().length, 2);
});
