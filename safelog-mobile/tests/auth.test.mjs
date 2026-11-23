// AuthService on both account stores (memory for web, SQLite via node:sqlite for the app).
import assert from 'node:assert/strict';
import { mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { MemoryAccountStore } from '../src/features/auth/account-store.ts';
import {
  AccountValidationError, AuthenticationError, AuthService, LOGIN_FAILED_MESSAGE, PermissionError,
} from '../src/features/auth/auth-service.ts';
import { DEMO_PASSWORD, demoAliases, profiles } from '../src/features/auth/demo.ts';
import { DEFAULT_SCRYPT_PARAMS, PasswordHasher } from '../src/features/auth/password.ts';
import { SqliteAccountStore } from '../src/repositories/sqlite/account-store.ts';
import { migrate } from '../src/repositories/sqlite/schema.ts';
import { openNodeDatabase } from './support/node-database.mjs';

const random = size => globalThis.crypto.getRandomValues(new Uint8Array(size));
const FAST = { logN: 6, r: 8, p: 1 };  // cheap cost for tests; the default is checked separately

class Clock {
  time = Date.parse('2026-11-23T09:00:00.000Z');
  now = () => new Date(this.time);
  advance(ms) { this.time += ms; }
}

function tempDir(t) {
  const dir = mkdtempSync(join(tmpdir(), 'safelog-auth-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
}

function openSqlite(path) {
  const db = openNodeDatabase(path);
  migrate(db);
  return { db, store: new SqliteAccountStore(db) };
}

const backends = {
  memory: () => ({ store: new MemoryAccountStore(), addProfile: (s, p) => s.addProfile(p), close() {} }),
  sqlite: () => {
    const dir = mkdtempSync(join(tmpdir(), 'safelog-auth-'));
    const { db, store } = openSqlite(join(dir, 'safelog.db'));
    return {
      store, db,
      close() {  // close before deleting: Windows refuses to delete an open database file
        db.close();
        rmSync(dir, { recursive: true, force: true });
      },
      addProfile: (_s, p) => db.run('INSERT INTO profiles (id, display_name, role) VALUES (?, ?, ?)', [p.id, p.name, p.role.toLowerCase()]),
    };
  },
};

let ids = 0;
function service(store, clock = new Clock()) {
  return new AuthService(store, new PasswordHasher(random, FAST), {
    now: clock.now, aliases: demoAliases, newProfileId: () => `profile-${++ids}`,
  });
}

async function failsWithGenericMessage(promise) {
  await assert.rejects(promise, error => error instanceof AuthenticationError && error.message === LOGIN_FAILED_MESSAGE);
}

for (const [name, makeBackend] of Object.entries(backends)) {
  test(`[${name}] demo login succeeds by id or alias; every failure has one message`, async t => {
    const backend = makeBackend(t);
    t.after(() => backend.close());
    const auth = service(backend.store);
    await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);

    const result = await auth.login('inspector-1', DEMO_PASSWORD);
    assert.deepEqual(result, { profile: profiles[0], mustChangePassword: false });
    assert.equal((await auth.login('inspector1', DEMO_PASSWORD)).profile.id, 'inspector-1');
    assert.equal((await auth.login('  MANAGER-1 ', DEMO_PASSWORD)).profile.role, 'Manager');

    await failsWithGenericMessage(auth.login('inspector-1', 'wrong-password'));
    await failsWithGenericMessage(auth.login('nobody', DEMO_PASSWORD));
    await failsWithGenericMessage(auth.login('x', DEMO_PASSWORD));
    await failsWithGenericMessage(auth.login('inspector-1', ''));
    await failsWithGenericMessage(auth.login('inspector-1', 'a'.repeat(129)));
    await failsWithGenericMessage(auth.login('inspector-1', 'DEMO1234'));
  });

  test(`[${name}] five failures lock for one minute; success resets the count`, async t => {
    const backend = makeBackend(t);
    t.after(() => backend.close());
    const clock = new Clock();
    const auth = service(backend.store, clock);
    await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);
    const record = () => backend.store.findCredentialByProfileId('assignee-1');

    for (let i = 0; i < 4; i++) await failsWithGenericMessage(auth.login('assignee-1', 'wrong-pass'));
    assert.equal(record().failedCount, 4);
    await failsWithGenericMessage(auth.login('assignee-1', 'wrong-pass'));
    assert.equal(record().lockedUntil, new Date(clock.time + 60_000).toISOString());
    assert.equal(record().failedCount, 0);

    const locked = record().lockedUntil;
    await failsWithGenericMessage(auth.login('assignee-1', DEMO_PASSWORD));  // correct password refused
    await failsWithGenericMessage(auth.login('assignee1', 'wrong-pass'));    // neither counts nor extends
    assert.equal(record().lockedUntil, locked);
    assert.equal(record().failedCount, 0);
    assert.equal((await auth.login('manager-1', DEMO_PASSWORD)).profile.id, 'manager-1');

    clock.advance(59_000);
    await failsWithGenericMessage(auth.login('assignee-1', DEMO_PASSWORD));
    clock.advance(2_000);
    assert.equal((await auth.login('assignee-1', DEMO_PASSWORD)).profile.id, 'assignee-1');
    assert.equal(record().lockedUntil, undefined);

    for (let i = 0; i < 4; i++) await failsWithGenericMessage(auth.login('assignee-1', 'wrong-pass'));
    await auth.login('assignee-1', DEMO_PASSWORD);
    for (let i = 0; i < 4; i++) await failsWithGenericMessage(auth.login('assignee-1', 'wrong-pass'));
    assert.equal((await auth.login('assignee-1', DEMO_PASSWORD)).profile.id, 'assignee-1');

    // Concurrent attempts are serialized: six parallel failures lock exactly once.
    await Promise.allSettled(Array.from({ length: 6 }, () => auth.login('assignee-1', 'wrong-pass')));
    assert.ok(record().lockedUntil);
    assert.equal(record().failedCount, 0);
  });

  test(`[${name}] only managers create accounts; new accounts must change their password`, async t => {
    const backend = makeBackend(t);
    t.after(() => backend.close());
    const auth = service(backend.store);
    await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);

    for (const actor of ['inspector-1', 'assignee-1', 'nobody']) {
      await assert.rejects(auth.createAccount(actor, '홍길동', 'Assignee', 'hong', 'temp-pass-1'), PermissionError);
    }
    assert.equal(backend.store.findCredentialByLoginId('hong'), undefined);

    const created = await auth.createAccount('manager-1', ' 홍길동 ', 'Assignee', 'Hong.GD', 'temp-pass-1');
    assert.deepEqual(backend.store.findProfile(created.id), { id: created.id, name: '홍길동', role: 'Assignee' });
    assert.equal(backend.store.findCredentialByProfileId(created.id).loginId, 'hong.gd');
    assert.deepEqual(await auth.login('hong.gd', 'temp-pass-1'), { profile: created, mustChangePassword: true });

    const invalid = [
      ['', 'empty-name', 'temp-pass-1'], ['가', 'bad id!', 'temp-pass-1'], ['가', 'ab', 'temp-pass-1'],
      ['가', 'short-pw', '1234567'], ['가', 'long-pw', 'a'.repeat(129)], ['가', 'inspector1', 'temp-pass-1'],
    ];
    for (const [displayName, loginId, password] of invalid) {
      await assert.rejects(auth.createAccount('manager-1', displayName, 'Inspector', loginId, password), AccountValidationError);
    }
    await auth.createAccount('manager-1', '한글', 'Inspector', 'korean-pw', '가나다라마바사아');  // 8 chars, 24 bytes
    assert.equal((await auth.login('korean-pw', '가나다라마바사아')).mustChangePassword, true);
  });

  test(`[${name}] duplicate login ids are rejected without leaving a profile behind`, async t => {
    const backend = makeBackend(t);
    t.after(() => backend.close());
    const auth = service(backend.store);
    await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);
    await assert.rejects(auth.createAccount('manager-1', '중복', 'Inspector', 'inspector-1', 'temp-pass-1'), AccountValidationError);
    await assert.rejects(auth.createAccount('manager-1', '중복', 'Inspector', ' Inspector-1 ', 'temp-pass-1'), AccountValidationError);

    const existing = backend.store.findCredentialByProfileId('inspector-1');
    const orphan = { ...existing, profileId: 'orphan', loginId: 'INSPECTOR-1' };
    assert.throws(() => backend.store.createAccount({ id: 'orphan', name: '고아', role: 'Inspector' }, orphan));
    assert.equal(backend.store.findProfile('orphan'), undefined);
    assert.throws(() => backend.store.addCredential({ ...existing, loginId: 'second-login' }));
    assert.equal(backend.store.findCredentialByLoginId('second-login'), undefined);
  });

  test(`[${name}] password change needs the current password and counts toward the lock`, async t => {
    const backend = makeBackend(t);
    t.after(() => backend.close());
    const auth = service(backend.store);
    await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);
    const person = await auth.createAccount('manager-1', '변경', 'Inspector', 'changer', 'temp-pass-1');

    await failsWithGenericMessage(auth.changePassword(person.id, 'wrong-current', 'new-pass-123'));
    assert.throws(() => auth.changePassword(person.id, 'temp-pass-1', 'short'), AccountValidationError);
    const before = backend.store.findCredentialByProfileId(person.id);
    await auth.changePassword(person.id, 'temp-pass-1', 'new-pass-123');
    const after = backend.store.findCredentialByProfileId(person.id);
    assert.notDeepEqual(after.salt, before.salt);
    assert.notEqual(after.passwordHash, before.passwordHash);
    assert.equal(after.mustChange, false);
    await failsWithGenericMessage(auth.login('changer', 'temp-pass-1'));
    assert.equal((await auth.login('changer', 'new-pass-123')).mustChangePassword, false);

    for (let i = 0; i < 5; i++) await failsWithGenericMessage(auth.changePassword(person.id, 'wrong-current', 'other-pass-1'));
    await failsWithGenericMessage(auth.login('changer', 'new-pass-123'));
    await failsWithGenericMessage(auth.changePassword('nobody', 'x', 'other-pass-1'));
  });

  test(`[${name}] demo seed links existing people, creates missing ones and never overwrites`, async t => {
    const backend = makeBackend(t);
    t.after(() => backend.close());
    const auth = service(backend.store);
    assert.throws(() => auth.seedDemoAccounts(profiles, 'short'), AccountValidationError);

    backend.addProfile(backend.store, { id: 'manager-1', name: '박관리(수정됨)', role: 'Manager' });
    await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);
    assert.equal(backend.store.findProfile('manager-1').name, '박관리(수정됨)');
    assert.deepEqual(backend.store.findProfile('inspector-1'), profiles[0]);

    await auth.seedDemoAccounts(profiles, 'another-pass-1');  // next start
    assert.equal((await auth.login('manager-1', DEMO_PASSWORD)).profile.id, 'manager-1');
    await failsWithGenericMessage(auth.login('manager-1', 'another-pass-1'));
    if (backend.db) {
      assert.equal(backend.db.get('SELECT COUNT(*) AS n FROM profiles').n, 3);
      assert.equal(backend.db.get('SELECT COUNT(*) AS n FROM credentials').n, 3);
    }
  });

  test(`[${name}] each user gets a random salt; hashes record their parameters`, async t => {
    const backend = makeBackend(t);
    t.after(() => backend.close());
    const auth = service(backend.store);
    await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);
    const a = await auth.createAccount('manager-1', 'A', 'Inspector', 'user-a', 'Same-Pass-123');
    const b = await auth.createAccount('manager-1', 'B', 'Inspector', 'user-b', 'Same-Pass-123');
    const ca = backend.store.findCredentialByProfileId(a.id);
    const cb = backend.store.findCredentialByProfileId(b.id);
    assert.equal(ca.salt.length, 16);
    assert.notDeepEqual(ca.salt, cb.salt);
    assert.notEqual(ca.passwordHash, cb.passwordHash);
    assert.match(ca.passwordHash, /^\$scrypt\$ln=6,r=8,p=1\$[A-Za-z0-9+/]{43}$/);
    assert.ok(!ca.passwordHash.includes('Same-Pass-123'));
  });
}

test('[sqlite] no plaintext password reaches the database files', async t => {
  const dir = tempDir(t);
  const { db, store } = openSqlite(join(dir, 'safelog.db'));
  const auth = service(store);
  await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);
  const secret = 'Unique-Plaintext-Marker-42';
  const person = await auth.createAccount('manager-1', '디스크', 'Inspector', 'disk', secret);
  await auth.changePassword(person.id, secret, secret + '-2');
  db.close();
  for (const file of readdirSync(dir)) {
    const bytes = readFileSync(join(dir, file)).toString('latin1');
    assert.ok(!bytes.includes(secret), file);
    assert.ok(!bytes.includes(DEMO_PASSWORD), file);
  }
});

test('[sqlite] counters, locks and changed passwords survive a restart', async t => {
  const path = join(tempDir(t), 'safelog.db');
  const clock = new Clock();
  let { db, store } = openSqlite(path);
  let auth = service(store, clock);
  await auth.seedDemoAccounts(profiles, DEMO_PASSWORD);
  const person = await auth.createAccount('manager-1', '재시작', 'Assignee', 'restart', 'temp-pass-1');
  await failsWithGenericMessage(auth.login('restart', 'wrong-pass'));
  await failsWithGenericMessage(auth.login('restart', 'wrong-pass'));
  db.close();

  ({ db, store } = openSqlite(path));
  auth = service(store, clock);
  assert.equal(store.findCredentialByProfileId(person.id).failedCount, 2);
  assert.equal(store.findCredentialByProfileId(person.id).mustChange, true);
  for (let i = 0; i < 3; i++) await failsWithGenericMessage(auth.login('restart', 'wrong-pass'));
  db.close();

  ({ db, store } = openSqlite(path));
  auth = service(store, clock);
  await failsWithGenericMessage(auth.login('restart', 'temp-pass-1'));  // still locked
  clock.advance(61_000);
  await auth.changePassword(person.id, 'temp-pass-1', 'final-pass-1');
  db.close();

  ({ db, store } = openSqlite(path));
  auth = service(store, clock);
  assert.deepEqual(await auth.login('restart', 'final-pass-1'),
    { profile: { id: person.id, name: '재시작', role: 'Assignee' }, mustChangePassword: false });
  db.close();
});

test('default scrypt cost (N=2^14, r=8) verifies and rejects corrupted hashes', async t => {
  const hasher = new PasswordHasher(random);
  assert.deepEqual(hasher.params, DEFAULT_SCRYPT_PARAMS);
  const started = Date.now();
  const { encoded, salt } = await hasher.hash('correct horse battery');
  t.diagnostic(`one default hash on this machine (Node/V8): ${Date.now() - started} ms`);
  assert.match(encoded, /^\$scrypt\$ln=14,r=8,p=1\$/);
  assert.equal(await hasher.verify('correct horse battery', salt, encoded), true);
  assert.equal(await hasher.verify('correct horse batterY', salt, encoded), false);

  const fast = new PasswordHasher(random, FAST).hash('pw-12345678');
  const { encoded: fastEncoded, salt: fastSalt } = await fast;
  assert.equal(await hasher.verify('pw-12345678', fastSalt, fastEncoded), true);  // parameters come from the string
  assert.equal(await hasher.verify('pw-12345678', fastSalt, 'garbage'), false);
  assert.equal(await hasher.verify('pw-12345678', fastSalt, '$scrypt$ln=30,r=8,p=1$AAAA'), false);
  assert.equal(await hasher.verify('pw-12345678', new Uint8Array(8), fastEncoded), false);
  assert.equal(await hasher.verify('pw-12345678', fastSalt, fastEncoded.slice(0, -1)), false);
});
