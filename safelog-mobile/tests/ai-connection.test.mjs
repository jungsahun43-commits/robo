import assert from 'node:assert/strict';
import test from 'node:test';
import { createAiConnectionStore } from '../src/features/ai/connection-store.ts';
import { createFixedCpuAnalyzer, FIXED_MODEL_SHA256 } from '../src/features/ai/fixed-cpu.ts';

const token = 'x'.repeat(43); // Isolated protocol value; never used by the running server.
const defaults = { generalBaseUrl: 'http://127.0.0.1:8780/general', pcbBaseUrl: 'http://127.0.0.1:8780/pcb' };
const reply = value => new Response(JSON.stringify(value), { status: 200 });
const signal = () => new AbortController().signal;

test('binds a runtime token to one HTTPS gateway and resets without persisting it', () => {
  const store = createAiConnectionStore(defaults);
  const initial = store.getSnapshot();
  assert.equal(initial, store.getSnapshot());
  let notifications = 0;
  const unsubscribe = store.subscribe(() => { notifications++; });
  store.configure('https://demo.example.test/', token);
  const active = store.getSnapshot();
  assert.equal(active.generalBaseUrl, 'https://demo.example.test/general');
  assert.equal(active.pcbBaseUrl, 'https://demo.example.test/pcb');
  assert.equal(active.demoToken, token);
  assert.equal(active.revision, 1);
  assert.ok(Object.isFrozen(active));
  assert.equal(createAiConnectionStore(defaults).getSnapshot().demoToken, undefined);
  store.reset();
  assert.deepEqual(store.getSnapshot(), { ...defaults, revision: 2 });
  assert.equal(notifications, 2);
  unsubscribe(); store.reset(); assert.equal(notifications, 2);
});

test('rejects non-HTTPS, credential/query/path URLs and malformed tokens atomically', () => {
  const store = createAiConnectionStore(defaults);
  const initial = store.getSnapshot();
  for (const url of ['http://192.168.1.2:8780', 'https://user:pass@example.test', 'https://example.test/general',
    'https://example.test/?token=hidden', 'https://example.test/#fragment', 'not-a-url']) {
    assert.throws(() => store.configure(url, token));
    assert.equal(store.getSnapshot(), initial);
  }
  for (const bad of ['', 'x'.repeat(42), 'x'.repeat(257), token + '\nheader:value']) {
    assert.throws(() => store.configure('https://demo.example.test', bad));
    assert.equal(store.getSnapshot(), initial);
  }
});

test('sends authentication to both pinned-model health routes without reading a photo', async () => {
  const store = createAiConnectionStore(defaults);
  store.configure('https://demo.example.test', token);
  const calls = [];
  for (const task of ['facility_surface', 'robustad_pcb']) {
    const analyzer = createFixedCpuAnalyzer({ ...store.getSnapshot(), task,
      readPhoto: async () => { assert.fail('Connection check must not read any photo'); },
      fetchImpl: async (url, options) => {
        calls.push({ url, options });
        return reply(task === 'facility_surface'
          ? { ready: true, neuralModelCount: 1, modelSha256: FIXED_MODEL_SHA256.general }
          : { ready: true, neuralModelCount: 1, models: { robustad_pcb: { modelSha256: FIXED_MODEL_SHA256.pcb } } });
      } });
    await analyzer.checkConnection(signal());
  }
  assert.deepEqual(calls.map(c => c.url), ['https://demo.example.test/general/health', 'https://demo.example.test/pcb/health']);
  for (const call of calls) {
    assert.equal(call.options.method, 'GET');
    assert.equal(call.options.headers.Authorization, 'Bearer ' + token);
    assert.equal(call.options.redirect, 'error');
    assert.equal(call.options.body, undefined);
  }
});

test('refuses sending a demo secret to plaintext LAN or unrelated origins', () => {
  const options = { task: 'facility_surface', demoToken: token, readPhoto: async () => '' };
  for (const pair of [
    ['http://192.168.1.2/general', 'http://192.168.1.2/pcb'],
    ['https://one.example.test/general', 'https://two.example.test/pcb'],
  ]) assert.throws(() => createFixedCpuAnalyzer({ ...options, generalBaseUrl: pair[0], pcbBaseUrl: pair[1] }), /동일한 HTTPS/);
});

test('does not read or send a photo when authentication fails and does not echo the token', async () => {
  let requests = 0;
  const analyzer = createFixedCpuAnalyzer({ task: 'facility_surface', ...defaults, demoToken: token,
    readPhoto: async () => { assert.fail('401 must stop before photo read'); },
    fetchImpl: async () => { requests++; return new Response(token, { status: 401 }); } });
  await assert.rejects(analyzer.hazard({ uri: 'file:///not-read.jpg', mimeType: 'image/jpeg' }, signal()), error => {
    assert.match(error.message, /인증에 실패/);
    assert.ok(!error.message.includes(token));
    return true;
  });
  assert.equal(requests, 1);
});

test('connection checks still reject another model SHA and can be cancelled', async () => {
  const analyzer = createFixedCpuAnalyzer({ task: 'facility_surface', ...defaults,
    readPhoto: async () => '', fetchImpl: async () => reply({ ready: true, neuralModelCount: 1, modelSha256: '0'.repeat(64) }) });
  await assert.rejects(analyzer.checkConnection(signal()), /고정 모델과 일치하지 않습니다/);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(analyzer.checkConnection(controller.signal), /취소/);
});
