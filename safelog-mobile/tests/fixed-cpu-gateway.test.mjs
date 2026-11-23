import assert from 'node:assert/strict';
import { createServer as createHttpServer } from 'node:http';
import { once } from 'node:events';
import test from 'node:test';
import { createServer } from '../scripts/fixed-cpu-gateway.mjs';

const origin = 'http://localhost:8081';

async function listen(server, context) {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  context.after(async () => {
    const closed = new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    server.closeAllConnections();
    await closed;
  });
  return `http://127.0.0.1:${server.address().port}`;
}

async function fixture(context, handle, options = {}) {
  const calls = [];
  const upstream = await listen(createHttpServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = Buffer.concat(chunks).toString('utf8');
    calls.push({ path: request.url, method: request.method, body, headers: request.headers });
    await handle(request, response, body);
  }), context);
  const address = await listen(createServer({
    upstreams: { general: upstream, pcb: upstream },
    ...options,
  }), context);
  return { address, calls };
}

function json(response, value, status = 200) {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(value));
}

test('forwards only the fixed health and model paths, preserving raw model JSON and CORS', async context => {
  const health = { ready: true, neuralModelCount: 1, modelSha256: 'fixed-model-sha' };
  const { address, calls } = await fixture(context, (_request, response) => json(response, health));
  const result = await fetch(address + '/general/health', { headers: { Origin: origin } });
  assert.equal(result.status, 200);
  assert.deepEqual(await result.json(), health);
  assert.equal(result.headers.get('access-control-allow-origin'), origin);
  assert.equal(result.headers.get('access-control-allow-credentials'), null);
  assert.equal(result.headers.get('cache-control'), 'no-store');
  assert.equal(calls[0].path, '/health');

  const payload = { image: 'test-only-bytes', equipmentCategory: 'robustad_pcb' };
  const pcb = await fetch(address + '/pcb/v1/equipment/pcb/analyze', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Origin: origin }, body: JSON.stringify(payload),
  });
  assert.equal(pcb.status, 200);
  assert.equal(calls[1].path, '/v1/equipment/pcb/analyze');
  assert.deepEqual(JSON.parse(calls[1].body), payload);
  assert.equal(calls[1].headers.origin, undefined);
  assert.equal(calls[1].headers.cookie, undefined);
});

test('blocks unlisted origins, routes, query strings and methods without contacting a model', async context => {
  const { address, calls } = await fixture(context, (_request, response) => json(response, {}));
  for (const [path, init, status] of [
    ['/general/health', { headers: { Origin: 'https://example.org' } }, 403],
    ['/general/health', { headers: { Origin: 'null' } }, 403],
    ['/general/health?url=http://example.org', {}, 404],
    ['/pcb/v1/analyze-hazard', {}, 404],
    ['/general/admin/shutdown', {}, 404],
    ['/general/health', { method: 'POST', body: '{}' }, 405],
  ]) {
    const result = await fetch(address + path, init);
    assert.equal(result.status, status, path);
    assert.equal(result.headers.get('access-control-allow-credentials'), null);
    assert.ok((await result.json()).code);
  }
  assert.equal(calls.length, 0);
});

test('allows explicit JSON browser preflight, but not credentials or an unrelated method', async context => {
  const { address, calls } = await fixture(context, (_request, response) => json(response, {}));
  const headers = { Origin: origin, 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type' };
  const result = await fetch(address + '/general/v1/analyze-hazard', { method: 'OPTIONS', headers });
  assert.equal(result.status, 204);
  assert.equal(result.headers.get('access-control-allow-origin'), origin);
  assert.equal(result.headers.get('access-control-allow-methods'), 'POST');
  assert.equal(result.headers.get('access-control-allow-headers'), 'content-type');
  const credential = await fetch(address + '/general/v1/analyze-hazard', {
    method: 'OPTIONS', headers: { ...headers, 'Access-Control-Request-Headers': 'authorization' },
  });
  assert.equal(credential.status, 403);
  const unrelated = await fetch(address + '/general/v1/analyze-hazard', {
    method: 'OPTIONS', headers: { ...headers, 'Access-Control-Request-Method': 'DELETE' },
  });
  assert.equal(unrelated.status, 403);
  assert.equal(calls.length, 0);
});

test('rejects malformed, non-object, oversized and non-JSON uploads before forwarding', async context => {
  const { address, calls } = await fixture(context, (_request, response) => json(response, {}), { maxBodyBytes: 64 });
  for (const [body, contentType, expected] of [
    ['{broken', 'application/json', 400],
    ['[]', 'application/json', 400],
    ['null', 'application/json', 400],
    [JSON.stringify({ image: 'x'.repeat(80) }), 'application/json', 413],
    ['{}', 'text/plain', 415],
  ]) {
    const result = await fetch(address + '/general/v1/analyze-hazard', {
      method: 'POST', headers: { 'Content-Type': contentType, Origin: origin }, body,
    });
    assert.equal(result.status, expected);
    assert.equal(result.headers.get('access-control-allow-origin'), origin);
    assert.ok((await result.json()).code);
  }
  assert.equal(calls.length, 0);
});

test('preserves an unavailable model error instead of reporting no candidate', async context => {
  const failure = { detail: 'Model is not ready.' };
  const { address } = await fixture(context, (_request, response) => json(response, failure, 503));
  const result = await fetch(address + '/general/health');
  assert.equal(result.status, 503);
  assert.deepEqual(await result.json(), failure);
});

test('rejects malformed and oversized upstream responses', async context => {
  const { address } = await fixture(context, (request, response) => {
    if (request.url === '/health') response.end('not-json');
    else json(response, { value: 'x'.repeat(80) });
  }, { maxResponseBytes: 64 });
  const invalid = await fetch(address + '/general/health');
  assert.equal(invalid.status, 502);
  assert.equal((await invalid.json()).code, 'invalid_upstream_json');
  const oversized = await fetch(address + '/pcb/v1/equipment/pcb/analyze', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
  });
  assert.equal(oversized.status, 502);
  assert.equal((await oversized.json()).code, 'upstream_too_large');
});

test('enforces the model deadline without changing it to a normal result', async context => {
  const { address } = await fixture(context, () => {}, { timeoutMs: 50 });
  const result = await fetch(address + '/general/health', { headers: { Origin: origin } });
  assert.equal(result.status, 504);
  assert.equal(result.headers.get('access-control-allow-origin'), origin);
  assert.equal((await result.json()).code, 'model_timeout');
});

test('aborts an upstream request when the caller cancels', async context => {
  let started;
  let disconnected;
  const upstreamStarted = new Promise(resolve => { started = resolve; });
  const upstreamDisconnected = new Promise(resolve => { disconnected = resolve; });
  const { address } = await fixture(context, (_request, response) => {
    response.once('close', disconnected);
    started();
  });
  const controller = new AbortController();
  const request = fetch(address + '/general/health', { signal: controller.signal });
  await upstreamStarted;
  controller.abort();
  await assert.rejects(request, { name: 'AbortError' });
  let deadline;
  try {
    await Promise.race([
      upstreamDisconnected,
      new Promise((_, reject) => { deadline = setTimeout(() => reject(new Error('Upstream was not cancelled.')), 1500); }),
    ]);
  } finally {
    clearTimeout(deadline);
  }
});

test('factory rejects remote upstreams, wildcard origins and invalid resource budgets', () => {
  assert.throws(() => createServer({ upstreams: { general: 'http://example.org' } }), /loopback/);
  assert.throws(() => createServer({ upstreams: { pcb: 'http://127.0.0.1:8774/private' } }), /loopback/);
  assert.throws(() => createServer({ allowedOrigins: ['*'] }));
  assert.throws(() => createServer({ allowedOrigins: ['http://192.168.1.2:8081'] }), /loopback/);
  assert.throws(() => createServer({ allowedOrigins: ['http://localhost:8081/'] }), /exact/);
  assert.throws(() => createServer({ timeoutMs: 0 }), /positive/);
  assert.throws(() => createServer({ maxBodyBytes: Infinity }), /positive/);
  assert.throws(() => createServer({ maxResponseBytes: -1 }), /positive/);
});
