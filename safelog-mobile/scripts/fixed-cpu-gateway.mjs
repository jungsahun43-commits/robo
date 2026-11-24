import { createServer as createHttpServer } from 'node:http';
import { createHash, timingSafeEqual } from 'node:crypto';
import { closeSync, fstatSync, openSync, readSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const DEFAULT_UPSTREAMS = Object.freeze({
  general: 'http://127.0.0.1:8775',
  pcb: 'http://127.0.0.1:8774',
});

export const DEFAULT_ALLOWED_ORIGINS = Object.freeze(
  ['8081', '8082', '19006'].flatMap(port => [
    `http://127.0.0.1:${port}`, `http://localhost:${port}`,
  ]),
);

const ROUTES = new Map([
  ['/general/health', { method: 'GET', upstream: 'general', path: '/health' }],
  ['/general/v1/equipment-categories', { method: 'GET', upstream: 'general', path: '/v1/equipment-categories' }],
  ['/general/v1/analyze-hazard', { method: 'POST', upstream: 'general', path: '/v1/analyze-hazard' }],
  ['/general/v1/compare-action', { method: 'POST', upstream: 'general', path: '/v1/compare-action' }],
  ['/pcb/health', { method: 'GET', upstream: 'pcb', path: '/health' }],
  ['/pcb/v1/equipment/pcb/analyze', { method: 'POST', upstream: 'pcb', path: '/v1/equipment/pcb/analyze' }],
]);

const TOKEN_PATTERN = /^[A-Za-z0-9_-]{43,256}$/;
const MAX_DEMO_LIFETIME_MS = 24 * 60 * 60 * 1000;

function demoAuthConfig(value) {
  if (value === undefined) return undefined;
  const expiresAt = typeof value?.expiresAt === 'string' ? Date.parse(value.expiresAt) : value?.expiresAt;
  if (!value || typeof value.token !== 'string' || !TOKEN_PATTERN.test(value.token) ||
      typeof expiresAt !== 'number' || !Number.isFinite(expiresAt)) {
    throw new TypeError('Demo authentication configuration is invalid.');
  }
  return { digest: createHash('sha256').update(value.token).digest(), expiresAt };
}

/** Read a short-lived token from a bounded local file, never from a command argument or public app variable. */
export function loadDemoAuthFromEnv(env = process.env, now = Date.now()) {
  const file = env.SAFELOG_DEMO_TOKEN_FILE;
  const expiry = env.SAFELOG_DEMO_TOKEN_EXPIRES_AT;
  if (file === undefined && expiry === undefined) return undefined;
  let descriptor;
  try {
    if (typeof file !== 'string' || !file || typeof expiry !== 'string' || !expiry) throw new Error();
    const expiresAt = Date.parse(expiry);
    if (!Number.isFinite(expiresAt) || expiresAt <= now || expiresAt > now + MAX_DEMO_LIFETIME_MS) throw new Error();
    descriptor = openSync(file, 'r');
    const stats = fstatSync(descriptor);
    if (!stats.isFile() || stats.size < 43 || stats.size > 258) throw new Error();
    const bytes = Buffer.alloc(259);
    const count = readSync(descriptor, bytes, 0, bytes.length, 0);
    if (count !== stats.size || count > 258) throw new Error();
    const token = new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(0, count)).replace(/\r?\n$/, '');
    demoAuthConfig({ token, expiresAt });
    return { token, expiresAt };
  } catch {
    throw new TypeError('Demo authentication configuration is invalid or unreadable.');
  } finally {
    if (descriptor !== undefined) closeSync(descriptor);
  }
}

function authenticated(request, auth) {
  if (!auth) return true;
  const authorization = request.headers.authorization;
  if (typeof authorization !== 'string' || Date.now() >= auth.expiresAt) return false;
  const match = /^Bearer ([A-Za-z0-9_-]{43,256})$/i.exec(authorization);
  if (!match) return false;
  const digest = createHash('sha256').update(match[1]).digest();
  return timingSafeEqual(digest, auth.digest);
}

class GatewayError extends Error {
  constructor(status, code, detail) {
    super(detail);
    this.status = status;
    this.code = code;
  }
}

function positiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 1) throw new TypeError(`${name} must be a positive integer.`);
  return value;
}

function loopbackUpstream(value) {
  const url = new URL(value);
  if (url.protocol !== 'http:' || !['127.0.0.1', '[::1]'].includes(url.hostname) ||
      url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new TypeError('Upstreams must be plain HTTP loopback origins without credentials, paths or queries.');
  }
  return url.origin;
}

function webOrigin(value) {
  const url = new URL(value);
  if (url.protocol !== 'http:' || !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) ||
      url.username || url.password || url.search || url.hash || url.pathname !== '/' ||
      value !== url.origin) {
    throw new TypeError('Web origins must be exact HTTP loopback origins, for example http://localhost:8081.');
  }
  return url.origin;
}

function jsonObject(bytes, status, code, detail) {
  try {
    const value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Not an object.');
    return value;
  } catch {
    throw new GatewayError(status, code, detail);
  }
}

function readRequest(request, limit, signal) {
  return new Promise((resolveBody, rejectBody) => {
    let size = 0;
    const chunks = [];
    const declaredLength = request.headers['content-length'];
    if (declaredLength !== undefined &&
        (!/^\d+$/.test(declaredLength) || Number(declaredLength) > limit)) {
      request.resume();
      rejectBody(new GatewayError(413, 'request_too_large', 'The JSON request exceeds the gateway byte limit.'));
      return;
    }

    function cleanup() {
      request.off('data', onData);
      request.off('end', onEnd);
      request.off('error', onError);
      request.off('aborted', onAborted);
      signal.removeEventListener('abort', onAborted);
    }
    function fail(error) {
      cleanup();
      // Drain without retaining the remaining upload, so a 413/504 JSON reply
      // can reach the caller rather than destroying its socket immediately.
      request.resume();
      rejectBody(error);
    }
    function onData(chunk) {
      size += chunk.length;
      if (size > limit) {
        fail(new GatewayError(413, 'request_too_large', 'The JSON request exceeds the gateway byte limit.'));
        return;
      }
      chunks.push(chunk);
    }
    function onEnd() {
      cleanup();
      resolveBody(Buffer.concat(chunks, size));
    }
    function onError() {
      fail(new GatewayError(400, 'request_interrupted', 'The JSON upload was interrupted.'));
    }
    function onAborted() {
      fail(signal.reason ?? new Error('The request was cancelled.'));
    }

    if (signal.aborted) {
      fail(signal.reason);
      return;
    }
    request.on('data', onData);
    request.once('end', onEnd);
    request.once('error', onError);
    request.once('aborted', onAborted);
    signal.addEventListener('abort', onAborted, { once: true });
  });
}

async function readResponse(response, limit) {
  const declaredLength = response.headers.get('content-length');
  if (declaredLength !== null && Number(declaredLength) > limit) {
    await response.body?.cancel();
    throw new GatewayError(502, 'upstream_too_large', 'The model response exceeds the gateway byte limit.');
  }
  if (!response.body) throw new GatewayError(502, 'invalid_upstream_json', 'The model returned an empty response.');
  const reader = response.body.getReader();
  let size = 0;
  const chunks = [];
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return Buffer.concat(chunks, size);
      size += value.byteLength;
      if (size > limit) {
        await reader.cancel();
        throw new GatewayError(502, 'upstream_too_large', 'The model response exceeds the gateway byte limit.');
      }
      chunks.push(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Construct an unbound server. The CLI below binds only to 127.0.0.1.
 * Upstream overrides exist for isolated loopback tests; the CLI always uses
 * the fixed CPU servers on 8775 and 8774.
 */
export function createServer({
  upstreams = DEFAULT_UPSTREAMS,
  allowedOrigins = DEFAULT_ALLOWED_ORIGINS,
  timeoutMs = 60_000,
  maxBodyBytes = 24 * 1024 * 1024,
  maxResponseBytes = 8 * 1024 * 1024,
  demoAuth,
} = {}) {
  positiveInteger(timeoutMs, 'timeoutMs');
  positiveInteger(maxBodyBytes, 'maxBodyBytes');
  positiveInteger(maxResponseBytes, 'maxResponseBytes');
  const targets = Object.fromEntries(Object.entries(DEFAULT_UPSTREAMS).map(([name, fallback]) =>
    [name, loopbackUpstream(upstreams[name] ?? fallback)]));
  const origins = new Set(allowedOrigins.map(webOrigin));
  const auth = demoAuthConfig(demoAuth);

  return createHttpServer({
    requestTimeout: timeoutMs + 1000,
    headersTimeout: Math.min(10_000, timeoutMs + 1000),
  }, async (request, response) => {
    // An interrupted IncomingMessage may emit error after its body listener
    // has been removed. Cancellation still aborts fetch below; this listener
    // prevents a disconnected upload from becoming an uncaught process error.
    request.on('error', () => {});
    const headers = { 'Cache-Control': 'no-store', Vary: 'Origin' };
    const sendError = (status, code, detail) => {
      if (response.destroyed || response.writableEnded) return;
      const body = JSON.stringify({ code, detail });
      response.writeHead(status, { ...headers, 'Content-Type': 'application/json; charset=utf-8' });
      response.end(body);
    };
    const origin = request.headers.origin;
    if (origin !== undefined) {
      if (!origins.has(origin)) {
        request.resume();
        sendError(403, 'origin_not_allowed', 'This browser origin is not allowed.');
        return;
      }
      headers['Access-Control-Allow-Origin'] = origin;
    }

    // Match the complete raw path. Queries, encoded traversal and arbitrary
    // URLs cannot turn this gateway into a general-purpose proxy.
    const route = ROUTES.get(request.url);
    if (!route) {
      request.resume();
      sendError(404, 'route_not_allowed', 'This gateway route does not exist.');
      return;
    }
    if (request.method === 'OPTIONS') {
      request.resume();
      const requestedHeaders = (request.headers['access-control-request-headers'] ?? '')
        .split(',').map(value => value.trim().toLowerCase()).filter(Boolean);
      if (!origin || request.headers['access-control-request-method'] !== route.method ||
          requestedHeaders.some(value => value !== 'content-type' && !(auth && value === 'authorization'))) {
        sendError(403, 'preflight_not_allowed', 'This browser preflight is not allowed.');
        return;
      }
      response.writeHead(204, {
        ...headers,
        'Access-Control-Allow-Methods': route.method,
        'Access-Control-Allow-Headers': auth ? 'content-type, authorization' : 'content-type',
      });
      response.end();
      return;
    }
    // Authenticate before reading or validating uploads and before any upstream request.
    // The bearer credential stays here: it is never forwarded to model servers or logged.
    if (!authenticated(request, auth)) {
      request.resume();
      headers['WWW-Authenticate'] = 'Bearer realm="safelog-demo"';
      sendError(401, 'demo_unauthorized', 'Demo access is missing, invalid or expired.');
      return;
    }
    if (request.method !== route.method) {
      request.resume();
      sendError(405, 'method_not_allowed', 'This method is not allowed on the gateway route.');
      return;
    }
    if (route.method === 'GET' &&
        (request.headers['transfer-encoding'] !== undefined || Number(request.headers['content-length'] ?? 0) > 0)) {
      request.resume();
      sendError(400, 'unexpected_request_body', 'Health and category requests must not contain a body.');
      return;
    }
    if (route.method === 'POST' &&
        !/^application\/json(?:\s*;|$)/i.test(request.headers['content-type'] ?? '')) {
      request.resume();
      sendError(415, 'json_required', 'Use application/json for model requests.');
      return;
    }

    const controller = new AbortController();
    const timeoutError = new GatewayError(504, 'model_timeout', 'The fixed CPU model did not respond before the deadline.');
    const deadline = setTimeout(() => controller.abort(timeoutError), timeoutMs);
    deadline.unref();
    const cancel = () => controller.abort(new Error('The client disconnected.'));
    const disconnect = () => { if (!response.writableEnded) cancel(); };
    request.once('aborted', cancel);
    response.once('close', disconnect);

    try {
      let body;
      if (route.method === 'POST') {
        body = await readRequest(request, maxBodyBytes, controller.signal);
        jsonObject(body, 400, 'invalid_request_json', 'The request must contain a valid JSON object.');
      }
      const upstream = await fetch(targets[route.upstream] + route.path, {
        method: route.method,
        headers: { Accept: 'application/json', ...(body ? { 'Content-Type': 'application/json' } : {}) },
        body,
        redirect: 'error',
        signal: controller.signal,
      });
      const bytes = await readResponse(upstream, maxResponseBytes);
      jsonObject(bytes, 502, 'invalid_upstream_json', 'The model response is not a valid JSON object.');
      if (controller.signal.aborted) throw controller.signal.reason;
      if (response.destroyed) return;
      // Preserve error statuses and model JSON exactly. An unavailable model
      // must never become an HTTP 200 "no candidate" result.
      response.writeHead(upstream.status, { ...headers, 'Content-Type': 'application/json; charset=utf-8' });
      response.end(bytes);
    } catch (error) {
      if (controller.signal.reason === timeoutError) error = timeoutError;
      if (error instanceof GatewayError) sendError(error.status, error.code, error.message);
      else sendError(502, 'model_unavailable', 'The fixed CPU model could not be reached or its response was interrupted.');
    } finally {
      clearTimeout(deadline);
      request.off('aborted', cancel);
      response.off('close', disconnect);
    }
  });
}

function main(args) {
  let port = 8780;
  let requireDemoAuth = false;
  const origins = [...DEFAULT_ALLOWED_ORIGINS];
  for (let index = 0; index < args.length; index += 1) {
    const name = args[index];
    if (name === '--port') {
      const value = args[++index];
      if (!/^\d+$/.test(value ?? '')) throw new TypeError('--port requires a number.');
      port = Number(value);
      if (port < 1 || port > 65535) throw new TypeError('--port must be between 1 and 65535.');
    } else if (name === '--web-origin') {
      origins.push(webOrigin(args[++index]));
    } else if (name === '--require-demo-auth') {
      requireDemoAuth = true;
    } else {
      throw new TypeError('Supported options are --port, --web-origin and --require-demo-auth. The bind address is always 127.0.0.1.');
    }
  }
  const demoAuth = loadDemoAuthFromEnv();
  if (requireDemoAuth && !demoAuth) throw new TypeError('Demo authentication must be configured before starting this gateway.');
  const server = createServer({ allowedOrigins: origins, demoAuth });
  server.once('error', error => {
    console.error(`Fixed CPU gateway could not start (${error.code ?? 'error'}).`);
    process.exitCode = 1;
  });
  server.listen(port, '127.0.0.1', () => {
    console.log(`SafeLog fixed CPU gateway: http://127.0.0.1:${port}`);
    console.log('General: /general -> 127.0.0.1:8775; PCB: /pcb -> 127.0.0.1:8774');
    if (demoAuth) console.log('Short-lived demo authentication is enabled.');
  });
  const close = () => {
    server.close();
    server.closeAllConnections();
  };
  process.once('SIGINT', close);
  process.once('SIGTERM', close);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
