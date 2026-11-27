import { isFixedTask } from '@/contracts/inspection';
import type { Photo } from '@/contracts/types';
import type { AiConnection } from './connection-store';
import { normalizeAiBaseUrl, validatePhotoDataUrl } from './fixed-cpu';

interface PhotoRouteOptions {
  connection: AiConnection;
  readPhoto: (photo: Photo, signal: AbortSignal) => Promise<string>;
  fetchImpl?: typeof fetch;
  signal: AbortSignal;
}

export async function routePhoto(photo: Photo, { connection, readPhoto, fetchImpl = fetch, signal }: PhotoRouteOptions) {
  const general = new URL(normalizeAiBaseUrl(connection.generalBaseUrl));
  const pcb = new URL(normalizeAiBaseUrl(connection.pcbBaseUrl));
  if (general.origin !== pcb.origin || general.pathname !== '/general' || pcb.pathname !== '/pcb') {
    throw new Error('자동 판별을 사용하려면 내 계정에서 두 모델을 연결하는 서버 주소를 적용하세요.');
  }
  const token = connection.demoToken;
  const loopback = general.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(general.hostname);
  if (token ? general.protocol !== 'https:' || !/^[A-Za-z0-9_-]{43,256}$/.test(token) : !loopback) {
    throw new Error('내 계정에서 HTTPS 서버 주소와 시연 토큰을 적용하세요.');
  }

  if (signal.aborted) throw new Error('사진 분석을 취소했습니다.');
  const controller = new AbortController();
  const abort = () => controller.abort();
  const timeout = setTimeout(abort, 75_000);
  signal.addEventListener('abort', abort, { once: true });
  try {
    const encoded = validatePhotoDataUrl(await readPhoto(photo, controller.signal));
    if (controller.signal.aborted) throw new Error('사진 분석을 취소했습니다.');
    const separator = encoded.indexOf(',');
    const mimeType = encoded.slice(5, encoded.indexOf(';'));
    let response: Response;
    try {
      response = await fetchImpl(new URL('/classify', general.origin).href, {
        method: 'POST', redirect: 'error', signal: controller.signal,
        headers: { 'Content-Type': 'application/json', Accept: 'application/json',
          ...(token ? { Authorization: 'Bearer ' + token } : {}) },
        body: JSON.stringify({ imageBase64: encoded.slice(separator + 1), mimeType }),
      });
    } catch {
      throw new Error('사진 자동 판별 서버에 연결할 수 없습니다.');
    }
    if (response.status === 401 || response.status === 403) {
      throw new Error('자동 판별 인증에 실패했습니다. 내 계정에서 서버 주소와 토큰을 확인하세요.');
    }
    if (response.status === 404 || response.status === 503) throw new Error('연결된 서버에서 사진 자동 판별을 할 수 없습니다.');
    if (!response.ok) throw new Error('사진의 점검 대상을 판별하지 못했습니다.');
    const declaredLength = Number(response.headers.get('content-length'));
    if (declaredLength > 4096) throw new Error('사진 자동 판별 응답 형식이 올바르지 않습니다.');
    const raw = await response.text();
    if (!raw || raw.length > 4096) throw new Error('사진 자동 판별 응답 형식이 올바르지 않습니다.');
    const value: unknown = JSON.parse(raw);
    if (typeof value === 'object' && value !== null && 'task' in value) {
      if (value.task === null) return null;
      if (isFixedTask(value.task)) return value.task;
    }
    throw new Error('사진 자동 판별 응답에 지원되는 점검 분야가 없습니다.');
  } catch (error) {
    if (!signal.aborted && controller.signal.aborted) throw new Error('사진 자동 판별 시간이 초과됐습니다.');
    if (error instanceof SyntaxError) throw new Error('사진 자동 판별 응답을 읽을 수 없습니다.');
    throw error;
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener('abort', abort);
  }
}
