import type { AiAnalyzer } from '@/contracts/interfaces';
import type { InspectionTask, Photo, Suggestion } from '@/contracts/types';
import { isEquipmentTask, isFixedTask, isPcbTask } from '@/contracts/inspection';

export const FIXED_MODEL_SHA256 = {
  general: '5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e',
  pcb: '7a6c9070c437417ff4fb285a548913c9f272afea030fd8a1e00799204a892302',
} as const;

export const FIXED_PROMPT = {
  hazard: 'safelog-hazard-v1',
  comparison: 'safelog-comparison-v1',
  pcb: 'pcb-multisource-app-v1',
} as const;

const SHA256 = /^[a-f0-9]{64}$/;
const MAX_PHOTO_BYTES = 8 * 1024 * 1024;
const MAX_RESPONSE_CHARS = 32 * 1024 * 1024;

export interface FixedCpuOptions {
  task: InspectionTask;
  generalBaseUrl: string;
  pcbBaseUrl: string;
  readPhoto: (photo: Photo, signal: AbortSignal) => Promise<string>;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
  demoToken?: string;
}

export interface FixedCpuAnalyzer extends AiAnalyzer {
  checkConnection: (signal: AbortSignal) => Promise<void>;
}

type JsonObject = Record<string, unknown>;

function invalid(field: string): never {
  throw new Error('AI 응답의 형식이나 점검 항목이 올바르지 않습니다: ' + field);
}

function object(value: unknown, field: string): JsonObject {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(field);
  return value as JsonObject;
}

function text(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value.trim()) invalid(field);
  return value;
}

function number(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) invalid(field);
  return value;
}

function boolean(value: unknown, field: string): boolean {
  if (typeof value !== 'boolean') invalid(field);
  return value;
}

function strings(value: unknown, field: string): string[] {
  if (!Array.isArray(value)) invalid(field);
  return value.map(item => text(item, field));
}

function hash(value: unknown, field: string): string {
  const digest = text(value, field);
  if (!SHA256.test(digest)) invalid(field);
  return digest;
}

function cancelled() {
  return new Error('분석을 취소했습니다.');
}

function checkAbort(signal: AbortSignal) {
  if (signal.aborted) throw cancelled();
}

/** Only a deliberately configured local HTTP or HTTPS server may receive a photo. */
export function normalizeAiBaseUrl(value: string): string {
  let url: URL;
  try { url = new URL(value.trim()); } catch { throw new Error('AI 서버 주소가 올바르지 않습니다.'); }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error('AI 서버 주소에 계정, 쿼리 또는 fragment를 넣을 수 없습니다.');
  }
  const host = url.hostname.toLowerCase();
  const octets = host.split('.').map(Number);
  const ipv4 = octets.length === 4 && octets.every(octet => Number.isInteger(octet) && octet >= 0 && octet <= 255);
  const privateHost = host === 'localhost' || host === '[::1]' || (ipv4 && (
    octets[0] === 127 || octets[0] === 10 || (octets[0] === 192 && octets[1] === 168) ||
    (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31)
  ));
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && privateHost)) {
    throw new Error('AI 서버는 HTTPS 또는 개발용 로컬 HTTP 주소여야 합니다.');
  }
  return url.toString().replace(/\/+$/, '');
}

export function validatePhotoDataUrl(value: string): string {
  if (typeof value !== 'string' || value.length > 12_000_000) {
    throw new Error('AI 사진은 JPEG/PNG 8MiB 이하여야 합니다.');
  }
  const comma = value.indexOf(',');
  const prefix = value.slice(0, comma);
  const body = value.slice(comma + 1);
  if (comma < 0 || !['data:image/jpeg;base64', 'data:image/png;base64'].includes(prefix) ||
      !body || body.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(body)) {
    throw new Error('읽을 수 있는 JPEG/PNG 사진을 선택하세요. HEIC는 직접 전송할 수 없습니다.');
  }
  const padding = body.endsWith('==') ? 2 : body.endsWith('=') ? 1 : 0;
  if (body.length / 4 * 3 - padding > MAX_PHOTO_BYTES) {
    throw new Error('AI 사진은 JPEG/PNG 8MiB 이하여야 합니다.');
  }
  if ((prefix.includes('/jpeg;') && !body.startsWith('/9j/')) ||
      (prefix.includes('/png;') && !body.startsWith('iVBORw0KGgo'))) {
    throw new Error('사진의 실제 바이트와 JPEG/PNG 형식이 일치하지 않습니다.');
  }
  return value;
}

function validateTask(value: JsonObject, task: InspectionTask) {
  const equipment = isEquipmentTask(task);
  if (equipment ? value.inspectionTask !== 'equipment_surface' || value.equipmentCategory !== task
    : value.inspectionTask !== task) invalid('inspectionTask');
}

function validateIdentity(value: JsonObject, expectedSha: string, prompt: string) {
  if (hash(value.modelSha256, 'modelSha256') !== expectedSha) {
    throw new Error('AI 서버의 모델이 시연용 고정 모델과 일치하지 않습니다.');
  }
  if (value.promptVersion !== prompt) invalid('promptVersion');
  text(value.modelName, 'modelName');
}

function validateInput(value: JsonObject) {
  hash(value.inputSha256, 'inputSha256');
  for (const name of ['inputWidth', 'inputHeight']) {
    const dimension = number(value[name], name);
    if (!Number.isInteger(dimension) || dimension < 1) invalid(name);
  }
}

function generalHazard(value: JsonObject, rawJson: string, task: InspectionTask): Suggestion {
  validateIdentity(value, FIXED_MODEL_SHA256.general, FIXED_PROMPT.hazard);
  validateTask(value, task);
  validateInput(value);
  const candidate = boolean(value.candidateDetected, 'candidateDetected');
  if (!Array.isArray(value.classResults) || value.classResults.length === 0) invalid('classResults');
  const classes = value.classResults.map(entry => {
    const row = object(entry, 'classResults');
    text(row.className, 'className');
    const score = number(row.score, 'score');
    const threshold = number(row.threshold, 'threshold');
    const detected = boolean(row.candidateDetected, 'class candidateDetected');
    if (threshold < 0 || detected !== (score > threshold)) invalid('class score/threshold');
    return detected;
  });
  if (candidate !== classes.some(Boolean)) invalid('candidateDetected/classResults');
  const risk = number(value.riskLevel, 'riskLevel');
  const confidence = number(value.confidence, 'confidence');
  if (!Number.isInteger(risk) || risk < 1 || risk > 5 || confidence < 0 || confidence > 1) invalid('riskLevel/confidence');
  strings(value.detectedHazards, 'detectedHazards');
  if (value.humanConfirmationRequired !== true) invalid('humanConfirmationRequired');
  const description = text(value.suggestedDescription, 'suggestedDescription');
  return {
    source: 'fixed-cpu', inspectionTask: task, model: text(value.modelName, 'modelName'),
    modelSha256: FIXED_MODEL_SHA256.general, inputSha256: hash(value.inputSha256, 'inputSha256'),
    prompt: FIXED_PROMPT.hazard, rawJson, confidence, candidateDetected: candidate,
    category: text(value.hazardCategory, 'hazardCategory'), risk,
    assessment: description + (candidate ? '' : ' 후보 미검출은 현장의 안전을 보증하지 않습니다.'),
    action: text(value.suggestedAction, 'suggestedAction'),
  };
}

function pcbHazard(value: JsonObject, rawJson: string, task: InspectionTask): Suggestion {
  validateIdentity(value, FIXED_MODEL_SHA256.pcb, FIXED_PROMPT.pcb);
  validateInput(value);
  if (value.schema !== 'safelog.equipment.pcb-visual.v3' || value.equipmentCategory !== task ||
      value.sceneType !== 'registered_pcb_closeup' || value.scoreIsProbability !== false ||
      value.localizationReadyForApp !== false || value.heatmapIsConfirmedDefectLocalization !== false ||
      value.humanConfirmationRequired !== true) invalid('PCB schema/scope');
  const candidate = boolean(value.candidateDetected, 'candidateDetected');
  const score = number(value.anomalyScore, 'anomalyScore');
  const threshold = number(value.threshold, 'threshold');
  if (score < 0 || threshold <= 0 || candidate !== (score > threshold)) invalid('PCB score/threshold');
  return {
    source: 'fixed-cpu', inspectionTask: task, model: text(value.modelName, 'modelName'),
    modelSha256: FIXED_MODEL_SHA256.pcb, inputSha256: hash(value.inputSha256, 'inputSha256'),
    prompt: FIXED_PROMPT.pcb, rawJson, candidateDetected: candidate, anomalyScore: score, threshold,
    category: 'PCB 외관 후보',
    assessment: candidate ? '회로기판 외관 이상 후보가 있습니다. 실제 손상 여부를 직접 확인하세요.'
      : '회로기판 외관 이상 후보를 찾지 못했습니다. 후보 미검출은 정상 동작을 보증하지 않습니다.',
    action: '등록된 유형의 PCB 근접 사진에 대한 외관 참고 결과입니다. 내부 전기 고장과 확정 결함 위치는 판단하지 않습니다.',
  };
}

function comparison(value: JsonObject, rawJson: string, task: InspectionTask): Suggestion {
  validateIdentity(value, FIXED_MODEL_SHA256.general, FIXED_PROMPT.comparison);
  validateTask(value, task);
  const before = boolean(value.beforeCandidateDetected, 'beforeCandidateDetected');
  const after = boolean(value.afterCandidateDetected, 'afterCandidateDetected');
  const resolved = boolean(value.likelyResolved, 'likelyResolved');
  if (resolved !== (before && !after) || value.humanConfirmationRequired !== true ||
      value.comparisonMethod !== 'same_fixed_model_scores_not_trained_change_detector') invalid('comparison scope');
  strings(value.remainingRisks, 'remainingRisks');
  const confidence = number(value.confidence, 'confidence');
  if (confidence < 0 || confidence > 1) invalid('confidence');
  return {
    source: 'fixed-cpu', inspectionTask: task, model: text(value.modelName, 'modelName'),
    modelSha256: FIXED_MODEL_SHA256.general, prompt: FIXED_PROMPT.comparison, rawJson,
    candidateDetected: after, likelyResolved: resolved,
    assessment: text(value.assessment, 'assessment'),
    action: '동일 고정 모델의 전후 외관 후보 비교입니다. 점검자가 사진과 조치 내용을 직접 확인하고 최종 판단하세요.',
  };
}

export function createFixedCpuAnalyzer(options: FixedCpuOptions): FixedCpuAnalyzer {
  if (!isFixedTask(options.task)) throw new Error('지원하지 않는 AI 점검 분야입니다. 직접 검토를 사용하세요.');
  const generalBase = normalizeAiBaseUrl(options.generalBaseUrl);
  const pcbBase = normalizeAiBaseUrl(options.pcbBaseUrl);
  const fetcher = options.fetchImpl ?? fetch;
  const token = options.demoToken?.trim();
  if (token && !/^[A-Za-z0-9_-]{43,256}$/.test(token)) {
    throw new Error('시연 연결 토큰 형식을 확인하세요.');
  }
  if (token) {
    const endpoints = [new URL(generalBase), new URL(pcbBase)];
    if (endpoints[0].origin !== endpoints[1].origin || endpoints.some(url => url.protocol !== 'https:'
      && !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname))) {
      throw new Error('시연 토큰은 동일한 HTTPS 서버 또는 이 기기의 로컬 서버에만 전송할 수 있습니다.');
    }
  }
  const timeout = options.timeoutMs ?? 60_000;
  if (!Number.isInteger(timeout) || timeout < 1 || timeout > 300_000) {
    throw new Error('AI 제한 시간은 1~300000ms 범위여야 합니다.');
  }

  async function json(base: string, path: string, signal: AbortSignal, body?: JsonObject) {
    checkAbort(signal);
    let response: Response;
    try {
      response = await fetcher(base + path, {
        method: body ? 'POST' : 'GET', signal, redirect: 'error',
        headers: { Accept: 'application/json', ...(body ? { 'Content-Type': 'application/json' } : {}),
          ...(token ? { Authorization: 'Bearer ' + token } : {}) },
        ...(body ? { body: JSON.stringify(body) } : {}),
      });
    } catch {
      checkAbort(signal);
      throw new Error('AI 서버에 연결할 수 없습니다. 서버 주소와 연결을 확인하거나 직접 검토하세요.');
    }
    checkAbort(signal);
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        throw new Error('AI 서버 연결 인증에 실패했습니다. 내 계정에서 시연 주소와 토큰을 확인하세요.');
      }
      if (response.status === 413) throw new Error('사진이 너무 큽니다. JPEG/PNG 8MiB 이하 사진을 선택하세요.');
      if (response.status === 422) throw new Error('AI가 입력 사진이나 점검 조건을 처리할 수 없습니다. 사진을 확인하고 직접 검토하세요.');
      if (response.status === 503) throw new Error('고정 AI 모델 서버가 준비되지 않았습니다. 직접 검토하거나 연결 후 다시 시도하세요.');
      throw new Error('AI 서버 요청에 실패했습니다(HTTP ' + response.status + '). 직접 검토하거나 다시 시도하세요.');
    }
    const declaredSize = response.headers.get('content-length');
    if (declaredSize && Number(declaredSize) > MAX_RESPONSE_CHARS) invalid('response size');
    const raw = await response.text();
    checkAbort(signal);
    if (!raw || raw.length > MAX_RESPONSE_CHARS) invalid('response size');
    let value: unknown;
    try { value = JSON.parse(raw); } catch { throw new Error('AI 서버 응답을 읽을 수 없습니다. 직접 검토하거나 다시 시도하세요.'); }
    return { value: object(value, 'response'), raw };
  }

  async function health(signal: AbortSignal) {
    const pcb = isPcbTask(options.task);
    const { value } = await json(pcb ? pcbBase : generalBase, '/health', signal);
    if (value.ready !== true || value.neuralModelCount !== 1) {
      throw new Error('고정 AI 모델 서버가 준비되지 않았습니다.');
    }
    const digest = pcb
      ? object(object(value.models, 'models')[options.task], 'models.' + options.task).modelSha256
      : value.modelSha256;
    if (digest !== (pcb ? FIXED_MODEL_SHA256.pcb : FIXED_MODEL_SHA256.general)) {
      throw new Error('AI 서버의 모델이 시연용 고정 모델과 일치하지 않습니다.');
    }
  }

  async function run<T>(operation: (signal: AbortSignal) => Promise<T>, external: AbortSignal): Promise<T> {
    if (external.aborted) throw cancelled();
    const controller = new AbortController();
    let timedOut = false;
    let rejectAbort: (reason: Error) => void = () => undefined;
    const abortPromise = new Promise<never>((_resolve, reject) => { rejectAbort = reject; });
    const abort = () => {
      controller.abort();
      rejectAbort(timedOut ? new Error('AI 응답 시간이 초과되었습니다. 직접 검토하거나 다시 시도하세요.') : cancelled());
    };
    external.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(() => { timedOut = true; abort(); }, timeout);
    try {
      return await Promise.race([operation(controller.signal), abortPromise]);
    } finally {
      clearTimeout(timer);
      external.removeEventListener('abort', abort);
    }
  }

  async function image(photo: Photo, signal: AbortSignal) {
    checkAbort(signal);
    const encoded = await options.readPhoto(photo, signal);
    checkAbort(signal);
    return validatePhotoDataUrl(encoded);
  }

  return {
    checkConnection: signal => run(health, signal),
    hazard: (before, signal) => run(async active => {
      await health(active);
      const encoded = await image(before, active);
      const pcb = isPcbTask(options.task);
      const result = await json(pcb ? pcbBase : generalBase, pcb ? '/v1/equipment/pcb/analyze' : '/v1/analyze-hazard', active,
        pcb ? { image: encoded, equipmentCategory: options.task, sceneType: 'registered_pcb_closeup', promptVersion: FIXED_PROMPT.pcb }
          : { image: encoded, inspectionTask: options.task, promptVersion: FIXED_PROMPT.hazard });
      return pcb ? pcbHazard(result.value, result.raw, options.task) : generalHazard(result.value, result.raw, options.task);
    }, signal),
    compare: (before, after, signal) => run(async active => {
      if (isPcbTask(options.task)) {
        throw new Error('PCB 모델은 전후 자동 비교를 지원하지 않습니다. 전후 사진을 직접 검토하고 최종 판단을 기록하세요.');
      }
      await health(active);
      const beforeImage = await image(before, active);
      const afterImage = await image(after, active);
      const result = await json(generalBase, '/v1/compare-action', active, {
        beforeImage, afterImage, inspectionTask: options.task, promptVersion: FIXED_PROMPT.comparison,
      });
      return comparison(result.value, result.raw, options.task);
    }, signal),
  };
}
