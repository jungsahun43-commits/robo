import assert from 'node:assert/strict';
import test from 'node:test';
import {
  createFixedCpuAnalyzer, FIXED_MODEL_SHA256, FIXED_PROMPT, normalizeAiBaseUrl, validatePhotoDataUrl,
} from '../src/features/ai/fixed-cpu.ts';
import { fixedTaskIds, equipmentTaskIds, pcbTaskIds } from '../src/contracts/inspection.ts';
import { demoTasks, inspectionTasks, researchTopTasks } from '../src/features/ai/tasks.ts';

// Protocol fixtures only: these bytes and responses are never sent to a model.
const encoded = 'data:image/jpeg;base64,/9j/2Q==';
const photo = { uri: 'file:///test-photo.jpg', mimeType: 'image/jpeg' };
const signal = () => new AbortController().signal;
const inputSha = 'a'.repeat(64);
const reply = value => new Response(JSON.stringify(value), { status: 200, headers: { 'content-type': 'application/json' } });
const ready = () => ({ ready: true, neuralModelCount: 1, modelSha256: FIXED_MODEL_SHA256.general });

function hazard(task = 'facility_surface', candidate = false) {
  const equipment = equipmentTaskIds.includes(task);
  return {
    schema: 'safelog.unified-main.v1',
    modelName: 'fixed-research-model', modelSha256: FIXED_MODEL_SHA256.general,
    inspectionTask: equipment ? 'equipment_surface' : task,
    ...(equipment ? { equipmentCategory: task } : {}),
    promptVersion: FIXED_PROMPT.hazard, inputSha256: inputSha, inputWidth: 640, inputHeight: 480,
    classResults: [{ className: 'test_appearance', score: candidate ? 0.9 : 0.1, threshold: 0.5, candidateDetected: candidate }],
    candidateDetected: candidate, hazardCategory: candidate ? '외관 이상 후보' : '외관 이상 후보 미검출',
    riskLevel: candidate ? 3 : 1, detectedHazards: candidate ? ['외관 이상 후보'] : [],
    suggestedDescription: candidate ? '사진에 외관 이상 후보가 있습니다.' : '외관 후보를 찾지 못했습니다.',
    suggestedAction: '점검자가 직접 확인하세요.', confidence: candidate ? 0.9 : 0.1,
    humanConfirmationRequired: true,
  };
}

function pcb(candidate = true, task = 'robustad_pcb') {
  return {
    schema: 'safelog.equipment.pcb-visual.v3', equipmentCategory: task, sceneType: 'registered_pcb_closeup',
    modelName: 'pcb-multisource-v3', modelSha256: FIXED_MODEL_SHA256.pcb, promptVersion: FIXED_PROMPT.pcb,
    inputSha256: inputSha, inputWidth: 640, inputHeight: 480, candidateDetected: candidate,
    anomalyScore: candidate ? 0.9 : 0.1, threshold: 0.4, scoreIsProbability: false,
    localizationReadyForApp: false, heatmapIsConfirmedDefectLocalization: false, humanConfirmationRequired: true,
  };
}

function comparison(task = 'conveyor_foreign') {
  const equipment = equipmentTaskIds.includes(task);
  return {
    modelName: 'fixed-research-model', modelSha256: FIXED_MODEL_SHA256.general,
    inspectionTask: equipment ? 'equipment_surface' : task, ...(equipment ? { equipmentCategory: task } : {}), promptVersion: FIXED_PROMPT.comparison,
    beforeCandidateDetected: true, afterCandidateDetected: false, likelyResolved: true,
    remainingRisks: [], assessment: '후보 변화를 사람이 확인하세요.', confidence: 0.5,
    humanConfirmationRequired: true, comparisonMethod: 'same_fixed_model_scores_not_trained_change_detector',
  };
}

function fixture(task = 'facility_surface', result = hazard(task)) {
  const calls = [];
  const analyzer = createFixedCpuAnalyzer({
    task, generalBaseUrl: 'http://127.0.0.1:8780/general/', pcbBaseUrl: 'http://127.0.0.1:8780/pcb/',
    readPhoto: async () => encoded,
    fetchImpl: async (url, options) => {
      calls.push({ url, ...options, body: options.body ? JSON.parse(options.body) : undefined });
      return reply(url.endsWith('/health') ? pcbTaskIds.includes(task)
        ? { ready: true, neuralModelCount: 1, models: { [task]: { modelSha256: FIXED_MODEL_SHA256.pcb } } }
        : ready() : result);
    },
  });
  return { analyzer, calls };
}

for (const task of fixedTaskIds.filter(task => !pcbTaskIds.includes(task))) {
  test('routes the pinned ' + task + ' task and preserves candidate absence without safety claims', async () => {
    const response = hazard(task);
    const { analyzer, calls } = fixture(task, response);
    const result = await analyzer.hazard(photo, signal());
    assert.equal(calls[0].url, 'http://127.0.0.1:8780/general/health');
    assert.equal(calls[1].url, 'http://127.0.0.1:8780/general/v1/analyze-hazard');
    assert.equal(calls[1].body.inspectionTask, task);
    assert.equal(calls[1].body.promptVersion, FIXED_PROMPT.hazard);
    assert.equal(calls[1].body.image, encoded);
    assert.equal(result.source, 'fixed-cpu');
    assert.equal(result.inspectionTask, task);
    assert.equal(result.candidateDetected, false);
    assert.match(result.assessment, /안전을 보증하지 않습니다/);
    assert.equal(result.modelSha256, FIXED_MODEL_SHA256.general);
    assert.equal(result.inputSha256, inputSha);
    assert.deepEqual(JSON.parse(result.rawJson), response);
    assert.equal(result.decision, undefined);
  });
}

for (const task of pcbTaskIds) {
  test('keeps the exact PCB category and health identity for ' + task, async () => {
    const { analyzer, calls } = fixture(task, pcb(true, task));
    const result = await analyzer.hazard(photo, signal());
    assert.equal(calls[0].url, 'http://127.0.0.1:8780/pcb/health');
    assert.equal(calls[1].body.equipmentCategory, task);
    assert.equal(result.inspectionTask, task);
    assert.equal(result.candidateDetected, true);
    const wrong = fixture(task, pcb(true, task === 'pcb1' ? 'pcb2' : 'pcb1'));
    await assert.rejects(wrong.analyzer.hazard(photo, signal()), /schema|scope|형식/);
    await assert.rejects(analyzer.compare(photo, photo, signal()), /직접 검토/);
    assert.equal(calls.length, 2);
  });
}

for (const task of equipmentTaskIds) {
  test('uses the equipment response contract in comparison for ' + task, async () => {
    const { analyzer, calls } = fixture(task, comparison(task));
    const result = await analyzer.compare(photo, photo, signal());
    assert.equal(calls[1].body.inspectionTask, task);
    assert.equal(result.inspectionTask, task);
    assert.equal(result.likelyResolved, true);
  });
}

test('offers all 20 fields and preserves research ranking separately from the selected cable demonstration', () => {
  const active = inspectionTasks.filter(task => task.id !== 'manual_other');
  assert.equal(active.length, 20);
  assert.deepEqual([...active.map(task => task.id)].sort(), [...fixedTaskIds].sort());
  assert.equal(new Set(active.map(task => task.id)).size, 20);
  assert.deepEqual(demoTasks.map(task => task.id), [
    'ball_screw_pitting', 'facility_surface', 'conveyor_foreign', 'robustad_pcb', 'ad_cable',
  ]);
  assert.deepEqual(researchTopTasks.map(task => task.id), [
    'ball_screw_pitting', 'facility_surface', 'conveyor_foreign', 'robustad_pcb', 'pcb4',
  ]);
  assert(demoTasks.every(task => task.photoRates.normal >= 20 && task.photoRates.defect >= 20));
  assert(!demoTasks.some(task => ['belt_damage','mpdd_connector','mpdd_tubes'].includes(task.id)));
});

test('cannot create a fake AI adapter for an unconnected manual field', () => {
  assert.throws(() => fixture('manual_other'), /지원하지 않는/);
});

test('uses PCB-specific route, scope and score without a fabricated confidence', async () => {
  const { analyzer, calls } = fixture('robustad_pcb', pcb());
  const result = await analyzer.hazard(photo, signal());
  assert.equal(calls[0].url, 'http://127.0.0.1:8780/pcb/health');
  assert.equal(calls[1].url, 'http://127.0.0.1:8780/pcb/v1/equipment/pcb/analyze');
  assert.deepEqual(calls[1].body, {
    image: encoded, equipmentCategory: 'robustad_pcb', sceneType: 'registered_pcb_closeup', promptVersion: FIXED_PROMPT.pcb,
  });
  assert.equal(result.anomalyScore, 0.9);
  assert.equal(result.threshold, 0.4);
  assert.equal(result.confidence, undefined);
  assert.equal(result.candidateDetected, true);
  assert.match(result.action, /내부 전기 고장/);
  assert.equal(result.modelSha256, FIXED_MODEL_SHA256.pcb);
});

test('candidate detection remains a pending human suggestion', async () => {
  const { analyzer } = fixture('facility_surface', hazard('facility_surface', true));
  const result = await analyzer.hazard(photo, signal());
  assert.equal(result.candidateDetected, true);
  assert.equal(result.risk, 3);
  assert.equal(result.decision, undefined);
  assert.equal(result.status, undefined);
});

test('PCB candidate absence does not create a normal-operation verdict', async () => {
  const { analyzer } = fixture('robustad_pcb', pcb(false));
  const result = await analyzer.hazard(photo, signal());
  assert.equal(result.candidateDetected, false);
  assert.equal(result.risk, undefined);
  assert.equal(result.confidence, undefined);
  assert.match(result.assessment, /정상 동작을 보증하지 않습니다/);
});

test('compares the same pinned task without presenting fixed 0.5 as a probability', async () => {
  const { analyzer, calls } = fixture('conveyor_foreign', comparison());
  const result = await analyzer.compare(photo, photo, signal());
  assert.equal(calls[1].url, 'http://127.0.0.1:8780/general/v1/compare-action');
  assert.equal(calls[1].body.promptVersion, 'safelog-comparison-v1');
  assert.equal(calls[1].body.beforeImage, encoded);
  assert.equal(calls[1].body.afterImage, encoded);
  assert.equal(result.likelyResolved, true);
  assert.equal(result.candidateDetected, false);
  assert.equal(result.confidence, undefined);
  assert.match(result.action, /최종 판단/);
});

test('PCB comparison requires direct human review and sends no unsupported request', async () => {
  const { analyzer, calls } = fixture('robustad_pcb', pcb());
  await assert.rejects(analyzer.compare(photo, photo, signal()), /전후 자동 비교를 지원하지 않습니다/);
  assert.equal(calls.length, 0);
});

test('rejects unready or wrong-SHA health before reading or transmitting a photo', async () => {
  for (const health of [{ ...ready(), ready: false }, { ...ready(), modelSha256: 'b'.repeat(64) }, { ...ready(), neuralModelCount: 2 }]) {
    let read = false;
    let requests = 0;
    const analyzer = createFixedCpuAnalyzer({
      task: 'facility_surface', generalBaseUrl: 'http://127.0.0.1:8775', pcbBaseUrl: 'http://127.0.0.1:8774',
      readPhoto: async () => { read = true; return encoded; },
      fetchImpl: async () => { requests++; return reply(health); },
    });
    await assert.rejects(analyzer.hazard(photo, signal()), /준비되지|고정 모델/);
    assert.equal(read, false);
    assert.equal(requests, 1);
  }
});

test('invalid general results never become candidate absence or a Mock success', async () => {
  const mutations = [
    { candidateDetected: 'false' }, { candidateDetected: true }, { riskLevel: 6 }, { riskLevel: 1.2 },
    { confidence: Infinity }, { confidence: -1 }, { modelSha256: 'b'.repeat(64) },
    { promptVersion: 'wrong' }, { inputSha256: 'not-a-hash' }, { inputWidth: 0 },
    { inspectionTask: 'conveyor_foreign' }, { detectedHazards: [12] }, { humanConfirmationRequired: false },
    { classResults: [] }, { classResults: [{ className: 'test', score: 0.1, threshold: 0.5, candidateDetected: true }] },
  ];
  for (const mutation of mutations) {
    const { analyzer } = fixture('facility_surface', { ...hazard(), ...mutation });
    await assert.rejects(analyzer.hazard(photo, signal()));
  }
});

test('rejects wrong equipment category even when the model SHA matches', async () => {
  const { analyzer } = fixture('mpdd_metal_plate', { ...hazard('mpdd_metal_plate'), equipmentCategory: 'ad_screw' });
  await assert.rejects(analyzer.hazard(photo, signal()), /inspectionTask/);
});

test('PCB score, candidate flag and localization contract must agree', async () => {
  for (const mutation of [
    { schema: 'safelog.equipment.pcb-visual.v2' }, { threshold: 0 }, { anomalyScore: NaN },
    { candidateDetected: false }, { scoreIsProbability: true }, { localizationReadyForApp: true },
    { heatmapIsConfirmedDefectLocalization: true }, { sceneType: 'arbitrary_photo' },
  ]) {
    const { analyzer } = fixture('robustad_pcb', { ...pcb(), ...mutation });
    await assert.rejects(analyzer.hazard(photo, signal()));
  }
});

test('rejects contradictory comparison claims', async () => {
  const { analyzer } = fixture('conveyor_foreign', { ...comparison(), afterCandidateDetected: true });
  await assert.rejects(analyzer.compare(photo, photo, signal()), /comparison scope/);
});

test('HTTP input and preparation errors stay errors', async () => {
  for (const status of [413, 422, 503, 500]) {
    let requests = 0;
    const analyzer = createFixedCpuAnalyzer({
      task: 'facility_surface', generalBaseUrl: 'http://localhost:8775', pcbBaseUrl: 'http://localhost:8774',
      readPhoto: async () => encoded,
      fetchImpl: async () => ++requests === 1 ? reply(ready()) : new Response('error', { status }),
    });
    await assert.rejects(analyzer.hazard(photo, signal()));
    assert.equal(requests, 2);
  }
});

test('invalid JSON and non-object JSON stay errors', async () => {
  for (const body of ['not json', '[]', 'null', '']) {
    let requests = 0;
    const analyzer = createFixedCpuAnalyzer({
      task: 'facility_surface', generalBaseUrl: 'http://localhost:8775', pcbBaseUrl: 'http://localhost:8774',
      readPhoto: async () => encoded,
      fetchImpl: async () => ++requests === 1 ? reply(ready()) : new Response(body),
    });
    await assert.rejects(analyzer.hazard(photo, signal()));
  }
});

test('network error does not fall back to Mock', async () => {
  const analyzer = createFixedCpuAnalyzer({
    task: 'facility_surface', generalBaseUrl: 'http://localhost:8775', pcbBaseUrl: 'http://localhost:8774',
    readPhoto: async () => encoded, fetchImpl: async () => { throw new Error('network'); },
  });
  await assert.rejects(analyzer.hazard(photo, signal()), /서버에 연결할 수 없습니다/);
});

test('already-cancelled requests perform no network calls', async () => {
  const controller = new AbortController();
  controller.abort();
  const { analyzer, calls } = fixture();
  await assert.rejects(analyzer.hazard(photo, controller.signal), /취소/);
  assert.equal(calls.length, 0);
});

test('cancellation aborts an active HTTP request and cannot return a late result', async () => {
  const controller = new AbortController();
  let reachedPost;
  const started = new Promise(resolve => { reachedPost = resolve; });
  let activeSignal;
  const analyzer = createFixedCpuAnalyzer({
    task: 'facility_surface', generalBaseUrl: 'http://localhost:8775', pcbBaseUrl: 'http://localhost:8774',
    readPhoto: async () => encoded,
    fetchImpl: async (url, options) => {
      if (url.endsWith('/health')) return reply(ready());
      activeSignal = options.signal;
      reachedPost();
      return new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true }));
    },
  });
  const pending = analyzer.hazard(photo, controller.signal);
  await started;
  controller.abort();
  await assert.rejects(pending, /취소/);
  assert.equal(activeSignal.aborted, true);
});

test('deadline covers a health transport that ignores cancellation', async () => {
  const analyzer = createFixedCpuAnalyzer({
    task: 'facility_surface', generalBaseUrl: 'http://localhost:8775', pcbBaseUrl: 'http://localhost:8774',
    readPhoto: async () => encoded, timeoutMs: 10, fetchImpl: () => new Promise(() => {}),
  });
  await assert.rejects(analyzer.hazard(photo, signal()), /시간이 초과/);
});

test('deadline covers photo loading and prevents a later POST', async () => {
  let finishRead;
  const loading = new Promise(resolve => { finishRead = resolve; });
  let calls = 0;
  const analyzer = createFixedCpuAnalyzer({
    task: 'facility_surface', generalBaseUrl: 'http://localhost:8775', pcbBaseUrl: 'http://localhost:8774',
    readPhoto: () => loading, timeoutMs: 10,
    fetchImpl: async () => { calls++; return reply(ready()); },
  });
  await assert.rejects(analyzer.hazard(photo, signal()), /시간이 초과/);
  finishRead(encoded);
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(calls, 1);
});

test('accepts explicit local/prefix and HTTPS endpoints and rejects accidental public HTTP or embedded credentials', () => {
  assert.equal(normalizeAiBaseUrl('http://127.0.0.1:8780/general/'), 'http://127.0.0.1:8780/general');
  assert.equal(normalizeAiBaseUrl('http://10.0.2.2:8780/pcb'), 'http://10.0.2.2:8780/pcb');
  assert.equal(normalizeAiBaseUrl('http://192.168.0.4:8780'), 'http://192.168.0.4:8780');
  assert.equal(normalizeAiBaseUrl('https://example.com/ai'), 'https://example.com/ai');
  for (const invalid of ['http://example.com', 'http://user:pass@localhost', 'http://localhost?token=x', 'http://localhost/#x', 'file:///tmp', 'not a URL']) {
    assert.throws(() => normalizeAiBaseUrl(invalid));
  }
});

test('rejects HEIC, corrupt base64, mismatched signatures and oversized image bytes', () => {
  assert.equal(validatePhotoDataUrl(encoded), encoded);
  for (const bad of ['data:image/heic;base64,/9j/2Q==', 'data:image/jpeg;base64,broken', 'data:image/png;base64,/9j/2Q==', 'data:image/jpeg;base64,abcd']) {
    assert.throws(() => validatePhotoDataUrl(bad));
  }
  const oversized = 'data:image/jpeg;base64,/9j/' + 'AAAA'.repeat(Math.ceil(8 * 1024 * 1024 / 3));
  assert.throws(() => validatePhotoDataUrl(oversized), /8MiB/);
});
