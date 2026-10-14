"""Cross-device validation of the frozen wear policy, no model/threshold search."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/industrial-surfaces-v9'
ISOLATED = RUN/'isolated-zip'
sys.path.insert(0, str(ISOLATED))
import safelog_ai.industrial_batch_server as runtime
from safelog_ai.industrial_supervised import wear_descriptor
from safelog_ai.equipment_expansion_server import prepare


def classification_rates(labels, scores, threshold):
    y = np.array(labels); predicted = np.array(scores) > threshold
    tp, fn = int((predicted & (y == 1)).sum()), int((~predicted & (y == 1)).sum())
    fp, tn = int((predicted & (y == 0)).sum()), int((~predicted & (y == 0)).sum())
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'positive_photos': tp+fn, 'negative_photos': fp+tn, 'fnr': fn/(tp+fn), 'fpr': fp/(fp+tn)}


def main():
    torch.set_num_threads(4)
    assert Path(runtime.__file__).resolve().is_relative_to(ISOLATED.resolve())
    meta = json.loads((ISOLATED/'model.json').read_text(encoding='utf-8'))
    engine = runtime.IndustrialEngine(ISOLATED/'model.pt', meta['sha256'], ISOLATED/'evaluation.json')
    manifest = json.loads((ROOT/'data/industrial-bsclass-v1/manifest.json').read_text(encoding='utf-8'))
    rows = [r for r in manifest['rows'] if r['phase'] == 'reserved_test']
    gold = json.loads((ROOT/'reports/industrial-bsclass-supervised-v8-validation.json').read_text(encoding='utf-8'))
    reference = {r['id']: r for r in gold['observations']}
    assert len(rows) == len(reference) == 3236 and {r['id'] for r in rows} == set(reference)
    def image(row):
        raw = (ROOT/row['image']).read_bytes(); assert hashlib.sha256(raw).hexdigest() == row['image_sha256']
        with Image.open(io.BytesIO(raw)) as im: return prepare(im.convert('RGB'))[0]
    scores = []; start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for i in range(0, len(rows), 32):
            x = torch.stack(list(pool.map(image, rows[i:i+32])))
            descriptor = wear_descriptor(engine.model.backbone(x)).half().float()
            scores.extend(engine.model.forward_wear_descriptor(descriptor).sigmoid().tolist())
            if i % 512 == 0: print(json.dumps({'CPU_wear_photos_checked': min(i+32, len(rows)), 'total': len(rows), 'seconds': round(time.monotonic()-start, 1)}), flush=True)
    threshold = engine.wear_point['photo_threshold']; labels = [r['label'] for r in rows]
    rates = classification_rates(labels, scores, threshold)
    mismatches = [{'id': row['id'], 'CPU_score': score, 'GPU_score': reference[row['id']]['score']} for row, score in zip(rows, scores) if (score > threshold) != reference[row['id']]['candidate_detected']]
    error = max(abs(score-reference[row['id']]['score']) for row, score in zip(rows, scores))
    assert max(rates['fpr'], rates['fnr']) <= .20
    result = {'status': 'passed', 'model_sha256': meta['sha256'], 'isolated_ZIP_runtime_imported': True,
        'photos': len(rows), 'CPU_photo_rates': rates, 'original_GPU_photo_rates': gold['photo_rates'],
        'all_CPU_GPU_decisions_equal': not mismatches, 'decision_mismatches': mismatches,
        'maximum_absolute_score_difference': error, 'threshold_unchanged': threshold, 'new_optimizer_updates': 0,
        'feature_precision': 'FP16 roundtrip matching original frozen feature cache', 'same_known_reserved_TEST_reused_for_cross_device_verification': True,
        'new_independent_field_validation': False, 'CPU_total_seconds_not_latency_benchmark': time.monotonic()-start}
    with (ROOT/'reports/industrial-night-v9-wear-CPU.json').open('x', encoding='utf-8', newline='\n') as out: json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
