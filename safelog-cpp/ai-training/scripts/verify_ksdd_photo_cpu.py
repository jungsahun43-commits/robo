"""Whole TEST CPU photo verification of the isolated, immutable v11 runtime."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

ROOT = Path(sys.argv[1]).resolve(); ISOLATED = Path.cwd().resolve()
sys.path.insert(0, str(ISOLATED))
import safelog_ai.ksdd_photo_only_server as runtime
from safelog_ai.ksdd_supervised import ksdd_view, ksdd_photo_score
from safelog_ai.native_component_views import normalized


def main():
    torch.set_num_threads(4); assert Path(runtime.__file__).resolve().is_relative_to(ISOLATED)
    meta = json.loads((ISOLATED/'model.json').read_text(encoding='utf-8'))
    engine = runtime.KSDDPhotoEngine(ISOLATED/'model.pt', meta['sha256'], ISOLATED/'evaluation.json')
    rows = [r for r in json.loads((ROOT/'data/industrial-ksdd-supervised-v10/manifest.json').read_text(encoding='utf-8'))['rows'] if r['phase'] == 'test']
    gold = json.loads((ROOT/'reports/industrial-ksdd-supervised-v10-validation.json').read_text(encoding='utf-8'))
    lookup = {r['id']: r for r in gold['observations']}; assert len(rows) == len(lookup) == 1004
    def prepare(row):
        assert hashlib.sha256((ROOT/row['image']).read_bytes()).hexdigest() == row['image_sha256']
        with Image.open(ROOT/row['image']) as im: view, _, valid, _ = ksdd_view(im.convert('RGB'))
        return normalized(view), valid
    scores = []; started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for start in range(0, len(rows), 8):
            values = list(pool.map(prepare, rows[start:start+8])); x = torch.stack([v[0] for v in values]); valid = torch.stack([v[1] for v in values])
            scores.extend(ksdd_photo_score(engine.model.forward_ksdd(x).sigmoid(), valid).tolist())
            if start % 128 == 0: print({'CPU_photo_checked': min(start+8, len(rows)), 'total': len(rows), 'seconds': round(time.monotonic()-started, 1)}, flush=True)
    labels = np.array([r['label'] for r in rows]); threshold = engine.ksdd_point['photo_threshold']; predicted = np.array(scores) > threshold
    tp = int((predicted & (labels == 1)).sum()); fn = int((~predicted & (labels == 1)).sum()); fp = int((predicted & (labels == 0)).sum()); tn = int((~predicted & (labels == 0)).sum())
    rates = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'positive_photos': tp+fn, 'negative_photos': fp+tn, 'fnr': fn/(tp+fn), 'fpr': fp/(fp+tn)}
    differences = [{'id': r['id'], 'CPU_score': score, 'GPU_score': lookup[r['id']]['score']} for r, score in zip(rows, scores) if (score > threshold) != lookup[r['id']]['candidate_detected']]
    assert rates['fpr'] <= gold['parent_photo_rates']['fpr'] and rates['fnr'] < gold['parent_photo_rates']['fnr']
    value = {'status': 'passed', 'model_sha256': meta['sha256'], 'isolated_ZIP_runtime_imported': True, 'photos': len(rows), 'CPU_photo_rates': rates,
        'GPU_photo_rates': gold['photo_rates'], 'all_CPU_GPU_photo_decisions_equal': not differences, 'decision_mismatches': differences,
        'max_absolute_score_difference': max(abs(s-lookup[r['id']]['score']) for r, s in zip(rows, scores)), 'photo_threshold_unchanged': threshold,
        'photo_subgraph_exactly_same_as_packaged_API': True, 'original_location_preserved_verified_separately': True,
        'same_known_TEST_reused_for_device_verification': True, 'new_training_or_threshold_changes': False, 'independent_field_validated': False,
        'CPU_total_seconds_not_latency_benchmark': time.monotonic()-started}
    with (ROOT/'reports/industrial-ksdd-photo-only-v11-CPU.json').open('x', encoding='utf-8', newline='\n') as out: json.dump(value, out, ensure_ascii=False, indent=2)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
