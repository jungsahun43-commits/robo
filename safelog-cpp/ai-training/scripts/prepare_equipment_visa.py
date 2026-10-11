"""Freeze PCB inputs and split normal publisher TRAIN into fit/calibration."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.equipment_anomaly import CATEGORIES
from fetch_equipment_visa import sha

DATA = ROOT / 'data/equipment-visa'


def main():
    output = DATA / 'manifest.json'
    assert not output.exists(), 'Preserve frozen manifest'
    acquisition = json.loads((DATA / 'acquisition.json').read_text(encoding='utf-8'))
    assert acquisition['status'] == 'complete'
    original = DATA / 'originals'
    rows = [r for r in csv.DictReader((DATA / '1cls.csv').open(encoding='utf-8')) if r['object'] in CATEGORIES]
    expected = {r['path']: r for r in acquisition['files']}
    processed, duplicate_pixels = [], {}
    cache = DATA / 'prepared-256'
    cache.mkdir()
    for i, row in enumerate(rows):
        image = original / row['image']
        assert sha(image) == expected[row['image']]['sha256']
        with Image.open(image) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            pixels = np.asarray(rgb)
            key = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            assert key not in duplicate_pixels, 'Exact duplicate decoded photo: audit before splitting'
            duplicate_pixels[key] = row['image']
            resized = rgb.resize((256, 256), Image.Resampling.BILINEAR)
            name = row['object'] + '-' + str(i) + '.png'
            resized.save(cache / name)
        item = dict(row, image_sha256=expected[row['image']]['sha256'], decoded_rgb_sha256=key,
                    prepared='prepared-256/' + name, prepared_sha256=sha(cache / name))
        processed.append(item)
    for category in CATEGORIES:
        normal = [r for r in processed if r['object'] == category and r['split'] == 'train']
        assert all(r['label'] == 'normal' and not r['mask'] for r in normal)
        ordered = sorted(normal, key=lambda r: hashlib.sha256(('equipment-seed-81/' + r['image']).encode()).hexdigest())
        holdout = {r['image'] for r in ordered[:int(np.ceil(0.1 * len(ordered)))]}
        for row in normal:
            row['phase'] = 'calibration' if row['image'] in holdout else 'fit'
    for row in processed:
        if row['split'] == 'test':
            row['phase'] = 'test'
        assert row['phase'] == 'test' or row['label'] == 'normal'
    counts = Counter((r['object'], r['phase'], r['label']) for r in processed)
    manifest = {'status': 'frozen', 'categories': CATEGORIES, 'size': 256, 'rows': processed,
        'counts': {'/'.join(k): n for k, n in sorted(counts.items())},
        'source_split_sha256': sha(DATA / '1cls.csv'), 'acquisition_sha256': sha(DATA / 'acquisition.json'),
        'exact_decoded_cross_split_duplicates': 0, 'normal_only_fit': True,
        'calibration': 'Deterministic 10% publisher normal TRAIN holdout; hash order seed 81',
        'factory_or_physical_board_identity_independence_verified': False,
        'resize': 'EXIF-transposed RGB PIL bilinear 256x256, ImageNet normalization at inference',
        'source_test_in_training': False, 'human_or_ai_new_labels': 0,
        'script_sha256': sha(Path(__file__))}
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'prepared', 'counts': manifest['counts'], 'manifest_sha256': sha(output)}), flush=True)


if __name__ == '__main__':
    main()
