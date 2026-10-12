"""Combine two official PCB sources, preserving labels and excluding exact overlaps."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.pcb_multisource import CATEGORIES

DATA = ROOT / 'data/pcb-multisource-v1'
ROBUST = ROOT / 'data/equipment-robustad-pcb-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not DATA.exists(), 'Never overwrite a prepared study'
    DATA.mkdir(); (DATA / 'prepared-256').mkdir()
    visa = json.loads((ROOT / 'data/equipment-visa/manifest.json').read_text(encoding='utf-8'))
    acquisition = json.loads((ROBUST / 'acquisition.json').read_text(encoding='utf-8'))
    assert acquisition['status'] == 'downloaded_verified' and acquisition['license'] == 'CC-BY-4.0'
    expected = {r['path']: r for r in acquisition['files']}
    rows = []
    for r in visa['rows']:
        assert r['object'] in CATEGORIES[:4]
        rows.append({'id': 'VisA/' + r['image'], 'source': 'VisA', 'category': r['object'], 'phase': r['phase'],
            'split': r['split'], 'label': int(r['label'] == 'anomaly'),
            'image': 'data/equipment-visa/originals/' + r['image'],
            'image_sha256': r['image_sha256'], 'decoded_rgb_sha256': r['decoded_rgb_sha256'],
            'prepared': 'data/equipment-visa/' + r['prepared'], 'prepared_sha256': r['prepared_sha256'],
            'mask': 'data/equipment-visa/originals/' + r['mask'] if r['mask'] else ''})
    visa_keys = {r['decoded_rgb_sha256'] for r in rows}; candidates = []
    for split in ['train', 'test0']:
        for r in acquisition['rows'][split]['selected_rows']:
            name = f'PCB/pcb_data_dir_{split}/' + r['file_name']; image = ROBUST / 'originals' / name
            assert sha(image) == expected[name]['sha256']
            assert r['label'] == int(r['file_name'].startswith('anomaly/'))
            with Image.open(image) as original:
                assert original.getexif().get(274, 1) == 1, 'Author mask orientation must match image'
                rgb = ImageOps.exif_transpose(original).convert('RGB'); pixels = np.asarray(rgb)
                key = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
                geometry = list(rgb.size)
            mask = ROBUST / 'originals' / r['mask'] if r['mask'] else None
            if r['label']:
                assert mask and sha(mask) == expected[r['mask']]['sha256']
                with Image.open(mask) as ground:
                    assert ground.size == tuple(geometry) and np.any(np.asarray(ground.convert('L')) > 0)
            else:
                assert mask is None
            candidates.append({'id': 'RobustAD/' + name, 'source': 'RobustAD', 'category': 'robustad_pcb',
                'phase': 'unassigned' if split == 'train' else 'test', 'split': split, 'label': r['label'],
                'image': image.relative_to(ROOT).as_posix(), 'image_sha256': sha(image), 'decoded_rgb_sha256': key,
                'mask': mask.relative_to(ROOT).as_posix() if mask else '', 'native_geometry': geometry})
    by_key = {}
    for r in candidates:
        by_key.setdefault(r['decoded_rgb_sha256'], []).append(r)
    excluded = []; included = []
    for key, records in by_key.items():
        reason = ('exact_overlap_with_VisA' if key in visa_keys else
                  'RobustAD_cross_split_exact_overlap' if len({r['split'] for r in records}) > 1 else
                  'RobustAD_within_split_duplicate_or_label_conflict' if len(records) > 1 else '')
        if reason:
            excluded.extend({'id': r['id'], 'reason': reason} for r in records)
        else:
            included.extend(records)
    normal_train = sorted([r for r in included if r['split'] == 'train'],
                          key=lambda r: hashlib.sha256(('pcb-pool-CAL-v1/' + r['id']).encode()).hexdigest())
    assert len(normal_train) >= 100, 'Insufficient unique added normal TRAIN photos'
    for index, r in enumerate(normal_train):
        r['phase'] = 'calibration' if index < 40 else 'fit'
    for i, r in enumerate(included):
        cache = DATA / 'prepared-256' / f'robust-{i:04d}.png'
        with Image.open(ROOT / r['image']) as image:
            ImageOps.exif_transpose(image).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR).save(cache)
        r['prepared'] = cache.relative_to(ROOT).as_posix(); r['prepared_sha256'] = sha(cache)
    rows.extend(included)
    assert len({r['decoded_rgb_sha256'] for r in rows}) == len(rows)
    counts = dict(Counter(f"{r['source']}/{r['category']}/{r['phase']}/{r['label']}" for r in rows))
    assert all(r['split'] == 'train' and r['label'] == 0 for r in rows if r['phase'] in ['fit', 'calibration'])
    manifest = {'schema': 'pcb-two-source-normal-only-v1', 'rows': rows, 'counts': counts, 'sources': ['VisA', 'RobustAD'],
        'normal_only_fit': True, 'cross_source_and_split_exact_rgb_duplicates_retained': 0, 'excluded': excluded,
        'robustad_train_calibration_count': 40, 'robustad_test_domains_used': ['test0'],
        'physical_board_or_factory_independence_verified': False, 'new_ai_or_human_labels': 0,
        'known_VisA_test_reused': True, 'RobustAD_test0_not_previously_scored': True,
        'robustad_revision': acquisition['revision'], 'license_both_sources': 'CC-BY-4.0',
        'VisA_manifest_sha256': sha(ROOT / 'data/equipment-visa/manifest.json'),
        'RobustAD_acquisition_sha256': sha(ROBUST / 'acquisition.json')}
    with (DATA / 'manifest.json').open('x', encoding='utf-8') as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps({'counts': counts, 'excluded_count': len(excluded), 'unique_photos': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
