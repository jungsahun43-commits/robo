"""Inventory new curator masks and screen all retained TEST images; no model work."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import sys
import zipfile
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_neu_micro_candidate import DATA
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.prepare_magnetic_tiles import Bands
from safelog_ai.belt_normal_roi import fingerprints, distance


def main():
    output = ROOT / 'reports/industrial-neu-micro-v22-candidate-audit.json'
    assert not output.exists() and not (DATA / 'candidate-manifest.json').exists()
    receipt = read(DATA / 'download-receipt.json')
    path = DATA / 'publisher-archive.zip'
    assert sha(path) == receipt['archive_sha256_local']
    raw = DATA / 'originals'; raw.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for item in archive.infolist():
            name = PurePosixPath(item.filename.replace('\\', '/'))
            assert not name.is_absolute() and '..' not in name.parts and not any(':' in p for p in name.parts)
            if item.is_dir() or name.suffix.lower() not in ('.jpg', '.png', '.bmp', '.jpeg'):
                continue
            target = raw.joinpath(*name.parts)
            assert target.resolve().is_relative_to(raw.resolve())
            data = archive.read(item)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == hashlib.sha256(data).hexdigest()
            else:
                with target.open('xb') as out: out.write(data)
    rows = []
    images = sorted(p for p in raw.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.bmp'))
    assert images
    for image in images:
        mask = image.with_suffix('.png')
        assert mask.is_file(), image
        with Image.open(image) as im: rgb = im.convert('RGB')
        with Image.open(mask) as im:
            assert im.size == rgb.size
            truth = np.array(im.convert('L'))
        values = np.unique(truth)
        rows.append({'id': 'neu_curator/' + image.relative_to(raw).as_posix(),
            'source_folder': image.relative_to(raw).parts[0], 'image': image.relative_to(ROOT).as_posix(),
            'image_sha256': sha(image), 'mask': mask.relative_to(ROOT).as_posix(), 'mask_sha256': sha(mask),
            'size': list(rgb.size), 'mask_values': values.tolist(), 'positive_pixels_if_bright': int((truth > 0).sum()),
            'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'orientation_phashes': fingerprints(rgb),
            'phase': 'unassigned_candidate_no_model_use', 'physical_object_identity': 'not_verified',
            'label_scope': 'curator visible-surface masks, not whole equipment functional health'})
    assert len(list(raw.rglob('*.png'))) == len(rows)
    names = ['data/equipment-expansion-v1/manifest.json', 'data/industrial-ksdd-supervised-v10/manifest.json',
             'data/industrial-bsclass-v1/manifest.json', 'data/industrial-magnetic-tiles-v16/manifest.json',
             'data/industrial-ksdd1-v18/manifest-r1.json']
    heldout = {}
    for name in names:
        for row in read(ROOT / name)['rows']:
            if row['phase'] in ('test', 'reserved_test'):
                heldout[row['image']] = dict(row)
    def prepare(row):
        assert sha(ROOT / row['image']) == row['image_sha256']
        if not row.get('orientation_phashes') or not row.get('rgb_sha256'):
            with Image.open(ROOT / row['image']) as im:
                rgb = im.convert('RGB')
                row['orientation_phashes'] = fingerprints(rgb)
                row['rgb_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
        return row
    print({'new_pairs': len(rows), 'retained_TEST_to_check': len(heldout)}, flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        old = list(pool.map(prepare, heldout.values()))
    index = Bands(); exact = defaultdict(set)
    for i, row in enumerate(old):
        index.add(row['orientation_phashes'], i); exact[row['rgb_sha256']].add(i)
    matches = []
    for row in rows:
        ids = set(exact.get(row['rgb_sha256'], ()))
        for i in index.candidates(row['orientation_phashes']):
            if distance(row['orientation_phashes'], old[i]['orientation_phashes']) <= 2: ids.add(i)
        matches.extend({'query': row['id'], 'retained_TEST': old[i]['image']} for i in sorted(ids))
    within = []
    for i, a in enumerate(rows):
        for b in rows[:i]:
            d = distance(a['orientation_phashes'], b['orientation_phashes'])
            if a['rgb_sha256'] == b['rgb_sha256'] or d <= 2:
                within.append({'a': a['id'], 'b': b['id'], 'min_orientation_phash_distance': d,
                               'exact_RGB': a['rgb_sha256'] == b['rgb_sha256']})
    audit = {'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'candidate_only_split_and_mask_visual_QA_pending',
        'source_sha256': sha(DATA / 'source.json'), 'download_receipt_sha256': sha(DATA / 'download-receipt.json'),
        'image_mask_pairs': len(rows), 'source_folder_counts': dict(Counter(r['source_folder'] for r in rows)),
        'mask_value_sets': sorted({tuple(r['mask_values']) for r in rows}),
        'empty_bright_masks': sum(not r['positive_pixels_if_bright'] for r in rows),
        'retained_TEST_images_checked': len(old), 'retained_TEST_manifests_sha256': {n: sha(ROOT / n) for n in names},
        'exact_RGB_or_any8orientation_pHash2_TEST_matches': matches, 'within_source_exact_or_near_links': within,
        'physical_capture_groups_verified': False, 'source_near_groups_are_not_physical_identity_proof': True,
        'train_VAL_CAL_split_fixed': False, 'new_encoder_features': 0, 'new_model_scores': 0, 'new_optimizer_updates': 0,
        'archive_code_executed': False, 'raw_data_or_masks_uploaded_to_Git': False,
        'mask_polarity_and_image_quality_visual_QA_done': False,
        'not_the_1800image_NEU_CLS_or_SDsaliency900': True,
        'citation_and_use': 'NEU original images and abin24 added masks; publisher explicitly academic research only/no commercial/cite original provider; no invented SPDX',
        'next': 'Audit source filename/capture/near groups, fix FIT/VAL/CAL and reserve heldout or source-only study before model scores; all TEST-linked groups excluded; visual mask polarity QA only predesignated FIT; do not label backgrounds as healthy machinery'}
    new(DATA / 'candidate-manifest.json', {'rows': rows, 'audit': audit})
    new(output, audit)
    print({k: v for k, v in audit.items() if k not in ('within_source_exact_or_near_links', 'retained_TEST_manifests_sha256')}, flush=True)
    print({'within_source_near_links': len(within), 'source_first_rows': [{k: r[k] for k in ('id', 'size', 'mask_values', 'positive_pixels_if_bright')} for r in rows[:4]]}, flush=True)


if __name__ == '__main__':
    main()
