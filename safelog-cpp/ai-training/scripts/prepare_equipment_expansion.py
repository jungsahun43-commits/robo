"""Select original photos/masks only; keep publisher tests out of training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import zipfile
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.belt_normal_roi import fingerprints
from safelog_ai.equipment_expansion import CATEGORIES
from scripts.fetch_equipment_expansion import DATA, SOURCES, sha


def new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2)


def relative_member(name, source, config):
    parts = Path(name.replace('\\', '/')).parts
    if name.replace('\\', '/').startswith('/') or Path(name).is_absolute() or '..' in parts or any(':' in p for p in parts):
        raise ValueError('Unsafe archive path')
    if not parts: return None
    if source == 'kolektor_sdd2':
        if parts[0] == 'KolektorSDD2': parts = parts[1:]
        if len(parts) != 2 or parts[0] not in ('train', 'test') or not re.fullmatch(r'\d+(?:_GT)?\.png', parts[1]): return None
    else:
        if parts[0] not in config['categories']: parts = parts[1:]
        if not parts or parts[0] not in config['categories'] or len(parts) < 4: return None
        if parts[1] not in ('train', 'test', 'ground_truth', 'validation'): return None
        if Path(parts[-1]).suffix.lower() not in ('.png', '.jpg', '.bmp', '.tif'): return None
    return Path(*parts)


def extract(source, config):
    folder = DATA/source; receipt = json.loads((folder/'download.json').read_text())
    if config.get('kind') == 'hf_selected':
        assert all(sha(folder/'originals'/p) == h for p, h in receipt['file_sha256'].items()); return
    archive = folder/config['archive']; assert sha(archive) == receipt['archive_sha256']
    output = folder/'originals'; output.mkdir(exist_ok=True)
    def write(name, stream):
        rel = relative_member(name, source, config)
        if rel is None: return
        path = output/rel; assert path.resolve().is_relative_to(output.resolve())
        if path.exists(): return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f: shutil.copyfileobj(stream, f)
    if archive.name.endswith('.zip'):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not member.is_dir() and relative_member(member.filename, source, config) is not None:
                    if (member.external_attr >> 16) & 0o170000 == 0o120000: raise ValueError('Archive symlink rejected')
                    with z.open(member) as stream: write(member.filename, stream)
    else:
        with tarfile.open(archive, 'r|xz') as tar:
            for member in tar:
                if member.isfile() and relative_member(member.name, source, config) is not None:
                    with tar.extractfile(member) as stream: write(member.name, stream)
                elif member.issym() or member.islnk(): raise ValueError('Archive links rejected')
    print(json.dumps({'extracted': source, 'only_original_selected_categories': True}), flush=True)


def row_for(source, category, image, masks, split, defect):
    with Image.open(image) as im:
        im.load(); rgb = im.convert('RGB'); rgb_bytes = rgb.tobytes(); size = list(rgb.size)
        fingerprint = min(fingerprints(rgb))
    truth = np.zeros((size[1], size[0]), dtype=bool)
    for path in masks:
        with Image.open(path) as mask:
            assert list(mask.size) == size, (image, path)
            truth |= np.array(mask.convert('L')) > 0
    label = int(truth.any()) if source == 'kolektor_sdd2' else int(defect != 'good')
    if label and not masks: raise ValueError('Publisher defect photo needs its masks: '+str(image))
    if label and not truth.any(): raise ValueError('Positive publisher mask must contain pixels')
    return {'id': source+'/'+image.relative_to(DATA/source/'originals').as_posix(), 'source': source, 'category': category,
            'image': image.relative_to(ROOT).as_posix(), 'image_sha256': sha(image), 'masks': [p.relative_to(ROOT).as_posix() for p in masks],
            'mask_sha256': [sha(p) for p in masks], 'size': size, 'rgb_sha256': hashlib.sha256(rgb_bytes).hexdigest(),
            'canonical_phash': fingerprint, 'publisher_split': split, 'defect': defect, 'label': label,
            'publisher_mask_pixels': int(truth.sum()), 'phase': 'test' if split == 'test' else 'pending',
            'whole_photo_scope': 'publisher_inspected_component_only', 'mechanical_health': 'unknown'}


def collect():
    config = json.loads(SOURCES.read_text()); rows = []
    for source, info in config.items():
        base = DATA/source/'originals'
        if source == 'kolektor_sdd2':
            for split in ('train', 'test'):
                for image in sorted((base/split).glob('*.png')):
                    if not re.fullmatch(r'\d+\.png', image.name): continue
                    mask = image.with_name(image.stem+'_GT.png'); assert mask.exists()
                    rows.append(row_for(source, 'ksdd2_surface', image, [mask], split, 'publisher_mask'))
        else:
            prefix = {'mvtec_ad': 'ad_', 'mvtec_loco': 'loco_', 'mpdd': 'mpdd_'}[source]
            for cat in info['categories']:
                category = prefix+cat; assert category in CATEGORIES
                for split in ('train', 'validation', 'test'):
                    for image in sorted((base/cat/split).glob('*/*')):
                        if image.suffix.lower() not in ('.png', '.jpg', '.bmp', '.tif'): continue
                        defect = image.parent.name; masks = []
                        if defect != 'good':
                            gt = base/cat/'ground_truth'/defect
                            masks = list(gt.glob(image.stem+'_mask.png')) or list(gt.glob(image.stem+'.png'))
                            if not masks and (gt/image.stem).is_dir(): masks = sorted((gt/image.stem).glob('*.png'))
                        rows.append(row_for(source, category, image, masks, split, defect))
        print(json.dumps({'audited_source': source, 'rows_so_far': len(rows)}), flush=True)
    assert set(r['category'] for r in rows) == set(CATEGORIES)
    return rows


def assign(rows):
    excluded = Counter(); near_test = 0
    for category in CATEGORIES:
        local = [r for r in rows if r['category'] == category]
        test = [r for r in local if r['publisher_split'] == 'test']; test_rgb = {r['rgb_sha256'] for r in test}
        test_hash = [int(r['canonical_phash'], 16) for r in test]
        normal = []
        for r in local:
            if r['publisher_split'] == 'test': continue
            if r['label']:
                # This first architecture learns normal appearance; preserve
                # labeled TRAIN defects for later supervised experiments.
                r['phase'] = 'reserved_training_defect'; continue
            if r['rgb_sha256'] in test_rgb:
                r['phase'] = 'excluded_exact_test_duplicate'; excluded[r['phase']] += 1; continue
            h = int(r['canonical_phash'], 16)
            if any((h ^ t).bit_count() <= 2 for t in test_hash):
                r['phase'] = 'excluded_near_test'; near_test += 1; excluded[r['phase']] += 1; continue
            normal.append(r)
        # Exact RGB group split, no duplicate can enter both FIT and CAL.
        groups = defaultdict(list)
        for r in normal: groups[r['rgb_sha256']].append(r)
        if category.startswith('loco_'):
            for group in groups.values():
                phases = {r['publisher_split'] for r in group}
                phase = 'cal' if 'validation' in phases else 'fit'
                for r in group: r['phase'] = phase
        else:
            keys = sorted(groups, key=lambda x: hashlib.sha256((category+':443:'+x).encode()).hexdigest())
            cal_count = min(max(20, round(len(keys)*.2)), max(0, len(keys)-20))
            cal = set(keys[:cal_count])
            for key, group in groups.items():
                group[0]['phase'] = 'cal' if key in cal else 'fit'
                for r in group[1:]: r['phase'] = 'excluded_same_phase_duplicate'; excluded[r['phase']] += 1
        fit_rgb = {r['rgb_sha256'] for r in local if r['phase'] == 'fit'}
        cal_rgb = {r['rgb_sha256'] for r in local if r['phase'] == 'cal'}
        assert not (fit_rgb & cal_rgb or fit_rgb & test_rgb or cal_rgb & test_rgb)
    assert all(r['label'] == 0 and r['publisher_split'] != 'test' for r in rows if r['phase'] in ('fit', 'cal'))
    counts = {cat: dict(Counter(r['phase'] for r in rows if r['category'] == cat)) for cat in CATEGORIES}
    return counts, dict(excluded), near_test


def prepare():
    assert not (DATA/'manifest.json').exists(), 'Completed manifest is immutable'
    config = json.loads(SOURCES.read_text())
    for source, info in config.items(): extract(source, info)
    rows = collect(); counts, exclusions, near_test = assign(rows)
    receipts = {name: json.loads((DATA/name/'download.json').read_text()) for name in config}
    new(DATA/'manifest.json', {'schema': 'safelog-equipment-expansion-v1', 'rows': rows, 'counts': counts,
        'sources': config, 'download_receipts': receipts, 'exclusions': exclusions, 'near_test_excluded': near_test,
        'normal_only_first_fit': True, 'publisher_test_used_for_training_or_calibration': False,
        'split_rule': 'exact RGB groups; exclude canonical rotation-aware pHash<=2 matches to publisher TEST; seeded hash20percent normalTRAIN CAL with at least20 if possible; LOCO publisher normal validation retained',
        'original_physical_component_or_site_groups_verified': False, 'field_validation': False,
        'source_sha256': {n: sha(ROOT/n) for n in ('configs/equipment_expansion_sources.json', 'scripts/prepare_equipment_expansion.py', 'scripts/fetch_equipment_expansion.py')}})
    summary = {'counts': counts, 'source_photos': len(rows), 'exclusions': exclusions,
               'pixel_masks_from_publisher': True, 'mirror_extras_prompts_and_annotated_photos_excluded': True,
               'new_independent_factory_photos_verified': 0, 'publisher_test_used_for_training_or_calibration': False}
    new(ROOT/'reports/equipment-expansion-v1-data-audit.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--extract-only', choices=['mpdd','kolektor_sdd2'])
    args = parser.parse_args()
    if args.extract_only: extract(args.extract_only, json.loads(SOURCES.read_text())[args.extract_only])
    else: prepare()
