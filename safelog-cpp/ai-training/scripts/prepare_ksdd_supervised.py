"""KSDD TRAIN-only groups; quarantine exact/near TEST links before supervised fit."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.belt_normal_roi import fingerprints
from scripts.study_equipment_expansion import read, MANIFEST
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
DATA = ROOT/'data/industrial-ksdd-supervised-v10'


def assign_groups(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j); parent[max(a, b)] = min(a, b)
    exact = {}; blocks = {}; hashes = {}; buckets = defaultdict(set); near_links = 0
    for i, row in enumerate(rows):
        key = row['rgb_sha256']
        if key in exact: union(i, exact[key])
        else: exact[key] = i
        # Numeric adjacency is a conservative proxy, never a claimed physical ID.
        block = (row['publisher_split'], int(Path(row['id']).stem)//10)
        if block in blocks: union(i, blocks[block])
        else: blocks[block] = i
        for encoded in row['orientation_phashes']:
            value = int(encoded, 16)
            if value in hashes: union(i, hashes[value]); continue
            candidates = set()
            for b in range(4): candidates.update(buckets[(b, (value >> (16*b)) & 65535)])
            for other in candidates:
                if (value ^ other).bit_count() <= 2: union(i, hashes[other]); near_links += 1
            hashes[value] = i
            for b in range(4): buckets[(b, (value >> (16*b)) & 65535)].add(value)
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[find(i)].append(row)
    label_sets = defaultdict(set)
    for row in rows: label_sets[row['rgb_sha256']].add(row['label'])
    conflicts = {k for k, labels in label_sets.items() if len(labels) > 1}
    quarantined = 0
    for group in groups.values():
        group_id = min(r['rgb_sha256'] for r in group)
        has_test = any(r['publisher_split'] == 'test' for r in group)
        has_conflict = any(r['rgb_sha256'] in conflicts for r in group)
        coin = int(hashlib.sha256(('ksdd-v10:'+group_id).encode()).hexdigest()[:16], 16)/2**64
        phase = 'fit' if coin < .60 else 'validation' if coin < .80 else 'calibration'
        for row in group:
            row['group_id'] = group_id; row['original_phase'] = row['phase']
            row['phase'] = 'test' if row['publisher_split'] == 'test' else 'excluded_TEST_link' if has_test else 'excluded_label_conflict' if has_conflict else phase
            if row['publisher_split'] != 'test' and row['phase'].startswith('excluded'): quarantined += 1
    return {'groups': len(groups), 'orientation_near_hash_links': near_links, 'exact_label_conflict_groups': len(conflicts), 'TRAIN_quarantined': quarantined}


def main():
    assert not DATA.exists(); rows = [dict(r) for r in read(MANIFEST)['rows'] if r['category'] == 'ksdd2_surface']
    def inspect(row):
        raw = (ROOT/row['image']).read_bytes(); assert hashlib.sha256(raw).hexdigest() == row['image_sha256']
        with Image.open(io.BytesIO(raw)) as im:
            rgb = im.convert('RGB'); assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['rgb_sha256']
            row['orientation_phashes'] = fingerprints(rgb)
        assert len(row['masks']) == 1 and sha(ROOT/row['masks'][0]) == row['mask_sha256'][0]
        with Image.open(ROOT/row['masks'][0]) as mask:
            assert list(mask.size) == row['size'] and int((np.array(mask.convert('L')) > 0).sum()) == row['publisher_mask_pixels']
        return row
    with ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(inspect, rows))
    summary = assign_groups(rows)
    counts = {phase: dict(Counter(str(r['label']) for r in rows if r['phase'] == phase)) for phase in sorted({r['phase'] for r in rows})}
    group_phases = defaultdict(set)
    for r in rows:
        if not r['phase'].startswith('excluded'): group_phases[r['group_id']].add(r['phase'])
    assert all(len(v) == 1 for v in group_phases.values())
    enough = all(len(counts.get(p, {})) == 2 and min(counts[p].values()) >= 20 for p in ('fit', 'validation', 'calibration', 'test'))
    audit = {'status': 'ready_for_supervised_protocol' if enough else 'insufficient_grouped_data_no_training', **summary,
        'counts': counts, 'source_rows': len(rows), 'reserved_publisher_TRAIN_defects_before_quarantine': 246,
        'group_split': '60/20/20 SHA256 group split of publisher TRAIN only; old normal FIT/CAL reassigned for this new head only',
        'grouping': 'exactRGB + any8orientation pHash distance<=2 + publisher-split-specific numeric filename blocks10',
        'numeric_adjacency_is_proxy_only': True, 'physical_acquisition_groups_verified': False, 'independent_factory_validated': False,
        'current_TEST_or_near_duplicates_in_fit_validation_calibration': False, 'group_phases_disjoint': True,
        'source_images_and_real_masks_unmodified': True, 'new_model_predictions_or_training_performed': False,
        'existing_parent_thresholds_unchanged': True}
    DATA.mkdir()
    new(DATA/'manifest.json', {'schema': 'safelog-ksdd-supervised-groups-v10', 'rows': rows, 'audit': audit, 'parent_manifest_sha256': sha(MANIFEST),
        'source_sha256': {'scripts/prepare_ksdd_supervised.py': sha(Path(__file__))}})
    new(ROOT/'reports/industrial-ksdd-supervised-v10-data-audit.json', audit)
    print(audit, flush=True)


if __name__ == '__main__': main()
