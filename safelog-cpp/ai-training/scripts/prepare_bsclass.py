"""Publisher N/P labels; keep rotated/exact/near duplicates in one fresh split."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.belt_normal_roi import fingerprints
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
DATA = ROOT/'data/industrial-bsclass-v1'


def publisher_label(name):
    match = re.fullmatch(r'([NP]) \((\d+)\)(_)?\.png', Path(name).name)
    if not match: raise ValueError('Explicit publisher N/P filename label required')
    return int(match[1] == 'P'), match[1]+':'+match[2], bool(match[3])


def split_groups(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j); parent[max(a, b)] = min(a, b)
    origin = {}; exact = {}; phash_first = {}; buckets = defaultdict(set); near_links = 0
    for i, row in enumerate(rows):
        for key, mapping in [(row['publisher_original_id'], origin), (row['canonical_rgb_sha256'], exact)]:
            if key in mapping: union(i, mapping[key])
            else: mapping[key] = i
        value = int(row['canonical_phash'], 16)
        if value in phash_first: union(i, phash_first[value]); continue
        candidates = set()
        for block in range(4): candidates.update(buckets[(block, (value >> (16*block)) & 65535)])
        for other in candidates:
            if (value ^ other).bit_count() <= 2: union(i, phash_first[other]); near_links += 1
        phash_first[value] = i
        for block in range(4): buckets[(block, (value >> (16*block)) & 65535)].add(value)
    exact_labels = defaultdict(set)
    for row in rows: exact_labels[row['canonical_rgb_sha256']].add(row['label'])
    conflicts = {k for k, labels in exact_labels.items() if len(labels) > 1}
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[find(i)].append(row)
    for group in groups.values():
        key = min(r['canonical_rgb_sha256'] for r in group); coin = int.from_bytes(hashlib.sha256(('bsclass-v1:'+key).encode()).digest()[:8], 'big')/2**64
        phase = 'fit' if coin < .70 else 'cal' if coin < .85 else 'reserved_test'
        for row in group: row.update(group_id=key, phase='excluded_label_conflict' if row['canonical_rgb_sha256'] in conflicts else phase)
    return {'groups': len(groups), 'near_hash_links': near_links, 'exact_label_conflict_groups': len(conflicts)}


def main():
    assert not (DATA/'manifest.json').exists() and not (DATA/'originals-v2').exists()
    receipt = read(DATA/'download.json'); archive = DATA/'publisher-bagit.archive'; assert sha(archive) == receipt['archive_sha256']
    output = DATA/'originals-v2'; output.mkdir()
    with zipfile.ZipFile(archive) as z:
        entries = [n for n in z.infolist() if not n.is_dir()]
        assert len(entries) == 21835 and sum(n.file_size for n in entries) < 2*1024**3
        def worker(info):
            name = info.filename.replace('\\', '/'); path = output/name
            assert path.resolve().is_relative_to(output.resolve()) and '..' not in Path(name).parts
            label, original, rotated = publisher_label(name); raw = z.read(info)
            with Image.open(io.BytesIO(raw)) as image:
                rgb = image.convert('RGB'); assert min(rgb.size) > 0 and max(rgb.size) <= 2048; size = list(rgb.size); values = np.array(rgb)
                canonical = min(hashlib.sha256(np.ascontiguousarray(np.rot90(values, k)).tobytes()).hexdigest() for k in range(4))
                phash = min(fingerprints(rgb))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as out: out.write(raw)
            return {'id': name, 'image': path.relative_to(ROOT).as_posix(), 'image_sha256': hashlib.sha256(raw).hexdigest(), 'label': label,
                'publisher_original_id': original, 'publisher_rotation_variant': rotated, 'canonical_rgb_sha256': canonical,
                'canonical_phash': phash, 'size': size, 'domain': 'ball_screw_visible_surface_pitting', 'mechanical_health': 'not_assessed'}
        rows = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, row in enumerate(pool.map(worker, entries), 1):
                rows.append(row)
                if i % 3000 == 0: print(json.dumps({'prepared_new_labelled_crops': i, 'total': len(entries)}), flush=True)
    grouping = split_groups(rows); counts = {phase: dict(Counter(str(r['label']) for r in rows if r['phase'] == phase)) for phase in sorted({r['phase'] for r in rows})}
    assigned = defaultdict(set)
    for row in rows:
        if not row['phase'].startswith('excluded'): assigned[row['group_id']].add(row['phase'])
    assert all(len(v) == 1 for v in assigned.values())
    audit = {'status': 'fresh_labelled_data_prepared_no_model_predictions', 'source': receipt['primary_source'], 'archive_sha256': receipt['archive_sha256'],
        'license': receipt['license'], 'source_images': len(rows), 'label_counts': dict(Counter(str(r['label']) for r in rows)), 'split_counts': counts,
        'publisher_rotation_variants': sum(r['publisher_rotation_variant'] for r in rows), **grouping,
        'actual_size_counts': dict(Counter(str(r['size']) for r in rows)), 'not_nominal150x150': sum(r['size'] != [150, 150] for r in rows),
        'initial_exact150size_check_failure_preserved': True, 'original_pixels_not_resized_or_modified': True,
        'split_rule': 'seeded70/15/15 group hash; publisher rotation origin, exact90degree-canonical RGB and canonicalpHash<=2 linked before splitting',
        'group_phases_disjoint': True, 'new_reserved_TEST_model_predictions_run': False, 'new_training_performed': False,
        'physical_spindle_or_acquisition_frame_groups_verified': False, 'independent_factory_or_subject_split_claimed': False,
        'not_same_as_MVTec_fastener_screw': True, 'whole_factory_photo_health_labels_created': False}
    new(DATA/'manifest.json', {'schema': 'safelog-bsclass-fresh-v1', 'rows': rows, 'audit': audit,
        'source_sha256': {n: sha(ROOT/n) for n in ('scripts/fetch_bsclass_originals.py', 'scripts/prepare_bsclass.py')}})
    new(ROOT/'reports/bsclass-v1-data-audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__': main()
