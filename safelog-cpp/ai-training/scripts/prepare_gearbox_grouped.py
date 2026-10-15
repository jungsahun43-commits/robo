"""Reserve physical/near groups before Re-GBC pixels, scores or training."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path
import shutil
import sys
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.belt_normal_roi import fingerprints, distance
from scripts.prepare_magnetic_tiles import Bands
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

DATA = ROOT / 'data/industrial-gearbox-source-v25'
RUN = ROOT / 'runs/industrial-gearbox-v25'
POLICY = ROOT / 'reports/industrial-gearbox-v25-data-policy.json'
MANIFEST = DATA / 'manifest.json'
HELDOUT = ('data/equipment-expansion-v1/manifest.json',
           'data/industrial-ksdd-supervised-v10/manifest.json',
           'data/industrial-bsclass-v1/manifest.json',
           'data/industrial-magnetic-tiles-v16/manifest.json',
           'data/industrial-ksdd1-v18/manifest-r1.json')


def group_allocation(groups):
    """No label/model dependent retries: sorted SHA group names, ceil15/15/rest."""
    ordered = sorted(groups, key=lambda g: (hashlib.sha256(('gearbox25:' + str(g)).encode()).hexdigest(), str(g)))
    count = math.ceil(.15 * len(ordered))
    return {g: 'validation' if i < count else 'cal' if i < 2 * count else 'fit'
            for i, g in enumerate(ordered)}


def freeze():
    assert not POLICY.exists() and not RUN.exists() and not MANIFEST.exists()
    assert shutil.disk_usage(ROOT).free > 4 * 1024**3
    reservation = DATA / 'reserved-author-S3-fold0.json'
    assert sha(reservation) == 'f1104c7f9c85899457608720162ea1bf5ef9d077dda9d26c28b7b2ae5f44e0a5'
    RUN.mkdir()
    p = {'UTC': datetime.now(timezone.utc).isoformat(),
         'fixed_before_PNG_decoding_model_features_scores_optimizer': True,
         'archive_sha256': '4c01475a48a8b2fa28a9bd8e0f11ccacf14cc8dc2651b5b72df21f072a241775',
         'reservation_sha256': sha(reservation), 'metadata_audit_sha256': sha(DATA / 'metadata-labels-audit.json'),
         'script_sha256': sha(Path(__file__)),
         'existing_TEST_manifest_SHA256': {n: sha(ROOT / n) for n in HELDOUT},
         'holdout': 'exact author functional-component-id fold0(S3) TEST3109 remains TEST; never switch fold',
         'group_rule': 'union publisher physical_part_id, functional_component_id, acq_id and exact RGB or any8orientation pHash<=2',
         'exclusion': 'all nonTEST in TEST-connected group excluded; any group matching retained old5746TEST excluded from nonTEST; original TEST list retained and any oldTEST matches explicitly flagged',
         'split_rule': 'remaining connected group sorted SHA256(gearbox25:+str(min_idx)); first ceil(.15*N) validation, next ceil(.15*N) calibration, rest FIT; no redraw based on labels or scores',
         'data_gate': 'FIT each label>=100 and>=8groups; VAL/CAL each label>=20 and>=3groups; TEST each label>=20 and>=3groups; no reservedTEST/oldTEST-near group in FIT/VAL/CAL',
         'unannotated_PNGs_excluded': 103,
         'label_scope': 'publisher128x128 visible-wear acceptance patch OK/nOK, not whole equipment functioning or location mask',
         'new_features_scores_optimizer_updates': 0, 'training_protocol_fixed': False}
    new(POLICY, p)
    print(p, flush=True)


def prepare():
    p = read(POLICY)
    assert sha(Path(__file__)) == p['script_sha256']
    assert sha(DATA / 'publisher-v1.zip') == p['archive_sha256']
    assert not MANIFEST.exists() and not (DATA / 'originals').exists()
    assert sha(DATA / 'reserved-author-S3-fold0.json') == p['reservation_sha256']
    for n, h in p['existing_TEST_manifest_SHA256'].items(): assert sha(ROOT / n) == h
    reservation = read(DATA / 'reserved-author-S3-fold0.json')
    author = read(DATA / 'publisher-annotations/gbc_annotations_v20_functional_component_id.json')['0']
    records = {r['idx']: r for r in author['train'] + author['test']}
    test = set(reservation['TEST_ids'])
    (DATA / 'originals').mkdir()
    with zipfile.ZipFile(DATA / 'publisher-v1.zip') as archive:
        def decode(i):
            raw = archive.read(reservation['idx_to_archive_member'][str(i)])
            with Image.open(io.BytesIO(raw)) as image:
                assert image.format == 'PNG' and image.size == (128, 128)
                assert image.getexif().get(274) in (None, 1)
                rgb = image.convert('RGB')
            path = DATA / 'originals' / f'{i:05d}.png'
            with path.open('xb') as out: out.write(raw)
            return {**records[i], 'id': 'gearbox/' + str(i), 'label': records[i]['label_int'],
                    'image': path.relative_to(ROOT).as_posix(), 'image_sha256': hashlib.sha256(raw).hexdigest(),
                    'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'orientation_phashes': fingerprints(rgb),
                    'metadata_group': reservation['idx_to_metadata_group'][str(i)]}
        with ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(decode, sorted(records)))
    new(DATA / 'decoded-fingerprints.json', {'rows': rows, 'policy_sha256': sha(POLICY)})
    print({'decoded_annotated_patches': len(rows)}, flush=True)
    old = {}
    for name in HELDOUT:
        for row in read(ROOT / name)['rows']:
            if row['phase'] in ('test', 'reserved_test'): old[row['image']] = row
    assert len(old) == 5746
    def old_fingerprint(row):
        assert sha(ROOT / row['image']) == row['image_sha256']
        with Image.open(ROOT / row['image']) as image: rgb = image.convert('RGB')
        return {'image': row['image'], 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'orientation_phashes': fingerprints(rgb)}
    with ThreadPoolExecutor(max_workers=4) as pool: oldrows = list(pool.map(old_fingerprint, old.values()))
    index = Bands(); exact = defaultdict(set)
    for i, r in enumerate(oldrows):
        index.add(r['orientation_phashes'], i); exact[r['rgb_sha256']].add(i)
    roots = {r['idx']: r['idx'] for r in rows}
    def find(i):
        while roots[i] != i: roots[i] = roots[roots[i]]; i = roots[i]
        return i
    def union(a, b):
        a, b = find(a), find(b); roots[max(a, b)] = min(a, b)
    metadata = {}; local = Bands(); local_rgb = defaultdict(set); near = []; matches = {}
    for i, r in enumerate(rows):
        if r['metadata_group'] in metadata: union(r['idx'], metadata[r['metadata_group']])
        else: metadata[r['metadata_group']] = r['idx']
        candidates = set(local_rgb[r['rgb_sha256']])
        for j in local.candidates(r['orientation_phashes']):
            if distance(r['orientation_phashes'], rows[j]['orientation_phashes']) <= 2: candidates.add(j)
        for j in sorted(candidates):
            union(r['idx'], rows[j]['idx'])
            near.append([r['idx'], rows[j]['idx']])
        local.add(r['orientation_phashes'], i); local_rgb[r['rgb_sha256']].add(i)
        hit = set(exact[r['rgb_sha256']])
        for j in index.candidates(r['orientation_phashes']):
            if distance(r['orientation_phashes'], oldrows[j]['orientation_phashes']) <= 2: hit.add(j)
        if hit: matches[r['idx']] = [oldrows[j]['image'] for j in sorted(hit)]
        if i % 3000 == 0: print({'grouped_patches': i, 'near_pairs': len(near), 'old_TEST_matches': len(matches)}, flush=True)
    testgroups = {find(i) for i in test}; oldgroups = {find(i) for i in matches}
    eligible = {find(r['idx']) for r in rows} - testgroups - oldgroups
    phases = group_allocation(eligible)
    for r in rows:
        g = find(r['idx']); r['group_id'] = str(g)
        r['phase'] = ('test' if r['idx'] in test else 'excluded_TEST_group' if g in testgroups
                      else 'excluded_old_TEST_group' if g in oldgroups else phases[g])
    counts = {phase: dict(Counter(str(r['label']) for r in rows if r['phase'] == phase))
              for phase in ('fit', 'validation', 'cal', 'test', 'excluded_TEST_group', 'excluded_old_TEST_group')}
    groups = {phase: sorted({r['group_id'] for r in rows if r['phase'] == phase}) for phase in counts}
    active = ('fit', 'validation', 'cal', 'test')
    assert all(set(groups[a]).isdisjoint(groups[b]) for i, a in enumerate(active) for b in active[i+1:])
    ready = (all(counts['fit'].get(str(y), 0) >= 100 for y in (0, 1)) and len(groups['fit']) >= 8
             and all(all(counts[k].get(str(y), 0) >= 20 for y in (0, 1)) and len(groups[k]) >= 3 for k in ('validation', 'cal', 'test'))
             and not any(r['idx'] in matches for r in rows if r['phase'] in ('fit', 'validation', 'cal')))
    audit = {'UTC': datetime.now(timezone.utc).isoformat(), 'ready': ready, 'policy_sha256': sha(POLICY),
             'annotated_patches': len(rows), 'counts': counts, 'groups_by_phase': groups,
             'metadata_connected_groups': 35, 'pixel_and_metadata_connected_groups': len({find(r['idx']) for r in rows}),
             'internal_exact_or_orientation_near_pairs': len(near), 'old_TEST_fingerprinted': len(oldrows),
             'old_TEST_direct_matches': matches, 'old_TEST_matches_in_reserved_TEST': sorted(test & set(matches)),
             'fixed_author_TEST_preserved': sorted(r['idx'] for r in rows if r['phase'] == 'test') == sorted(test),
             'unannotated_PNGs_excluded': 103, 'no_TEST_masks_or_model_scores_used': True,
             'model_features_scores_optimizer_updates': 0, 'label_scope': p['label_scope'],
             'physical_metadata_provided_by_author_not_new_independent_field_validation': True,
             'near_hash_is_not_physical_identity_proof': True}
    new(DATA / 'near-pairs.json', near)
    new(MANIFEST, {'schema': 'safelog-gearbox-grouped-v25', 'rows': rows, 'audit': audit})
    new(ROOT / 'reports/industrial-gearbox-v25-data-audit.json', audit)
    new(RUN / 'data-ready.json', {'ready': ready, 'manifest_sha256': sha(MANIFEST), 'audit': audit})
    print(audit, flush=True)
    assert ready, 'Fixed data gate failed; preserve state and do not run new features/optimizer'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('freeze', 'prepare'))
    args = parser.parse_args(); globals()[args.stage]()
