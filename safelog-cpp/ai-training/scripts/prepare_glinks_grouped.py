"""Original G-Link normal TRAIN, retained publisher TEST, grouped pixel audit."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.glinks_data import publisher_record, normal_group_split
from safelog_ai.belt_normal_roi import fingerprints, distance
from scripts.prepare_magnetic_tiles import Bands
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.fetch_glinks_source import ARCHIVE, OUT, SIZE

POLICY = ROOT / 'reports/industrial-glinks-v35-data-policy.json'
HELDOUT = ('data/equipment-expansion-v1/manifest.json',
           'data/industrial-ksdd-supervised-v10/manifest.json',
           'data/industrial-bsclass-v1/manifest.json',
           'data/industrial-magnetic-tiles-v16/manifest.json',
           'data/industrial-ksdd1-v18/manifest-r1.json',
           'data/industrial-gearbox-source-v25/manifest.json')
SOURCES = ('scripts/prepare_glinks_grouped.py', 'safelog_ai/glinks_data.py', 'tests/test_glinks_data.py')


def metadata_audit():
    archive = read(OUT / 'archive-members-before-pixels.json')
    rows = [{'member': entry['member'], **publisher_record(entry['member']), 'bytes': entry['bytes']}
            for entry in archive['members'] if not entry['is_directory']]
    counts = Counter(row['publisher_role'] + '/' + str(row['label']) for row in rows)
    assert counts == {'train/0': 3424, 'test/0': 403, 'test/1': 165}
    by = {role: {row['acquisition_uuid_proxy'] for row in rows if row['publisher_role'] == role}
          for role in ('train', 'test')}
    result = {'UTC': datetime.now(timezone.utc).isoformat(), 'original_archive_SHA256': sha(ARCHIVE),
              'photographs': len(rows), 'normal_TRAIN': 3424, 'normal_TEST': 403, 'defect_TEST': 165,
              'actual_mask_files_in_original_archive': 0, 'image_decoding': 0,
              'TRAIN_uuid_proxies': len(by['train']), 'TEST_uuid_proxies': len(by['test']),
              'TRAIN_TEST_uuid_intersection': sorted(by['train'] & by['test']),
              'uuid_is_filename_acquisition_proxy_not_verified_physical_identity': True,
              'publisher_web_page_normal_TEST_464_disagrees_with_actual_original_403': True,
              'source_description_mask_claim_not_local_mask_availability': True,
              'initial_prefix_counter_included_filenames_not_directory_counts': True,
              'publisher_TEST_reserved_before_pixel_viewing': True,
              'current_new_features_scores_optimizer': 0, 'no_other_variants_as_extra_photos': True}
    return rows, result


def freeze():
    assert not POLICY.exists() and not (OUT / 'manifest.json').exists()
    assert sha(ARCHIVE) == 'f838533cf779a20af519f61f56b5bfd2f4df0c951c79f36f95d230d0b884793a'
    rows, audit = metadata_audit()
    new(OUT / 'metadata-audit-r1.json', audit)
    new(ROOT / 'reports/industrial-glinks-v35-metadata-audit-r1.json', audit)
    new(OUT / 'reserved-original-TEST-before-pixels.json', {'rows': rows, 'audit': audit})
    policy = {'UTC': datetime.now(timezone.utc).isoformat(), 'fixed_before_new_image_decoding_or_viewing': True,
              'script_sha256': {name: sha(ROOT / name) for name in SOURCES},
              'archive_SHA256': audit['original_archive_SHA256'],
              'metadata_reservation_SHA256': sha(OUT / 'reserved-original-TEST-before-pixels.json'),
              'retained_TEST_manifests_SHA256': {name: sha(ROOT / name) for name in HELDOUT},
              'original_publisher_TEST': 'all568 original TEST images preserved; masks absent in this ZIP, not training targets',
              'grouping': 'union same filename UUID acquisition proxy OR exact RGB OR any8orientation pHash distance<=2. Similarity does not prove physical identity. Exclude TRAIN in TEST-connected group or connected to any retained8855TEST.',
              'split': 'remaining normal TRAIN connected groups sorted SHA256(glinks35:+str(groupminindex)); firstceil15percent VAL,nextceil15percent CAL,restFIT. No model/label-dependent retries or changed limits.',
              'minimums': 'normal FIT>=300 and>=8groups; normal VAL andCAL each>=100 and>=3groups; TEST eachlabel>=20 and>=3groups. Failure stops before model features/optimizer; original TEST remains retained.',
              'pixels': 'raw JPEG RGB convert no reencode; EXIF None/1, store bytesSHA/RGBSHA/8orientationpHash/size; extract to fixed numeric filenames, never member-provided filesystem paths',
              'disk': 'current3.5GB; JPEG originals~307MB, small receipts/manifest<25MB, no dense features yet; require extractionbytes+1GiB reserve before decode',
              'training_protocol_fixed': False, 'optimizer_or_model_scores': 0,
              'source_scope': 'visible coupling-link surface condition in area-camera robot acquisition, not smartphone field/internal damage/load safety',
              'license': 'CC BY-NC-SA4.0 dataset v1, DOI10.5281/zenodo.15693611, author/version/change attribution and sharealike; existing sources unchanged',
              'new_units_passed': 2}
    new(POLICY, policy)
    print(audit, flush=True)


def prepare():
    policy = read(POLICY)
    for name, expected in policy['script_sha256'].items(): assert sha(ROOT / name) == expected
    git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory=' + ROOT.parents[1].as_posix(), '-C', str(ROOT.parents[1])]
    for name in (*SOURCES, POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git + ['show', 'HEAD:safelog-cpp/ai-training/' + name])).hexdigest() == sha(ROOT / name)
    assert sha(ARCHIVE) == policy['archive_SHA256']
    assert sha(OUT / 'reserved-original-TEST-before-pixels.json') == policy['metadata_reservation_SHA256']
    for name, expected in policy['retained_TEST_manifests_SHA256'].items(): assert sha(ROOT / name) == expected
    assert shutil.disk_usage(ROOT).free > SIZE + 1024 ** 3
    assert not (OUT / 'manifest.json').exists() and not (OUT / 'originals').exists()
    (OUT / 'originals').mkdir()
    entries = read(OUT / 'reserved-original-TEST-before-pixels.json')['rows']
    with zipfile.ZipFile(ARCHIVE) as archive:
        def decode(item):
            index, entry = item; raw = archive.read(entry['member'])
            with Image.open(io.BytesIO(raw)) as image:
                assert image.format == 'JPEG' and image.getexif().get(274) in (None, 1)
                rgb = image.convert('RGB')
            output = OUT / 'originals' / f'{index:05d}.jpg'
            with output.open('xb') as target: target.write(raw)
            return {**entry, 'id': 'glinks/' + entry['member'], 'image': output.relative_to(ROOT).as_posix(),
                    'image_sha256': hashlib.sha256(raw).hexdigest(), 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                    'orientation_phashes': fingerprints(rgb), 'size': list(rgb.size), 'mask': None,
                    'label_scope': 'publisher visible surface condition, not whole-machine function or locationGT'}
        with ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(decode, enumerate(entries)))
    new(OUT / 'decoded-fingerprints.json', {'rows': rows, 'policy_SHA256': sha(POLICY)})
    heldout = {}
    for name in HELDOUT:
        for row in read(ROOT / name)['rows']:
            if row['phase'] in ('test', 'reserved_test'): heldout[row['image']] = row
    assert len(heldout) == 8855
    def old_fingerprint(row):
        assert sha(ROOT / row['image']) == row['image_sha256']
        if row.get('rgb_sha256') and row.get('orientation_phashes'):
            return {key: row[key] for key in ('image', 'rgb_sha256', 'orientation_phashes')}
        with Image.open(ROOT / row['image']) as image: rgb = image.convert('RGB')
        return {'image': row['image'], 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'orientation_phashes': fingerprints(rgb)}
    with ThreadPoolExecutor(max_workers=4) as pool: old = list(pool.map(old_fingerprint, heldout.values()))
    old_index = Bands(); old_exact = defaultdict(set)
    for index, row in enumerate(old):
        old_index.add(row['orientation_phashes'], index); old_exact[row['rgb_sha256']].add(index)
    parent = list(range(len(rows)))
    def find(index):
        while parent[index] != index: parent[index] = parent[parent[index]]; index = parent[index]
        return index
    def union(a, b):
        a, b = find(a), find(b); parent[max(a, b)] = min(a, b)
    uuid = {}; exact = {}; local = Bands(); near = []; matches = {}
    for i, row in enumerate(rows):
        proxy = row['acquisition_uuid_proxy']
        if proxy in uuid: union(i, uuid[proxy])
        else: uuid[proxy] = i
        if row['rgb_sha256'] in exact: union(i, exact[row['rgb_sha256']])
        else: exact[row['rgb_sha256']] = i
        for j in sorted(local.candidates(row['orientation_phashes'])):
            if distance(row['orientation_phashes'], rows[j]['orientation_phashes']) <= 2:
                union(i, j); near.append([i, j])
        local.add(row['orientation_phashes'], i)
        hit = set(old_exact.get(row['rgb_sha256'], ()))
        for j in old_index.candidates(row['orientation_phashes']):
            if distance(row['orientation_phashes'], old[j]['orientation_phashes']) <= 2: hit.add(j)
        if hit: matches[i] = [old[j]['image'] for j in sorted(hit)]
    test_groups = {find(i) for i, row in enumerate(rows) if row['publisher_role'] == 'test'}
    old_groups = {find(i) for i in matches}
    candidates = {find(i) for i, row in enumerate(rows) if row['publisher_role'] == 'train'} - test_groups - old_groups
    allocation = normal_group_split(candidates)
    for i, row in enumerate(rows):
        group = find(i); row['group_id'] = str(group)
        row['phase'] = ('test' if row['publisher_role'] == 'test' else 'excluded_TEST_group' if group in test_groups
                        else 'excluded_old_TEST_group' if group in old_groups else allocation[group])
    phases = ('fit', 'validation', 'cal', 'test', 'excluded_TEST_group', 'excluded_old_TEST_group')
    counts = {phase: dict(Counter(str(row['label']) for row in rows if row['phase'] == phase)) for phase in phases}
    groups = {phase: sorted({row['group_id'] for row in rows if row['phase'] == phase}) for phase in phases}
    active = phases[:4]
    assert all(set(groups[a]).isdisjoint(groups[b]) for i, a in enumerate(active) for b in active[i+1:])
    ready = counts['fit'].get('0', 0) >= 300 and len(groups['fit']) >= 8
    ready = ready and all(counts[phase].get('0', 0) >= 100 and len(groups[phase]) >= 3 for phase in ('validation', 'cal'))
    ready = ready and all(counts['test'].get(str(label), 0) >= 20 and len({row['group_id'] for row in rows if row['phase'] == 'test' and row['label'] == label}) >= 3 for label in (0, 1))
    audit = {'UTC': datetime.now(timezone.utc).isoformat(), 'ready': bool(ready), 'counts': counts, 'groups_by_phase': groups,
             'TEST_groups_by_label': {str(label): len({row['group_id'] for row in rows if row['phase'] == 'test' and row['label'] == label}) for label in (0,1)},
             'retained_previous_TEST_photos_checked': len(old), 'old_TEST_direct_matches': matches,
             'internal_orientation_near_pairs': len(near), 'connected_groups': len({find(i) for i in range(len(rows))}),
             'publisher_TEST_all568_preserved': sum(row['phase'] == 'test' for row in rows) == 568,
             'actual_mask_files': 0, 'metadata_UUID_not_confirmed_physical_id': True,
             'near_hash_not_identity_proof': True, 'agent_image_content_viewing': False,
             'new_encoder_features_scores_optimizer': 0, 'training_protocol_fixed': False,
             'independent_field_validation': False, 'policy_SHA256': sha(POLICY)}
    new(OUT / 'orientation-near-pairs.json', near)
    new(OUT / 'manifest.json', {'schema': 'safelog-glinks-original-grouped-v35', 'rows': rows, 'audit': audit})
    new(ROOT / 'reports/industrial-glinks-v35-data-audit.json', audit)
    print({key: value for key, value in audit.items() if key not in ('old_TEST_direct_matches', 'groups_by_phase')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('freeze', 'prepare'))
    args = parser.parse_args(); globals()[args.stage]()
