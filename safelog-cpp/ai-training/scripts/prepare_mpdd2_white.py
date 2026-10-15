"""Public original MPDD2 white-bracket download and fixed, model-free data audit."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import threading
import time

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.prepare_glinks_grouped import HELDOUT
from scripts.prepare_magnetic_tiles import Bands
from scripts.study_equipment_expansion import read
from safelog_ai.belt_normal_roi import fingerprints, distance

OUT = ROOT / 'data/industrial-mpdd2-white-v39'
INVENTORY = ROOT / 'data/industrial-mpdd2-complete-v38/inventory.json'
POLICY = ROOT / 'reports/industrial-mpdd2-white-v39-data-policy.json'
SOURCES = ('scripts/prepare_mpdd2_white.py', 'tests/test_mpdd2_white_data.py')
MAX_FILE = 16 * 1024 ** 2
MAX_TOTAL = 512 * 1024 ** 2


def source_role(name):
    parts = PurePosixPath(name).parts
    if len(parts) != 4 or parts[0] != 'bracket_white' or not re.fullmatch(r'\d{3}\.png', parts[3]):
        raise ValueError('Only observed white-bracket PNG names permitted')
    if parts[1:3] not in (('train', 'normal'), ('test', 'normal'), ('test', 'anomalous')):
        raise ValueError('Unknown publisher role or label')
    return parts[1], int(parts[2] == 'anomalous')


def split_groups(groups):
    ordered = sorted(set(groups), key=lambda group: (hashlib.sha256(('mpdd2white39:' + str(group)).encode()).hexdigest(), str(group)))
    count = math.ceil(.15 * len(ordered))
    return {group: 'validation' if i < count else 'cal' if i < count * 2 else 'fit'
            for i, group in enumerate(ordered)}


def committed():
    git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory=' + ROOT.parents[1].as_posix(), '-C', str(ROOT.parents[1])]
    for name in (*SOURCES, POLICY.relative_to(ROOT).as_posix()):
        content = subprocess.check_output(git + ['show', 'HEAD:safelog-cpp/ai-training/' + name])
        assert hashlib.sha256(content).hexdigest() == sha(ROOT / name), name


def freeze():
    assert not POLICY.exists() and not OUT.exists()
    assert sha(INVENTORY) == 'e3beff9e6e3615c973d4f8a3073856e1485ab106f5479d80c70d392a5365492b'
    entries = []
    for row in sorted(read(INVENTORY)['files'], key=lambda item: item['relative_path']):
        if row['relative_path'].startswith('bracket_white/'):
            role, label = source_role(row['relative_path'])
            assert re.fullmatch(r'[A-Za-z0-9_-]{10,100}', row['id'])
            entries.append({**row, 'publisher_role': role, 'label': label})
    assert Counter((row['publisher_role'], row['label']) for row in entries) == {('train', 0): 195, ('test', 0): 49, ('test', 1): 91}
    assert len({row['id'] for row in entries}) == 335
    OUT.mkdir()
    new(OUT / 'reserved-before-download.json', {'rows': entries, 'UTC': datetime.now(timezone.utc).isoformat(),
        'original_TEST_reserved_all140': True, 'source_inventory_SHA256': sha(INVENTORY),
        'photo_downloads_decodes_viewing_model_features_scores_optimizer': 0})
    policy = {'UTC': datetime.now(timezone.utc).isoformat(), 'frozen_before_MPDD2_image_download_or_decoding': True,
        'source_code_SHA256': {name: sha(ROOT / name) for name in SOURCES},
        'reservation_SHA256': sha(OUT / 'reserved-before-download.json'),
        'retained_TEST_manifest_SHA256': {name: sha(ROOT / name) for name in HELDOUT},
        'source': 'author stepanje/MPDD2 public Drive, white bracket only; DOI10.1109/ICUMT57764.2022.9943437',
        'license': 'CC BY-NC-SA4.0, noncommercial academic research, author/source/version/change attribution, sharealike; other source licenses unchanged',
        'download': {'workers': 4, 'max_file_bytes': MAX_FILE, 'max_total_bytes': MAX_TOTAL, 'required_reserve_bytes': 1024 ** 3,
                     'attempts': 3, 'access': 'public uc download without accounts, cookies, private keys, personal information or access bypass'},
        'image_audit': 'raw bytes local SHA256, PNG signature/format, RGB pixels/size/orientation hashes; no original reencode; no mask labels available',
        'groups': 'union exactRGB or any8orientation pHash<=2; unknown physical/capture groups, numeric filenames not physical IDs. Similarity is conservative exclusion, not identity proof.',
        'exclusion': 'original publisherTEST140 remains TEST; TRAIN in a newTEST group or linked to any retained8855TEST excluded. Never move TEST or near-TEST to FIT.',
        'split': 'eligible normal TRAIN connected groups sorted SHA256(mpdd2white39:+groupminindex); firstceil15percent VAL,nextceil15percent CAL,restFIT, once only',
        'minimums': {'normal_FIT': 30, 'normal_VAL': 20, 'normal_CAL': 20, 'groups_each_active_split': 3, 'TEST_each_label': 20},
        'reason_for_minimums': 'new small normal-reference candidate, not changing failed GLinks35 requirements; sufficient support does not prove field accuracy',
        'failed_gate': 'preserve all records and stop before encoder features or training; no thresholds/splits/minimums retried',
        'semantic_scope': 'publisher white-bracket visible assembly/surface anomaly, not internal failure, mechanical operation, or safety certification',
        'other_sources': 'brown/grey/rosette normalTEST below20; not combined with white to satisfy gate; GLinks quarantine unchanged',
        'new_training_protocol_fixed': False, 'new_model_features_scores_optimizer': 0,
        'agent_photo_viewing': 'only after split, fixed FIT examples; no TEST selection or relabeling'}
    new(POLICY, policy)
    print({'policy_SHA256': sha(POLICY), 'reserved_files': len(entries)}, flush=True)


def load_policy():
    value = read(POLICY)
    for name, expected in value['source_code_SHA256'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(OUT / 'reserved-before-download.json') == value['reservation_SHA256']
    committed()
    return value


def fetch():
    load_policy()
    assert not (OUT / 'download.json').exists()
    assert shutil.disk_usage(ROOT).free > MAX_TOTAL + 1024 ** 3
    (OUT / 'originals').mkdir(exist_ok=True)
    (OUT / 'receipts').mkdir(exist_ok=True)
    entries = read(OUT / 'reserved-before-download.json')['rows']
    budget_lock = threading.Lock()
    used_bytes = sum(path.stat().st_size for path in (OUT / 'originals').iterdir() if path.is_file())
    assert used_bytes <= MAX_TOTAL

    def one(item):
        nonlocal used_bytes
        index, entry = item
        target = OUT / 'originals' / f'{index:04d}.png'
        receipt_path = OUT / 'receipts' / f'{index:04d}.json'
        if receipt_path.exists():
            saved = read(receipt_path)
            assert target.exists() and sha(target) == saved['image_sha256'] and saved['source_id'] == entry['id']
            return saved
        assert not target.exists()
        failures = []
        for attempt in range(3):
            part = target.with_suffix(f'.attempt{attempt}.partial')
            if part.exists():
                failures.append({'attempt': attempt, 'error': 'previous_partial_preserved'})
                continue
            try:
                url = 'https://drive.google.com/uc?export=download&id=' + entry['id']
                with requests.get(url, stream=True, timeout=(20, 45)) as response:
                    response.raise_for_status()
                    from urllib.parse import urlsplit
                    host = urlsplit(response.url).hostname
                    assert host and (host == 'drive.google.com' or host == 'drive.usercontent.google.com' or host.endswith('.googleusercontent.com'))
                    assert int(response.headers.get('Content-Length', '0')) <= MAX_FILE
                    size = 0
                    with part.open('xb') as output:
                        for chunk in response.iter_content(65536):
                            if chunk:
                                size += len(chunk)
                                assert size <= MAX_FILE
                                with budget_lock:
                                    used_bytes += len(chunk)
                                    assert used_bytes <= MAX_TOTAL, 'Total raw/partial download budget exceeded'
                                output.write(chunk)
                with part.open('rb') as source:
                    assert source.read(8) == b'\x89PNG\r\n\x1a\n', 'Expected public PNG, not login/confirmation page'
                part.rename(target)
                result = {**entry, 'source_id': entry['id'], 'image': target.relative_to(ROOT).as_posix(),
                          'image_sha256': sha(target), 'bytes': size, 'URL': url,
                          'HTTP_status': response.status_code, 'downloaded_UTC': datetime.now(timezone.utc).isoformat(),
                          'failures': failures, 'SHA256_is_local_not_author_supplied': True}
                new(receipt_path, result)
                if index % 25 == 0: print({'completed_index': index, 'bytes': size}, flush=True)
                return result
            except Exception as error:
                failures.append({'attempt': attempt, 'error': repr(error)})
                time.sleep(1 + attempt)
        new(OUT / 'receipts' / f'{index:04d}-failure.json', {'entry': entry, 'failures': failures})
        raise RuntimeError(f'Public download failed for reserved index {index}; partials preserved')

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, enumerate(entries)))
    assert sum(row['bytes'] for row in rows) <= MAX_TOTAL
    result = {'UTC': datetime.now(timezone.utc).isoformat(), 'rows': rows, 'files': len(rows),
              'total_bytes': sum(row['bytes'] for row in rows), 'policy_SHA256': sha(POLICY),
              'photo_decoding_viewing_or_model_execution': 0, 'public_no_login_or_personal_information': True}
    new(OUT / 'download.json', result)
    print({key: value for key, value in result.items() if key != 'rows'}, flush=True)


def audit():
    policy = load_policy()
    assert not (OUT / 'manifest.json').exists()
    for name, expected in policy['retained_TEST_manifest_SHA256'].items():
        assert sha(ROOT / name) == expected
    download = read(OUT / 'download.json')
    assert download['policy_SHA256'] == sha(POLICY)

    def decode(row):
        path = ROOT / row['image']
        assert sha(path) == row['image_sha256']
        with Image.open(path) as image:
            assert image.format == 'PNG' and image.getexif().get(274) in (None, 1)
            image.load()
            rgb = image.convert('RGB')
        return {**row, 'id': 'mpdd2/' + row['relative_path'], 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                'orientation_phashes': fingerprints(rgb), 'size': list(rgb.size), 'mask': None,
                'category': 'mpdd2_bracket_white', 'defect': row['relative_path'].split('/')[2]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(decode, download['rows']))
    new(OUT / 'decoded-fingerprints.json', {'rows': rows, 'download_SHA256': sha(OUT / 'download.json')})
    # Reuse an already audited old TEST fingerprint only after checking its original file SHA.
    heldout = {}
    for name in HELDOUT:
        for row in read(ROOT / name)['rows']:
            if row['phase'] in ('test', 'reserved_test'):
                heldout[row['image']] = row
    assert len(heldout) == 8855

    def old_fingerprint(row):
        assert sha(ROOT / row['image']) == row['image_sha256']
        if row.get('rgb_sha256') and row.get('orientation_phashes'):
            return {key: row[key] for key in ('image', 'rgb_sha256', 'orientation_phashes')}
        with Image.open(ROOT / row['image']) as image:
            rgb = image.convert('RGB')
        return {'image': row['image'], 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'orientation_phashes': fingerprints(rgb)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        old = list(pool.map(old_fingerprint, heldout.values()))
    old_index = Bands()
    old_exact = defaultdict(set)
    for i, row in enumerate(old):
        old_index.add(row['orientation_phashes'], i)
        old_exact[row['rgb_sha256']].add(i)
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        parent[max(i, j)] = min(i, j)

    local = Bands()
    exact = {}
    pairs = []
    matches = {}
    for i, row in enumerate(rows):
        if row['rgb_sha256'] in exact:
            union(i, exact[row['rgb_sha256']])
        else:
            exact[row['rgb_sha256']] = i
        for j in sorted(local.candidates(row['orientation_phashes'])):
            if distance(row['orientation_phashes'], rows[j]['orientation_phashes']) <= 2:
                union(i, j)
                pairs.append([i, j])
        local.add(row['orientation_phashes'], i)
        hits = set(old_exact.get(row['rgb_sha256'], ()))
        for j in old_index.candidates(row['orientation_phashes']):
            if distance(row['orientation_phashes'], old[j]['orientation_phashes']) <= 2:
                hits.add(j)
        if hits:
            matches[i] = [old[j]['image'] for j in sorted(hits)]
    test_groups = {find(i) for i, row in enumerate(rows) if row['publisher_role'] == 'test'}
    old_groups = {find(i) for i in matches}
    eligible = {find(i) for i, row in enumerate(rows) if row['publisher_role'] == 'train'} - test_groups - old_groups
    allocation = split_groups(eligible)
    for i, row in enumerate(rows):
        group = find(i)
        row['group_id'] = str(group)
        row['phase'] = ('test' if row['publisher_role'] == 'test' else 'excluded_TEST_group' if group in test_groups
                        else 'excluded_old_TEST_group' if group in old_groups else allocation[group])
    phases = ('fit', 'validation', 'cal', 'test', 'excluded_TEST_group', 'excluded_old_TEST_group')
    counts = {phase: dict(Counter(str(row['label']) for row in rows if row['phase'] == phase)) for phase in phases}
    groups = {phase: sorted({row['group_id'] for row in rows if row['phase'] == phase}) for phase in phases}
    assert all(set(groups[a]).isdisjoint(groups[b]) for i, a in enumerate(phases[:4]) for b in phases[:4][i+1:])
    ready = counts['fit'].get('0', 0) >= 30
    ready = ready and all(counts[phase].get('0', 0) >= 20 for phase in ('validation', 'cal'))
    ready = ready and all(len(groups[phase]) >= 3 for phase in phases[:4])
    ready = ready and all(counts['test'].get(str(label), 0) >= 20 and len({row['group_id'] for row in rows if row['phase'] == 'test' and row['label'] == label}) >= 3 for label in (0, 1))
    result = {'UTC': datetime.now(timezone.utc).isoformat(), 'ready': bool(ready), 'counts': counts, 'groups': groups,
        'TEST_groups_each_label': {str(label): len({row['group_id'] for row in rows if row['phase'] == 'test' and row['label'] == label}) for label in (0, 1)},
        'new_photos': len(rows), 'original_TEST140_preserved': sum(row['phase'] == 'test' for row in rows) == 140,
        'photo_sizes': dict(Counter(str(row['size']) for row in rows)), 'retained_previous_TEST_checked': len(old),
        'old_TEST_direct_matching_new_photos': len(matches), 'old_TEST_matching_TRAIN_photos': sum(rows[i]['publisher_role'] == 'train' for i in matches),
        'internal_near_links': len(pairs), 'connected_groups': len({find(i) for i in range(len(rows))}),
        'numeric_filenames_not_physical_ids': True, 'physical_capture_independence_unverified': True,
        'no_new_encoder_features_scores_gradients_optimizer': True, 'no_photo_contents_visually_viewed': True,
        'independent_field_validation': False, 'new_training_protocol_fixed': False,
        'policy_SHA256': sha(POLICY), 'download_SHA256': sha(OUT / 'download.json')}
    new(OUT / 'near-pairs-and-old-matches.json', {'near_pairs': pairs, 'old_TEST_matches': matches})
    new(OUT / 'manifest.json', {'schema': 'safelog-mpdd2-white-v39', 'rows': rows, 'audit': result})
    result['manifest_SHA256'] = sha(OUT / 'manifest.json')
    new(ROOT / 'reports/industrial-mpdd2-white-v39-data-audit.json', result)
    print({key: value for key, value in result.items() if key != 'groups'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'fetch', 'audit'))
    globals()[parser.parse_args().stage]()
