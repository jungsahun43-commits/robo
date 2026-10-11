"""Acquire a predetermined sequence-balanced TRAIN/VAL sample from official V1."""
import hashlib
import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/equipment-beltcrack'
INDEX = DATA / 'range-index'
OUT = DATA / 'starter-v1'
AVAILABLE_ONLY = False


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(members):
    groups = defaultdict(set)
    for row in members:
        match = re.fullmatch(r'BeltCrack14ks/json/(train|val)/(data\d+)/(\d+)\.json', row['name'])
        if match:
            groups[(match[1], match[2])].add(row['name'])
    assert len([k for k in groups if k[0] == 'train']) == 23
    assert len([k for k in groups if k[0] == 'val']) == 10
    selected = []
    for (split, group), names in sorted(groups.items()):
        ordered = sorted(names, key=lambda name: int(Path(name).stem))
        indices = np.linspace(0, len(ordered) - 1, min(32, len(ordered)), dtype=int)
        selected.extend({'name': ordered[i], 'split': split, 'sequence': group} for i in indices)
    assert len(selected) == 1031
    return selected


def fetch_segment(args):
    row, entry = args
    start, end = row['offset'], row['offset'] + row['header_size'] + row['packed'] - 1
    target = INDEX / f'selected-{entry["fileName"]}-{start}-{end}.bin'
    # Read and compare the real source header to the quick-open header used in
    # indexing, to prevent displaying one name and extracting another.
    with (INDEX / entry['fileName']).open('rb') as handle:
        handle.seek(start)
        expected = handle.read(row['header_size'])
    partial = DATA / 'archives' / (entry['fileName'] + '.partial')
    if not target.exists() and partial.exists() and partial.stat().st_size > end:
        with partial.open('rb') as handle:
            handle.seek(start)
            content = handle.read(end - start + 1)
        assert len(content) == end - start + 1 and content[:len(expected)] == expected
        with target.open('xb') as handle:
            handle.write(content)
    if not target.exists():
        assert not AVAILABLE_ONLY, 'Availability-only mode must never send another network request'
        for attempt in range(5):
            try:
                # Respect the publisher's rate limit; keep one transfer at a time.
                time.sleep(1.0)
                response = requests.get('https://china.scidb.cn/download?fileId=' + entry['id'],
                    headers={'Range': f'bytes={start}-{end}'}, timeout=(30, 60))
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After', '')
                    delay = max(30, float(retry_after)) if retry_after.isdigit() else 30 * (attempt + 1)
                    print(json.dumps({'publisher_rate_limit': True, 'retry_after_seconds': delay}), flush=True)
                    time.sleep(delay)
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers['Content-Range'] == f'bytes {start}-{end}/{entry["size"]}'
                assert len(response.content) == end - start + 1
                assert response.content[:len(expected)] == expected, 'Real/cache header mismatch'
                with target.open('xb') as handle:
                    handle.write(response.content)
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(min(30, 5 * (attempt + 1)))
    actual = target.read_bytes()
    assert len(actual) == end - start + 1 and actual[:len(expected)] == expected
    with (INDEX / entry['fileName']).open('r+b') as handle:
        handle.seek(start)
        handle.write(actual)
    return {'member': row['name'], 'volume': row['volume'], 'start': start, 'end': end,
            'range_sha256': sha(target), 'bytes': len(actual), 'source_header_equals_cache': True}


def main():
    OUT.mkdir(exist_ok=True)
    members = json.loads((INDEX / 'members.json').read_text(encoding='utf-8'))['members']
    entries = json.loads((DATA / 'source-files.json').read_text(encoding='utf-8'))['data']
    plan = OUT / 'selection.json'
    if not plan.exists():
        chosen = read_available = None
        if AVAILABLE_ONLY:
            read_available = json.loads((DATA / 'starter-v1/byte-availability-audit.json').read_text(encoding='utf-8'))
            chosen = read_available['available']
            assert len(chosen) == 679 and sum(r['split'] == 'train' for r in chosen) == 552
            assert sum(r['split'] == 'val' for r in chosen) == 127
        else:
            chosen = select(members)
        plan.write_text(json.dumps({'selected': chosen, 'max_frames_per_sequence': 32,
            'selection_rule': 'Evenly spaced numerical frame indices, before reading selected labels or model scores',
            'byte_availability_filter': AVAILABLE_ONLY, 'original_planned_photos': 1031,
            'availability_filter_rule': read_available['rule'] if read_available else None,
            'availability_audit_sha256': sha(DATA / 'starter-v1/byte-availability-audit.json') if AVAILABLE_ONLY else None,
            'source_test_selected': False, 'source_index_sha256': sha(INDEX / 'members.json'),
            'observed_archive': {'images': 9645, 'train': 5736, 'val': 1961, 'test': 1948,
                'sequence_counts': {'train': 23, 'val': 10, 'test': 9}},
            'publication_archive_count_mismatch': True}, indent=2), encoding='utf-8')
    selected = json.loads(plan.read_text(encoding='utf-8'))['selected']
    names = {r['name'] for r in selected}
    parts = [row for row in members if row['name'] in names]
    assert set(r['name'] for r in parts) == names
    start = time.monotonic()
    ranges = []
    # Sequential requests also stop immediately on an exhausted retry budget;
    # do not queue hundreds of retries behind a rate-limited request.
    for i, row in enumerate(parts, 1):
        value = fetch_segment((row, entries[row['volume']]))
        ranges.append(value)
        if i % 50 == 0:
            print(json.dumps({'selected_parts_verified': i, 'total': len(parts),
                'minutes': (time.monotonic() - start) / 60}), flush=True)
    listing = OUT / 'selected-members.txt'
    listing.write_text('\n'.join(r['name'] for r in selected), encoding='utf-8')
    dest = OUT / 'original-annotations'
    dest.mkdir(exist_ok=True)
    executable = ROOT / 'runs/equipment-tools/7zip-26.04/7z.exe'
    command = [str(executable), 'x', str(INDEX / entries[0]['fileName']), '-i@' + str(listing),
               '-pcv205', '-o' + str(dest), '-aos', '-spd', '-scsUTF-8']
    process = subprocess.run(command, capture_output=True)
    (OUT / 'extraction.log').write_bytes(process.stdout + process.stderr)
    assert process.returncode == 0, (process.stdout + process.stderr).decode(errors='replace')[-2000:]
    files = []
    for row in selected:
        path = dest.joinpath(*row['name'].split('/'))
        assert path.resolve().is_relative_to(dest.resolve())
        assert path.is_file()
        files.append({**row, 'sha256': sha(path), 'bytes': path.stat().st_size})
    output = OUT / 'acquisition.json'
    assert not output.exists(), 'Preserve completed acquisition'
    output.write_text(json.dumps({'status': 'complete', 'source': 'https://doi.org/10.57760/sciencedb.31181',
        'license': 'CC-BY-SA-4.0', 'archive_label': 'BeltCrack14ks', 'observed_total_images': 9645,
        'publication_count_matches': False, 'selection_sha256': sha(plan), 'members': files,
        'ranges': ranges, 'actual_selected_original_headers_verified': True,
        'selected_members_extractor_checksum_passed': True, 'whole_archive_md5_verified': False,
        'source_test_photos_or_labels_acquired': 0, 'embedded_images_to_be_audited': True,
        'script_sha256': sha(Path(__file__))}, indent=2), encoding='utf-8')
    print('Belt annotation/image starter acquired: ' + str(len(files)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--available-only', action='store_true')
    args = parser.parse_args()
    AVAILABLE_ONLY = args.available_only
    if AVAILABLE_ONLY:
        OUT = DATA / 'starter-available-v1'
    main()
