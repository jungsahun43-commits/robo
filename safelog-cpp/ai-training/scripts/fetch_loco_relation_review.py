"""Fetch a bounded normal-FIT review sample from the author's public ZIP.

Never extracts an archive wholesale, executes publisher code, or reads TEST labels.
Run `freeze` before `fetch`. Each downloaded entry is checked against its ZIP CRC.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import zlib

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/industrial-relation-sources-v14'
DATA = ROOT / 'data/industrial-relation-review-v14'
MANIFEST = ROOT / 'data/industrial-loco-context-v13/manifest.json'
PREFIX = 'LOCO-Annotations/splicing_connectors/'
AUTHOR = 'https://github.com/22strongestme/LOCO-Annotations'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def eligible(row):
    return (row['category'] == 'loco_splicing_connectors' and row['phase'] == 'fit'
            and row['publisher_split'] == 'train' and row['label'] == 0
            and '/train/good/' in row['id'] and not row['masks'])


def freeze():
    assert not DATA.exists(), 'Existing review protocol must not be replaced.'
    m = read(MANIFEST)
    pool = [r for r in m['rows'] if eligible(r)]
    assert len(pool) == 199
    selected = sorted(pool, key=lambda r: hashlib.sha256(
        ('loco-relation-review-v14:' + r['id']).encode()).hexdigest())[:24]
    assert len({r['appearance_group_proxy'] for r in selected}) == 24
    report = {
        'schema': 'loco_relation_review_protocol_v1',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'source_manifest_sha256': sha(MANIFEST),
        'source_archive_index_sha256': sha(SOURCE / 'logicode_zip_index.json'),
        'author_url': AUTHOR,
        'author_repo_commit': read(SOURCE / 'logicode_commit.json')['sha'],
        'selection': 'First24 SHA256(loco-relation-review-v14:sourceID), only clean v13 normal FIT199',
        'rows': selected,
        'review_photos': 24,
        'TRAIN_defects_acquired': 0,
        'TEST_or_VAL_or_CAL_annotations_to_fetch': 0,
        'fetch_policy': 'Public HTTP206 bounded ranges, per-entry size/CRC/SHA, no author code execution',
        'image_alignment_gate': 'All24 author PNG RGB SHA must equal local v13 source image RGB SHA',
        'annotation_acceptance': 'Publisher polygons are candidates; derived grouping/counts/endpoints require review',
        'license_gate': 'Archive contains MVTec CC BY-NC-SA4.0; separate new annotation rights not expressly stated',
        'optimizer_updates_planned': 0,
        'training_ready': False,
        'physical_acquisition_groups_verified': False,
        'inherited_parent_exposure_undone': False,
    }
    write_new(DATA / 'protocol.json', report)
    write_new(ROOT / 'reports/industrial-loco-relation-review-v14-protocol.json', report)
    print({'protocol_frozen': True, 'photos': len(selected), 'training_ready': False})


def decode_entry(blob, entry):
    name = entry['name']
    assert PurePosixPath(name).parts[:2] == ('LOCO-Annotations', 'splicing_connectors')
    assert '..' not in PurePosixPath(name).parts and '\\' not in name
    assert blob[:4] == b'PK\x03\x04'
    flags, method = struct.unpack_from('<HH', blob, 6)
    assert not flags & 1 and method == entry['method'] and method in (0, 8)
    n, extra = struct.unpack_from('<HH', blob, 26)
    assert blob[30:30+n].decode('utf-8') == name
    content = blob[30+n+extra:30+n+extra+entry['compressed']]
    assert len(content) == entry['compressed'] and 0 < entry['size'] < 5_000_000
    if method == 8:
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(content, entry['size'] + 1)
        assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    else:
        raw = content
    assert len(raw) == entry['size'] and zlib.crc32(raw) == entry['CRC']
    return raw


def fetch():
    protocol = read(DATA / 'protocol.json')
    assert sha(MANIFEST) == protocol['source_manifest_sha256']
    assert sha(SOURCE / 'logicode_zip_index.json') == protocol['source_archive_index_sha256']
    assert all(eligible(r) for r in protocol['rows'])
    index = read(SOURCE / 'logicode_zip_index.json')
    entries = {r['name']: r for r in index['entries']}
    warning = (SOURCE / 'logicode_drive_response.html').read_text(encoding='utf-8')
    params = dict(re.findall(r'name="([^"]+)" value="([^"]+)"', warning))
    assert params.get('id') == '1LjsX6bmpC1IIipQSlcq1RwfNC78R0iU-'
    tasks = []
    for row in protocol['rows']:
        for suffix in ('.json', '.png'):
            name = PREFIX + 'train/good/' + Path(row['image']).stem + suffix
            assert name in entries
            tasks.append((name, row))
    for name in ('license.txt', 'readme.txt'):
        tasks.append((PREFIX + name, None))

    def download(task):
        name, row = task
        entry = entries[name]
        assert 0 < entry['compressed'] < 5_000_000
        start = entry['header_offset']
        end = start + 30 + len(name.encode()) + 512 + entry['compressed'] - 1
        with requests.get('https://drive.usercontent.google.com/download', params=params,
                          headers={'Range': f'bytes={start}-{end}',
                                   'User-Agent': 'SafeLog-research-source-audit/1.0'},
                          timeout=45, stream=True) as response:
            assert response.status_code == 206
            assert response.headers['Content-Range'] == f'bytes {start}-{end}/{index["total_bytes"]}'
            chunks = []; size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                assert size <= end - start + 1
                chunks.append(chunk)
            blob = b''.join(chunks)
        assert len(blob) == end - start + 1
        raw = decode_entry(blob, entry)
        target = DATA / 'publisher' / name.removeprefix(PREFIX)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == raw, 'Existing source bytes differ; retain for investigation.'
        else:
            with target.open('xb') as f: f.write(raw)
        receipt = {'entry': name, 'downloaded_bytes': len(blob), 'content_bytes': len(raw),
                   'content_sha256': hashlib.sha256(raw).hexdigest(), 'CRC_verified': True,
                   'authentication_used': False, 'utc': datetime.now(timezone.utc).isoformat()}
        if name.endswith('.png'):
            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert('RGB')
                assert list(im.size) == row['size']
                assert hashlib.sha256(im.tobytes()).hexdigest() == row['rgb_sha256']
            assert sha(ROOT / row['image']) == row['image_sha256']
            receipt['local_source_RGB_exact'] = True
            receipt['local_source_file_bytes_exact'] = row['image_sha256'] == receipt['content_sha256']
        return receipt

    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(download, tasks))
    write_new(DATA / 'download-receipts.json', receipts)
    print({'downloaded_entries': len(receipts), 'PNG_RGB_equal': sum(r.get('local_source_RGB_exact', False) for r in receipts),
           'downloaded_bytes': sum(r['downloaded_bytes'] for r in receipts), 'optimizer_updates': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'fetch'))
    args = parser.parse_args()
    freeze() if args.mode == 'freeze' else fetch()
