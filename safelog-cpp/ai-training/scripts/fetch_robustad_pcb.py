"""Fetch official, ungated RobustAD PCB TRAIN normals and TEST0 with author hashes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/equipment-robustad-pcb-v1'
REV = '82c642b2d18be60f2dab104f8dd4dea1f4ccfacb'
REPO = 'AmazonScience/RobustAD'
API = 'https://huggingface.co/api/datasets/' + REPO
BASE = 'https://huggingface.co/datasets/' + REPO + '/resolve/' + REV + '/'


def json_new(path, value):
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, ensure_ascii=False)


def metadata(split):
    name = 'PCB/pcb_data_dir_' + split + '/metadata.jsonl'
    response = requests.get(BASE + name, timeout=40); response.raise_for_status()
    rows = [json.loads(line) for line in response.text.splitlines()]
    assert all(row['label'] in [0, 1] for row in rows)
    return name, response.content, rows


def fetch(record):
    name = record['path']; path = DATA / 'originals' / name; path.parent.mkdir(parents=True, exist_ok=True)
    expected = record.get('lfs', {}).get('oid')
    if path.exists():
        raw = path.read_bytes()
    else:
        response = requests.get(BASE + quote(name, safe='/'), timeout=(15, 90)); response.raise_for_status()
        raw = response.content
        assert len(raw) == record['size'], name
        if expected:
            assert hashlib.sha256(raw).hexdigest() == expected, name
        with path.open('xb') as output:
            output.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert len(raw) == record['size'] and (not expected or digest == expected), name
    return {'path': name, 'size_bytes': len(raw), 'sha256': digest, 'author_lfs_sha256_verified': bool(expected)}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    assert not (DATA / 'acquisition.json').exists(), 'Preserve completed acquisition'
    info = requests.get(API + '/revision/' + REV, timeout=40); info.raise_for_status()
    assert info.json()['sha'] == REV and not info.json().get('gated')
    readme = requests.get(BASE + 'README.md', timeout=40); readme.raise_for_status()
    assert 'cc-by-4.0' in readme.text
    (DATA / 'AUTHOR_README.md').write_bytes(readme.content)
    selected = []; records = {}
    for split in ['train', 'test0']:
        name, raw, rows = metadata(split)
        path = DATA / 'originals' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        usable = [row for row in rows if split != 'train' or row['label'] == 0]
        needed = {f'PCB/pcb_data_dir_{split}/' + row['file_name'] for row in usable}
        needed.update(row['mask'] for row in usable if row['mask'])
        tree = requests.get(API + '/tree/' + REV + '/PCB/pcb_data_dir_' + split,
                            params={'recursive': 'true', 'limit': 1000}, timeout=40)
        tree.raise_for_status(); inventory = {r['path']: r for r in tree.json() if r['type'] == 'file'}
        assert needed <= inventory.keys()
        selected.extend(inventory[name] for name in sorted(needed))
        records[split] = {'metadata_path': name, 'metadata_sha256': hashlib.sha256(raw).hexdigest(),
                          'author_rows': len(rows), 'selected_rows': usable,
                          'selected_label_counts': dict(Counter(row['label'] for row in usable))}
    size = sum(row['size'] for row in selected); assert size < 1_000_000_000
    json_new(DATA / 'download-plan.json', {'revision': REV, 'rows': records, 'files': selected, 'bytes': size})
    print(json.dumps({'selected_files': len(selected), 'bytes': size,
                      'label_counts': {s: r['selected_label_counts'] for s, r in records.items()}}), flush=True)
    downloaded = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, row) for row in selected]
        for future in as_completed(futures):
            downloaded.append(future.result())
            if len(downloaded) % 40 == 0:
                print(json.dumps({'downloaded': len(downloaded), 'total': len(selected)}), flush=True)
    json_new(DATA / 'acquisition.json', {'status': 'downloaded_verified', 'revision': REV,
        'source': 'https://huggingface.co/datasets/AmazonScience/RobustAD', 'license': 'CC-BY-4.0',
        'author_readme_sha256': hashlib.sha256(readme.content).hexdigest(), 'rows': records,
        'files': sorted(downloaded, key=lambda r: r['path']), 'total_bytes': size,
        'author_train_anomalies_not_downloaded_for_fit': True, 'only_official_test0_selected': True})
    print('Official PCB TRAIN normals and TEST0 downloaded with all author LFS hashes verified', flush=True)


if __name__ == '__main__':
    main()
