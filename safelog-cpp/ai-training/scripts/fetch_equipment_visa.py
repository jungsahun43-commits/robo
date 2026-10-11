"""Official VisA archive, selecting only the four electronic PCB categories."""
import csv
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/equipment-visa'
URL = 'https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar'
SPLIT_URL = 'https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv'
CATEGORIES = ('pcb1', 'pcb2', 'pcb3', 'pcb4')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive path')
    return path


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / 'VisA_20220922.tar'
    part = DATA / 'VisA_20220922.tar.partial'
    if not archive.exists():
        start = part.stat().st_size if part.exists() else 0
        with requests.get(URL, headers={'Range': f'bytes={start}-'}, stream=True, timeout=(30, 60)) as response:
            response.raise_for_status()
            assert response.status_code == 206
            assert response.headers['Content-Range'] == f'bytes {start}-1929840639/1929840640'
            last = time.monotonic()
            with part.open('ab') as handle:
                for block in response.iter_content(8 << 20):
                    handle.write(block)
                    start += len(block)
                    if time.monotonic() - last > 25:
                        print(json.dumps({'archive_bytes': start, 'total': 1929840640}), flush=True)
                        last = time.monotonic()
        assert part.stat().st_size == 1929840640
        part.rename(archive)
    archive_sha = sha(archive)
    split = DATA / '1cls.csv'
    if not split.exists():
        response = requests.get(SPLIT_URL, timeout=45)
        response.raise_for_status()
        split.write_bytes(response.content)
    rows = [r for r in csv.DictReader(split.open(encoding='utf-8')) if r['object'] in CATEGORIES]
    assert len(rows) == 4416
    selected = {r['image'] for r in rows} | {r['mask'] for r in rows if r['mask']}
    unpacked = DATA / 'originals'
    unpacked.mkdir(exist_ok=True)
    seen = set()
    with tarfile.open(archive, 'r:') as handle:
        for member in handle:
            path = safe_member(member.name)
            name = str(path)
            if name not in selected:
                continue
            assert member.isfile(), 'Selected member must be regular'
            assert name not in seen, 'Duplicate tar member'
            seen.add(name)
            dest = unpacked.joinpath(*path.parts)
            assert dest.resolve().is_relative_to(unpacked.resolve())
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                with handle.extractfile(member) as source, dest.open('xb') as output:
                    shutil.copyfileobj(source, output)
            assert dest.stat().st_size == member.size
    assert selected == seen, 'Split paths missing from official archive'
    ledger = [{'path': n, 'sha256': sha(unpacked / n), 'bytes': (unpacked / n).stat().st_size}
              for n in sorted(selected)]
    output = DATA / 'acquisition.json'
    assert not output.exists(), 'Preserve completed acquisition'
    output.write_text(json.dumps({'status': 'complete', 'source': 'https://github.com/amazon-science/spot-diff',
        'license': 'CC-BY-4.0', 'archive_url': URL, 'archive_sha256': archive_sha,
        'archive_bytes': archive.stat().st_size, 'split_url': SPLIT_URL, 'split_sha256': sha(split),
        'categories': CATEGORIES, 'official_rows': len(rows), 'files': ledger,
        'script_sha256': sha(Path(__file__))}, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'pcb_rows': len(rows), 'files': len(ledger), 'archive_sha256': archive_sha}), flush=True)


if __name__ == '__main__':
    main()
