"""Acquire the authors' public BeltCrack14ks volumes; never touch building data."""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/equipment-beltcrack'
SOURCE = 'https://doi.org/10.57760/sciencedb.31181'
DATASET = '46e3906e446f4d7bb7e200b72fc4ca98'
API = 'https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath'


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def fetch(entry):
    name = entry['fileName']
    assert Path(name).name == name and name.startswith('BeltCrack14ks.part')
    dest = DATA / 'archives' / name
    part = dest.with_suffix('.rar.partial')
    url = 'https://china.scidb.cn/download?fileId=' + entry['id']
    if not dest.exists():
        for attempt in range(3):
            start = part.stat().st_size if part.exists() else 0
            if start == entry['size']:
                break
            assert start < entry['size'], 'Partial exceeds official length'
            try:
                with requests.get(url, headers={'Range': f'bytes={start}-'}, stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    assert response.status_code == 206
                    assert response.headers['Content-Range'].startswith(f'bytes {start}-')
                    assert response.headers['Content-Range'].endswith('/' + str(entry['size']))
                    assert int(response.headers['Content-Length']) == entry['size'] - start
                    last = time.monotonic()
                    with part.open('ab') as handle:
                        for block in response.iter_content(4 << 20):
                            handle.write(block)
                            start += len(block)
                            if time.monotonic() - last > 25:
                                print(json.dumps({'file': name, 'bytes': start, 'total': entry['size']}), flush=True)
                                last = time.monotonic()
                assert part.stat().st_size == entry['size']
                break
            except requests.RequestException as exc:
                print(json.dumps({'file': name, 'attempt': attempt + 1, 'retry': str(exc)}), flush=True)
                if attempt == 2:
                    raise
        assert part.stat().st_size == entry['size']
        assert digest(part, 'md5') == entry['md5'], 'Official volume MD5 mismatch'
        part.rename(dest)
    assert dest.stat().st_size == entry['size']
    assert digest(dest, 'md5') == entry['md5']
    result = {'file': name, 'bytes': entry['size'], 'official_md5': entry['md5'],
              'sha256': digest(dest, 'sha256'), 'url': url}
    print(json.dumps({'status': 'verified', **result}), flush=True)
    return result


def main():
    DATA.joinpath('archives').mkdir(parents=True, exist_ok=True)
    record = DATA / 'source-files.json'
    if not record.exists():
        response = requests.post(API, json={'dataSetId': DATASET, 'version': 'V1',
                                 'path': '/V1/BeltCrack14ks(all)', 'lastIndex': 0, 'pageSize': 100}, timeout=45)
        response.raise_for_status()
        content = response.json()
        assert content['code'] == 20000 and len(content['data']) == 8
        record.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding='utf-8')
    entries = json.loads(record.read_text(encoding='utf-8'))['data']
    assert sum(e['size'] for e in entries) == 16143115776
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, entries))
    output = DATA / 'acquisition.json'
    assert not output.exists(), 'Preserve completed acquisition'
    output.write_text(json.dumps({'status': 'complete', 'source': SOURCE,
        'dataset_license': 'CC-BY-SA-4.0', 'source_version': 'V1', 'volumes': results,
        'bytes': sum(e['bytes'] for e in results), 'official_md5_verified': True,
        'source_file_listing_sha256': digest(record, 'sha256'),
        'script_sha256': digest(Path(__file__), 'sha256')}, indent=2), encoding='utf-8')
    print('BeltCrack14ks acquisition complete', flush=True)


if __name__ == '__main__':
    main()
