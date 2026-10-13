"""Fetch public research archives into a new, isolated dataset namespace."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import urllib.parse
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/equipment-expansion-v1'
SOURCES = ROOT/'configs/equipment_expansion_sources.json'


def fetch_selected(name, source):
    folder = DATA/name; folder.mkdir(parents=True, exist_ok=True)
    receipt = folder/'download.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert all(sha(folder/'originals'/n) == h for n, h in saved['file_sha256'].items())
        return saved
    repo, revision = source['hf_repo'], source['mirror_revision']
    url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&expand=true&limit=100'
    catalog = []
    while url:
        with urllib.request.urlopen(url, timeout=45) as response:
            catalog.extend(json.load(response)); links = response.headers.get('Link', '')
        url = None
        for link in links.split(','):
            if 'rel="next"' in link: url = link.split('<', 1)[1].split('>', 1)[0]
    selected = [x for x in catalog if x['type'] == 'file' and
                (x['path'].split('/')[0] in source['categories'] or x['path'] in ('license.txt', 'readme.txt')) and
                Path(x['path']).suffix.lower() in ('.png', '.jpg', '.bmp', '.txt')]
    assert selected and all('..' not in Path(x['path']).parts and not Path(x['path']).is_absolute() for x in selected)
    def one(item):
        path = folder/'originals'/item['path']; path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): payload = path.read_bytes()
        else:
            address = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/'+urllib.parse.quote(item['path'])
            with urllib.request.urlopen(address, timeout=45) as response: payload = response.read()
        assert len(payload) == item['size']
        digest = hashlib.sha256(payload).hexdigest()
        if item.get('lfs'): assert digest == item['lfs']['oid']
        else: assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\x00'+payload).hexdigest() == item['oid']
        if not path.exists():
            with path.open('xb') as f: f.write(payload)
        return item['path'], digest
    files = {}; started = time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(one, item) for item in selected]
        for future in as_completed(jobs):
            n, h = future.result(); files[n] = h
            if len(files)%100 == 0: print(json.dumps({'dataset': name, 'files_downloaded': len(files), 'total': len(selected)}), flush=True)
    result = {'dataset': name, 'status': 'downloaded', 'source': source, 'file_sha256': files,
              'files': len(files), 'reference_checksum_matched': True, 'publisher_checksum_verified': False,
              'seconds': time.monotonic()-started, 'bytes': sum(x['size'] for x in selected)}
    with receipt.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2)
    print(json.dumps({'dataset': name, 'files_complete': len(files)}), flush=True)
    return result


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, source):
    if source.get('kind') == 'hf_selected': return fetch_selected(name, source)
    folder = DATA/name; folder.mkdir(parents=True, exist_ok=True)
    path = folder/source['archive']; receipt = folder/'download.json'
    if path.exists():
        saved = json.loads(receipt.read_text()); assert sha(path) == saved['archive_sha256']
        return saved
    partial = path.with_suffix(path.suffix+'.partial')
    started = time.monotonic()
    req = urllib.request.Request(source['url'], headers={'User-Agent': 'SafeLog-research/1.0'})
    with urllib.request.urlopen(req, timeout=45) as response:
        expected = int(response.headers.get('Content-Length') or 0)
        assert expected > 1000000, 'Expected a dataset archive, not an HTML login'
        h = hashlib.sha256(); received = 0; notified = time.monotonic()
        # Only this task's partial file can be restarted. Existing final archives
        # are verified above and never overwritten.
        with partial.open('wb') as f:
            while chunk := response.read(4*1024*1024):
                if received == 0:
                    assert chunk.startswith(b'PK\x03\x04') or chunk.startswith(b'\xfd7zXZ\x00'), 'Invalid archive signature'
                f.write(chunk); h.update(chunk); received += len(chunk)
                if time.monotonic()-notified >= 25:
                    print(json.dumps({'dataset': name, 'download_bytes': received, 'total_bytes': expected}), flush=True)
                    notified = time.monotonic()
        assert received == expected
        digest = h.hexdigest()
        if source.get('sha256'):
            assert digest == source['sha256'], 'Reference archive checksum differs'
        result = {'dataset': name, 'status': 'downloaded', 'source': source, 'bytes': received,
                  'archive_sha256': digest, 'reference_checksum_matched': bool(source.get('sha256')),
                  'publisher_checksum_verified': False, 'seconds': time.monotonic()-started}
    partial.rename(path)
    with receipt.open('x', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'dataset': name, 'completed_bytes': received, 'archive_sha256': digest}), flush=True)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset'); args = parser.parse_args()
    assert shutil.disk_usage(ROOT).free > 25*1024**3, 'Insufficient disk for selected research categories'
    sources = json.loads(SOURCES.read_text(encoding='utf-8')); results = {}
    if args.dataset:
        assert args.dataset in sources
        fetch(args.dataset, sources[args.dataset]); return
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch, name, source): name for name, source in sources.items()}
        for future in as_completed(pending):
            name = pending[future]
            try: results[name] = future.result()
            except Exception as e:
                results[name] = {'status': 'failed', 'error': str(e)}
                print(json.dumps({'dataset': name, **results[name]}), flush=True)
    path = ROOT/'runs/equipment-expansion-v1/download-summary.json'
    if path.exists(): path = path.with_name('download-summary-completed.json')
    if path.exists(): assert json.loads(path.read_text()) == results
    else:
        with path.open('x', encoding='utf-8') as f: json.dump(results, f, ensure_ascii=False, indent=2)
    assert all(x['status'] == 'downloaded' for x in results.values()), 'See download summary; successful downloads retained'


if __name__ == '__main__': main()
