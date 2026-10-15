"""Fetch curator-published real steel image/mask candidate; never train or execute archive code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
DATA = ROOT / 'data/industrial-neu-micro-v22'
REPO = 'abin24/Surface-Inspection-defect-detection-dataset'
ARCHIVE = 'NEUMicro surface defect database and Oil pollution defect database.zip'


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'SafeLog-academic-dataset-audit'}), timeout=60)


def metadata():
    DATA.mkdir(exist_ok=True)
    assert not (DATA / 'source.json').exists()
    with request('https://api.github.com/repos/' + REPO + '/commits/master') as response:
        commit = json.load(response)['sha']
    with request('https://api.github.com/repos/' + REPO + '/git/trees/' + commit + '?recursive=1') as response:
        tree = json.load(response)
    assert not tree.get('truncated')
    entry = next(r for r in tree['tree'] if r['path'] == ARCHIVE and r['type'] == 'blob')
    assert 0 < entry['size'] <= 500 * 1024**2
    with request('https://raw.githubusercontent.com/' + REPO + '/' + commit + '/README.md') as response:
        readme = response.read()
    assert b'only for academic research' in readme and b'no commercial purposes' in readme
    with (DATA / 'publisher-README.md').open('xb') as out:
        out.write(readme)
    new(DATA / 'publisher-tree.json', tree)
    source = {'UTC': datetime.now(timezone.utc).isoformat(), 'publisher_repository': 'https://github.com/' + REPO,
        'publisher_commit': commit, 'archive_entry': entry,
        'download_url': 'https://raw.githubusercontent.com/' + REPO + '/' + commit + '/' + urllib.parse.quote(ARCHIVE),
        'publisher_README_sha256': sha(DATA / 'publisher-README.md'),
        'source_attribution': 'NEU Kechen Song original surface images; abin24 curator explicitly provides added pixel labels in this archive',
        'usage': 'academic research only, no commercial purpose; cite original provider; no invented SPDX license',
        'rights_evidence_url': 'https://github.com/' + REPO + '/blob/' + commit + '/README.md',
        'raw_files_downloaded': False, 'new_optimizer_updates': 0}
    new(DATA / 'source.json', source)
    print(source, flush=True)


def fetch():
    source = json.loads((DATA / 'source.json').read_text(encoding='utf-8'))
    target = DATA / 'publisher-archive.zip'
    assert not target.exists() and not (DATA / 'download-receipt.json').exists()
    assert shutil.disk_usage(ROOT).free > 2 * 1024**3
    partial = DATA / 'publisher-archive.partial.zip'
    assert not partial.exists(), 'Preserve and inspect a previous partial download before resuming'
    with request(source['download_url']) as response, partial.open('xb') as out:
        for chunk in iter(lambda: response.read(1024**2), b''):
            out.write(chunk)
    entry = source['archive_entry']
    assert partial.stat().st_size == entry['size']
    h = hashlib.sha1(('blob ' + str(entry['size']) + '\0').encode())
    with partial.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    assert h.hexdigest() == entry['sha'], 'GitHub git-blob identity mismatch'
    inventory = []
    with zipfile.ZipFile(partial) as archive:
        infos = archive.infolist()
        assert len(infos) <= 4096 and sum(i.file_size for i in infos) <= 1024**3
        assert archive.testzip() is None
        for item in infos:
            name = PurePosixPath(item.filename.replace('\\', '/'))
            assert not name.is_absolute() and '..' not in name.parts and not any(':' in p for p in name.parts)
            assert ((item.external_attr >> 16) & 0o170000) != 0o120000
            inventory.append({'name': item.filename, 'bytes': item.file_size, 'CRC': item.CRC})
    partial.rename(target)
    receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'archive_sha256_local': sha(target),
        'archive_bytes': target.stat().st_size, 'git_blob_SHA1_matches_publisher_tree': True,
        'ZIP_CRC_all_members_passed': True, 'publisher_supplied_SHA256': False,
        'members': inventory, 'source_json_sha256': sha(DATA / 'source.json'),
        'archive_code_executed': False, 'archive_files_extracted': 0, 'training_split_fixed': False,
        'old_TEST_cross_duplicate_audit_done': False, 'new_optimizer_updates': 0}
    new(DATA / 'download-receipt.json', receipt)
    print({k: v for k, v in receipt.items() if k != 'members'}, flush=True)
    print({'member_count': len(inventory), 'first_members': inventory[:20]}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('metadata', 'fetch'))
    globals()[parser.parse_args().mode]()
