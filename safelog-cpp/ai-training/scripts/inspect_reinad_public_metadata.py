"""v101: bounded anonymous ReinAD file metadata and TEST-file reservation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.process_probe import process_exists

REPO = 'Tocmac/ReinAD'
API = 'https://huggingface.co/api/datasets/' + REPO
OUT = ROOT / 'data/industrial-reinad-v101-metadata'
RUN = ROOT / 'runs/industrial-reinad-public-metadata-v101'
POLICY = ROOT / 'reports/industrial-reinad-v101-metadata-policy.json'
CODE = 'scripts/inspect_reinad_public_metadata.py'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def safe_path(value):
    assert isinstance(value, str) and 0 < len(value) <= 160
    assert '\\' not in value and ':' not in value and '\x00' not in value
    path = PurePosixPath(value)
    assert not path.is_absolute() and '..' not in path.parts
    assert str(path) == value and all(part not in ('', '.') for part in path.parts)
    return value


def lfs_sha(item):
    value = item['lfs'].get('sha256', item['lfs'].get('oid'))
    assert isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value)
    return value


def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    assert shutil.disk_usage(ROOT).free >= 1024**3 + 8 * 1024**2
    OUT.mkdir()
    RUN.mkdir()
    save(POLICY, {
        'UTC': utc(), 'scope': 'Public author-linked ReinAD metadata only; no HDF5/binary/image/mask/model acquisition',
        'repository': REPO, 'API': API, 'code_SHA256': sha(ROOT / CODE),
        'process_probe_SHA256': sha(ROOT / 'safelog_ai/process_probe.py'),
        'v100_report_SHA256': sha(ROOT / 'reports/REINAD_V100_METADATA_KO.md'),
        'provenance': 'https://tocmac.github.io/ReinAD/ directly links this exact Hugging Face dataset. No mirrors or arbitrary archive/ID guesses.',
        'network': 'Anonymous HTTPS GET repository info with blobs=true; pin returned full revision; retrieve same immutable revision info, recursive tree and README.md raw text only. No auth tokens, cookies, redirects, retries, repository code execution or data viewer.',
        'schema': 'Require public ungated repo id, full40hex revision, consistent info/tree filenames sizes/gitblob/LFS-SHA256, no duplicate/unsafe paths or unresolved pagination. Only root README.md/.gitattributes plus train/*.h5 and test/*.h5 accepted. HDF5 contents not opened.',
        'max_HTTP_response_bytes': 1024**2, 'max_total_network_bytes': 4 * 1024**2,
        'max_readme_bytes': 64 * 1024, 'max_tree_entries': 256,
        'reserve_bytes': 1024**3, 'storage_bound_bytes': 8 * 1024**2,
        'reservation': 'Before any future binary/data contents, reserve every listed publisher test HDF5 path/size/server LFS SHA256/revision and all records contained in each file. This is a file-level byte binding, not yet per-photo IDs or verified source hashes.',
        'selection': 'No source selected or acquired now. Report smallest train HDF5 by declared byte size then lexical path only as a future structural-metadata prospect, never admit it as readyFIT or replace original test categories.',
        'data_contract': 'Author data card CC BY-NC-SA4.0 noncommercial research/attribution/share-alike. Do not substitute project-page/code/paper license. Native HDF5 coordinate/normal/pixel/physical grouping still unverified.',
        'preservation': 'Keep all prior originals/failures/roles/TEST/graphs/models. No old global RGB/pHash audit, features, scores, gradient, optimizer, CAL/TEST model evaluation or app changes.',
        'ready_for_training': False, 'new_source_photos_masks_pixels_features_gradient_optimizer': 0,
    })
    print(json.dumps({'policy_SHA256': sha(POLICY), 'source_bytes': 0}), flush=True)


def run():
    policy = read(POLICY)
    git = ['C:/Program Files/Git/cmd/git.exe', '-C', str(ROOT.parents[1]),
           '-c', 'safe.directory=' + ROOT.parents[1].as_posix()]
    for name in (CODE, POLICY.relative_to(ROOT).as_posix(), 'reports/REINAD_V100_METADATA_KO.md'):
        assert subprocess.check_output(git + ['show', 'HEAD:safelog-cpp/ai-training/' + name]) == (ROOT / name).read_bytes()
    assert sha(ROOT / CODE) == policy['code_SHA256']
    assert sha(ROOT / 'safelog_ai/process_probe.py') == policy['process_probe_SHA256']
    assert sha(ROOT / 'reports/REINAD_V100_METADATA_KO.md') == policy['v100_report_SHA256']
    assert shutil.disk_usage(ROOT).free >= policy['reserve_bytes'] + policy['storage_bound_bytes']
    assert not (OUT / 'metadata-complete.json').exists()
    for lock in (ROOT / 'runs').glob('**/*.lock'):
        if process_exists(int(lock.read_text(encoding='utf-8').strip())):
            raise RuntimeError('Live operation: ' + str(lock))
    with (RUN / 'operation.lock').open('x') as stream:
        stream.write(str(os.getpid()))
    received = 0
    receipts = []
    try:
        session = requests.Session()
        session.trust_env = False
        session.headers.update({'Accept-Encoding': 'identity', 'User-Agent': 'SafeLog-public-metadata-audit'})

        def get(url, name, limit=None):
            nonlocal received
            assert urlparse(url).scheme == 'https' and urlparse(url).netloc == 'huggingface.co'
            assert url.startswith(API) or url.startswith('https://huggingface.co/datasets/' + REPO + '/raw/')
            size = 0
            dest = OUT / name
            with session.get(url, stream=True, timeout=(15, 40), allow_redirects=False) as response:
                header = {'UTC': utc(), 'URL': url, 'status': response.status_code,
                          'ETag': response.headers.get('ETag'), 'Content-Length': response.headers.get('Content-Length'),
                          'Content-Encoding': response.headers.get('Content-Encoding'), 'Link': response.headers.get('Link')}
                save(OUT / (name + '.headers.json'), header)
                assert response.status_code == 200, 'HTTP status: ' + str(response.status_code)
                assert response.headers.get('Content-Encoding', 'identity') == 'identity'
                assert 'rel="next"' not in response.headers.get('Link', '')
                with dest.open('xb') as stream:
                    for block in response.iter_content(65536):
                        size += len(block)
                        received += len(block)
                        assert size <= (limit or policy['max_HTTP_response_bytes'])
                        assert received <= policy['max_total_network_bytes']
                        assert shutil.disk_usage(ROOT).free >= policy['reserve_bytes'] + len(block)
                        stream.write(block)
                if response.headers.get('Content-Length'):
                    assert int(response.headers['Content-Length']) == size
            receipts.append({'URL': url, 'bytes': size, 'local_SHA256': sha(dest)})
            return dest.read_bytes()

        info = json.loads(get(API + '?blobs=true', 'repository-info.json'))
        assert info['id'] == REPO and info.get('private') is False and info.get('gated', False) is False
        revision = info['sha']
        assert re.fullmatch('[a-f0-9]{40}', revision)
        pinned = json.loads(get(API + '/revision/' + revision + '?blobs=true', 'pinned-info.json'))
        assert pinned['sha'] == revision and pinned['id'] == REPO
        assert pinned.get('private') is False and pinned.get('gated', False) is False
        assert pinned['cardData']['license'] == 'cc-by-nc-sa-4.0'
        tree = json.loads(get(API + '/tree/' + revision + '?recursive=true', 'pinned-tree.json'))
        assert isinstance(tree, list) and 0 < len(tree) <= policy['max_tree_entries']
        by_path = {}
        for item in tree:
            name = safe_path(item['path'])
            assert name not in by_path
            by_path[name] = item
            if item['type'] == 'directory':
                assert name in ('train', 'test')
            else:
                assert item['type'] == 'file' and (name in ('README.md', '.gitattributes') or re.fullmatch(r'(train|test)/[A-Za-z0-9_.-]+\.h5', name))
        siblings = {safe_path(item['rfilename']): item for item in pinned['siblings']}
        assert len(siblings) == len(pinned['siblings'])
        assert set(siblings) == {name for name, item in by_path.items() if item['type'] == 'file'}
        data_files = []
        for name, item in sorted(siblings.items()):
            entry = by_path[name]
            assert item['size'] == entry['size'] and item['blobId'] == entry['oid']
            assert re.fullmatch('[a-f0-9]{40}', item['blobId'])
            if name.endswith('.h5'):
                digest = lfs_sha(item)
                assert digest == lfs_sha(entry) and item['lfs']['size'] == item['size'] == entry['lfs']['size']
                assert 0 < item['size'] <= 32 * 1024**3
                data_files.append({'path': name, 'bytes': item['size'], 'server_LFS_SHA256': digest,
                                   'git_blob_SHA1': item['blobId'], 'role': name.split('/')[0]})
        train = [item for item in data_files if item['role'] == 'train']
        test = [item for item in data_files if item['role'] == 'test']
        assert train and test and len(data_files) <= 128
        reservation = {'UTC': utc(), 'repository': REPO, 'revision': revision, 'policy_SHA256': sha(POLICY),
                       'scope': 'All records and copies contained in these publisher TEST files reserved; no binary content read',
                       'TEST_files': test, 'server_digests_not_locally_verified': True, 'per_photo_IDs_not_yet_known': True}
        save(OUT / 'reserved-publisher-TEST-files-before-binary-content.json', reservation)
        raw = get('https://huggingface.co/datasets/' + REPO + '/raw/' + revision + '/README.md',
                  'publisher-README.md', policy['max_readme_bytes'])
        assert len(raw) == siblings['README.md']['size']
        assert hashlib.sha1(('blob ' + str(len(raw)) + '\0').encode() + raw).hexdigest() == siblings['README.md']['blobId']
        text = raw.decode('utf-8')
        assert '\0' not in text and 'cc-by-nc-sa-4.0' in text
        prospect = min(train, key=lambda item: (item['bytes'], item['path']))
        result = {'UTC': utc(), 'repository': REPO, 'revision': revision, 'policy_SHA256': sha(POLICY),
                  'data_license': 'CC BY-NC-SA4.0', 'author_project': 'https://tocmac.github.io/ReinAD/',
                  'data_files': data_files, 'TRAIN_file_count': len(train), 'TEST_file_count': len(test),
                  'total_declared_source_bytes': sum(item['bytes'] for item in data_files),
                  'smallest_TRAIN_future_structure_prospect_only': prospect,
                  'TEST_file_reservation_SHA256': sha(OUT / 'reserved-publisher-TEST-files-before-binary-content.json'),
                  'actual_HTTP_bytes': received, 'receipts': receipts, 'local_README_SHA256': sha(OUT / 'publisher-README.md'),
                  'source_bytes_acquired': 0, 'source_digests_locally_verified': False,
                  'new_source_photos_masks_pixels_views_features_scores_gradient_optimizer_CAL_TEST': 0,
                  'HDF5_coordinate_dtype_counts_GT_support_and_physical_groups_verified': False,
                  'full_source_transitive_graph_completed': False, 'ready_for_training': False,
                  'app_models_roles_GT_and_all_previous_files_unchanged': True,
                  'external_author_code_executed': False, 'disk_free_bytes': shutil.disk_usage(ROOT).free}
        save(OUT / 'metadata-complete.json', result)
        print(json.dumps({key: result[key] for key in ('revision', 'TRAIN_file_count', 'TEST_file_count',
              'total_declared_source_bytes', 'smallest_TRAIN_future_structure_prospect_only', 'actual_HTTP_bytes', 'ready_for_training')}), flush=True)
    except BaseException as exc:
        save(RUN / ('failure-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'),
             {'UTC': utc(), 'error': repr(exc), 'traceback': traceback.format_exc(), 'received_bytes': received, 'receipts': receipts})
        raise
    finally:
        (RUN / 'operation.lock').unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'run'))
    globals()[parser.parse_args().stage]()
