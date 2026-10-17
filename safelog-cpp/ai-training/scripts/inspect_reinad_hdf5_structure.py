"""v102: one preregistered TRAIN file; verified bytes and HDF5 metadata, no array reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.process_probe import process_exists

META = ROOT / 'data/industrial-reinad-v101-metadata/metadata-complete.json'
RESERVED = ROOT / 'data/industrial-reinad-v101-metadata/reserved-publisher-TEST-files-before-binary-content.json'
OUT = ROOT / 'data/industrial-reinad-v102-structure'
RUN = ROOT / 'runs/industrial-reinad-structure-v102'
POLICY = ROOT / 'reports/industrial-reinad-v102-structure-policy.json'
CODE = 'scripts/inspect_reinad_hdf5_structure.py'
PYPI = ROOT.parents[3] / 'outputs/H5PY_3_16_0_PYPI_METADATA_2026-10-17.json'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    meta = read(META)
    item = meta['smallest_TRAIN_future_structure_prospect_only']
    assert item['path'] == 'train/cable_2.h5' and item['bytes'] == 8343076
    assert sha(RESERVED) == meta['TEST_file_reservation_SHA256']
    pypi = read(PYPI)
    wheel, = [item for item in pypi['urls'] if item['filename'] == 'h5py-3.16.0-cp312-cp312-win_amd64.whl']
    assert wheel['size'] == 3182868
    assert wheel['digests']['sha256'] == '96b422019a1c8975c2d5dadcf61d4ba6f01c31f92bbde6e4649607885fe502d6'
    assert sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32
    assert shutil.disk_usage(ROOT).free >= 1024**3 + 64 * 1024**2
    OUT.mkdir()
    RUN.mkdir()
    save(POLICY, {
        'UTC': utc(), 'scope': 'Single smallest declared TRAIN HDF5, byte verification and structural metadata only; no RGB/mask dataset values',
        'code_SHA256': sha(ROOT / CODE), 'process_probe_SHA256': sha(ROOT / 'safelog_ai/process_probe.py'),
        'v101_metadata_SHA256': sha(META), 'TEST_reservation_SHA256': sha(RESERVED), 'PyPI_metadata_SHA256': sha(PYPI),
        'repository': meta['repository'], 'revision': meta['revision'], 'source': item,
        'source_URL': 'https://huggingface.co/datasets/' + meta['repository'] + '/resolve/' + meta['revision'] + '/' + item['path'] + '?download=true',
        'h5py_wheel': {'filename': wheel['filename'], 'bytes': wheel['size'], 'URL': wheel['url'], 'SHA256': wheel['digests']['sha256']},
        'reader': 'Standard h5py3.16.0 compatible WindowsCPython3.12 wheel, verified PyPI SHA; install --no-deps --no-index --no-compile into this OUT/reader-runtime only. Do not change existing venv/numpy or install dataset-author code/plugins.',
        'network': 'Anonymous HTTPS, source pinned public resolve URL and server-issued redirects only to huggingface.co subdomains, cas-bridge.xethub.hf.co or documented us.aws.cdn.hf.co/us.gcp.cdn.hf.co/cdn-lfs-us-1.hf.co/cdn-lfs-eu-1.hf.co; wheel exact files.pythonhosted.org. At most5redirects; no auth/cookies/retries/range guesses. Final200 exact declared length/SHA; preserve partial/header/failure, never overwrite completed originals.',
        'CDN_documentation': 'https://huggingface.co/docs/hub/en/datasets-downloading',
        'max_total_download_bytes': 16 * 1024**2, 'storage_bound_bytes': 64 * 1024**2, 'reserve_bytes': 1024**3,
        'HDF5_read_contract': 'Read-only; at most256objects/depth4, hard links only/no alias cycles, no external/virtual datasets. Names ASCII<=80, root scalar split/category values only for fixed-length strings <=256bytes; variable/unknown attributes record shape/dtype without value reads. Dataset shape/dtype/chunks/filter descriptors only. Block Dataset.__getitem__/__array__/read_direct: any value read attempt fails. No dataset-author loader/import/PIL/decoder/array slice.',
        'metadata_limits': 'Root open/file cache bounded1MiB. Unknown dtype/shape recorded, not repaired or inferred as RGB/GT. Dataset logical sizes are not expanded into RAM/disk.',
        'future_pixel_policy_screen_only': 'At least70declarednormal/50declaredanomaly rows, matching NHWCuint8C3 and NHWuint8 masks, H/W32..8192 and <=40MP. This is NOT a group/pixel-label/independence/minimum or training pass. If below, stop this smallest-file direction without selecting a larger substitute from observed values.',
        'TEST_and_roles': 'All publisher22TEST files/allcontainedrecords remain reserved by v101 LFS bytes. No TEST HDF5 acquired/opened, no roles/splits assigned or changed. Full-source123668352496byte closure/oldheld links not done, so never readyFIT/independent evaluation.',
        'data_terms': 'ReinAD author dataCC BY-NC-SA4.0; noncommercial research/author-source-version-changes/share-alike; h5py dependency terms separate',
        'ready_for_training': False, 'gradient_optimizer_features_scores_CAL_TEST_API_CPP_ZIP_APK': 0,
    })
    print(json.dumps({'policy_SHA256': sha(POLICY), 'source_bytes_planned': item['bytes']}), flush=True)


def run():
    p = read(POLICY)
    git = ['C:/Program Files/Git/cmd/git.exe', '-C', str(ROOT.parents[1]), '-c', 'safe.directory=' + ROOT.parents[1].as_posix()]
    for name in (CODE, POLICY.relative_to(ROOT).as_posix()):
        assert subprocess.check_output(git + ['show', 'HEAD:safelog-cpp/ai-training/' + name]) == (ROOT / name).read_bytes()
    for path, key in ((ROOT / CODE, 'code_SHA256'), (ROOT / 'safelog_ai/process_probe.py', 'process_probe_SHA256'),
                      (META, 'v101_metadata_SHA256'), (RESERVED, 'TEST_reservation_SHA256'), (PYPI, 'PyPI_metadata_SHA256')):
        assert sha(path) == p[key]
    assert shutil.disk_usage(ROOT).free >= p['reserve_bytes'] + p['storage_bound_bytes']
    assert not (OUT / 'structure-complete.json').exists()
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
        session.headers.update({'Accept-Encoding': 'identity', 'User-Agent': 'SafeLog-verified-structure-audit'})

        def download(url, filename, expected_size, expected_sha, kind):
            nonlocal received
            dest = OUT / filename
            if dest.exists():
                assert dest.stat().st_size == expected_size and sha(dest) == expected_sha
                receipts.append({'kind': kind, 'verified_existing_reused': True, 'bytes': expected_size, 'SHA256': expected_sha})
                return dest
            partial = OUT / (filename + '.partial')
            for redirect in range(6):
                parsed = urlparse(url)
                assert parsed.scheme == 'https'
                allowed = parsed.hostname == 'files.pythonhosted.org' if kind == 'wheel' else (
                    parsed.hostname == 'huggingface.co' or parsed.hostname.endswith('.huggingface.co') or parsed.hostname in (
                        'cas-bridge.xethub.hf.co', 'us.aws.cdn.hf.co', 'us.gcp.cdn.hf.co', 'cdn-lfs-us-1.hf.co', 'cdn-lfs-eu-1.hf.co'))
                assert allowed
                with session.get(url, stream=True, timeout=(15, 60), allow_redirects=False) as response:
                    save(OUT / (filename + f'.HTTP{redirect}.json'), {'UTC': utc(), 'URL': url, 'status': response.status_code,
                        'Location': response.headers.get('Location'), 'Content-Length': response.headers.get('Content-Length'), 'ETag': response.headers.get('ETag')})
                    if response.status_code in (301, 302, 303, 307, 308):
                        assert kind == 'source' and redirect < 5
                        url = urljoin(url, response.headers['Location'])
                        continue
                    assert response.status_code == 200 and response.headers.get('Content-Encoding', 'identity') == 'identity'
                    if response.headers.get('Content-Length'):
                        assert int(response.headers['Content-Length']) == expected_size
                    count = 0
                    with partial.open('xb') as stream:
                        for block in response.iter_content(65536):
                            count += len(block)
                            received += len(block)
                            assert count <= expected_size and received <= p['max_total_download_bytes']
                            assert shutil.disk_usage(ROOT).free >= p['reserve_bytes'] + len(block)
                            stream.write(block)
                    assert count == expected_size and sha(partial) == expected_sha
                    assert not dest.exists()
                    partial.rename(dest)
                    receipts.append({'kind': kind, 'bytes': count, 'SHA256': expected_sha, 'verified_existing_reused': False})
                    return dest
            raise RuntimeError('Redirect limit')

        wheel_info = p['h5py_wheel']
        wheel = download(wheel_info['URL'], wheel_info['filename'], wheel_info['bytes'], wheel_info['SHA256'], 'wheel')
        runtime = OUT / 'reader-runtime'
        if not runtime.exists():
            with (RUN / 'reader-install.log').open('x', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--no-compile', '--no-cache-dir',
                                '--disable-pip-version-check', '--target', str(runtime), str(wheel)], check=True, stdout=stream, stderr=subprocess.STDOUT)
        sys.path.insert(0, str(runtime))
        import h5py
        assert h5py.__version__ == '3.16.0' and Path(h5py.__file__).resolve().is_relative_to(runtime.resolve())
        source = p['source']
        original = download(p['source_URL'], 'publisher-train-cable_2.h5', source['bytes'], source['server_LFS_SHA256'], 'source')
        with original.open('rb') as stream:
            assert stream.read(8) == b'\x89HDF\r\n\x1a\n'
        value_read_attempts = 0

        def blocked(*args, **kwargs):
            nonlocal value_read_attempts
            value_read_attempts += 1
            raise RuntimeError('Dataset value read forbidden in v102')

        h5py.Dataset.__getitem__ = blocked
        h5py.Dataset.__array__ = blocked
        h5py.Dataset.read_direct = blocked
        objects = []
        seen = set()
        root_attrs = {}
        with h5py.File(original, 'r', rdcc_nbytes=1024**2) as handle:
            for name in ('split', 'category'):
                if name in handle.attrs:
                    attr = handle.attrs.get_id(name)
                    descriptor = {'shape': list(attr.shape), 'dtype': str(attr.dtype), 'value_read': False}
                    if attr.shape == () and attr.dtype.kind in ('S', 'U') and attr.dtype.itemsize <= 256:
                        value = handle.attrs[name]
                        descriptor.update(value=value.decode('utf-8') if isinstance(value, bytes) else str(value), value_read=True)
                    root_attrs[name] = descriptor

            def visit(group, depth=0):
                assert depth <= 4
                for name in sorted(group.keys()):
                    assert re.fullmatch('[A-Za-z0-9_.-]{1,80}', name)
                    assert isinstance(group.get(name, getlink=True), h5py.HardLink)
                    obj = group[name]
                    address = h5py.h5o.get_info(obj.id).addr
                    assert address not in seen
                    seen.add(address)
                    assert len(seen) <= 256
                    if isinstance(obj, h5py.Group):
                        objects.append({'path': obj.name, 'kind': 'group'})
                        visit(obj, depth + 1)
                    else:
                        assert isinstance(obj, h5py.Dataset) and not obj.is_virtual and not obj.external
                        objects.append({'path': obj.name, 'kind': 'dataset', 'shape': list(obj.shape), 'dtype': str(obj.dtype),
                                        'chunks': obj.chunks, 'compression': obj.compression, 'logical_bytes': obj.nbytes})

            visit(handle)
        assert value_read_attempts == 0 and sha(original) == source['server_LFS_SHA256']
        datasets = {item['path']: item for item in objects if item['kind'] == 'dataset'}
        normal_rows = anomaly_rows = 0
        geometry_ok = True
        for name, item in datasets.items():
            if not name.startswith('/Images/'):
                continue
            shape = item['shape']
            good = (len(shape) == 4 and item['dtype'] == 'uint8' and shape[-1] == 3 and
                    all(32 <= size <= 8192 for size in shape[1:3]) and shape[1] * shape[2] <= 40000000)
            geometry_ok = geometry_ok and good
            if re.fullmatch('/Images/Normal_[0-9]+', name):
                normal_rows += shape[0] if shape else 0
            elif re.fullmatch('/Images/Anomaly_[0-9]+', name):
                anomaly_rows += shape[0] if shape else 0
                mask = datasets.get(name.replace('/Images/', '/Masks/', 1))
                geometry_ok = geometry_ok and mask is not None and mask['dtype'] == 'uint8' and mask['shape'] == shape[:3]
            else:
                geometry_ok = False
        result = {'UTC': utc(), 'policy_SHA256': sha(POLICY), 'repository': p['repository'], 'revision': p['revision'],
                  'source_path': source['path'], 'verified_source_bytes': original.stat().st_size,
                  'source_SHA256': sha(original), 'root_attributes': root_attrs, 'objects': objects,
                  'dependency': {'h5py': h5py.__version__, 'HDF5': h5py.version.hdf5_version, 'isolated_runtime': str(runtime)},
                  'actual_download_bytes': received, 'receipts': receipts, 'dataset_value_read_attempts': value_read_attempts,
                  'declared_NORMAL_rows_not_decoded': normal_rows, 'declared_ANOMALY_rows_not_decoded': anomaly_rows,
                  'future_pixel_policy_structural_minimum_only': geometry_ok and normal_rows >= 70 and anomaly_rows >= 50,
                  'RGB_masks_decoded_or_viewed': 0, 'new_TEST_files_acquired_or_opened': 0,
                  'label_values_native_GT_alignment_physical_groups_and_full_graph_verified': False,
                  'ready_for_training': False, 'new_features_gradient_optimizer_CAL_TEST_API_CPP_ZIP_APK': 0,
                  'prior_GT_roles_models_and_all_originals_preserved': True, 'disk_free_bytes': shutil.disk_usage(ROOT).free}
        save(OUT / 'structure-complete.json', result)
        print(json.dumps({key: result[key] for key in ('verified_source_bytes', 'root_attributes', 'objects', 'actual_download_bytes',
                          'dataset_value_read_attempts', 'ready_for_training')}), flush=True)
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
