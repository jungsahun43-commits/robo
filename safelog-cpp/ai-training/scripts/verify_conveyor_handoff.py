"""Verify delivered ZIP in an isolated unpack directory, with real HTTP for both models."""
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
UNPACK = ROOT/'runs/conveyor-dual-handoff-v3-standalone'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def smoke(module, model, photo, request, endpoint):
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)  # No fallback to the source workspace package.
    with (UNPACK/(module.rsplit('.', 1)[-1]+'-smoke.log')).open('xb') as log:
        server = subprocess.Popen([sys.executable, '-m', module, '--checkpoint', model['path'],
            '--sha256', model['sha256'], '--port', str(port)], cwd=UNPACK, env=env,
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            url = f'http://127.0.0.1:{port}'
            for attempt in range(45):
                if server.poll() is not None:
                    raise RuntimeError('Delivered server failed: '+module)
                try:
                    health = requests.get(url+'/health', timeout=2)
                    health.raise_for_status()
                    assert health.json()['modelSha256']==model['sha256']
                    break
                except requests.RequestException:
                    time.sleep(1)
            else:
                raise RuntimeError('Delivered server did not become ready')
            raw = photo.read_bytes()
            start = time.monotonic()
            response = requests.post(url+endpoint, json={**request, 'image':base64.b64encode(raw).decode()}, timeout=30)
            response.raise_for_status()
            result = response.json()
            assert result['modelSha256']==model['sha256'] and result['inputSha256']==hashlib.sha256(raw).hexdigest()
            assert result['mechanicalFailureState']=='not_assessed' and result['humanConfirmationRequired']
            result.pop('heatmapPngBase64', None)
            return {'status':'passed', 'http_status':response.status_code, 'module':module, 'port':port,
                'elapsed_post_seconds':time.monotonic()-start, 'result':result, 'owned_server_stopped':True}
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(timeout=10)


def main():
    release = read(ROOT/'reports/conveyor-dual-handoff-release-v3.json')
    package = Path(release['package'])
    assert hashlib.sha256(package.read_bytes()).hexdigest()==release['package_sha256']
    assert not UNPACK.exists()
    UNPACK.mkdir()
    with zipfile.ZipFile(package) as archive:
        for name in archive.namelist():
            assert (UNPACK/name).resolve().is_relative_to(UNPACK.resolve()), 'Unexpected archive path'
        archive.extractall(UNPACK)
    for name, digest in read(UNPACK/'file-sha256.json').items():
        assert hashlib.sha256((UNPACK/name).read_bytes()).hexdigest()==digest
    models = read(UNPACK/'models.json')
    belt = smoke('safelog_ai.equipment_server', models['belt_crack'], ROOT/'runs/belt-author-embedded-sample.jpg',
                 {'equipmentCategory':'conveyor_belt', 'promptVersion':'belt-appearance-v1'}, '/v1/equipment/belt/analyze')
    base = ROOT/'data/equipment-conveyor-foreign/grouped-photo-v1'
    row = next(r for r in read(base/'manifest.json')['rows'] if r['split']=='train')
    foreign = smoke('safelog_ai.conveyor_foreign_server', models['conveyor_foreign_object'], base/row['native'],
        {'equipmentCategory':'conveyor_belt', 'sceneType':'iron_ore_closeup', 'promptVersion':'conveyor-foreign-v1'},
        '/v1/equipment/conveyor/foreign/analyze')
    proof = {'status':'passed', 'package_sha256':release['package_sha256'], 'unpack_directory':str(UNPACK.resolve()),
        'all_package_file_hashes_passed':True, 'imports_from_unpacked_package_without_workspace_pythonpath':True,
        'actual_http_cpu_inference':{'belt_crack':belt, 'conveyor_foreign_object':foreign},
        'benchmark_test_images_used':False, 'additional_test_model_scores':0, 'owned_servers_stopped':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (ROOT/'reports/conveyor-dual-handoff-standalone-v3.json').open('x', encoding='utf-8') as handle:
        json.dump(proof, handle, indent=2)
    print(json.dumps(proof, indent=2))


if __name__=='__main__':
    main()
