"""Actual loopback CPU HTTP inference; use TRAIN only, stop the owned server."""
import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    proof = ROOT/'reports/conveyor-foreign-grouped-api-smoke.json'
    assert not proof.exists()
    training = read(ROOT/'reports/conveyor-foreign-grouped-training.json')
    base = ROOT/'data/equipment-conveyor-foreign/grouped-photo-v1'
    manifest = read(base/'manifest.json')
    # Frozen manifest order, first TRAIN photo of each label. No TEST inference here.
    selected = [next(r for r in manifest['rows'] if r['split']=='train' and r['foreign_object_photo_label']==label)
                for label in (0, 1)]
    observations = []
    with (ROOT/'runs/conveyor-foreign-grouped-http-smoke.log').open('xb') as log:
        server = subprocess.Popen([sys.executable, '-m', 'safelog_ai.conveyor_foreign_server', '--checkpoint',
            str(ROOT/'runs/conveyor-foreign-grouped-v1/final.pt'), '--sha256', training['weights_sha256'], '--port', '8772'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            url = 'http://127.0.0.1:8772'
            for attempt in range(45):
                if server.poll() is not None:
                    raise RuntimeError('Owned HTTP server failed; inspect smoke log')
                try:
                    response = requests.get(url+'/health', timeout=2)
                    response.raise_for_status()
                    health = response.json()
                    assert health['modelSha256']==training['weights_sha256']
                    break
                except requests.RequestException:
                    time.sleep(1)
            else:
                raise RuntimeError('HTTP server did not become ready')
            for row in selected:
                raw = (base/row['native']).read_bytes()
                start = time.monotonic()
                response = requests.post(url+'/v1/equipment/conveyor/foreign/analyze', json={
                    'equipmentCategory':'conveyor_belt', 'sceneType':'iron_ore_closeup',
                    'promptVersion':'conveyor-foreign-v1', 'image':base64.b64encode(raw).decode('ascii')}, timeout=30)
                response.raise_for_status()
                value = response.json()
                assert value['modelSha256']==training['weights_sha256']
                assert value['inputSha256']==hashlib.sha256(raw).hexdigest()
                assert value['className']=='conveyor_foreign_object'
                assert value['mechanicalFailureState']=='not_assessed' and value['humanConfirmationRequired']
                assert not value['localizationAvailable'] and not value['fieldPerformanceValidated']
                assert value['candidateDetected']==(value['presenceScore']>0.5)
                observations.append({'source_id':row['id'], 'split':'train', 'publisher_label':row['foreign_object_photo_label'],
                    'http_status':response.status_code, 'elapsed_post_seconds':time.monotonic()-start, 'result':value})
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(timeout=10)
    value = {'status':'passed', 'transport':'ActualHTTP loopback127.0.0.1:8772; CPU inference',
        'observations':observations, 'weights_sha256':training['weights_sha256'],
        'held_out_performance_measurement':False, 'server_stopped':True, 'server_tests_passed':5,
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('safelog_ai/conveyor_foreign_inference.py', 'safelog_ai/conveyor_foreign_server.py',
             'scripts/smoke_conveyor_foreign_api.py', 'tests/test_conveyor_foreign_server.py')}}
    with proof.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
    print(json.dumps(value, indent=2))


if __name__=='__main__':
    main()
