"""One real loopback HTTP/CPU inference on an author TRAIN photo; stop server."""
import base64
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    result_path = ROOT/'reports/beltcrack-expansion-api-smoke.json'
    assert not result_path.exists()
    training = json.loads((ROOT/'reports/beltcrack-expansion-training.json').read_text(encoding='utf-8'))
    image_path = ROOT/'runs/belt-author-embedded-sample.jpg'
    raw = image_path.read_bytes()
    # This is publisherTRAIN data1/0, already extracted before either evaluation.
    payload = {'equipmentCategory':'conveyor_belt','image':base64.b64encode(raw).decode('ascii'),
               'promptVersion':'belt-appearance-v1'}
    log_path = ROOT/'runs/beltcrack-expansion-http-smoke.log'
    with log_path.open('xb') as log:
        server = subprocess.Popen([sys.executable,'-m','safelog_ai.equipment_server','--checkpoint',
            str(ROOT/'runs/beltcrack-cached-expansion-v2/final.pt'),'--sha256',training['weights_sha256'],
            '--port','8771'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            url = 'http://127.0.0.1:8771'
            for attempt in range(60):
                if server.poll() is not None:
                    raise RuntimeError('HTTP server failed; inspect '+str(log_path))
                try:
                    health_response = requests.get(url+'/health',timeout=2)
                    health_response.raise_for_status()
                    health = health_response.json()
                    assert health['modelSha256']==training['weights_sha256']
                    break
                except requests.RequestException:
                    time.sleep(1)
            else:
                raise RuntimeError('Local server did not become ready')
            started = time.monotonic()
            response = requests.post(url+'/v1/equipment/belt/analyze',json=payload,timeout=60)
            response.raise_for_status()
            value = response.json()
            heatmap = base64.b64decode(value.pop('heatmapPngBase64'),validate=True)
            with Image.open(io.BytesIO(heatmap)) as decoded, Image.open(io.BytesIO(raw)) as photo:
                assert decoded.size==photo.size and decoded.mode=='L'
            assert value['modelSha256']==training['weights_sha256']
            assert value['inputSha256']==hashlib.sha256(raw).hexdigest()
            assert value['equipmentCategory']=='conveyor_belt' and value['className']=='belt_crack'
            assert value['mechanicalFailureState']=='not_assessed' and value['humanConfirmationRequired']
            assert not value['normalPhotoFprValidated'] and not value['scoreIsMechanicalFailureProbability']
            with (ROOT/'runs/beltcrack-expansion-api-heatmap.png').open('xb') as handle:
                handle.write(heatmap)
            result = {'status':'passed','transport':'ActualHTTP on127.0.0.1:8771; CPU inference',
                'http_status':response.status_code,'health':health,'result':value,
                'elapsed_post_seconds':time.monotonic()-started,'source_image':'PublisherTRAIN/data1/0 embedded image',
                'new_held_out_performance_measurement':False,'heatmap_sha256':hashlib.sha256(heatmap).hexdigest(),
                'server_started_by_this_script':True,'server_stopped_after_smoke':True,
                'normal_audit_tests':3,'equipment_server_tests':4,
                'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                    ('scripts/smoke_equipment_belt_api.py','safelog_ai/belt_inference.py','safelog_ai/equipment_server.py',
                     'tests/test_equipment_server.py','scripts/audit_belt_normal_coco.py','tests/test_belt_normal_audit.py')}}
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
    with result_path.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
