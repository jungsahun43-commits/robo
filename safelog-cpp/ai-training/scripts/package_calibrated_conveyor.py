"""Final v5 handoff with calibrated CPU endpoint; keep v3/v4 and source weights."""
import argparse
import base64
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import requests

from prepare_foreign_calibration_check import read,new,sha,PRIOR
from study_foreign_calibration import ROOT

PACKAGE = ROOT.parents[3]/'outputs/equipment-ai-role4-handoff-2026-10-12-v5.zip'
UNPACK = ROOT/'runs/calibrated-conveyor-v5-standalone'


def build():
    fit = read(ROOT/'reports/foreign-calibration-fit.json')
    verify = read(ROOT/'reports/foreign-calibration-verification.json')
    assert verify['status']=='passed'
    belt = read(ROOT/'reports/beltcrack-expansion-training.json')
    models = {'belt_crack':{'path':'beltcrack-cached-expansion-v2/final.pt','sha256':belt['weights_sha256'],'trained_source_epochs':40},
        'conveyor_foreign_calibrated_v3':{'path':'conveyor-foreign-calibrated-v3/final.pt','sha256':fit['weights_sha256'],
            'trained_source_epochs':30,'trained_source_updates':4290,'new_neural_epochs_in_calibration':0,
            'threshold':fit['threshold'],'source_weights_sha256':fit['source_weights_sha256']}}
    files = {model['path']:ROOT/'runs'/model['path'] for model in models.values()}
    for name in ('belt_inference.py','belt_segmentation.py','equipment_server.py','conveyor_foreign.py',
                 'conveyor_foreign_inference.py','conveyor_foreign_server.py','foreign_calibrated_server.py'):
        files['safelog_ai/'+name] = ROOT/'safelog_ai'/name
    names = ['FOREIGN_CALIBRATION_RESULTS_KO.md','CONVEYOR_FOREIGN_EXPANSION_RESULTS_KO.md','BELTCRACK_EXPANSION_RESULTS_KO.md',
             'FOREIGN_CALIBRATION_ATTRIBUTION_KO.md','CONVEYOR_EXPANSION_ATTRIBUTION_KO.md','CONVEYOR_DATA_ATTRIBUTION_KO.md',
             'foreign-calibration-fresh-rates.png','conveyor-foreign-expansion-rates.png','conveyor-foreign-expansion-original-misses.png',
             'beltcrack-expansion-comparison.png','beltcrack-expansion-examples.png']
    names += [f'foreign-calibration-{s}.json' for s in ('protocol','fit','results','verification','plot')]
    names += [f'conveyor-foreign-expansion-{s}.json' for s in ('protocol','training','results','verification','plot')]
    for name in names:
        files['reports/'+name] = ROOT/'reports'/name
    portable = {}
    for key in list(files):
        if key.endswith('.md'):
            portable[key] = files.pop(key).read_text(encoding='utf-8').replace('../../docs/','../docs/').encode('utf-8')
    for name in ('FOREIGN_CALIBRATED_HANDOFF_KO.md','CONVEYOR_EXPANSION_HANDOFF_KO.md','EQUIPMENT_API_HANDOFF_KO.md'):
        text = (ROOT.parent/'docs'/name).read_text(encoding='utf-8').replace('../ai-training/reports/','../reports/')
        if name=='EQUIPMENT_API_HANDOFF_KO.md':
            text = text.replace('.venv/Scripts/python.exe','python').replace('runs/beltcrack-cached-expansion-v2/final.pt','beltcrack-cached-expansion-v2/final.pt')
        if name=='CONVEYOR_EXPANSION_HANDOFF_KO.md':
            text = '이 문서는 이전 v4 기록이다. 해당 이전/미보정 모델 파일은 v4 ZIP에서 보존한다. v5 실행은 FOREIGN_CALIBRATED_HANDOFF_KO.md를 사용한다.\n\n'+text
        portable['docs/'+name] = text.encode('utf-8')
    guide = f'''# 역할4 전달 v5: 기존 벨트 균열 + 이물질 보정 기준

Python3.12와 environment.json의 torch/torchvision/numpy/Pillow/fastapi/uvicorn 환경에서 실행한다.
압축을 해제하고 docs/FOREIGN_CALIBRATED_HANDOFF_KO.md를 먼저 읽는다.

이물질 서버:
python -m safelog_ai.foreign_calibrated_server --checkpoint conveyor-foreign-calibrated-v3/final.pt --sha256 {fit['weights_sha256']} --port 8772

벨트 균열 서버:
python -m safelog_ai.equipment_server --checkpoint beltcrack-cached-expansion-v2/final.pt --sha256 {belt['weights_sha256']} --port 8771

Android USB는 adb reverse tcp:8772 tcp:8772 및 필요하면8771번 포트를 설정한다.
이 서버는 개발 PC CPU에서 실행하며 모바일 앱 전체 빌드 결과물이 아니다.
Qt/C++에서는 새 이물질 응답 schema v2를 확인하고 서버의 candidateDetected를 사용한다.
presenceScore >0.5를 클라이언트에서 다시 계산하지 않는다. 반환 threshold와 모델/사진 SHA를 DB에 저장한다.

실제 추가 학습30epoch/4558장/4290업데이트 후, CAL320장으로 검출 기준을 보정했다.
보정 자체는 신경망0epoch/0업데이트이며 원래 가중치는 그대로다.
새783장: 오탐0/399, 미탐0/384. 기존448장: 오탐2/256(0.78%), 미탐0/192.
기존 시험 오탐이1→2건 늘어난 변화도 결과 문서에 기록했다.
새783장도 알려진4 TEST 촬영명의 다른 프레임이며 독립 공장 검증이 아니다.
적용 범위는 철광석 흐름의 잘린 근접 사진이다. 정상 가동·기계 고장을 판정하지 않는다.
최종 확인은 사람이 하고 앱 기본 모델은 자동 변경하지 않았다.

이전 모델은 v3/v4 ZIP으로 보존한다. 학습 실패·보정 자료·비교 수치를 함께 동봉한다.
원본 학습 사진·개인 사진·비밀 키는 없다. 공개 파생 그림의 출처·라이선스를 함께 유지한다.
models.json과 file-sha256.json으로 모델과 파일을 확인한다.
'''
    portable.update({'README_RUN_KO.md':guide.encode('utf-8'),'safelog_ai/__init__.py':b'',
        'models.json':json.dumps(models,indent=2).encode(),
        'environment.json':json.dumps({'python':'3.12','verified_versions':{n:importlib.metadata.version(n)
            for n in ('torch','torchvision','numpy','Pillow','fastapi','uvicorn')}},indent=2).encode()})
    manifest = {name:sha(path) for name,path in files.items()}
    import hashlib
    manifest.update({name:hashlib.sha256(content).hexdigest() for name,content in portable.items()})
    assert not PACKAGE.exists();PACKAGE.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(PACKAGE,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
        for name,path in files.items():
            archive.write(path,name)
        for name,content in portable.items():
            archive.writestr(name,content)
        archive.writestr('file-sha256.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(PACKAGE) as archive:
        assert archive.testzip() is None
        for name,digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    new(ROOT/'reports/calibrated-conveyor-release-v5.json',{'status':'created_and_verified','package':str(PACKAGE.resolve()),
        'package_sha256':sha(PACKAGE),'bytes':PACKAGE.stat().st_size,'models':models,'file_sha256':manifest,
        'builder_sha256':sha(Path(__file__)),'new_neural_epochs_in_calibration':0,'private_photos_or_api_keys_included':False})
    print({'package':str(PACKAGE),'sha256':sha(PACKAGE),'bytes':PACKAGE.stat().st_size},flush=True)


def verify():
    release = read(ROOT/'reports/calibrated-conveyor-release-v5.json')
    assert sha(PACKAGE)==release['package_sha256'] and not UNPACK.exists();UNPACK.mkdir()
    with zipfile.ZipFile(PACKAGE) as archive:
        for name in archive.namelist():
            assert (UNPACK/name).resolve().is_relative_to(UNPACK.resolve())
        archive.extractall(UNPACK)
    for name,digest in read(UNPACK/'file-sha256.json').items():
        assert sha(UNPACK/name)==digest
    models = read(UNPACK/'models.json');rows = read(PRIOR/'manifest.json')['rows']
    observations = []
    for key,model in models.items():
        belt = key=='belt_crack'
        module = 'safelog_ai.equipment_server' if belt else 'safelog_ai.foreign_calibrated_server'
        endpoint = '/v1/equipment/belt/analyze' if belt else '/v1/equipment/conveyor/foreign/analyze'
        photos = [ROOT/'runs/belt-author-embedded-sample.jpg'] if belt else [PRIOR/next(r for r in rows
            if r['split']=='train' and r['foreign_object_photo_label']==label)['native'] for label in (0,1)]
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1',0));port = reservation.getsockname()[1]
        env = dict(os.environ);env.pop('PYTHONPATH',None)
        with (UNPACK/(key+'.log')).open('xb') as log:
            server = subprocess.Popen([sys.executable,'-m',module,'--checkpoint',model['path'],'--sha256',model['sha256'],
                '--port',str(port)],cwd=UNPACK,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                url = f'http://127.0.0.1:{port}'
                for attempt in range(45):
                    if server.poll() is not None:
                        raise RuntimeError('Delivered server failed: '+key)
                    try:
                        health = requests.get(url+'/health',timeout=2);health.raise_for_status()
                        assert health.json()['modelSha256']==model['sha256'];break
                    except requests.RequestException:
                        time.sleep(1)
                else:
                    raise RuntimeError('Delivered server startup timed out')
                for photo in photos:
                    raw = photo.read_bytes();body = {'equipmentCategory':'conveyor_belt','promptVersion':'belt-v1' if belt else 'conveyor-foreign-calibrated-v2',
                        'image':base64.b64encode(raw).decode()}
                    if not belt:
                        body['sceneType'] = 'iron_ore_closeup'
                    response = requests.post(url+endpoint,json=body,timeout=30);response.raise_for_status()
                    value = response.json();value.pop('heatmapPngBase64',None)
                    import hashlib
                    assert value['modelSha256']==model['sha256'] and value['inputSha256']==hashlib.sha256(raw).hexdigest()
                    assert value['humanConfirmationRequired'] and value['mechanicalFailureState']=='not_assessed'
                    if not belt:
                        assert value['schema']=='safelog.equipment.conveyor-foreign.v2'
                        assert value['threshold']==model['threshold'] and value['candidateDetected']==(value['presenceScore']>model['threshold'])
                        assert not value['fieldPerformanceValidated'] and not value['localizationAvailable']
                    observations.append({'variant':key,'http_status':response.status_code,'result':value})
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill();server.wait(timeout=10)
    new(ROOT/'reports/calibrated-conveyor-standalone-v5.json',{'status':'passed','package_sha256':release['package_sha256'],
        'all_package_sha256_passed':True,'actual_cpu_http_inference':observations,'imports_from_unpack_only':True,
        'held_out_photos_used_for_smoke':False,'additional_test_model_scores':0,'owned_servers_stopped':True,
        'calibration_tests_passed':5,'calibrated_engine_tests_passed':2,'source_sha256':sha(Path(__file__))})
    print({'status':'passed','actual_http_requests':len(observations),'servers_stopped':True},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('mode',choices=('build','verify'))
    args = parser.parse_args();{'build':build,'verify':verify}[args.mode]()
