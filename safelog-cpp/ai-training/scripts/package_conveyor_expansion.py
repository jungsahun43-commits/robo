"""Deliver old and expanded variants together; preserve the completed v3 ZIP."""
import argparse
import base64
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[3]/'outputs'
PACKAGE = OUT/'equipment-ai-role4-handoff-2026-10-12-v4.zip'
UNPACK = ROOT/'runs/conveyor-expansion-v4-standalone'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    old_release = read(ROOT/'reports/conveyor-dual-handoff-release-v3.json')
    assert sha(Path(old_release['package']))==old_release['package_sha256']
    files,models = {},{}
    variants = [('belt_crack','beltcrack-expansion','beltcrack-cached-expansion-v2'),
                ('conveyor_foreign_v1','conveyor-foreign-grouped','conveyor-foreign-grouped-v1'),
                ('conveyor_foreign_expanded_v2','conveyor-foreign-expansion','conveyor-foreign-expansion-v2')]
    for key,prefix,folder in variants:
        training,result,verified = [read(ROOT/f'reports/{prefix}-{suffix}.json') for suffix in ('training','results','verification')]
        assert verified['status']=='passed'
        model = ROOT/'runs'/folder/'final.pt'
        assert sha(model)==training['weights_sha256']==result['weights_sha256']==verified['weights_sha256']
        files[folder+'/final.pt'] = model
        models[key] = {'path':folder+'/final.pt','sha256':sha(model),'class':training['class'],
                       'actual_epochs':training['epochs'],'actual_optimizer_updates':training['optimizer_updates']}
        for suffix in ('protocol','training','results','verification','plot'):
            files[f'reports/{prefix}-{suffix}.json'] = ROOT/f'reports/{prefix}-{suffix}.json'
    result = read(ROOT/'reports/conveyor-foreign-expansion-results.json')
    reference = result['common_val_reference_passed'] and result['extra_val_fpr_fnr_below_5_percent']
    for name in ('belt_inference.py','belt_segmentation.py','equipment_server.py','conveyor_foreign.py',
                 'conveyor_foreign_inference.py','conveyor_foreign_server.py'):
        files['safelog_ai/'+name] = ROOT/'safelog_ai'/name
    for name in ('BELTCRACK_EXPANSION_RESULTS_KO.md','CONVEYOR_FOREIGN_RESULTS_KO.md','CONVEYOR_FOREIGN_EXPANSION_RESULTS_KO.md',
                 'CONVEYOR_DATA_ATTRIBUTION_KO.md','CONVEYOR_FOREIGN_ATTRIBUTION_KO.md','CONVEYOR_EXPANSION_ATTRIBUTION_KO.md',
                 'beltcrack-expansion-comparison.png','beltcrack-expansion-examples.png',
                 'conveyor-foreign-grouped-rates.png','conveyor-foreign-grouped-examples.png',
                 'conveyor-foreign-expansion-rates.png','conveyor-foreign-expansion-original-misses.png',
                 'conveyor-foreign-expansion-feedback.json'):
        files['reports/'+name] = ROOT/'reports'/name
    portable = {}
    for name in ('EQUIPMENT_API_HANDOFF_KO.md','CONVEYOR_FOREIGN_API_HANDOFF_KO.md','CONVEYOR_EXPANSION_HANDOFF_KO.md'):
        content = (ROOT.parent/'docs'/name).read_text(encoding='utf-8')
        content = content.replace('../ai-training/reports/','../reports/').replace('.venv/Scripts/python.exe','python')
        for model in models.values():
            content = content.replace('runs/'+model['path'],model['path'])
        portable['docs/'+name] = content.encode('utf-8')
    for name in ('BELTCRACK_EXPANSION_RESULTS_KO.md','CONVEYOR_FOREIGN_RESULTS_KO.md','CONVEYOR_FOREIGN_EXPANSION_RESULTS_KO.md'):
        key = 'reports/'+name
        portable[key] = files.pop(key).read_text(encoding='utf-8').replace('../../docs/','../docs/').encode('utf-8')
    guide = f'''# 역할4 전달 v4: 벨트 균열 + 이물질 이전/보강 모델

1. 압축을 해제한다. Python3.12, torch/torchvision/numpy/Pillow/fastapi/uvicorn 환경을 사용한다.
   검증된 환경 버전은 environment.json에 있다. 기존 역할3 .venv도 사용 가능하다.
2. docs/CONVEYOR_EXPANSION_HANDOFF_KO.md에 있는 PowerShell 명령으로 모델을 선택한다.
   models.json의 경로·SHA로 명시적으로 실행한다. 역할4는 이 모델 서버와 Qt/C++ 앱을 연결한다.
3. 이전 모델 conveyor_foreign_v1과 새 모델 conveyor_foreign_expanded_v2를 모두 보존한다.
   두 모델은 같은 API, 입력384, 임계값0.5, 사진 단위 이물질 유무 분류를 사용한다.
4. 벨트 표면 균열 모델은 기존 별도 API를 사용한다. 이물질 모델은 위치/heatmap을 제공하지 않는다.
5. 역할2는 사진/모델 SHA, AI 제안과 사람의 채택/수정/거절, 조치 이력을 각각 저장한다.

새 모델은 실제30epoch/4,558장/{models['conveyor_foreign_expanded_v2']['actual_optimizer_updates']:,}업데이트다.
VAL 기준(오탐 악화 없음·미탐 감소)과 추가VAL5% 참고 기준을 함께 충족했는가: {reference}.
모델 추천 참고 값이며 앱 기본 모델을 자동 교체하지 않는다.
이전 모델은 실제30epoch/1,230장/1,170업데이트다. 모든 수치는 두 결과 문서에 있다.

기존 VAL/TEST는 알려진 공개 비교 자료다. 추가VAL309장도 같은3 촬영명이다.
독립 공장·스마트폰 전체 장면·기계 고장 성능으로 해석하지 않는다.
정상은 이물질 없음이며 안전 가동 판정이 아니다. 최종 확인은 사람이 한다.

이 ZIP은 모델·서버·연결 설명이다. 모바일 앱/로그인/DB/전체APK는 팀의 앱 구현과 연결한다.
원본 학습 데이터·개인 사진·비밀 키는 없다. 예시 그림은 공개 사진과 점수를 함께 표시한다.
원본 출처와 CC BY4.0/CC BY-SA4.0 변경 설명을 reports/에 동봉한다.
'''
    portable.update({'README_RUN_KO.md':guide.encode('utf-8'),'safelog_ai/__init__.py':b'',
        'models.json':json.dumps(models,indent=2).encode(),
        'environment.json':json.dumps({'python':'3.12','verified_versions':{n:importlib.metadata.version(n)
            for n in ('torch','torchvision','numpy','Pillow','fastapi','uvicorn')}},indent=2).encode()})
    manifest = {name:sha(path) for name,path in files.items()}
    manifest.update({name:hashlib.sha256(payload).hexdigest() for name,payload in portable.items()})
    OUT.mkdir(parents=True,exist_ok=True)
    assert not PACKAGE.exists()
    with zipfile.ZipFile(PACKAGE,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
        for name,path in files.items():
            archive.write(path,name)
        for name,payload in portable.items():
            archive.writestr(name,payload)
        archive.writestr('file-sha256.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(PACKAGE) as archive:
        assert archive.testzip() is None
        for name,digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    proof = {'status':'created_and_verified','package':str(PACKAGE.resolve()),'package_sha256':sha(PACKAGE),
        'bytes':PACKAGE.stat().st_size,'models':models,'file_sha256':manifest,'builder_sha256':sha(Path(__file__)),
        'reference_passed':reference,'old_v3_zip_sha256_preserved':old_release['package_sha256'],
        'app_default_promoted':False,'private_photos_or_api_keys_included':False}
    with (ROOT/'reports/conveyor-expansion-release-v4.json').open('x',encoding='utf-8') as handle:
        json.dump(proof,handle,indent=2)
    print(json.dumps({k:v for k,v in proof.items() if k!='file_sha256'},indent=2))


def verify():
    release = read(ROOT/'reports/conveyor-expansion-release-v4.json')
    assert sha(PACKAGE)==release['package_sha256'] and not UNPACK.exists()
    UNPACK.mkdir()
    with zipfile.ZipFile(PACKAGE) as archive:
        for name in archive.namelist():
            assert (UNPACK/name).resolve().is_relative_to(UNPACK.resolve())
        archive.extractall(UNPACK)
    for name,digest in read(UNPACK/'file-sha256.json').items():
        assert sha(UNPACK/name)==digest
    models = read(UNPACK/'models.json')
    base = ROOT/'data/equipment-conveyor-foreign/train-expansion-v2'
    rows = read(base/'manifest.json')['rows']
    sample = next(r for r in rows if r['split']=='train' and r['from_completed_v1'])
    observations = []
    for key in models:
        model = models[key]
        module = 'safelog_ai.equipment_server' if key=='belt_crack' else 'safelog_ai.conveyor_foreign_server'
        raw = (ROOT/'runs/belt-author-embedded-sample.jpg').read_bytes() if key=='belt_crack' else (base/sample['native']).read_bytes()
        request = {'equipmentCategory':'conveyor_belt','promptVersion':'belt-appearance-v1' if key=='belt_crack' else 'conveyor-foreign-v1',
                   'image':base64.b64encode(raw).decode()}
        if key!='belt_crack':
            request['sceneType'] = 'iron_ore_closeup'
        endpoint = '/v1/equipment/belt/analyze' if key=='belt_crack' else '/v1/equipment/conveyor/foreign/analyze'
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
                        assert health.json()['modelSha256']==model['sha256']
                        break
                    except requests.RequestException:
                        time.sleep(1)
                else:
                    raise RuntimeError('Delivered server startup timeout')
                response = requests.post(url+endpoint,json=request,timeout=30);response.raise_for_status()
                value = response.json();value.pop('heatmapPngBase64',None)
                assert value['modelSha256']==model['sha256'] and value['inputSha256']==hashlib.sha256(raw).hexdigest()
                assert value['humanConfirmationRequired'] and value['mechanicalFailureState']=='not_assessed'
                if key!='belt_crack':
                    assert not value['localizationAvailable'] and not value['fieldPerformanceValidated']
                observations.append({'variant':key,'http_status':response.status_code,'result':value,'owned_server_stopped':True})
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill();server.wait(timeout=10)
    proof = {'status':'passed','package_sha256':release['package_sha256'],'all_file_hashes_passed':True,
        'imports_from_unpacked_package_without_workspace_pythonpath':True,'actual_cpu_http_inference':observations,
        'held_out_photos_used_for_smoke':False,'additional_benchmark_model_scores':0,'owned_servers_stopped':True,
        'source_sha256':sha(Path(__file__))}
    with (ROOT/'reports/conveyor-expansion-standalone-v4.json').open('x',encoding='utf-8') as handle:
        json.dump(proof,handle,indent=2)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('build','verify'))
    args = parser.parse_args()
    {'build':build,'verify':verify}[args.mode]()
