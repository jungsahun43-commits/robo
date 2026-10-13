"""Package the broad RESEARCH prototype; test isolated CPU and Qt/C++ calls."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.study_equipment_expansion import read,RUN,MANIFEST,validate
from scripts.fetch_equipment_expansion import sha
from scripts.release_unified_main import MODULES,local_links
from safelog_ai.equipment_expansion import CATEGORIES
REPO=ROOT.parents[1];REPORTS=ROOT/'reports';OUTPUTS=REPO.parents[1]/'outputs'
NAME='unified-equipment-prototype-role4-handoff-2026-10-14-v1.1'
STAGE=RUN/'handoff-stage-v1.1';ISOLATED=RUN/'isolated-zip-v1.1';ARCHIVE=OUTPUTS/(NAME+'.zip')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def guide():
    t=read(REPORTS/'equipment-expansion-v1-training.json')
    text='''# 역할4 · 산업 설비 범위 확장 시제품 연결

기존 시설·벨트·이물질과 새 설비11유형을 같은 공통 분석부 하나·모델 파일 하나로 처리한다. 이 전달물은 전체 앱 흐름을 연결하기 위한 연구 시제품이다. 새 유형의 오탐·미탐 목표 통과 여부는 `/v1/equipment-categories`와 실제 성능표를 확인한다. 기존 전달 파일과 앱 기본 모델은 자동 교체하지 않았다. APK를 만든 전달물이 아니다.

## 실행

ZIP을 새 폴더에 풀고 Python3.12에서 실행한다. CPU 서버는127.0.0.1에만 바인딩한다. 제작환경CPU실행과 새로운 pip설치 검증은 구분한다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$equipmentPrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.equipment_expansion_server --checkpoint $equipmentPrototype.path --sha256 $equipmentPrototype.sha256 --evaluation evaluation.json --port 8775
```

실제 스마트폰에서는 팀의 서버 주소·인증·HTTPS를 연결한다. 사용환경을 선택한 뒤 설비 사진을 입력하는 화면에 연결한다.

## 기존 C++ 어댑터 연결

기존 `HttpAiSafetyAnalyzerConfig.inspectionTask` 또는 환경변수로 유형 ID를 전달하면 서버가 해당 설비 항목으로 분석한다. 모델을 따로11개 설치하는 설정이 아니다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="ksdd2_surface"
```

| 화면 항목 | inspectionTask 값 |
|---|---|
| 시설 표면 | facility_surface |
| 벨트 손상 | belt_damage |
| 컨베이어 이물질 | conveyor_foreign |
| 케이블 | ad_cable |
| 나사 | ad_screw |
| 너트 | ad_metal_nut |
| 전선 연결부 | loco_splicing_connectors |
| 검정/갈색/흰색 브래킷 | mpdd_bracket_black / mpdd_bracket_brown / mpdd_bracket_white |
| 금속 연결부 | mpdd_connector |
| 금속판 | mpdd_metal_plate |
| 금속 튜브 | mpdd_tubes |
| 부품 표면 | ksdd2_surface |

화면에서 선택한 유형을 런타임 config에 전달한다. 전역 환경변수는 개발 시 기본항목을 고르는 예시다. 다른 설비의 사진을 같은 유형으로 보내면 성능표와 같은 조건이 아니다.

## HTTP 계약

- GET `/health`: 모델SHA·모델수1·공통분석부수1.
- GET `/v1/equipment-categories`: 유형 ID·표시이름·학습여부·평가단계.
- GET `/v1/inference-policy`: 기존 벨트와 새 설비의 분석정책SHA.
- POST `/v1/analyze-hazard`: image·inspectionTask·userMemo·promptVersion.
- POST `/v1/compare-action`: beforeImage·afterImage·inspectionTask·actionNote·promptVersion.
- POST `/v1/summarize`: inspectionContext·promptVersion.

새 UI는아래명시적형식도사용할수있다. 기존C++별칭방식과같은모델출력이다.

```json
{
  "inspectionTask": "equipment_surface",
  "equipmentCategory": "ad_cable",
  "image": "<JPEG또는PNG base64>",
  "userMemo": "케이블 외관 점검",
  "promptVersion": "equipment-prototype-v1"
}
```

equipment_surface의유형누락·알수없는유형·별칭/유형충돌은422다. 요청하지않은다른설비를정상으로저장하지않는다. 입력은JPEG/PNG8MB이하·12MP이하이며원본을전달한다. 새설비는256RGB로분석하고벨트는기존원본구역정책을유지한다.

## 역할별 연결 작업

1. 역할1: 설비종류선택·사진입력·후보heatmap·점검자의채택/수정/거절 화면.
2. 역할2: 설비ID·종류·사진·점검의견·AI결과·사람최종확인·modelSha256·inferencePolicySha256·점검/조치이력 저장.
3. 역할3: 유형별데이터·학습·오탐/미탐/위치평가와새모델 전달. 20%넘는유형은보강상태를유지.
4. 역할4: 위화면/DB/API를연결하고조치전후·보고서·앱패키징을완성.

위험등급·조치문장·요약은규칙템플릿이며학습된LLM이나학습된위험도판정으로표현하지않는다. 새외관분석head는정상특징복원을실제학습했다. heatmap은거친후보확인자료이고확정고장위치가아니다. 내부고장은not_assessed,현장성능은미검증,사람최종확인은필수다.

[첫 실제 성능](../ai-training/reports/EQUIPMENT_EXPANSION_RESULTS_KO.md), [새 자료 출처](../ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md), [기존 자료 사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md).
'''
    write(REPO/'safelog-cpp/docs/EQUIPMENT_EXPANSION_HANDOFF_KO.md',text)


def package():
    validate();verified=read(REPORTS/'equipment-expansion-v1-verification.json');assert verified['status']=='passed'
    t=read(REPORTS/'equipment-expansion-v1-training.json');r=read(REPORTS/'equipment-expansion-v1-results.json')
    assert not STAGE.exists() and not ARCHIVE.exists();STAGE.mkdir()
    assert sha(RUN/'train/final.pt')==t['weights_sha256'];shutil.copy2(RUN/'train/final.pt',STAGE/'model.pt')
    write(STAGE/'model.json',{'path':'model.pt','sha256':t['weights_sha256'],'architecture':t['architecture'],'checkpoint_count':1,'backbone_count':1,
        'inspectionTasks':t['tasks'],'equipmentCategories':t['equipment_categories'],'newHeadEpochs':30,'newHeadOptimizerUpdates':2640,
        'stage20Categories':r['stage20_categories'],'allCategoriesWithin20':False,'researchPrototype':True,'appDefaultPromoted':False,'fieldPerformanceValidated':False})
    shutil.copy2(REPORTS/'equipment-expansion-v1-results.json',STAGE/'evaluation.json')
    write(STAGE/'requirements-api.txt','fastapi==0.142.1\nuvicorn==0.54.0\npydantic==2.13.5\npillow==12.3.0\nnumpy==2.5.2\n')
    write(STAGE/'safelog_ai/__init__.py','')
    for name in (*MODULES,'belt_tiles.py','unified_tiles_server.py','equipment_expansion.py','equipment_expansion_server.py'):
        shutil.copy2(ROOT/'safelog_ai'/name,STAGE/'safelog_ai'/name)
    write(STAGE/'README_KO.md','''# SafeLog 산업 설비 범위 확장 연구 시제품

모델 하나·공통 분석부 하나로기존3항목과새설비11유형을연결한다. 새11유형모두가20%목표를통과한모델은아니다. 유형별실제성능을확인하고점검자확인을유지한다. 기존앱기본모델은자동교체하지않았다.

- [역할4 실행·API·유형·DB 연결](safelog-cpp/docs/EQUIPMENT_EXPANSION_HANDOFF_KO.md)
- [실제 학습·첫 유형별 성능](safelog-cpp/ai-training/reports/EQUIPMENT_EXPANSION_RESULTS_KO.md)
- [연구 사용 조건·출처](safelog-cpp/ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md)
''')
    queue=[REPO/'safelog-cpp/docs/EQUIPMENT_EXPANSION_HANDOFF_KO.md',REPORTS/'EQUIPMENT_EXPANSION_RESULTS_KO.md',REPORTS/'EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md',
        REPO/'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp',REPO/'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp',
        REPO/'safelog-cpp/apps/mobile-qt/tools/unified_main_probe.cpp',REPO/'safelog-cpp/src/ai/include/safelog/ai/ai_analyzer.hpp']
    seen=set()
    while queue:
        source=queue.pop().resolve()
        if source in seen:continue
        assert source.is_relative_to(REPO) and source.is_file(),source
        seen.add(source);dest=STAGE/source.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if source.suffix=='.md':queue.extend(local_links(source))
    for doc in STAGE.rglob('*.md'):
        for link in local_links(doc):assert link.is_relative_to(STAGE) and link.is_file(),link
    write(STAGE/'package-sha256.json',{p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()})
    with zipfile.ZipFile(ARCHIVE,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(STAGE).as_posix())
    print(json.dumps({'archive':str(ARCHIVE),'sha256':sha(ARCHIVE)}),flush=True)


def http(path,body=None):
    req=Request('http://127.0.0.1:8775'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=30) as response:return json.load(response)


def smoke():
    assert not ISOLATED.exists()
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8775))!=0,'Preserve unrelated server on8775'
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist())==1
        for n in z.namelist():assert (ISOLATED/n).resolve().is_relative_to(ISOLATED.resolve())
        z.extractall(ISOLATED)
    metadata=read(ISOLATED/'model.json');digest=metadata['sha256'];assert sha(ISOLATED/'model.pt')==digest
    manifest=read(ROOT/'data/unified-main-v1/manifest.json');rows=read(MANIFEST)['rows']
    photos={'facility_surface':ROOT/manifest['fit']['facility_surface'][0]['image'],'belt_damage':ROOT/'runs/belt-author-embedded-sample.jpg','conveyor_foreign':ROOT/manifest['fit']['conveyor_foreign'][0]['image']}
    photos.update({cat:ROOT/next(r['image'] for r in rows if r['category']==cat and r['phase']=='fit') for cat in CATEGORIES})
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8';env['QT_QPA_PLATFORM']='offscreen'
    env['PATH']='C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH','')
    log=(ISOLATED/'server.log').open('x',encoding='utf-8');process=subprocess.Popen([sys.executable,'-m','safelog_ai.equipment_expansion_server','--checkpoint','model.pt','--sha256',digest,'--evaluation','evaluation.json','--port','8775'],
        cwd=ISOLATED,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    proofs=[];hazards=[]
    try:
        end=time.monotonic()+45
        while time.monotonic()<end:
            if process.poll() is not None:raise RuntimeError('Packaged CPU server exited: '+(ISOLATED/'server.log').read_text())
            try:health=http('/health');break
            except (OSError,ValueError):time.sleep(.25)
        else:raise RuntimeError('Owned packaged CPU server timed out')
        assert health['ready'] and health['neuralModelCount']==health['sharedBackboneCount']==1 and health['modelSha256']==digest
        categories=http('/v1/equipment-categories')['categories'];assert len(categories)==11 and all(x['trained'] for x in categories)
        assert sum(x['benchmarkStage']['ready_for_next_stage'] for x in categories)==1
        for task,image in photos.items():
            reply=http('/v1/analyze-hazard',{'image':base64.b64encode(image.read_bytes()).decode(),'inspectionTask':task,'promptVersion':'equipment-prototype-smoke-v1'})
            assert reply['modelSha256']==digest and reply['inputSha256']==sha(image) and reply['neuralModelCount']==reply['sharedBackboneCount']==1
            assert reply['humanConfirmationRequired'] and reply['researchPrototype'] and not reply['fieldPerformanceValidated']
            assert reply['mechanicalFailureState']=='not_assessed' and not reply['otherTasksEvaluated'] and len(reply['trainedDatasetSources'])==9
            reply.pop('heatmapPngBase64');hazards.append(reply)
            proof_path=ISOLATED/(task+'-cpp.json')
            result=subprocess.run([str(RUN/'qt-probe/build/unified_probe.exe'),task,str(image),digest,str(proof_path)],cwd=ISOLATED,env=env,timeout=45,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            assert result.returncode==0 and proof_path.exists(),result.stderr.decode(errors='replace')
            proof=read(proof_path);assert proof['status']=='passed' and proof['operations']==3 and proof['samePhotoDoesNotResolve'];proofs.append(proof)
            print(json.dumps({'CPU_CPP_passed':task,'operations':3}),flush=True)
        try:http('/v1/analyze-hazard',{'image':'x','inspectionTask':'equipment_surface','promptVersion':'missing'})
        except HTTPError as error:assert error.code==422
        else:raise AssertionError('Missing category must be rejected')
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        log.close()
    with zipfile.ZipFile(ARCHIVE) as z:
        hashes=json.loads(z.read('package-sha256.json'));assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
    write(REPORTS/'equipment-expansion-v1-release.json',{'status':'passed','archive':str(ARCHIVE),'archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,
        'model_sha256':digest,'one_checkpoint_one_backbone':True,'isolated_ZIP_CPU_API':True,'actual_CPP_operations':42,'CPP_cases':proofs,'CPU_hazard_cases':hazards,
        'all_smoke_photos_are_TRAIN':True,'missing_category_rejected422':True,'ZIP_CRC_member_SHA_and_links_verified':True,'owned_server_stopped':process.poll() is not None,
        'fresh_pip_install_tested':False,'Android_APK_built':False,'all_categories_photo_rates_within20percent':False,'app_default_promoted':False})
    print(json.dumps({'release':'passed','CPP_operations':42,'archive_sha256':sha(ARCHIVE)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['guide','package','smoke']);args=parser.parse_args();globals()[args.mode]()
