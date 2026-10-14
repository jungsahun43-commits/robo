"""Selected CAL20 prototype, explicit policy, isolated CPU and targeted C++ proof."""
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
from scripts.study_metal_reference import RUN,REPORTS,validate,read,START,VARIANTS
from scripts.fetch_equipment_expansion import sha
from scripts.release_unified_main import MODULES,local_links
REPO=ROOT.parents[1];OUTPUTS=REPO.parents[1]/'outputs'
NAME='unified-equipment-metal20-role4-handoff-2026-10-14-v1'
STAGE=RUN/'handoff-stage';ISOLATED=RUN/'isolated-zip';ARCHIVE=OUTPUTS/(NAME+'.zip')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def guide():
    write(REPO/'safelog-cpp/docs/METAL_CALIBRATION_HANDOFF_KO.md','''# 역할4 · 금속판20% 목표 보강 시제품

기존 범위 확장 시제품과 같은 신경망 가중치이며 금속판의 사진 경고 기준만 정상CAL20장으로 조정했다. 이 전달물은 금속판 오탐3.85%·미탐18.31%를 측정한 연구 연결용 버전이다. 부품 표면의15%통과 상태는 유지했다. 나머지9유형은 보강/평가부족 상태다. 기존 ZIP과 앱 기본 모델은 자동 교체하지 않았다.

## CPU 실행

ZIP을 새 폴더에 풀고 Python3.12에서 실행한다. 이전 시제품과 설치 의존성은 같다. 새 pip설치를 이번검사로 검증한 것은 아니다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$metalPrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.metal_calibration_server --checkpoint $metalPrototype.path --sha256 $metalPrototype.sha256 --evaluation evaluation.json --port 8775
```

새 서버 모듈은금속판의20%정상경고예산/rank17정책을명시한다. 이전서버의85%정상분위수정책으로표시하지않는다. 모델SHA와금속판정책SHA를함께기록한다. 서버는127.0.0.1에바인딩하며스마트폰의서버주소·HTTPS·인증은팀앱흐름에연결한다. APK를빌드한전달물이아니다.

## C++ 설정

기존 확장 C++ 어댑터의 유형 ID를 그대로 사용한다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="mpdd_metal_plate"
```

기존3항목과새11유형을같은모델파일/공통분석부하나로처리한다. 전체유형ID·역할별연결·DB계약은[기존전체틀가이드](EQUIPMENT_EXPANSION_HANDOFF_KO.md)를참고하되실행서버와모델파일은이번ZIP의값을사용한다.

- GET /health: 모델수1·공통분석부수1·새모델SHA.
- GET /v1/equipment-categories: 부품표면15%목표와금속판20%목표상태.
- GET /v1/inference-policy: 기존belt/equipment정책과이번metalPlate정책.
- POST /v1/analyze-hazard, /v1/compare-action, /v1/summarize: 기존계약유지.

새UI는inspectionTask=equipment_surface와equipmentCategory=mpdd_metal_plate를명시할수있다. 금속판사진응답의normalCALAlpha=.20·normalCALRank=17을확인한다. 다른설비는이전정책과기준을유지한다.

동일사진으로조치완료가나와서는안된다. 사진경고기준을조정해미탐을줄인것이며위치pixel미탐27.96%는그대로다. 위치heatmap은점검자확인자료이고확정고장위치가아니다. 내부기계고장은not_assessed,최종확인은사람이수행한다. 위험등급·조치·요약은규칙템플릿이다.

[금속판비교](../ai-training/reports/METAL_REFERENCE_RESULTS_KO.md), [자료출처·사용조건](../ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md).
''')


def package():
    validate();choice=read(REPORTS/'equipment-metal-reference-v1-selection.json');assert choice['selected']==VARIANTS[0]
    assert not STAGE.exists() and not ARCHIVE.exists();STAGE.mkdir();checkpoint=ROOT/choice['checkpoint_path'];digest=choice['checkpoint_sha256'];assert sha(checkpoint)==digest
    shutil.copy2(checkpoint,STAGE/'model.pt')
    state=torch_state(checkpoint);results=read(REPORTS/'equipment-expansion-v1-results.json');new_result=read(REPORTS/f'equipment-metal-reference-v1-{VARIANTS[0]}-validation.json')
    category=results['categories']['mpdd_metal_plate']
    category.update(photo_rates=new_result['photo_rates'],stage_gate=new_result['stage_gate'],photo_threshold=new_result['normalCAL_photo_threshold'],
        pixel_iou=new_result['pixel_iou'],pixel_fnr=new_result['pixel_fnr'],pixel_counts_model256=new_result['pixel_counts_model256'])
    observations={r['id']:r for r in new_result['observations']}
    for row in results['rows']:
        if row['category']=='mpdd_metal_plate':row.update(score=observations[row['id']]['score'],candidate_detected=observations[row['id']]['candidate'])
    results.update(status='known_prototype_metal_budget_refined',checkpoint_sha256=digest,known_TEST_reused=True,
        stage20_categories=['mpdd_metal_plate','ksdd2_surface'],all_categories_photo_rates_within20percent=False,parent_prototype_sha256=sha(START))
    write(STAGE/'evaluation.json',results)
    write(STAGE/'model.json',{'path':'model.pt','sha256':digest,'architecture':state['architecture'],'checkpoint_count':1,'backbone_count':1,
        'inspectionTasks':state['tasks'],'equipmentCategories':state['equipment_categories'],'baseHeadEpochs':30,'baseHeadOptimizerUpdates':2640,
        'newNeuralUpdatesThisRefinement':0,'refinementCategory':'mpdd_metal_plate','normalCALAlpha':.20,'normalCALRank':17,
        'stage20Categories':results['stage20_categories'],'allCategoriesWithin20':False,'researchPrototype':True,'appDefaultPromoted':False,'fieldPerformanceValidated':False})
    write(STAGE/'safelog_ai/__init__.py','')
    for name in (*MODULES,'belt_tiles.py','unified_tiles_server.py','equipment_expansion.py','equipment_expansion_server.py','metal_calibration_server.py'):
        shutil.copy2(ROOT/'safelog_ai'/name,STAGE/'safelog_ai'/name)
    shutil.copy2(ROOT/'runs/equipment-expansion-v1/handoff-stage-v1.1/requirements-api.txt',STAGE/'requirements-api.txt')
    write(STAGE/'README_KO.md','''# SafeLog 금속판20% 보강 연구 시제품

신경망파일하나·공통분석부하나로기존전체틀을유지하고금속판의정상경고기준을조정했다. 금속판사진오탐3.85%·미탐18.31%,부품표면의기존15%통과상태를유지한다. 다른유형과위치기능은별도로보강한다.

- [이번버전 실행·역할4 연결](safelog-cpp/docs/METAL_CALIBRATION_HANDOFF_KO.md)
- [금속판 보강·제외한 특징비교 결과](safelog-cpp/ai-training/reports/METAL_REFERENCE_RESULTS_KO.md)
- [출처·연구사용조건](safelog-cpp/ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md)
''')
    queue=[REPO/'safelog-cpp/docs/METAL_CALIBRATION_HANDOFF_KO.md',REPORTS/'METAL_REFERENCE_RESULTS_KO.md',
        REPO/'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp',REPO/'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp']
    seen=set()
    while queue:
        source=queue.pop().resolve()
        if source in seen:continue
        assert source.is_relative_to(REPO) and source.is_file(),source;seen.add(source)
        dest=STAGE/source.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if source.suffix=='.md':queue.extend(local_links(source))
    for doc in STAGE.rglob('*.md'):
        for link in local_links(doc):assert link.is_relative_to(STAGE) and link.is_file(),link
    write(STAGE/'package-sha256.json',{p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()})
    with zipfile.ZipFile(ARCHIVE,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for path in sorted(STAGE.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(STAGE).as_posix())
    print(json.dumps({'archive':str(ARCHIVE),'sha256':sha(ARCHIVE)}),flush=True)


def torch_state(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)


def http(path,body=None):
    req=Request('http://127.0.0.1:8775'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=30) as reply:return json.load(reply)


def smoke():
    assert not ISOLATED.exists()
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8775))!=0,'Preserve unrelated server'
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist())==1
        for n in z.namelist():assert (ISOLATED/n).resolve().is_relative_to(ISOLATED.resolve())
        z.extractall(ISOLATED)
    meta=read(ISOLATED/'model.json');digest=meta['sha256'];assert sha(ISOLATED/'model.pt')==digest
    old=read(ROOT/'data/unified-main-v1/manifest.json');rows=read(ROOT/'data/equipment-expansion-v1/manifest.json')['rows']
    photos={'facility_surface':ROOT/old['fit']['facility_surface'][0]['image'],'belt_damage':ROOT/'runs/belt-author-embedded-sample.jpg','conveyor_foreign':ROOT/old['fit']['conveyor_foreign'][0]['image']}
    photos.update({c:ROOT/next(r['image'] for r in rows if r['category']==c and r['phase']=='fit') for c in ('mpdd_metal_plate','ksdd2_surface')})
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8';env['QT_QPA_PLATFORM']='offscreen'
    env['PATH']='C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH','')
    log=(ISOLATED/'server.log').open('x',encoding='utf-8');process=subprocess.Popen([sys.executable,'-m','safelog_ai.metal_calibration_server','--checkpoint','model.pt','--sha256',digest,'--evaluation','evaluation.json','--port','8775'],
        cwd=ISOLATED,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    proofs=[];replies=[]
    try:
        end=time.monotonic()+45
        while time.monotonic()<end:
            if process.poll() is not None:raise RuntimeError('Packaged CPU server exited: '+(ISOLATED/'server.log').read_text())
            try:health=http('/health');break
            except (OSError,ValueError):time.sleep(.25)
        else:raise RuntimeError('Packaged CPU server timed out')
        assert health['ready'] and health['neuralModelCount']==health['sharedBackboneCount']==1 and health['modelSha256']==digest
        categories=http('/v1/equipment-categories')['categories'];assert len(categories)==11 and sum(c['benchmarkStage']['ready_for_next_stage'] for c in categories)==2
        policy=http('/v1/inference-policy')['metalPlate'];assert policy['normalCALAlpha']==.20 and policy['normalCALRank']==17 and not policy['TESTThresholdTuning']
        for task,path in photos.items():
            result=http('/v1/analyze-hazard',{'image':base64.b64encode(path.read_bytes()).decode(),'inspectionTask':task,'promptVersion':'metal-cal20-smoke'})
            assert result['modelSha256']==digest and result['inputSha256']==sha(path) and result['neuralModelCount']==result['sharedBackboneCount']==1
            assert not result['fieldPerformanceValidated'] and result['humanConfirmationRequired']
            if task=='mpdd_metal_plate':assert result['normalCALAlpha']==.20 and result['normalCALRank']==17 and result['inferencePolicySha256']==policy['sha256']
            result.pop('heatmapPngBase64');replies.append(result)
            output=ISOLATED/(task+'-cpp.json')
            cp=subprocess.run([str(ROOT/'runs/equipment-expansion-v1/qt-probe/build/unified_probe.exe'),task,str(path),digest,str(output)],cwd=ISOLATED,env=env,capture_output=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
            assert cp.returncode==0 and output.exists(),cp.stderr.decode(errors='replace');proof=read(output);assert proof['operations']==3 and proof['samePhotoDoesNotResolve'];proofs.append(proof)
            print(json.dumps({'CPU_CPP_passed':task,'operations':3}),flush=True)
        try:http('/v1/analyze-hazard',{'image':'x','inspectionTask':'equipment_surface','promptVersion':'missing'})
        except HTTPError as error:assert error.code==422
        else:raise AssertionError('Missingcategory must fail')
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        log.close()
    with zipfile.ZipFile(ARCHIVE) as z:
        hashes=json.loads(z.read('package-sha256.json'));assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
    write(REPORTS/'equipment-metal-reference-v1-release.json',{'status':'passed','archive':str(ARCHIVE),'archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,
        'model_sha256':digest,'metal_policy_sha256':policy['sha256'],'one_checkpoint_one_backbone':True,'isolated_ZIP_CPU_API_checked':True,
        'actual_CPP_operations_this_refinement':15,'CPP_cases':proofs,'hazard_replies':replies,'all_smoke_photos_are_FIT':True,'normal_budget_policy_checked':True,
        'missingcategory_rejected422':True,'ZIP_CRC_member_SHA_links_verified':True,'owned_server_stopped':process.poll() is not None,
        'fresh_pip_install_tested':False,'Android_APK_built':False,'independent_field_validated':False,'app_default_promoted':False})
    print(json.dumps({'release':'passed','CPP_operations':15,'ZIP_SHA':sha(ARCHIVE)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['guide','package','smoke']);args=parser.parse_args();globals()[args.mode]()
