"""Report both fixed candidates; export only the declared VAL-selected model."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import Request,urlopen
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.study_unified_preservation_typed import study
BASE,VARIANTS,FACILITY_CLASSES=study.BASE,study.VARIANTS,study.FACILITY_CLASSES
from scripts.unified_main_data import DATA,read,sha
from scripts.release_unified_main import local_links,MODULES
REPO=ROOT.parents[1];REPORTS=ROOT/'reports';OUTPUTS=REPO.parents[1]/'outputs'
NAME='unified-main-preservation-role4-handoff-2026-10-13-v2'
STAGE=BASE/'handoff-stage';ARCHIVE=OUTPUTS/(NAME+'.zip');ISOLATED=BASE/'isolated-zip'


def new(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('x',encoding='utf-8',newline='\n') as out:
        out.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def pct(value):return '미측정' if value is None else f'{100*value:.2f}%'


def alignment_audit():
    import numpy as np
    manifest=read(DATA/'manifest.json');draws=np.load(BASE/'draws.npz',allow_pickle=False)
    rows=manifest['fit']['facility_surface'];indices=sorted(set(draws['facility_surface'].flatten().tolist()))
    def count(index):
        row=rows[index]
        with np.load(ROOT/row['pixel_target'],allow_pickle=False) as target:
            known=target['known'];changed=np.count_nonzero(known!=known[...,::-1])
            return row['domain'],int(changed>0),int(changed),int(np.count_nonzero(known==0))
    domains={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for domain,affected,cells,unknown in pool.map(count,indices):
            entry=domains.setdefault(domain,{'selected_records':0,'asymmetric_known_grid_records':0,'mirrored_known_cells_changed':0,'unknown_cells':0})
            entry['selected_records']+=1;entry['asymmetric_known_grid_records']+=affected
            entry['mirrored_known_cells_changed']+=cells;entry['unknown_cells']+=unknown
    new(REPORTS/'unified-preservation-alignment-train-audit.json',{'status':'audited_TRAIN_only',
        'unique_scheduled_facility_records':len(indices),'domains':domains,'new_labels':0,
        'meaning':'Count changed known coordinates under horizontal mirroring; not additional defect labels',
        'heldout_images_or_annotations_used':False})
    print(json.dumps({'alignment_TRAIN_audit':domains}),flush=True)


def report():
    choice=read(REPORTS/'unified-preservation-selection.json');verified=read(REPORTS/'unified-preservation-verification.json')
    protocol=read(study.PROTOCOL);baseline=read(REPORTS/'unified-main-results.json')
    validation={v:read(REPORTS/f'unified-preservation-{v}-validation.json') for v in VARIANTS}
    foreign={v:read(REPORTS/f'unified-preservation-{v}-foreign.json')['foreign'] for v in VARIANTS}
    training={v:read(REPORTS/f'unified-preservation-{v}-training.json') for v in VARIANTS}
    lines=['# 通합 메인 보존 대조 · 실제 결과'.replace('通','통'),'','기준일: 2026-10-13','','## 수행한 작업','',
        '같은 통합 v1 모델에서 두 후보를 실제로 추가 학습했다. 두 후보 모두 시설·벨트·이물질을 같은 공통 MobileNet 분석부 하나에 연결한다. joint는 공통 분석부까지 업데이트했고 frozen은 공통 가중치를 정확히 고정한 채 기능별 출력부만 학습했다. 앱에 두 후보를 동시에 로드하는 방식은 아니다.','',
        '- **후보마다10epoch·3,840회, 합계20후보 epoch·7,680회 추가 업데이트**. 이전 v1의20epoch·6,400회를 반복 실행한 수치가 아니다. 선택한 새 후보의 학습 이력은 v1의20회 + 추가10회다.',
        f"- 완료 업데이트 journal 집계: {verified['actual_optimizer_calls_total']:,}회. 사전 GPU 점검2회는 제외한다. 중단 후 버린 업데이트가 있으면 완료 모델에 반영된7,680회와 journal 수를 구분한다.",
        '- 시설·벨트·이물질을 epoch당 각각1,024회 뽑았다. 두 후보의 원본, 정답, 순서, 학습률, 손실은 같고 공통 분석부 업데이트 여부만 다르다.',
        '- DACL10k·DamSegment·CODEBRIM·BeltCrack·Mendeley 이물질5출처를 재사용했다. 고유 예약 TRAIN15,260 record이며 새 데이터·새 AI 정답은0개다. 일부는 작성자 patch로 독립 전체 사진 수와 다르다.',
        '- 실제 데이터의 known은 좌표 mask가 아니라7개 클래스 플래그다. 원래 v1 처리대로 이미지·pixel target만 뒤집고 클래스 플래그는 유지한다. 두 후보에 같은 검증을 적용했다.',
        '- 먼저 진행한 v2 부분 학습은 known을 좌표로 오해해 클래스 플래그를 반전한 오류가 있어 중단·제외했다. 해당2,179회 journal 업데이트와 중간 파일을 보존했으며 이번 유효 모델에 상속하지 않았다.',
        '- 64회 업데이트마다 가중치·옵티마이저·현재 순서·난수 상태를 저장했다. 완료된 과거 모델과 결과는 SHA로 보존했다.','',
        '| 후보 | 실제 추가 회차 | 실제 업데이트 | 학습·CAL 시간 | 공통 분석부 변경 |', '|---|---:|---:|---:|---|']
    for v in VARIANTS:
        t=training[v];lines.append(f"| {v} | {t['epochs']} | {t['optimizer_updates']:,} | {t['elapsed_training_and_CAL_minutes']:.2f}분 | {'변경' if t['backbone_changed'] else '전체 바이트 동일'} |")
    lines += ['','## 시설 · 같은 알려진 VAL1,745장','',
              '이전 시설 기준을 그대로 사용한다. 아래는 항목별 오탐/미탐이며 unknown 항목은 분모에서 제외했다. 전체 정답 분모·FP·FN은 후보별 validation JSON에 있다.','',
              '| 출처 | 항목 | 통합 v1 오탐 / 미탐 | joint 오탐 / 미탐 | frozen 오탐 / 미탐 |', '|---|---|---:|---:|---:|']
    for domain,rows in baseline['facility_fixed_old_thresholds'].items():
        for c,value in rows.items():
            a=value['unified'];b=validation['joint']['facility'][domain][c]['candidate'];d=validation['frozen']['facility'][domain][c]['candidate']
            lines.append(f"| {domain} | {c} | {pct(a['fpr'])} / {pct(a['fnr'])} | {pct(b['fpr'])} / {pct(b['fnr'])} | {pct(d['fpr'])} / {pct(d['fnr'])} |")
    lines += ['','## 벨트 · 같은 알려진 작성자 VAL353장','',
              '| 항목 | 통합 v1 | joint | frozen |','|---|---:|---:|---:|',
              '| 사진 미탐 | '+' | '.join(pct(x['photo_rates']['fnr']) for x in [baseline['belt_val353'],validation['joint']['belt'],validation['frozen']['belt']])+' |',
              '| 손상 위치 pixel IoU | '+' | '.join(pct(x['pixel_iou']) for x in [baseline['belt_val353'],validation['joint']['belt'],validation['frozen']['belt']])+' |',
              '| 정상 사진 오탐 | 미측정·정상0장 | 미측정·정상0장 | 미측정·정상0장 |','',
              '사진 미탐과 손상 위치 성능을 함께 본다. 원본 픽셀의 background FPR을 정상 사진 FPR로 바꾸어 발표하지 않는다.','',
              '## 연구 후보 선택과 이물질 시험','',
              f"사전에 등록한 VAL 조건에 따른 선택: **{choice['selected']}**. 시설 균열·박락의 최대 오류, 벨트 사진 미탐과 위치 IoU가 통합 v1보다 나빠지지 않는 조건을 모두 확인했다. 둘 다 통과하지 않으면 v1을 유지한다. 이 선택을 저장한 뒤 이물질 TEST를 평가했고 시험 수치로 선택을 바꾸지 않았다.",'',
              '| 알려진 시험 | 통합 v1 오탐 / 미탐 | joint 오탐 / 미탐 | frozen 오탐 / 미탐 |','|---|---:|---:|---:|']
    for name in ['common448','known783']:
        rates=[baseline['foreign'][name],foreign['joint'][name],foreign['frozen'][name]]
        lines.append('| '+name+' | '+' | '.join(pct(r['fpr'])+' / '+pct(r['fnr'])+f" (정상{r['negative_photos']} / 이물질{r['positive_photos']})" for r in rates)+' |')
    lines += ['','이물질 기준은 후보마다 별도 CAL320장에서 같은 알고리즘으로 정했다. CAL 재대입 수치는 성능 검증으로 쓰지 않는다. 두 TEST 묶음은 같은4개 알려진 촬영 기록의 프레임이며 독립 현장이나 이번 회차의 새 사진이 아니다.','',
              '## 전달과 다음 판단','',
              '연구 후보를 선택해도 기존 앱 기본 모델은 교체하지 않는다. 정상 벨트 정답과 독립 산업현장 검증이 없어 전체 항목 오탐·미탐5% 미만을 검증하지 못했다. 내부 모터·롤러 고장이나 정상 가동 여부도 사진으로 확정하지 않는다. 위험 등급·조치 문장·요약은 기존 규칙 템플릿이다.','',
              '추가 학습은 시설 비중을 높인 회차다. joint 대 frozen 비교는 같은 조건에서 공통 분석부 고정 효과를 보는 대조이며, v1 대비 결과를 데이터 추가만의 효과로 설명하지 않는다. 기존 v1의 known 처리는 올바른 처리였다.','',
              '![실제 추가 학습 비교](unified-preservation-comparison.png)','',
              '- [유효 대조의 사전 등록 조건](unified-preservation-typed-protocol.json)',
              '- [실제 정답 구조·클래스 플래그 보존](unified-preservation-known-scope.json)',
              '- [잘못된 부분 학습의 중단·제외 기록](unified-preservation-v2-aborted.json)',
              '- [선택 조건 통과·실패](unified-preservation-selection.json)',
              '- [학습·입력·가중치 보존 검증](unified-preservation-verification.json)',
              '- [joint 전체 VAL 계수](unified-preservation-joint-validation.json)',
              '- [frozen 전체 VAL 계수](unified-preservation-frozen-validation.json)',
              '- [출처와 사용 조건](UNIFIED_MAIN_ATTRIBUTION_KO.md)','']
    new(REPORTS/'UNIFIED_PRESERVATION_RESULTS_KO.md','\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Unified v1','Joint +10','Frozen +10'];results=[protocol['baseline_for_guards'],validation['joint'],validation['frozen']]
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    for axis,key,title in [(axes[0],'facility','Facility crack/spall max error'),(axes[1],'fnr','Belt photo FNR'),(axes[2],'iou','Belt pixel IoU (higher is better)')]:
        values=[r['facility_primary_max'] if key=='facility' else r['belt']['photo_rates']['fnr'] if key=='fnr' else r['belt']['pixel_iou'] for r in results]
        axis.bar(labels,[100*v for v in values],color=['#607684','#3e92bd','#47864d']);axis.set_ylabel('%');axis.set_ylim(0,max(values)*135+.5);axis.set_title(title,fontsize=11)
        for x,y in enumerate(values):axis.text(x,100*y+.3,f'{100*y:.2f}%',ha='center')
        axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Fixed +10 epoch candidates on known reused benchmarks',fontsize=14)
    fig.text(.5,.02,'Normal belt photo FPR: unavailable. Research selection: '+choice['selected']+'. No factory validation.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.92));fig.savefig(REPORTS/'unified-preservation-comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({'report':'created','selected':choice['selected']}),flush=True)


def package():
    choice=read(REPORTS/'unified-preservation-selection.json')
    if choice['selected']=='prior_v1':
        new(REPORTS/'unified-preservation-release.json',{'status':'prior_v1_preserved_no_new_model_recommended',
            'selection':choice,'app_default_promoted':False,'new_package':None})
        print('Both guards did not pass; keeping tested v1 handoff',flush=True);return
    assert not STAGE.exists() and not ARCHIVE.exists();STAGE.mkdir()
    model=ROOT/choice['checkpoint_path'];assert sha(model)==choice['checkpoint_sha256'];shutil.copy2(model,STAGE/'model.pt')
    new(STAGE/'model.json',{'path':'model.pt','sha256':sha(model),'researchVariant':choice['selected'],
        'apiArchitectureFamily':'unified-main-v1','checkpoint_count':1,'backbone_count':1,'inspectionTasks':['facility_surface','belt_damage','conveyor_foreign'],
        'base_epochs':20,'additional_epochs':10,'additional_optimizer_updates':3840,
        'app_default_promoted':False,'fieldPerformanceValidated':False})
    new(STAGE/'requirements-api.txt','fastapi==0.142.1\nuvicorn==0.54.0\npydantic==2.13.5\npillow==12.3.0\nnumpy==2.5.2\n')
    new(STAGE/'safelog_ai/__init__.py','')
    for name in MODULES:shutil.copy2(ROOT/'safelog_ai'/name,STAGE/'safelog_ai'/name)
    new(STAGE/'README_KO.md','''# SafeLog 후속 통합 메인 · 역할4

통합 v1에서 추가 학습한 VAL 선택 연구 후보 하나를 전달한다. 모델 하나·공통 분석부 하나로 시설·벨트 손상·이물질 항목을 처리한다.

- [실제 비교·개선과 악화](safelog-cpp/ai-training/reports/UNIFIED_PRESERVATION_RESULTS_KO.md)
- [실행·앱 연결](safelog-cpp/docs/UNIFIED_PRESERVATION_HANDOFF_KO.md)
- [출처와 연구·교육 사용 조건](safelog-cpp/ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md)

기존 앱 기본 모델은 자동 교체하지 않았다. 전체 산업현장5% 미만이나 내부 고장 판정 기능을 검증한 모델이 아니다.
''')
    guide=f'''# 역할4 · 후속 통합 메인 연결

연구 후보 `{choice['selected']}`를 모델 파일 하나로 제공한다. API 응답의 modelName은 기존 구조 계열 `unified-main-v1`이며 실제 학습 버전은 modelSha256과 model.json의 researchVariant로 구분한다.

## CPU 서버 실행

`{NAME}.zip`을 새 폴더에 풀고 Python3.12에서 실행한다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$mainModelRecord = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.unified_main_server --checkpoint $mainModelRecord.path --sha256 $mainModelRecord.sha256 --port 8775
```

제작 환경에서 CPU로 실제 실행한 결과와 완전히 새 환경의 pip 설치 시험은 구분한다. 서버는127.0.0.1에서 실행하며 실제 스마트폰에는 별도 서버 주소·인증·HTTPS 구성이 필요하다.

## 기존 C++ 어댑터 사용

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="belt_damage"
```

같은 Qt/C++ [설정](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp)과 [요청 코드](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp)를 사용한다. 점검 항목은 facility_surface·belt_damage·conveyor_foreign이다. 앱 시작 환경변수는 한 실행의 모드를 정하며 촬영 화면의 항목 선택·DB 연결 UI는 팀에서 이어 붙인다.

POST /v1/analyze-hazard는 image, inspectionTask, userMemo, promptVersion을 받는다. POST /v1/compare-action은 beforeImage, afterImage, inspectionTask, actionNote, promptVersion을 받는다. POST /v1/summarize는 inspectionContext와 promptVersion을 받는다. GET /health의 ready, neuralModelCount1, sharedBackboneCount1, modelSha256을 확인한다.

점검 항목이 없거나 지원하지 않으면422다. 다른 항목을 정상으로 저장하지 않는다. 벨트만 원본 크기의 확인용 heatmap을 내보내며 확정 결함 위치가 아니다. score는 보정된 고장 확률이 아니고, 사진 후보 변화로 수리 완료를 확정하지 않는다. 위험 등급·조치 문장·요약은 규칙 템플릿이며 점검자 최종 확인이 필요하다. 사진 외의 내부 고장은 not_assessed다.

이 전달은 CPU API·연결 코드이며 Android APK를 만든 버전은 아니다. [전체 성능](../ai-training/reports/UNIFIED_PRESERVATION_RESULTS_KO.md)과 [출처·사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md)을 함께 전달한다.
'''
    new(REPO/'safelog-cpp/docs/UNIFIED_PRESERVATION_HANDOFF_KO.md',guide)
    queue=[REPORTS/'UNIFIED_PRESERVATION_RESULTS_KO.md',REPO/'safelog-cpp/docs/UNIFIED_PRESERVATION_HANDOFF_KO.md']
    queue += [p for p in REPORTS.glob('unified-preservation-*') if p.suffix in ('.json','.png')]
    queue += [REPO/'safelog-cpp/apps/mobile-qt/tools/unified_main_probe.cpp',REPO/'safelog-cpp/src/ai/include/safelog/ai/ai_analyzer.hpp']
    seen=set()
    while queue:
        source=queue.pop().resolve()
        if source in seen:continue
        assert source.is_relative_to(REPO) and source.is_file(),source
        seen.add(source);target=STAGE/source.relative_to(REPO);target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source,target)
        if source.suffix=='.md':queue.extend(local_links(source))
    for doc in STAGE.rglob('*.md'):
        for link in local_links(doc):assert link.is_relative_to(STAGE) and link.is_file()
    members={p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()}
    new(STAGE/'package-sha256.json',members)
    with zipfile.ZipFile(ARCHIVE,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for file in sorted(STAGE.rglob('*')):
            if file.is_file():archive.write(file,file.relative_to(STAGE).as_posix())
    print(json.dumps({'archive':str(ARCHIVE),'sha256':sha(ARCHIVE),'bytes':ARCHIVE.stat().st_size}),flush=True)


def http(path,body=None):
    request=Request('http://127.0.0.1:8775'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=30) as response:return json.load(response)


def smoke():
    choice=read(REPORTS/'unified-preservation-selection.json')
    if choice['selected']=='prior_v1':return
    assert not ISOLATED.exists()
    with socket.socket() as port:assert port.connect_ex(('127.0.0.1',8775))!=0,'8775 occupied; preserve unrelated server'
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None and sum(p.endswith('.pt') for p in archive.namelist())==1
        for name in archive.namelist():assert (ISOLATED/name).resolve().is_relative_to(ISOLATED)
        archive.extractall(ISOLATED)
    digest=choice['checkpoint_sha256'];assert sha(ISOLATED/'model.pt')==digest
    manifest=read(DATA/'manifest.json');belt=ROOT/'runs/belt-author-embedded-sample.jpg'
    assert any(r['embedded_image_sha256']==sha(belt) for r in manifest['fit']['belt_damage'])
    photos={'facility_surface':ROOT/next(r['image'] for r in manifest['fit']['facility_surface'] if r['targets'][0]==1),
            'belt_damage':belt,'conveyor_foreign':ROOT/next(r['image'] for r in manifest['fit']['conveyor_foreign'] if r['label']==1)}
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8';env['PATH']='C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH','');env['QT_QPA_PLATFORM']='offscreen'
    log=(ISOLATED/'server.log').open('x',encoding='utf-8')
    process=subprocess.Popen([sys.executable,'-m','safelog_ai.unified_main_server','--checkpoint','model.pt','--sha256',digest,'--port','8775'],
        cwd=ISOLATED,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    proofs=[];hazards=[]
    try:
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Packaged CPU server exited')
            try:health=http('/health');break
            except (OSError,ValueError):time.sleep(.25)
        else:raise RuntimeError('Packaged server timeout')
        assert health['ready'] and health['neuralModelCount']==health['sharedBackboneCount']==1 and health['modelSha256']==digest
        for task,path in photos.items():
            reply=http('/v1/analyze-hazard',{'image':base64.b64encode(path.read_bytes()).decode(),'inspectionTask':task,'promptVersion':'preservation-release-v2'})
            assert reply['modelSha256']==digest and reply['inputSha256']==sha(path)
            assert reply['humanConfirmationRequired'] and not reply['fieldPerformanceValidated'] and not reply['otherTasksEvaluated']
            assert reply['mechanicalFailureState']=='not_assessed' and not reply['beltPhotoFprValidated']
            if task=='belt_damage':
                from PIL import Image
                with Image.open(io.BytesIO(base64.b64decode(reply['heatmapPngBase64']))) as heat:assert heat.size==(reply['inputWidth'],reply['inputHeight'])
            else:assert reply['heatmapPngBase64'] is None
            reply.pop('heatmapPngBase64');hazards.append(reply)
            output=ISOLATED/(task+'-cpp.json')
            result=subprocess.run([str(ROOT/'runs/unified-main-qt-probe/build/unified_probe.exe'),task,str(path),digest,str(output)],
                cwd=ISOLATED,env=env,timeout=45,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            assert result.returncode==0 and output.is_file()
            proof=read(output);assert proof['status']=='passed' and proof['modelSha256']==digest and proof['operations']==3;proofs.append(proof)
            print(task+': selected model actual CPU/Qt HTTP passed',flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        log.close()
    with zipfile.ZipFile(ARCHIVE) as archive:
        members=json.loads(archive.read('package-sha256.json'))
        assert all(hashlib.sha256(archive.read(n)).hexdigest()==h for n,h in members.items())
    for doc in ISOLATED.rglob('*.md'):
        for link in local_links(doc):assert link.is_relative_to(ISOLATED) and link.is_file()
    new(REPORTS/'unified-preservation-release.json',{'status':'passed','archive':str(ARCHIVE),'archive_sha256':sha(ARCHIVE),
        'archive_bytes':ARCHIVE.stat().st_size,'model_sha256':digest,'selected':choice['selected'],'checkpoint_count':1,'backbone_count':1,
        'isolated_zip_CPU_HTTP':True,'actual_CPP_http_operations':9,'cpp_proofs':proofs,'hazard_responses':hazards,
        'all_smoke_photos_are_TRAIN':True,'zip_crc_member_SHA_and_local_links_verified':True,'owned_server_stopped':process.poll() is not None,
        'fresh_pip_install_tested':False,'all_tasks_industrial_5percent_verified':False,'app_default_promoted':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['alignment-audit','report','package','smoke'])
    globals()[parser.parse_args().mode.replace('-','_')]()
