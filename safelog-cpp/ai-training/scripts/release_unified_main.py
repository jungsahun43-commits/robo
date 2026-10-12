"""Release evidence for the frozen main experiment; never refit or relabel data."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import unquote
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REPORTS = ROOT / 'reports'
RUN = ROOT / 'runs/unified-main-v1'
OUTPUTS = REPO.parents[1] / 'outputs'
NAME = 'unified-main-role4-handoff-2026-10-13-v1'
STAGE = ROOT / 'runs' / (NAME + '-stage')
ARCHIVE = OUTPUTS / (NAME + '.zip')
ISOLATED = ROOT / 'runs/unified-main-isolated-zip-v1'
EXPECTED = '362f9cab4a137419e63118f73b6b622dccfacc37c94a8f52af62f217b1491598'
MODULES = ('unified_main_server.py', 'unified_main.py', 'auxiliary_classifier.py',
           'spatial_classifier.py', 'belt_segmentation.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def percent(rate):
    return '측정 불가' if rate is None else f'{100*rate:.2f}%'


def ratecell(row, kind):
    denominator = row['negative_photos' if kind == 'fpr' else 'positive_photos']
    count = row['fp' if kind == 'fpr' else 'fn']
    return percent(row[kind]) + (f' ({count}/{denominator})' if denominator else ' (분모0)')


def report():
    r = read(REPORTS / 'unified-main-results.json')
    t = read(REPORTS / 'unified-main-training.json')
    old = read(REPORTS / 'foreign-calibration-results.json')
    names = dict(concrete_crack='균열', concrete_spalling='박락', rust_stain='녹 얼룩',
                 exposed_rebar='철근 노출', wet_surface='젖은 표면', efflorescence='백화', surface_cavity='표면 공동')
    lines = ['# 기존 메인 확장 · 실제 학습과 검증 결과', '', '기준일: 2026-10-13', '',
             '## 실제로 학습한 내용', '',
             '기존 시설 연구 메인 `facility-presence-target-roi-control`의 공통 MobileNet 분석부와 시설 출력을 정확히 복사해 시작했다. 벨트 출력만 기존 벨트 모델에서 복사했고 이물질 출력은 새로 초기화했다. **공통 분석부 하나·체크포인트 하나·점검 기능3개**다. 학습 교사는 기존 메인 보존 손실에만 사용하며 배포 추론에 포함하지 않는다.', '',
             f"- 실제 학습: **{t['epochs']}epoch·{t['optimizer_updates']:,}회 가중치 업데이트·{t['elapsed_training_and_CAL_minutes']:.2f}분**. 사전 점검5회와 기준 보정은 이 횟수에서 제외했다.",
             '- 실제 예약된 고유 TRAIN record: 시설9,305 + 벨트1,397 + 이물질4,558 = **15,260개**. CODEBRIM 출판자 patch도 포함되므로 모두 독립 전체 사진이라는 뜻은 아니다.',
             '- 20회 반복 샘플링은51,200 draw이다. 반복 사용 횟수를 새 사진 수로 세지 않았다.',
             '- 출처: DACL10k, DamSegment, CODEBRIM, BeltCrack, Mendeley 이물질 자료5개. PCB와 IDEA ND 보충자료는 이번 학습에 넣지 않았다.',
             '- 시설7종과 DACL 보조19종, 벨트 작성자 손상 mask, 이물질 사진 유무를 사용했다. 확인하지 않은 항목은 unknown으로 유지했다.',
             '- 최종 epoch20을 평가 전에 고정했다. 이물질 기준은 별도 CAL320장으로만 정했고 TEST는 학습·기준 보정에 사용하지 않았다.',
             f"- 기존 메인 SHA256: `{t['initial_main_sha256']}`.",
             f'- 최종 모델 SHA256: `{EXPECTED}`.', '',
             '## 시설 · 같은 알려진 검증 사진과 기존 기준', '',
             '시설 검증은 DACL710장·DamSegment424장·CODEBRIM611장이다. 균열 기준0.28985977, 박락0.46490818, 나머지5항목0.5를 이전 모델과 통합 모델에 동일하게 적용했다. 추가5항목의0.5는 최적화된 기준이라는 뜻이 아니다. unknown 항목은 분모에서 제외했다.', '',
             '| 출처 | 항목 | 이전 오탐 | 통합 오탐 | 이전 미탐 | 통합 미탐 |',
             '|---|---|---:|---:|---:|---:|']
    for domain, rows in r['facility_fixed_old_thresholds'].items():
        for name, values in rows.items():
            a, b = values['baseline'], values['unified']
            lines.append(f'| {domain} | {names[name]} | {ratecell(a,"fpr")} | {ratecell(b,"fpr")} | {ratecell(a,"fnr")} | {ratecell(b,"fnr")} |')
    lines.extend(['', '**균열·박락의12개 오탐·미탐률 중 최댓값은22.28%→25.13%로 악화했다.** 모든 시설 항목의 성능을 보존하거나 전체5% 미만을 달성한 모델로 해석할 수 없다.', '',
                  '## 벨트 · 같은 알려진 작성자 VAL353장', '',
                  '| 항목 | 이전 벨트 별도 모델 | 통합 메인 |', '|---|---:|---:|',
                  f"| 손상 사진 미탐 | {ratecell(r['belt_baseline_val353']['photo_rates'],'fnr')} | {ratecell(r['belt_val353']['photo_rates'],'fnr')} |",
                  '| 정상 사진 오탐 | 측정 불가·정상 정답0장 | 측정 불가·정상 정답0장 |',
                  f"| 손상 mask pixel IoU | {percent(r['belt_baseline_val353']['pixel_iou'])} | {percent(r['belt_val353']['pixel_iou'])} |", '',
                  '손상 사진을 놓치는 비율은 줄었지만 **위치 겹침 IoU는 낮아졌다**. 이물질 없음 사진을 정상 벨트 정답으로 바꾸지 않았다. 정상 벨트의 사진 오탐률은0%가 아니라 미측정이다. 벨트 heatmap은 점검자 확인 자료이며 확정 손상 위치가 아니다.', '',
                  '## 이물질 · 이전 보정 별도 모델과 비교', '',
                  '| 알려진 시험 묶음 | 이전 오탐 | 통합 오탐 | 이전 미탐 | 통합 미탐 |', '|---|---:|---:|---:|---:|'])
    for label, a, b in [('448장', old['known_test'], r['foreign']['common448']),
                         ('783장', old['fresh']['expanded_calibrated'], r['foreign']['known783'])]:
        lines.append(f'| {label} | {ratecell(a,"fpr")} | {ratecell(b,"fpr")} | {ratecell(a,"fnr")} | {ratecell(b,"fnr")} |')
    lines.extend(['', '두 묶음은 이미 이전 실험에서 평가한 알려진 사진이며 같은4개 TEST 촬영 기록에서 나온 프레임이다. 이번 회차의 새로운 산업현장 검증이 아니다. 783장도 현재 시점에서 새 자료라고 부르지 않는다. 통합 수치는 이 묶음에서5% 미만이지만 이전 보정 별도 모델보다 악화했다. 프레임을 독립 현장·독립 물체 표본으로 해석하지 않는다.', '',
                  '## 결론과 앱 전달 범위', '',
                  '한 모델로5개 출처를 학습하는 구조와3개 기능 연결은 구현했다. 성능은 항목별로 개선과 악화가 함께 있다. **기존 앱 기본 모델은 교체하지 않았고 연구·연결 비교 후보로 제공한다. 산업현장 모든 항목의 오탐·미탐5% 미만은 달성하지 않았다.** 사진으로 모터·롤러 내부 고장이나 가동 안전을 판정하지 않는다.', '',
                  '비교에는 구조·초기 가중치·해상도·표본 배분 차이가 포함된다. 데이터 추가만의 효과를 입증한 실험은 아니다. 다음 성능 실험은 기존 시설 보존과 벨트 위치 악화를 함께 해결하도록 TRAIN에서 손실 비중·공통 분석부 업데이트 범위를 대조하고, 최종 검증용 현장 자료와 정상 벨트 정답을 별도로 확보하는 방향이 적절하다.', '',
                  '위험 등급·조치 문장·보고서 요약은 현재 규칙 템플릿이다. 학습한 위험 심각도 모델이나 생성형 보고서 모델이라고 표시하지 않는다. 조치 전후 비교는 같은 점검 후보의 변화이며 별도 학습한 수리 완료 판정 모델이 아니다.', '',
                  '![실제 비교 통계](unified-main-comparison.png)', '',
                  '- [실제 학습 기록](unified-main-training.json)',
                  '- [전체 계수와 분모](unified-main-results.json)',
                  '- [모델·입력 보존 확인](unified-main-verification.json)',
                  '- [실제 Qt/C++9개 테스트](unified-main-qt-tests-verified.json)',
                  '- [출처·사용 조건](UNIFIED_MAIN_ATTRIBUTION_KO.md)',
                  '- [역할4 연결 설명](../../docs/UNIFIED_MAIN_HANDOFF_KO.md)', ''])
    write(REPORTS / 'UNIFIED_MAIN_RESULTS_KO.md', '\n'.join(lines))
    write(REPORTS / 'UNIFIED_MAIN_ATTRIBUTION_KO.md', '''# 통합 메인 · 자료 출처와 사용 조건

이 전달 파일은 비상업 학술 연구·교육용 모델이다. 모델에는 다음5출처의 학습 이력이 있다. 원본 데이터 사진·주석은 ZIP에 포함하지 않는다. 기존 교사 모델도 배포하지 않는다.

| 출처 | 원본·조건 | 이번 사용·변경 |
|---|---|---|
| DACL10k | [원본 프로젝트](https://arxiv.org/abs/2309.00460), CC BY-NC4.0. [배포 LICENSE 원문](LICENSE_DACL.md) | 시설 사진·격자 정답7종, 출판자19종 보조 태그. 다른 시설의 안전 정답으로 확대하지 않음 |
| DamSegment | [Mendeley V1](https://data.mendeley.com/datasets/z5z6gtt5t4/1), CC BY4.0 | 균열·박락과 출판자 intact 정의. 분류 Non-Crack의 두 항목 음성은 저자 정의에서 변환한 의미이며 별도 native spalling 음성 mask라고 표현하지 않음 |
| CODEBRIM | [원본 프로젝트](https://github.com/MrtnMndt/meta-learning-CODEBRIM), [원문 사용 조건](LICENSE_CODEBRIM.md) | 작성자 patch·5종 정답. 젖은 표면·표면 공동은 unknown. 조건5의 데이터 자체를 포함하지 않는 학습 모델 공유 범위에 따라 비상업 연구·교육용으로 전달 |
| BeltCrack | Huang Jianghong; Ji Luping; Ma Xin; Ye Mao. [ScienceDB DOI10.57760/sciencedb.31181](https://doi.org/10.57760/sciencedb.31181), [CC BY-SA4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 작성자 embedded 사진과 crack polygon/rectangle, 640 letterbox·padding 제외. 사진 없는 통계 그림만 전달. 데이터 라이선스는 작성자 코드의 Apache2와 별도 |
| 컨베이어 이물질 | Frederico Martins Sousa; Thiago Eustaquio Alves de Oliveira; Bruno Nazário Coelho; Saul Emanuel Delabrida Silva. [Mendeley V1](https://data.mendeley.com/datasets/s25x2bnshz/1), DOI10.17632/s25x2bnshz.1, [CC BY4.0](https://creativecommons.org/licenses/by/4.0/) | normal-op/anomaly 사진 정답. 철광석 흐름의 이물질 부재이며 벨트 기계 정상 정답이 아님 |

CODEBRIM 인용: Martin Mundt, Sagnik Majumder, Sreenivas Murali, Panagiotis Panetsos, Visvanathan Ramesh. *Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with the COncrete DEfect BRidge IMage Dataset*. CVPR2019.

이번 학습은 출판자 정답을 사용하며 새 AI 정답을 생성하지 않았다. 모르는 항목을 음성으로 만들지 않는다. 학습 파일에는 이전 시설 메인과 벨트 출력의 가중치 이력도 포함되며 동일 자료 조건을 유지한다. IDEA ND 보충자료와 PCB는 이번 메인에 넣지 않았다. 모델·출처·변경 설명과 원문 조건을 함께 유지한다.
''')
    shutil.copy2(ROOT/'data/dacl10k/dacl10k_v2_devphase/LICENSE.md', REPORTS/'LICENSE_DACL.md')
    shutil.copy2(ROOT/'data/codebrim/license.md', REPORTS/'LICENSE_CODEBRIM.md')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot = REPORTS/'unified-main-comparison.png'
    assert not plot.exists()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    comparisons = [('Facility max crack/spall error', 22.2797927461, 25.1295336788, axes[0]),
                   ('Belt damaged-photo FNR', 100*r['belt_baseline_val353']['photo_rates']['fnr'], 100*r['belt_val353']['photo_rates']['fnr'], axes[1]),
                   ('Belt mask IoU (higher is better)', 100*r['belt_baseline_val353']['pixel_iou'], 100*r['belt_val353']['pixel_iou'], axes[2])]
    for title, a, b, axis in comparisons:
        axis.bar(['Previous', 'Unified main'], [a,b], color=['#597084','#287bbb'])
        axis.set_ylim(0, max(a,b)*1.3); axis.set_ylabel('%'); axis.set_title(title, fontsize=11)
        for x,y in enumerate([a,b]): axis.text(x,y+.7,f'{y:.2f}%',ha='center')
        axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Known reused benchmarks: mixed improvements and regressions', fontsize=14)
    fig.text(.5,.02,'Belt photo FPR: unavailable (0 normal labels). No independent factory validation.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.92));fig.savefig(plot,dpi=150);plt.close(fig)
    print('Actual results, attribution and comparison plot written', flush=True)


def local_links(path):
    text = path.read_text(encoding='utf-8')
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
        target = target.strip().strip('<>')
        if re.match(r'^[a-zA-Z]+:',target) or target.startswith('#'):
            continue
        yield (path.parent / unquote(target.split('#')[0])).resolve()


def package():
    assert not STAGE.exists() and not ARCHIVE.exists()
    assert sha(RUN/'final.pt') == EXPECTED
    STAGE.mkdir();OUTPUTS.mkdir(exist_ok=True)
    shutil.copy2(RUN/'final.pt', STAGE/'model.pt')
    t=read(REPORTS/'unified-main-training.json')
    write(STAGE/'model.json', {'path':'model.pt','sha256':EXPECTED,'neuralModelCount':1,'sharedBackboneCount':1,
                             'inspectionTasks':t['tasks'],'thresholds':t['thresholds'],'epochs':20,'optimizer_updates':6400,
                             'app_default_promoted':False,'fieldPerformanceValidated':False})
    write(STAGE/'requirements-api.txt','fastapi==0.142.1\nuvicorn==0.54.0\npydantic==2.13.5\npillow==12.3.0\nnumpy==2.5.2\n')
    write(STAGE/'safelog_ai/__init__.py','')
    for module in MODULES: shutil.copy2(ROOT/'safelog_ai'/module,STAGE/'safelog_ai'/module)
    write(STAGE/'README_KO.md','''# SafeLog 통합 메인 · 역할4 전달

기존 시설 메인을 확장한 모델 하나가 시설·벨트 손상·이물질 점검을 지원한다. 실제20epoch·6,400회 업데이트를 완료했다. 공통 분석부 하나이며 점검 항목을 요청에 명시한다.

- [실행·Qt/C++ 앱 연결](safelog-cpp/docs/UNIFIED_MAIN_HANDOFF_KO.md)
- [실제 성능과 개선·악화](safelog-cpp/ai-training/reports/UNIFIED_MAIN_RESULTS_KO.md)
- [자료 출처·비상업 연구·교육 사용 조건](safelog-cpp/ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md)

기존 앱 기본 모델은 교체하지 않았다. 벨트 미탐 감소와 시설·위치 성능 악화가 함께 있다. API 서버용 연구 후보·C++ 연결 코드이며 완성 Android APK가 아니다. 같은 ZIP을 사용한 실제 CPU/HTTP와 Qt/C++ 연결 시험 기록은 저장소의 unified-main-standalone-v1.json으로 제공한다.
''')
    queue=[REPO/'safelog-cpp/docs/UNIFIED_MAIN_HANDOFF_KO.md']
    queue += [p for p in REPORTS.iterdir() if p.name.startswith('unified-main-') and p.suffix in ('.json','.png')]
    queue += [REPORTS/'UNIFIED_MAIN_RESULTS_KO.md',REPORTS/'UNIFIED_MAIN_ATTRIBUTION_KO.md']
    queue += [ROOT/'tests/test_unified_main_server.py',ROOT/'tests/test_unified_main.py', ROOT/'scripts/build_unified_qt_probe.py']
    queue += [REPO/'safelog-cpp/apps/mobile-qt/tests/http_ai_safety_analyzer_tests.cpp',
              REPO/'safelog-cpp/apps/mobile-qt/tools/unified_main_probe.cpp',
              REPO/'safelog-cpp/src/ai/include/safelog/ai/ai_analyzer.hpp']
    seen=set()
    while queue:
        source=queue.pop().resolve()
        if source in seen:continue
        assert source.is_relative_to(REPO) and source.is_file(), f'Missing local artifact {source}'
        seen.add(source);target=STAGE/source.relative_to(REPO)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if source.suffix=='.md':queue.extend(local_links(source))
    for doc in STAGE.rglob('*.md'):
        for link in local_links(doc):assert link.is_relative_to(STAGE) and link.is_file(),f'Broken package link {link}'
    members={p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()}
    write(STAGE/'package-sha256.json',members)
    with zipfile.ZipFile(ARCHIVE,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for source in sorted(STAGE.rglob('*')):
            if source.is_file():archive.write(source,source.relative_to(STAGE).as_posix())
    write(REPORTS/'unified-main-release-v1.json',{'status':'packaged_pending_isolated_http_test','archive':str(ARCHIVE),
        'sha256':sha(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'model_sha256':EXPECTED,'checkpoint_count':1,
        'source_dataset_images_included':False,'member_count':len(members)+1,'local_markdown_links_closed':True})
    print(json.dumps({'zip':str(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'sha256':sha(ARCHIVE)}),flush=True)


def http(path, body=None):
    request=Request('http://127.0.0.1:8775'+path,data=None if body is None else json.dumps(body).encode(),
                    headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=30) as response:return json.load(response)


def smoke():
    assert not ISOLATED.exists()
    with socket.socket() as port:assert port.connect_ex(('127.0.0.1',8775))!=0,'8775 already in use; do not stop an unrelated server'
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert sum(p.endswith('.pt') for p in archive.namelist())==1
        for name in archive.namelist():assert (ISOLATED/name).resolve().is_relative_to(ISOLATED)
        archive.extractall(ISOLATED)
    assert sha(ISOLATED/'model.pt')==EXPECTED
    manifest=read(ROOT/'data/unified-main-v1/manifest.json')
    facility=next(r for r in manifest['fit']['facility_surface'] if r['targets'][0]==1)
    foreign=next(r for r in manifest['fit']['conveyor_foreign'] if r['label']==1)
    belt_path=ROOT/'runs/belt-author-embedded-sample.jpg'
    belt=next(r for r in manifest['fit']['belt_damage'] if r['embedded_image_sha256']==sha(belt_path))
    photos={'facility_surface':ROOT/facility['image'],'belt_damage':belt_path,'conveyor_foreign':ROOT/foreign['image']}
    rows={'facility_surface':facility,'belt_damage':belt,'conveyor_foreign':foreign}
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8'
    env['PATH']='C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH','');env['QT_QPA_PLATFORM']='offscreen'
    # Verify the exact shipped preprocessing against the frozen TRAIN loader.
    import importlib.util
    import numpy as np
    from PIL import Image
    import torch
    spec=importlib.util.spec_from_file_location('frozen_data',ROOT/'scripts/unified_main_data.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    server_spec=importlib.util.spec_from_file_location('safelog_ai.shipped_main_server',ISOLATED/'safelog_ai/unified_main_server.py')
    shipped=importlib.util.module_from_spec(server_spec);server_spec.loader.exec_module(shipped)
    decode,tensor=shipped.decode,shipped.tensor
    parity={}
    for task,path in photos.items():
        rgb,_=decode(base64.b64encode(path.read_bytes()).decode());values,_=tensor(rgb,task)
        original=helper.normalize_batch(helper.photo(rows[task]).unsqueeze(0),'cpu')
        assert torch.equal(values,original),f'{task} preprocessing changed'
        parity[task]={'identical_tensor_to_frozen_TRAIN':True,'id':rows[task]['id'],
                      'native_size':list(rgb.size),'input_sha256':sha(path)}
    log=(ISOLATED/'server.log').open('x',encoding='utf-8')
    process=subprocess.Popen([sys.executable,'-m','safelog_ai.unified_main_server','--checkpoint','model.pt',
                              '--sha256',EXPECTED,'--port','8775'],cwd=ISOLATED,env=env,stdout=log,stderr=log,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    observed=[];cpp=[];stopped=False
    try:
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Packaged server exited: '+(ISOLATED/'server.log').read_text())
            try:
                health=http('/health');break
            except (OSError,ValueError):time.sleep(.25)
        else:raise RuntimeError('Packaged server readiness timeout')
        assert health['ready'] and health['neuralModelCount']==health['sharedBackboneCount']==1
        assert health['modelSha256']==EXPECTED
        for task,path in photos.items():
            body={'inspectionTask':task,'image':base64.b64encode(path.read_bytes()).decode(),'promptVersion':'unified-release-test-v1'}
            reply=http('/v1/analyze-hazard',body)
            assert reply['modelSha256']==EXPECTED and reply['inspectionTask']==task
            assert reply['neuralModelCount']==reply['sharedBackboneCount']==1
            assert reply['humanConfirmationRequired'] and not reply['otherTasksEvaluated']
            assert not reply['fieldPerformanceValidated'] and not reply['beltPhotoFprValidated']
            assert not reply['scoreIsCalibratedProbability'] and reply['mechanicalFailureState']=='not_assessed'
            assert reply['inputSha256']==sha(path)
            if task=='belt_damage':
                with Image.open(io.BytesIO(base64.b64decode(reply['heatmapPngBase64']))) as heat:
                    assert list(heat.size)==parity[task]['native_size']
            else:assert reply['heatmapPngBase64'] is None
            saved=dict(reply);saved.pop('heatmapPngBase64');observed.append(saved)
            proof=ISOLATED/(task+'-cpp.json')
            result=subprocess.run([str(ROOT/'runs/unified-main-qt-probe/build/unified_probe.exe'),task,str(path),EXPECTED,str(proof)],
                                  cwd=ISOLATED,env=env,timeout=45,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            assert result.returncode==0 and proof.is_file(),f'Actual Qt probe failed: exit {result.returncode}'
            item=read(proof);assert item['status']=='passed' and item['operations']==3 and item['modelSha256']==EXPECTED
            cpp.append(item)
            print(f'{task}: isolated CPU HTTP and actual Qt/C++3 operations passed',flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        stopped=process.poll() is not None;log.close()
    write(REPORTS/'unified-main-standalone-v1.json',{'status':'passed','isolated_zip':str(ISOLATED),
        'runtime_pythonpath_override_removed':True,'checkpoint_count':1,'neural_model_count':1,'shared_backbone_count':1,
        'model_sha256':EXPECTED,'preprocessing_parity':parity,'health':health,'actual_python_http_hazards':observed,
        'actual_Qt_CPP_probes':cpp,'actual_Qt_CPP_http_operations':9,'all_probe_photos_are_TRAIN':True,
        'normal_belt_fpr_validated':False,'independent_factory_validation':False,'owned_server_stopped':stopped,
        'fresh_environment_pip_install_tested':False})


def audit():
    verification=read(REPORTS/'unified-main-verification.json')
    standalone=read(REPORTS/'unified-main-standalone-v1.json')
    protocol=read(REPORTS/'unified-main-protocol.json')
    assert verification['status']==standalone['status']=='passed'
    assert sha(RUN/'final.pt')==EXPECTED
    for relative,digest in {**protocol['source_sha256'],**protocol['legacy_source_sha256']}.items():
        assert sha(ROOT/relative)==digest, f'Frozen source changed {relative}'
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        members=json.loads(archive.read('package-sha256.json'))
        assert all(hashlib.sha256(archive.read(name)).hexdigest()==digest for name,digest in members.items())
        assert sum(name.endswith('.pt') for name in archive.namelist())==1
        assert not any('/data/' in name for name in archive.namelist())
    for doc in ISOLATED.rglob('*.md'):
        for link in local_links(doc):assert link.is_relative_to(ISOLATED) and link.is_file()
    sources={str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in [ROOT/'safelog_ai/unified_main_server.py',
        REPO/'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp',
        REPO/'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp',
        REPO/'safelog-cpp/apps/mobile-qt/tools/unified_main_probe.cpp']}
    write(REPORTS/'unified-main-final-audit.json',{'status':'passed','archive_sha256':sha(ARCHIVE),
        'archive_bytes':ARCHIVE.stat().st_size,'zip_crc_and_member_sha256_verified':True,
        'local_markdown_links_closed_after_extraction':True,'one_checkpoint_only':True,
        'frozen_current_and_legacy_sources_preserved':len(protocol['source_sha256'])+len(protocol['legacy_source_sha256']),
        'training_inputs_and_previous_artifacts_preserved':verification,'isolated_CPU_HTTP_and_actual_Qt_CPP_passed':True,
        'actual_cpp_http_operations':standalone['actual_Qt_CPP_http_operations'],'release_source_sha256':sources,
        'all_tasks_5percent_goal_achieved':False,'app_default_promoted':False})
    print('ZIP CRC/SHA, one model, local docs, frozen sources and actual Qt HTTP audit passed',flush=True)


def revise_docs():
    """Keep the tested v1 ZIP and issue a docs-only correction with identical runtime."""
    revised=OUTPUTS/(NAME.replace('-v1','-v1.1')+'.zip')
    extracted=ROOT/'runs/unified-main-isolated-zip-v1.1'
    assert not revised.exists() and not extracted.exists()
    old_audit=read(REPORTS/'unified-main-final-audit.json')
    assert sha(ARCHIVE)==old_audit['archive_sha256']
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        contents={name:archive.read(name) for name in archive.namelist()}
    changed=[]
    for name in ['safelog-cpp/docs/UNIFIED_MAIN_HANDOFF_KO.md',
                 'safelog-cpp/ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md']:
        data=(REPO/name).read_bytes()
        if data!=contents[name]:changed.append(name)
        contents[name]=data
    old_members=json.loads(contents['package-sha256.json'])
    members={name:hashlib.sha256(data).hexdigest() for name,data in contents.items() if name!='package-sha256.json'}
    assert changed and all(name.endswith('.md') for name in changed)
    assert all(members[name]==digest for name,digest in old_members.items() if name not in changed)
    contents['package-sha256.json']=(json.dumps(members,ensure_ascii=False,indent=2)+'\n').encode()
    with zipfile.ZipFile(revised,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for name,data in sorted(contents.items()):archive.writestr(name,data)
    with zipfile.ZipFile(revised) as archive:
        assert archive.testzip() is None
        assert sum(name.endswith('.pt') for name in archive.namelist())==1
        for name in archive.namelist():
            assert (extracted/name).resolve().is_relative_to(extracted)
            if name!='package-sha256.json':assert hashlib.sha256(archive.read(name)).hexdigest()==members[name]
        archive.extractall(extracted)
    for doc in extracted.rglob('*.md'):
        for link in local_links(doc):assert link.is_relative_to(extracted) and link.is_file()
    current={**old_audit,'archive_sha256':sha(revised),'archive_bytes':revised.stat().st_size,
             'archive':str(revised),'docs_only_revision':True,'changed_documents':changed,
             'runtime_model_and_CPP_byte_identical_to_tested_v1':True,
             'actual_runtime_test_evidence':'unified-main-standalone-v1.json',
             'previous_archive_preserved_sha256':old_audit['archive_sha256']}
    write(REPORTS/'unified-main-final-audit-v1.1.json',current)
    write(REPORTS/'unified-main-release-v1.1.json',{'status':'passed','archive':str(revised),
         'sha256':sha(revised),'bytes':revised.stat().st_size,'model_sha256':EXPECTED,
         'checkpoint_count':1,'docs_only_revision':True,'source_dataset_images_included':False,
         'runtime_model_and_CPP_byte_identical_to_tested_v1':True,
         'actual_runtime_test_evidence':'unified-main-standalone-v1.json'})
    print(json.dumps({'status':'passed','archive':str(revised),'sha256':sha(revised),'bytes':revised.stat().st_size}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['report','package','smoke','audit','revise-docs'])
    globals()[parser.parse_args().mode.replace('-','_')]()
