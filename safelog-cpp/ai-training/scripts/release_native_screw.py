"""Measured native screw improvement, one-checkpoint role4 ZIP and CPU/C++ proof."""
import argparse
import base64
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
from urllib.error import HTTPError
import zipfile
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.study_native_screw_budget import RUN, REPORTS, validate, START, EXPECTED, read
from scripts.fetch_equipment_expansion import sha
from scripts.release_unified_main import MODULES, local_links
from scripts.release_metal_calibration import write, http, torch_state
REPO = ROOT.parents[1]; OUTPUTS = REPO.parents[1]/'outputs'
NAME = 'unified-equipment-native-screw-role4-handoff-2026-10-15-v7'
STAGE = RUN/'handoff-stage'; ISOLATED = RUN/'isolated-zip'; ARCHIVE = OUTPUTS/(NAME+'.zip')


def docs():
    validate(); choice = read(REPORTS/'equipment-native-screw-budget-v7-selection.json'); verification = read(REPORTS/'equipment-native-screw-budget-v7-verification.json')
    assert choice['selected'] == 'normalCAL15' and verification['status'] == 'passed'
    result = read(REPORTS/'equipment-native-screw-budget-v7-validation.json')['candidates']['normalCAL15']; data = read(REPORTS/'bsclass-v1-data-audit.json')
    guide = '''# 역할4 · 원본 세부 정보를 보존한 나사 AI 연결

## 현재 측정 결과

같은 공개 평가160장에서 나사 사진 오탐2/41=4.88%, 미탐15/119=12.61%를 측정했다. 기존 오탐19.51%·미탐85.71%에서 개선했다. 신규11유형 중5개가 사진20% 이내이며 그중4개가15% 이내다. 케이블·너트·금속판·부품 표면의 이전 결과를 유지했다.

**전체 나사 사진의 합산 수치다. 나사산 옆면 손상은 미탐13/23=56.52%가 남아 있다.** 모든 손상 종류가15% 이내라는 뜻으로 표시하면 안 된다. 알려진 공개 평가자료로 반복 개발한 연구 시제품이며 새로운 현장 검증은 남아 있다.

## CPU 서버 실행

새 ZIP을 새 폴더에 푼다. Python3.12와 기존 의존성에 이번 원본 처리용OpenCV가 추가된다. 아래는 현재 검사 환경의 버전이며 새 pip 설치 자체를 이번 검증에서 실행한 것은 아니다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$nativePrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.native_component_server --checkpoint $nativePrototype.path --sha256 $nativePrototype.sha256 --evaluation evaluation.json --port 8775
```

이번 서버와model.json을 함께 사용한다. 이전 서버는 새 모델 구조를 읽을 수 없다. 서버는127.0.0.1에서 실행되며 스마트폰 주소·HTTPS·인증은 기존 팀 앱 구조에 연결한다. 이번 전달물은 APK가 아니다.

## C++ 연결

기존 C++ 어댑터에서 유형ID는그대로다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="ad_screw"
```

나사 하나가 선명하게 나온 원본 사진을 보낸다. 서버가사진만으로배경과주요축을찾아방향을맞추고768×256비율을유지해비교한다. 작은256×256정사각형으로미리줄여보내지않는다. 판별사진과원래위치출력의보존을위해같은공통신경망을두번사용한다. 모델1개·공통분석부1개이며나사응답의backbonePassesPerPhoto=2다.

어두운부분을분리하거나방향을정하기어려우면422를반환한다. 화면은다시촬영/수동검수로이어가야한다. 이경우를정상또는고장미검출로저장하지않는다.

## 화면·DB 필드

| 필드 | 의미 |
|---|---|
| candidateDetected / classResults | 사진의 나사 외관 이상 후보 |
| heatmapPngBase64 | 기존 위치 출력. 이번에도 동일하며 위치 정확도는 낮음 |
| referenceHeatmapPngBase64 | 원본 좌표로 되돌린 정상 특징 차이 참고 이미지 |
| referenceValidRegionMaskPngBase64 | 참고 이미지가 비교한 영역. 바깥은 정상으로 판단한 영역이 아님 |
| nativeViewImagePngBase64 / nativeReferenceHeatmapPngBase64 | 방향 보정 사진과 같은 좌표의 차이 이미지 |
| nativeViewGeometry | 원본과 보정 사진 사이의 입력 기반 변환 |
| defectGroupBenchmark / allDefectGroupsWithin20 | 손상 종류별 한계. 나사산 옆면 미탐56.52%, 전체 종류20% 통과false |
| benchmarkStageScope | pooled_photo_category_on_known_TEST. 전체 현장·모든 결함의 보장이 아님 |
| modelSha256 / inferencePolicySha256 | 모델과 나사 경고 정책을 함께 저장 |
| normalCALAlpha / normalCALRank | 정상CAL24장 기준.15/rank22. 실제 오탐률 보장치가 아님 |

referenceHeatmapIsEvaluatedLocalization=false이며정상특징차이이미지는정답으로검증한손상분할결과가아니다. 기존위치출력의나사pixel미탐97.20%는그대로다. 사진미탐12.61%와구분한다. confidence/raw score도실제고장확률로표시하지않는다.

최종판정은점검자가한다. 내부기계고장은mechanicalFailureState=not_assessed이고위험등급·조치·요약은기존규칙템플릿이다. 새볼스크루정상/손상자료는준비했지만이번모델에학습한자료가아니다.

GET /v1/inference-policy의nativeComponents.ad_screw와GET /v1/equipment-categories에서정책·합산사진상태·손상종류별한계를확인한다. 나머지요청은기존계약을유지한다.

- [실제 비교 결과](../ai-training/reports/NATIVE_SCREW_RESULTS_KO.md)
- [기존 전체 연결 안내](EQUIPMENT_REFERENCE_HANDOFF_KO.md)
- [기존 학습자료 출처](../ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md)
- [이번 ZIP 연결 검사 기록](https://github.com/jungsahun43-commits/robo/blob/feature/ai-engine/safelog-cpp/ai-training/reports/equipment-native-screw-budget-v7-release.json)
'''
    write(REPO/'safelog-cpp/docs/NATIVE_SCREW_HANDOFF_KO.md', guide)
    def pct(x): return f'{100*x:.2f}%'
    lines = ['# 원본 정보·나사 방향 보정으로 사진 미탐 개선', '', '기준일: 2026-10-15', '',
        '## 선택 결과', '', '| 항목 | 기존 모델 | 선택 후보 |', '|---|---:|---:|',
        '| 사진 오탐 | 8/41 = 19.51% | 2/41 = 4.88% |', '| 사진 미탐 | 102/119 = 85.71% | 15/119 = 12.61% |',
        '| 사진 평가 | 정상41·결함119 | 같은160장 재사용 |', '| 나사 위치pixel 미탐 | 97.20% | 97.20% 유지 |', '',
        '나사 전체 사진에서 오탐·미탐15% 목표를 충족했다. 사진미탐87장을줄였고정상오경고6장을줄였다. 신규11유형중5개가20%이내,그중4개가15%이내다. 기존케이블·너트·금속판·부품표면결과는같다.', '',
        '**나사산 옆면 손상은 미탐13/23=56.52%가 남는다.** 종류별결과를합산사진수치와함께공개한다. 모든나사손상종류와새산업현장에대한15%성능을보장하지않는다.', '',
        '## 손상 종류별 결과', '', '| 종류 | 미탐/결함 사진 | 미탐률 |', '|---|---:|---:|']
    labels = {'manipulated_front': '앞부분 변형', 'scratch_head': '머리 긁힘', 'scratch_neck': '목 부분 긁힘', 'thread_side': '나사산 옆면', 'thread_top': '나사산 끝부분'}
    for group, value in result['by_defect'].items(): lines.append(f"| {labels[group]} | {value['fn']}/{value['defect_photos']} | {pct(value['fnr'])} |")
    lines += ['', '## 변경한 내용', '',
        '- 입력만으로Otsu배경분리·주축PCA·머리가넓은쪽정렬을수행했다. 마스크정답이나이상예측으로위치/방향을고르지않는다. 원본1024×1024에서나사주변을768×256으로등방변환한다.',
        '- 정상FIT96장의기존공통신경망low40/high960특징73,728개를같은모델파일의FP16참조버퍼에추가했다. 사진점수는정확한최근접cosine거리상위1%평균이다.',
        '- 기존256입력위치출력과새방향보정입력을같은신경망으로각1회처리한다. 한모델·한공통분석부·나사사진당2회다. 기존신경망가중치와케이블/너트참조는모두같다.',
        '- 이번v6/v7의추가신경망optimizer업데이트는0회다. 정상특징적합과입력처리·정상CAL기준개선이다. 앞선v5의40epoch·2560번학습은미채택상태로보존했다.',
        '- 입력품질검사976장중실패0장. 실사용에서객체방향/배경분리가불분명하면422와수동검수요청으로처리한다.', '',
        '## 평가 전에 정한 비교', '', '| 후보 | 사진 오탐 | 사진 미탐 | 선택 |', '|---|---:|---:|---|']
    for variant in ('screw_hilo', 'screw_fine'):
        for budget, value in read(REPORTS/f'equipment-native-components-v6-{variant}-validation.json')['candidates'].items():
            lines.append(f"| {variant}/{budget} | {pct(value['photo_rates']['fpr'])} | {pct(value['photo_rates']['fnr'])} | 제외 |")
    for budget, value in read(REPORTS/'equipment-native-screw-budget-v7-validation.json')['candidates'].items():
        lines.append(f"| 같은aligned_hilo/{budget} | {pct(value['photo_rates']['fpr'])} | {pct(value['photo_rates']['fnr'])} | {'선택' if budget=='normalCAL15' else '오탐 초과로 제외'} |")
    lines += ['', 'v6에서는4입력/특징후보와각CAL10/CAL05를먼저고정했다. 나사hilo가오탐0%·미탐21.01%였지만20%에못미쳐제외했다. 그결과를본뒤v7에서나사hilo에한해CAL15/CAL20두정상경고기준을추가로사전등록했다.',
        '선택기준은정상CAL24장의22번째값이다. TEST점수분위수로임계값을고르거나TEST결함을학습에넣지않았다. 다만이미본공개TEST로후보를비교하고다음회차를설계한개발과정이므로독립검증결과는아니다.', '',
        '## 연결부와 새 실제 손상 자료', '',
        '전선연결부는원래2:1비율을1024×512로보존하고지역특징과전체배치를시험했다. 결합후보CAL10은오탐19.33%·미탐39.38%로20%동시목표미달이어서반영하지않았다.',
        f"볼스크루분류자료{data['source_images']:,}장을공식경로에서받아정상/피팅손상라벨을확인했다. 회전본·정확중복·근접pHash를묶어학습/보정/예약평가를분리했고예약평가의모델예측은계산하지않았다. 이자료는일반체결나사사진과다른설비표면자료이며이번나사모델에는사용하지않았다.",
        '[볼스크루분류자료원출처](https://publikationen.bibliothek.kit.edu/1000133819)는CCBY-NC4.0이고파일명N/P라벨과90도회전변형을설명한다. 검수중명목150×150과다른크기가확인되어원본을그대로보존하고실제크기를기록했다. 촬영원본프레임·물리적설비별그룹은검증하지못했으므로새현장독립평가로표시하지않는다.',
        '[BSData원자료](https://github.com/2Obe/BSData)도고정리비전에서받아선택한JPEG/JSON/license의Git체크섬을확인했다. 외부코드나저장모델을추출·실행하지않았다. 두자료모두아직학습결과를주장하지않는다.', '',
        '## 보존·연결', '', f"기존자료·모델·보고서{verification['protected_files']:,}개의SHA/mtime과기존Python소스{verification['legacy_sources']}개의SHA를보존했다. 기존신경망tensor와정상참조는정확히같고다른10유형의기준·출력도유지했다.",
        '기존위치heatmap은그대로라나사pixel미탐97.20%는별도보강이필요하다. 새참고heatmap은입력변환을역으로적용해원본좌표에표시하며비교영역mask를함께준다. 새참고heatmap의위치IoU를측정한분할결과는아니다.',
        '역할4전달은새모델파일하나와새CPU서버모듈을사용한다. 기존기본앱을자동배포한것은아니며실제ZIP·CPU·C++검사기록을확인한다.', '',
        '선택모델SHA: `'+choice['checkpoint_sha256']+'`', '', '![나사 사진 비교](equipment-native-screw-budget-v7-comparison.png)', '',
        '- [v6 사전 조건](equipment-native-components-v6-protocol.json)', '- [v6 비교·제외](equipment-native-components-v6-selection.json)',
        '- [v7 사전 조건](equipment-native-screw-budget-v7-protocol.json)', '- [v7 전체 결과](equipment-native-screw-budget-v7-validation.json)',
        '- [v7 선택](equipment-native-screw-budget-v7-selection.json)', '- [v7 보존 검증](equipment-native-screw-budget-v7-verification.json)',
        '- [새 라벨 자료 검수](bsclass-v1-data-audit.json)', '- [역할4 실행·화면 연결](../../docs/NATIVE_SCREW_HANDOFF_KO.md)',
        '- [앞선 실제 학습과 실패 결과](COMPONENT_DEVELOPMENT_RESULTS_KO.md)', '- [기존 학습자료 출처](EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md)', '']
    write(REPORTS/'NATIVE_SCREW_RESULTS_KO.md', '\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, label, values in [(axes[0], 'Photo FPR', [19.51219512195122, 4.878048780487805]), (axes[1], 'Photo FNR', [85.71428571428571, 12.605042016806722])]:
        bars = ax.bar(['Prior', 'Native pose + CAL15'], values, color=['#778793', '#397c65']); ax.set_title(label); ax.set_ylabel('%'); ax.set_ylim(0, 105)
        ax.axhline(15, color='#ac6336', linestyle='--')
        for bar in bars: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f'{bar.get_height():.2f}%', ha='center')
    fig.suptitle('Screw photo improvement; known TEST160 reused')
    fig.text(.5, .02, 'Thread-side FNR remains13/23=56.52%. Location output unchanged. No independent field validation.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, .93]); fig.savefig(REPORTS/'equipment-native-screw-budget-v7-comparison.png', dpi=160); plt.close(fig)
    header = '## 나사 원본 보존 · 사진15% 목표 보강 · 2026-10-15'
    block = '\n'.join([header, '', '나사 방향·원본 세부 정보를 보존해 공개 평가 사진 오탐4.88%·미탐12.61%를 측정했다. 정상FIT96장의 참조 특징을 추가했으며 이번 신경망 추가 업데이트0회다.',
        '신규11유형 중5개 사진20% 이내, 그중4개15% 이내. 나사산 옆면 미탐56.52%와 기존 위치pixel 미탐97.20%는 남아 있다. 같은 공개TEST를 재사용한 개발 결과다.', ''])
    for path, link in [(REPO/'README.md', 'safelog-cpp/ai-training/reports/NATIVE_SCREW_RESULTS_KO.md'), (ROOT/'README_KO.md', 'reports/NATIVE_SCREW_RESULTS_KO.md'),
        (REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md', '../ai-training/reports/NATIVE_SCREW_RESULTS_KO.md'), (REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md', '../ai-training/reports/NATIVE_SCREW_RESULTS_KO.md')]:
        current = path.read_bytes().decode('utf-8'); assert header not in current; nl = '\r\n' if '\r\n' in current else '\n'; title, body = current.split(nl+nl, 1)
        path.write_bytes((title+nl+nl+(block+'\n- [원본 보존 나사 결과]('+link+')\n').replace('\n', nl)+nl+body).encode('utf-8'))


def package():
    validate(); choice = read(REPORTS/'equipment-native-screw-budget-v7-selection.json'); assert choice['selected'] == 'normalCAL15'
    assert not STAGE.exists() and not ARCHIVE.exists(); STAGE.mkdir(); checkpoint = ROOT/choice['checkpoint_path']; assert sha(checkpoint) == choice['checkpoint_sha256']
    shutil.copy2(checkpoint, STAGE/'model.pt'); shutil.copy2(RUN/'evaluation.json', STAGE/'evaluation.json'); state = torch_state(checkpoint)
    write(STAGE/'model.json', {'path': 'model.pt', 'sha256': choice['checkpoint_sha256'], 'architecture': state['architecture'], 'checkpoint_count': 1, 'backbone_count': 1,
        'nativeComponents': ['ad_screw'], 'nativeBackbonePassesPerPhoto': 2, 'normalFITPhotosAdded': 96, 'nativeReferencePatches': 73728,
        'newNeuralUpdates': 0, 'normalCALCount': 24, 'normalCALAlpha': .15, 'normalCALRank': 22, 'stage20Categories': choice['stage20_categories'],
        'allDefectGroupsWithin20': False, 'knownThreadSideFNR': 13/23, 'newBSClassDataUsedForThisModel': False, 'researchPrototype': True,
        'adaptiveDevelopmentOnKnownTEST': True, 'fieldPerformanceValidated': False, 'appDefaultPromoted': False})
    write(STAGE/'safelog_ai/__init__.py', '')
    for name in (*MODULES, 'belt_tiles.py', 'unified_tiles_server.py', 'equipment_expansion.py', 'equipment_expansion_server.py', 'metal_reference.py', 'metal_calibration_server.py',
        'equipment_reference_photo.py', 'equipment_reference_server.py', 'equipment_reference_budget_server.py', 'component_reference.py', 'component_detail.py', 'native_component_views.py', 'native_component_server.py'):
        shutil.copy2(ROOT/'safelog_ai'/name, STAGE/'safelog_ai'/name)
    requirements = (ROOT/'runs/equipment-reference-budget-v2/handoff-stage/requirements-api.txt').read_text()+'opencv-python==5.0.0.93\n'
    write(STAGE/'requirements-api.txt', requirements)
    write(STAGE/'README_KO.md', '# SafeLog 원본 보존 나사 연구 시제품\n\n나사 사진 오탐4.88%·미탐12.61%. 같은 공개TEST 재사용 결과이며 나사산 옆면 미탐56.52%는 남아 있습니다.\n\n'
        '- [역할4 실행·C++·화면 연결](safelog-cpp/docs/NATIVE_SCREW_HANDOFF_KO.md)\n- [실제 비교·한계](safelog-cpp/ai-training/reports/NATIVE_SCREW_RESULTS_KO.md)\n')
    queue = [REPO/'safelog-cpp/docs/NATIVE_SCREW_HANDOFF_KO.md', REPORTS/'NATIVE_SCREW_RESULTS_KO.md',
        REPO/'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp', REPO/'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp']; seen = set()
    while queue:
        source = queue.pop().resolve()
        if source in seen: continue
        assert source.is_relative_to(REPO) and source.is_file(), str(source); seen.add(source)
        target = STAGE/source.relative_to(REPO); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        if source.suffix == '.md': queue.extend(local_links(source))
    for path in STAGE.rglob('*.md'):
        for link in local_links(path): assert link.is_relative_to(STAGE) and link.is_file(), str(link)
    write(STAGE/'package-sha256.json', {p.relative_to(STAGE).as_posix(): sha(p) for p in STAGE.rglob('*') if p.is_file()})
    with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for path in sorted(STAGE.rglob('*')):
            if path.is_file(): z.write(path, path.relative_to(STAGE).as_posix())
    print(json.dumps({'archive': str(ARCHIVE), 'sha256': sha(ARCHIVE), 'bytes': ARCHIVE.stat().st_size}), flush=True)


def smoke():
    assert not ISOLATED.exists()
    with socket.socket() as s: assert s.connect_ex(('127.0.0.1', 8775)) != 0, 'Preserve unrelated server'
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist()) == 1
        for name in z.namelist(): assert (ISOLATED/name).resolve().is_relative_to(ISOLATED.resolve())
        z.extractall(ISOLATED)
    meta = read(ISOLATED/'model.json'); digest = meta['sha256']; assert sha(ISOLATED/'model.pt') == digest
    original = read(ROOT/'data/unified-main-v1/manifest.json'); rows = read(ROOT/'data/equipment-expansion-v1/manifest.json')['rows']; result = read(REPORTS/'equipment-native-screw-budget-v7-validation.json')['candidates']['normalCAL15']
    photos = {'facility_surface': ROOT/original['fit']['facility_surface'][0]['image'], 'belt_damage': ROOT/'runs/belt-author-embedded-sample.jpg', 'conveyor_foreign': ROOT/original['fit']['conveyor_foreign'][0]['image']}
    photos.update({c: ROOT/next(r['image'] for r in rows if r['category'] == c and r['phase'] == 'fit') for c in ('ad_cable', 'ad_metal_nut', 'mpdd_metal_plate', 'ksdd2_surface')})
    positive = next(r for r in result['observations'] if r['label'] and r['candidate_detected'] and r['score']-result['photo_threshold'] > .005)
    photos['ad_screw'] = ROOT/next(r['image'] for r in rows if r['id'] == positive['id'])
    env = dict(os.environ); env.pop('PYTHONPATH', None); env['PYTHONIOENCODING'] = 'utf-8'; env['QT_QPA_PLATFORM'] = 'offscreen'
    env['PATH'] = 'C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH', '')
    log = (ISOLATED/'server.log').open('x', encoding='utf-8'); process = subprocess.Popen([sys.executable, '-m', 'safelog_ai.native_component_server', '--checkpoint', 'model.pt', '--sha256', digest, '--evaluation', 'evaluation.json', '--port', '8775'],
        cwd=ISOLATED, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    proofs = []; replies = []; timings = {}
    try:
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError((ISOLATED/'server.log').read_text())
            try: health = http('/health'); break
            except (OSError, ValueError): time.sleep(.25)
        else: raise RuntimeError('Packaged CPU server timed out')
        assert health['modelSha256'] == digest and health['neuralModelCount'] == health['sharedBackboneCount'] == 1
        policies = http('/v1/inference-policy'); native_policy = policies['nativeComponents']['ad_screw']; assert native_policy['normalCALRank'] == 22 and native_policy['backbonePassesPerPhoto'] == 2
        prior = read(REPORTS/'equipment-reference-budget-v2-release.json'); assert policies['referencePhoto'] == prior['reference_policies']
        assert policies['metalPlate']['sha256'] == read(REPORTS/'equipment-metal-reference-v1-release.json')['metal_policy_sha256']
        categories = http('/v1/equipment-categories')['categories']; assert len(categories) == 11 and sum(c['benchmarkStage']['ready_for_next_stage'] for c in categories) == 5
        screw = next(c for c in categories if c['id'] == 'ad_screw'); assert not screw['allDefectGroupsWithin20'] and screw['defectGroupBenchmark']['thread_side']['fn'] == 13
        for task, photo in photos.items():
            started = time.monotonic(); reply = http('/v1/analyze-hazard', {'inspectionTask': task, 'image': base64.b64encode(photo.read_bytes()).decode(), 'promptVersion': 'native-screw-v7-smoke'}); timings[task] = time.monotonic()-started
            assert reply['modelSha256'] == digest and reply['inputSha256'] == sha(photo) and reply['neuralModelCount'] == reply['sharedBackboneCount'] == 1
            assert reply['humanConfirmationRequired'] and not reply['fieldPerformanceValidated']
            if task == 'ad_screw':
                assert reply['candidateDetected'] and abs(reply['classResults'][0]['score']-positive['score']) < 2e-4
                assert reply['backbonePassesPerPhoto'] == 2 and reply['inferencePolicySha256'] == native_policy['sha256'] and not reply['allDefectGroupsWithin20']
                for key in ('heatmapPngBase64', 'referenceHeatmapPngBase64', 'referenceValidRegionMaskPngBase64'):
                    with Image.open(io.BytesIO(base64.b64decode(reply[key]))) as im: assert im.size == (reply['inputWidth'], reply['inputHeight'])
                for key in ('nativeReferenceHeatmapPngBase64', 'nativeViewImagePngBase64'):
                    with Image.open(io.BytesIO(base64.b64decode(reply[key]))) as im: assert im.size == (768, 256)
            for key in [n for n in reply if n.endswith('PngBase64')]: reply.pop(key)
            replies.append({'task': task, 'source_phase': 'known_TEST_reused' if task == 'ad_screw' else 'FIT_or_TRAIN', 'result': reply})
            output = ISOLATED/(task+'-cpp.json'); cp = subprocess.run([str(ROOT/'runs/equipment-expansion-v1/qt-probe/build/unified_probe.exe'), task, str(photo), digest, str(output)], cwd=ISOLATED, env=env, capture_output=True, timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
            assert cp.returncode == 0 and output.exists(), cp.stderr.decode(errors='replace')
            proof = read(output); assert proof['operations'] == 3 and proof['samePhotoDoesNotResolve']; proofs.append(proof)
            print(json.dumps({'CPU_CPP_passed': task, 'operations': 3, 'CPU_hazard_seconds': timings[task]}), flush=True)
        normal = ROOT/next(r['image'] for r in rows if r['category'] == 'ad_screw' and r['phase'] == 'fit')
        response = http('/v1/analyze-hazard', {'inspectionTask': 'equipment_surface', 'equipmentCategory': 'ad_screw', 'image': base64.b64encode(normal.read_bytes()).decode(), 'promptVersion': 'native-normal-FIT'})
        assert not response['candidateDetected']
        buffer = io.BytesIO(); Image.new('RGB', (512, 512), 'white').save(buffer, format='PNG')
        try: http('/v1/analyze-hazard', {'inspectionTask': 'ad_screw', 'image': base64.b64encode(buffer.getvalue()).decode(), 'promptVersion': 'quality-unknown'})
        except HTTPError as error: assert error.code == 422
        else: raise AssertionError('Unassessable input must require review, not healthy')
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        log.close()
    with zipfile.ZipFile(ARCHIVE) as z:
        hashes = json.loads(z.read('package-sha256.json')); assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in hashes.items())
    write(REPORTS/'equipment-native-screw-budget-v7-release.json', {'status': 'passed', 'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
        'model_sha256': digest, 'one_checkpoint_one_shared_backbone': True, 'native_screw_backbone_passes': 2, 'actual_CPP_operations': 24, 'CPP_cases': proofs,
        'isolated_ZIP_CPU_API_checked': True, 'hazard_replies': replies, 'native_policy': native_policy, 'prior_cable_nut_and_metal_policy_SHA_preserved': True,
        'normal_FIT_explicit_category_checked': True, 'unassessable_input_rejected422': True, 'native_and_original_heatmap_sizes_checked': True,
        'subgroup_limit_exposed_to_app': True, 'CPU_single_request_seconds_not_latency_benchmark': timings, 'ZIP_CRC_member_SHA_links_verified': True,
        'owned_server_stopped': process.poll() is not None, 'fresh_pip_install_tested': False, 'Android_APK_built': False, 'independent_field_validated': False, 'app_default_promoted': False})
    print(json.dumps({'release': 'passed', 'actual_CPP_operations': 24, 'ZIP_SHA': sha(ARCHIVE)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['docs', 'package', 'smoke']); args = parser.parse_args(); globals()[args.mode]()
