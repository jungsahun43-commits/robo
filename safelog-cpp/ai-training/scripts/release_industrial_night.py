"""Package the measured night batch; verify isolated CPU and15 C++ task aliases."""
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
from scripts import study_industrial_surfaces as study
from scripts import study_bsclass_supervised as wear
from scripts.fetch_equipment_expansion import sha
from scripts.release_unified_main import local_links
from scripts.release_metal_calibration import write, http
from safelog_ai.equipment_expansion import CATEGORIES
from safelog_ai.industrial_supervised import WEAR_CATEGORY
from safelog_ai.equipment_expansion_server import LABELS

RUN, REPORTS = study.RUN, study.REPORTS
REPO = ROOT.parents[1]; OUTPUTS = REPO.parents[1]/'outputs'
NAME = 'unified-industrial-night-role4-handoff-2026-10-15-v9'
STAGE = RUN/'handoff-stage'; ISOLATED = RUN/'isolated-zip'; ARCHIVE = OUTPUTS/(NAME+'.zip')


def checked():
    study.validate(); choice = study.read(REPORTS/'industrial-surfaces-v9-selection.json')
    assert study.read(REPORTS/'industrial-surfaces-v9-verification.json')['status'] == 'passed'
    assert study.read(REPORTS/'industrial-bsclass-supervised-v8-verification.json')['status'] == 'passed'
    assert sha(ROOT/choice['checkpoint_path']) == choice['checkpoint_sha256']
    return choice


def docs():
    choice = checked(); evaluation = study.read(RUN/'evaluation.json'); before = study.read(ROOT/'runs/equipment-native-screw-budget-v7/evaluation.json')
    training = study.read(wear.RUN/'chosen-head.json'); fresh = study.read(REPORTS/'industrial-bsclass-supervised-v8-validation.json')
    candidates = study.read(REPORTS/'industrial-surfaces-v9-validation.json')['categories']
    def pct(v): return f'{100*v:.2f}%'
    labels = {**LABELS, WEAR_CATEGORY: '볼스크루 표면 피팅'}
    lines = ['# 설비별 야간 보강 결과', '', '기준일: 2026-10-15', '',
        '## 채택한 결과', '',
        '새 볼스크루 실제 손상 판별기와 갈색·흰색 브래킷 정상 비교를 같은 모델 파일에 통합했다. 원래 추가 설비 11유형 중 사진 오탐·미탐 20% 이내가 5→7개로 늘었다. 새 볼스크루 유형을 포함하면 12유형 중 8개이며, 그중 5개는 15% 이내다. 기존 시설·벨트·이물질 세 작업은 이 12유형 집계에 포함하지 않는다.', '',
        '| 유형 | 이전 오탐 / 미탐 | 현재 오탐 / 미탐 | 정상 / 결함 평가 사진 | 범위 |', '|---|---:|---:|---:|---|']
    for c in (*CATEGORIES, WEAR_CATEGORY):
        v = evaluation['categories'][c]; rates = v['photo_rates']; previous = before['categories'][c]['photo_rates'] if c in before['categories'] else fresh['baseline_photo_rates']
        scope = '새 예약 평가 · 부품 잘라낸 사진' if c == WEAR_CATEGORY else '같은 공개 TEST 재사용'
        if c == 'mpdd_connector': scope += ' · 결함 14장으로 평가 부족'
        lines.append(f"| {labels[c]} | {pct(previous['fpr'])} / {pct(previous['fnr'])} | {pct(rates['fpr'])} / {pct(rates['fnr'])} | {rates['negative_photos']} / {rates['positive_photos']} | {scope} |")
    lines += ['', '볼스크루의 이전 수치는 같은 고정 특징의 정상 중심 거리 기준선이다. 기존 일반 체결나사 모델의 수치를 볼스크루 이전 성능으로 바꾸어 적은 것이 아니다. 나머지 유형의 이전 수치는 이번 야간 작업 직전 v7 모델이다.', '',
        '## 실제 정상·손상 라벨 학습', '',
        '- BSClass 공개 자료 21,835장의 파일명 N/P 정답을 사용했다. 일반 체결나사와 다른 볼스크루 표면 피팅 유형으로 분리했다.',
        '- 회전본·정확 중복·근접 해시 그룹을 나누기 전에 연결했다. FIT 정상 7,793장·손상 7,540장, 검증 1,653장, 보정 1,613장, 예약 평가 3,236장이다.',
        '- 기존 공통 분석부와 모든 기존 출력·참조를 고정하고, 같은 분석부의 low40/high960 평균·표준편차·최댓값 특징 3,000개로 새 판별기를 학습했다. 특징 캐시의 FP16 반올림도 배포 서버에서 동일하게 적용한다.',
        '- 선형 판별기와 MLP를 각각 30epoch·1,800회 학습했다. 합계 실제 optimizer 업데이트는 3,600회다. 검증 자료만으로 고른 MLP의 23epoch·1,380회 시점 가중치를 채택했다.',
        '- FIT에서만 특징 표준화 통계를 구했다. 회전·중복 그룹당 학습 영향이 같도록 그룹 내 사진 수의 역수로 가중했다. 검증으로 모델·회차를 고른 뒤 별도 보정 자료에서 경고 기준을 정했다.',
        '- 모델·회차·기준을 파일에 고정한 뒤 예약 평가를 처음 실행했다. 정상 1,643장 중 오탐 14장, 손상 1,593장 중 미탐 10장이다. 실제 사진 정답이며 가상 이상 학습이 아니다.',
        '- 물리적 볼스크루·촬영 원본 프레임 그룹은 제공 정보만으로 확인하지 못했다. 잘라낸 공개 부품 사진의 평가이며 새로운 공장·전체 설비 사진의 성능은 아직 검증하지 않았다. 피팅 위치 마스크도 없어 위치 정확도를 주장하지 않는다.', '',
        '[BSClass 원출처](https://publikationen.bibliothek.kit.edu/1000133819)는 CC BY-NC 4.0 연구 자료다. 출처와 비상업 조건을 유지한다.', '',
        '## 브래킷·연결부·튜브 비교', '',
        '정상 FIT574장으로 256 입력의 모든 정상 특징과, 비율을 보존한 512 입력에서 사진당 고정 256개 정상 특징을 각각 적합했다. 정상 CAL의 15%·20% 예산 순위 기준을 모두 저장한 뒤 알려진 TEST로 후보를 비교했다. 이번 참조 적합의 신경망 업데이트는 0회다.', '',
        '| 유형 / 후보 | 오탐 | 미탐 | 채택 |', '|---|---:|---:|---|']
    for c, variants in candidates.items():
        for name, value in variants.items():
            r = value['photo_rates']; selected = choice['selected'].get(c) == name
            reason = '채택' if selected else '평가 표본 부족' if c == 'mpdd_connector' else '제외'
            lines.append(f"| {labels[c]} / {name} | {pct(r['fpr'])} | {pct(r['fnr'])} | {reason} |")
    lines += ['', '갈색은 256 비교, 흰색은 512 비교를 선택했다. 선택한 정상 FIT는 164장·41,984개 참조 특징이다. 검정 브래킷과 튜브는 미탐이 줄어도 오탐·미탐 20% 조건을 함께 충족하지 못해 기존 출력을 유지했다. 금속 연결부 후보의 미탐 0/14는 표본 조건 20장에 부족하여 채택하지 않았다.', '',
        '## 손상 종류와 위치 성능', '',
        '나사산 옆면 미탐 13/23=56.52%와 기존 나사 위치 pixel 미탐 97.20%는 남는다. 사진 전체의 합산 통과를 모든 손상 종류·위치의 정확도로 표시하지 않는다. 브래킷의 기존 위치 출력도 그대로이며 새 정상 차이 이미지는 위치 정답으로 검증한 분할 결과가 아니다.', '',
        '| 선택한 브래킷 | 손상 종류 | 미탐 / 결함 사진 | 미탐률 |', '|---|---|---:|---:|']
    for c, name in choice['selected'].items():
        for group, v in candidates[c][name]['by_defect'].items(): lines.append(f"| {labels[c]} | {group} | {v['fn']}/{v['defect_photos']} | {pct(v['fnr'])} |")
    lines += ['', '종류별 표본 수는 적을 수 있다. 다른 유형은 같은 공개 TEST를 반복 사용한 개발 비교이며, 새 예약 평가와 같은 독립성을 가진 수치로 합산하지 않는다.', '',
        '## 앱 연결과 검증', '',
        '하나의 모델 파일과 하나의 공통 분석부를 사용한다. 볼스크루·갈색 브래킷은 1회, 나사·흰색 브래킷은 원래 위치 출력 보존을 위해 같은 분석부를 2회 실행한다. 내부 기계 고장은 평가하지 않으며 점검자가 최종 확인한다. 위험 등급·조치·요약은 기존 규칙 템플릿이다.', '',
        '실제 ZIP CPU API·C++ 연결 결과는 별도 실행 기록을 확인한다. 새로운 pip 설치나 Android APK·기본 앱 배포는 이 작업에 포함되지 않았다.', '',
        f"선택 모델 SHA: `{choice['checkpoint_sha256']}`", '',
        '- [실제 라벨 학습 조건](industrial-bsclass-supervised-v8-protocol.json)', '- [볼스크루 예약 평가](industrial-bsclass-supervised-v8-validation.json)',
        '- [볼스크루 가중치·출력 보존](industrial-bsclass-supervised-v8-verification.json)', '- [5유형 비교 조건](industrial-surfaces-v9-protocol.json)',
        '- [5유형 전체 후보](industrial-surfaces-v9-validation.json)', '- [최종 선택](industrial-surfaces-v9-selection.json)',
        '- [최종 보존 검증](industrial-surfaces-v9-verification.json)', '- [역할4 연결 안내](../../docs/INDUSTRIAL_NIGHT_HANDOFF_KO.md)',
        '- [CPU·C++ 실행 기록](https://github.com/jungsahun43-commits/robo/blob/feature/ai-engine/safelog-cpp/ai-training/reports/industrial-night-v9-release.json)', '']
    write(REPORTS/'INDUSTRIAL_NIGHT_RESULTS_KO.md', '\n'.join(lines))
    guide = '''# 역할4 · 설비별 야간 개선 모델 연결

## 전달물과 실행

이 ZIP에는 모델 하나, Python CPU 서버와 C++ 어댑터 소스가 들어 있다. 전체 앱 또는 APK는 아니다. 이전 ZIP과 다른 새 폴더에 압축을 푼다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$industrialPrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.industrial_batch_server --checkpoint $industrialPrototype.path --sha256 $industrialPrototype.sha256 --evaluation evaluation.json --port 8775
```

현재 검증 환경의 의존성 버전이다. 이번에 새 pip 설치 자체를 검증한 것은 아니다. `GET /health`, `/v1/equipment-categories`, `/v1/inference-policy`에서 모델 SHA와 유형·경고 정책을 확인한다.

## 앱에서 선택할 점검 항목

| 항목 | 유형 ID | 입력과 결과 |
|---|---|---|
| 새 볼스크루 표면 피팅 | ball_screw_pitting | 볼스크루 표면을 잘라낸 선명한 사진. 사진 정상/손상 후보만 제공, 위치 표시는 없음 |
| 갈색 브래킷 | mpdd_bracket_brown | 지정 부품 사진. 기존 위치 출력 + 별도 정상 차이 참고 이미지 |
| 흰색 브래킷 | mpdd_bracket_white | 원본 비율·세부 정보 보존. 기존 위치 출력 + 별도 정상 차이 참고 이미지 |
| 일반 체결나사 | ad_screw | 나사 한 개를 담은 원본 사진. 입력 방향이 모호하면 422·재촬영/수동 확인 |

기존 시설·벨트·이물질과 나머지 설비 유형 ID도 유지한다. 볼스크루와 일반 체결나사를 같은 항목으로 합치지 않는다. 전체 공장 사진에서 자동으로 설비 종류를 골라내는 모델은 아니다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="ball_screw_pitting"
```

새 볼스크루 ID를 허용한 C++ 어댑터가 함께 들어 있다. 저장소의 같은 경로에 반영하거나 AI 브랜치를 합친다. 기존 HTTP API 경로와 분석·전후 비교·요약 요청은 같다. 기기 주소·HTTPS·로그인·권한은 팀 앱 서버 구조에 연결한다.

## 화면과 DB에 보관할 필드

- `modelSha256`, `inferencePolicySha256`, 원본 사진, 점검자 채택/수정 결과를 함께 보관한다.
- `benchmarkStage`는 사진 단위 합산 상태이며 `benchmarkStageScope`, `defectGroupBenchmark`의 범위·종류별 한계를 같이 표시한다.
- 볼스크루의 `requiredInput=cropped_ball_screw_surface`, `localizationAvailable=false`, `inputDomainAutoVerified=false`를 따른다. 위치 박스나 heatmap을 임의로 만들지 않는다.
- 브래킷의 `referenceHeatmapPngBase64`는 정상 차이 참고 이미지이며 `referenceHeatmapIsEvaluatedLocalization=false`다. 기존 `heatmapPngBase64`의 위치 정확도가 좋아졌다는 의미는 아니다.
- 나사산 옆면 미탐 56.52%는 남는다. 전체 나사 사진 미탐 12.61%와 구분한다.
- 판독 불가 422는 정상으로 저장하지 않고 다시 촬영하거나 수동 검수한다. 내부 기계 고장은 `mechanicalFailureState=not_assessed`다.
- confidence·sigmoid score는 실제 고장 확률로 보정된 값이 아니다. 위험 등급·조치·요약은 규칙 템플릿이고 점검자가 최종 확인한다.

볼스크루 공개 예약 평가의 오탐 0.85%·미탐 0.63%는 부품을 잘라낸 사진 기준이다. 물리적 설비·촬영 프레임 그룹을 확인하지 못해 새 현장 성능은 아직 검증하지 않았다. 기존 설비 11유형은 같은 공개 TEST를 재사용한 개발 수치다.

- [설비별 결과와 모든 후보](../ai-training/reports/INDUSTRIAL_NIGHT_RESULTS_KO.md)
- [C++ 어댑터](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp)
- [C++ 헤더](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp)
'''
    write(REPO/'safelog-cpp/docs/INDUSTRIAL_NIGHT_HANDOFF_KO.md', guide)
    header = '## 설비별 야간 보강 · 실제 손상 학습 · 2026-10-15'
    block = '\n'.join((header, '', '실제 볼스크루 정상·손상 라벨로 두 판별기를 합계 3,600회 학습했다. 부품을 잘라낸 예약 평가 사진에서 오탐 0.85%·미탐 0.63%를 측정했다.',
        '갈색·흰색 브래킷도 보강해 기존 추가 설비 11유형 중 사진 20% 이내가 5→7개로 늘었다. 새 볼스크루를 포함하면 12유형 중 8개다. 기존 통과 유형의 가중치를 보존했다.',
        '볼스크루는 촬영 설비별 독립성이 미검증이고, 다른 유형은 알려진 TEST를 반복 사용한 개발 결과다. 위치 정확도와 내부 기계 고장은 별도 범위다.', ''))
    for path, link in ((REPO/'README.md', 'safelog-cpp/ai-training/reports/INDUSTRIAL_NIGHT_RESULTS_KO.md'), (ROOT/'README_KO.md', 'reports/INDUSTRIAL_NIGHT_RESULTS_KO.md'),
        (REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md', '../ai-training/reports/INDUSTRIAL_NIGHT_RESULTS_KO.md'), (REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md', '../ai-training/reports/INDUSTRIAL_NIGHT_RESULTS_KO.md')):
        raw = path.read_bytes().decode('utf-8'); assert header not in raw; nl = '\r\n' if '\r\n' in raw else '\n'; title, body = raw.split(nl+nl, 1)
        path.write_bytes((title+nl+nl+(block+'\n- [이번 실제 학습·설비별 결과]('+link+')\n').replace('\n', nl)+nl+body).encode('utf-8'))


def package():
    choice = checked(); assert not STAGE.exists() and not ARCHIVE.exists(); STAGE.mkdir()
    shutil.copy2(ROOT/choice['checkpoint_path'], STAGE/'model.pt'); shutil.copy2(RUN/'evaluation.json', STAGE/'evaluation.json')
    write(STAGE/'model.json', {'path': 'model.pt', 'sha256': choice['checkpoint_sha256'], 'architecture': 'safelog-industrial-surface-reference-v9',
        'checkpoint_count': 1, 'backbone_count': 1, 'new_supervised_category': WEAR_CATEGORY, 'surface_methods': choice['selected'],
        'actual_new_supervised_optimizer_updates_all_registered_heads': 3600, 'selected_wear_epoch': 23, 'selected_wear_epoch_updates': 1380,
        'known_TEST_reused_for_original11': True, 'fresh_reserved_TEST_used_for_BSClass_selection_guard_only': True,
        'physical_spindle_groups_verified': False, 'field_performance_validated': False, 'app_default_promoted': False})
    write(STAGE/'safelog_ai/__init__.py', '')
    previous = ROOT/'runs/equipment-native-screw-budget-v7/handoff-stage'
    names = [p.name for p in (previous/'safelog_ai').glob('*.py') if p.name != '__init__.py']+['industrial_supervised.py', 'industrial_surface_reference.py', 'industrial_batch_server.py']
    for name in names: shutil.copy2(ROOT/'safelog_ai'/name, STAGE/'safelog_ai'/name)
    shutil.copy2(previous/'requirements-api.txt', STAGE/'requirements-api.txt')
    write(STAGE/'README_KO.md', '# SafeLog 설비별 야간 개선 모델\n\n모델 한 개·공통 분석부 한 개. 볼스크루 실제 손상 학습과 브래킷 개선을 포함한다.\n\n'
        '- [역할4 실행·화면·DB 연결](safelog-cpp/docs/INDUSTRIAL_NIGHT_HANDOFF_KO.md)\n- [범위와 실제 결과](safelog-cpp/ai-training/reports/INDUSTRIAL_NIGHT_RESULTS_KO.md)\n')
    queue = [REPO/'safelog-cpp/docs/INDUSTRIAL_NIGHT_HANDOFF_KO.md', REPORTS/'INDUSTRIAL_NIGHT_RESULTS_KO.md']; seen = set()
    while queue:
        source = queue.pop().resolve()
        if source in seen: continue
        assert source.is_file() and source.is_relative_to(REPO), str(source); seen.add(source)
        target = STAGE/source.relative_to(REPO); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        if source.suffix == '.md': queue.extend(local_links(source))
    for path in STAGE.rglob('*.md'):
        for link in local_links(path): assert link.is_file() and link.is_relative_to(STAGE), str(link)
    write(STAGE/'package-sha256.json', {p.relative_to(STAGE).as_posix(): sha(p) for p in STAGE.rglob('*') if p.is_file()})
    with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for path in sorted(STAGE.rglob('*')):
            if path.is_file(): z.write(path, path.relative_to(STAGE).as_posix())
    print(json.dumps({'archive': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size, 'sha256': sha(ARCHIVE)}), flush=True)


def smoke():
    choice = checked(); assert not ISOLATED.exists()
    with socket.socket() as s: assert s.connect_ex(('127.0.0.1', 8775)) != 0, 'Preserve unrelated server'
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist()) == 1
        for name in z.namelist(): assert (ISOLATED/name).resolve().is_relative_to(ISOLATED.resolve())
        z.extractall(ISOLATED)
    digest = choice['checkpoint_sha256']; assert sha(ISOLATED/'model.pt') == digest
    original = study.read(ROOT/'data/unified-main-v1/manifest.json'); rows = study.read(study.MANIFEST)['rows']
    photos = {'facility_surface': ROOT/original['fit']['facility_surface'][0]['image'], 'belt_damage': ROOT/'runs/belt-author-embedded-sample.jpg', 'conveyor_foreign': ROOT/original['fit']['conveyor_foreign'][0]['image']}
    photos.update({c: ROOT/next(r['image'] for r in rows if r['category'] == c and r['phase'] == 'fit') for c in CATEGORIES})
    expected = {}; values = study.read(REPORTS/'industrial-surfaces-v9-validation.json')['categories']
    for category, key in choice['selected'].items():
        result = values[category][key]; selected = next(r for r in result['observations'] if r['label'] and r['candidate_detected'] and r['score']-result['photo_threshold'] > .005)
        photos[category] = ROOT/next(r['image'] for r in rows if r['id'] == selected['id']); expected[category] = selected['score']
    fresh = study.read(REPORTS/'industrial-bsclass-supervised-v8-validation.json'); data = study.read(wear.MANIFEST)['rows']
    selected = next(r for r in fresh['observations'] if r['label'] and r['candidate_detected'] and r['score'] > .99)
    photos[WEAR_CATEGORY] = ROOT/next(r['image'] for r in data if r['id'] == selected['id']); expected[WEAR_CATEGORY] = selected['score']
    env = dict(os.environ); env.pop('PYTHONPATH', None); env.update(PYTHONIOENCODING='utf-8', QT_QPA_PLATFORM='offscreen', QT_PLUGIN_PATH='C:/Qt/6.11.2/mingw_64/plugins')
    env['PATH'] = 'C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH', '')
    log = (ISOLATED/'server.log').open('x', encoding='utf-8')
    process = subprocess.Popen([sys.executable, '-m', 'safelog_ai.industrial_batch_server', '--checkpoint', 'model.pt', '--sha256', digest, '--evaluation', 'evaluation.json', '--port', '8775'], cwd=ISOLATED, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    proofs = []; replies = []; timings = {}
    try:
        deadline = time.monotonic()+50
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError((ISOLATED/'server.log').read_text())
            try: health = http('/health'); break
            except (OSError, ValueError): time.sleep(.25)
        else: raise RuntimeError('Packaged server startup timed out')
        assert health['modelSha256'] == digest and health['neuralModelCount'] == health['sharedBackboneCount'] == 1
        policies = http('/v1/inference-policy'); prior = study.read(REPORTS/'equipment-native-screw-budget-v7-release.json')
        assert policies['nativeComponents']['ad_screw'] == prior['native_policy']
        assert policies['referencePhoto'] == study.read(REPORTS/'equipment-reference-budget-v2-release.json')['reference_policies']
        assert policies['metalPlate']['sha256'] == study.read(REPORTS/'equipment-metal-reference-v1-release.json')['metal_policy_sha256']
        categories = http('/v1/equipment-categories')['categories']; assert len(categories) == 12 and sum(c['benchmarkStage']['ready_for_next_stage'] for c in categories) == 8
        assert set(policies['surfaceReference']) == set(choice['selected']) and WEAR_CATEGORY in policies['supervisedWear']
        for task, photo in photos.items():
            start = time.monotonic(); reply = http('/v1/analyze-hazard', {'inspectionTask': task, 'image': base64.b64encode(photo.read_bytes()).decode(), 'promptVersion': 'industrial-night-v9-smoke'}); timings[task] = time.monotonic()-start
            assert reply['inputSha256'] == sha(photo) and reply['modelSha256'] == digest and reply['neuralModelCount'] == reply['sharedBackboneCount'] == 1
            assert reply['humanConfirmationRequired'] and not reply['fieldPerformanceValidated']
            if task in expected:
                assert reply['candidateDetected'] and abs(reply['classResults'][0]['score']-expected[task]) < 2e-4, task
            if task == WEAR_CATEGORY:
                assert not reply['localizationAvailable'] and not reply['inputDomainAutoVerified'] and 'heatmapPngBase64' not in reply
            if task in choice['selected']:
                for key in ('heatmapPngBase64', 'referenceHeatmapPngBase64'):
                    with Image.open(io.BytesIO(base64.b64decode(reply[key]))) as im: assert im.size == (reply['inputWidth'], reply['inputHeight'])
            if task == 'ad_screw': assert reply['backbonePassesPerPhoto'] == 2 and not reply['allDefectGroupsWithin20']
            for key in [k for k in reply if k.endswith('PngBase64')]: reply.pop(key)
            replies.append({'task': task, 'result': reply, 'source_phase': 'known_evaluation_reused_for_runtime_only' if task in expected else 'FIT_or_TRAIN'})
            output = ISOLATED/(task+'-cpp.json')
            cp = subprocess.run([str(RUN/'qt-probe/build/unified_probe.exe'), task, str(photo), digest, str(output)], cwd=ISOLATED, env=env, capture_output=True, timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
            assert cp.returncode == 0 and output.exists(), cp.stderr.decode(errors='replace')
            proof = study.read(output); assert proof['operations'] == 3 and proof['samePhotoDoesNotResolve']; proofs.append(proof)
            print(json.dumps({'CPU_CPP_passed': task, 'operations': 3, 'single_CPU_request_seconds': round(timings[task], 3)}), flush=True)
        normal = next(r for r in fresh['observations'] if not r['label'] and not r['candidate_detected'] and r['score'] < .01)
        photo = ROOT/next(r['image'] for r in data if r['id'] == normal['id'])
        reply = http('/v1/analyze-hazard', {'inspectionTask': 'equipment_surface', 'equipmentCategory': WEAR_CATEGORY, 'image': base64.b64encode(photo.read_bytes()).decode(), 'promptVersion': 'wear-normal-control'})
        assert not reply['candidateDetected'] and abs(reply['classResults'][0]['score']-normal['score']) < 2e-4
        buffer = io.BytesIO(); Image.new('RGB', (32, 32), 'white').save(buffer, format='PNG'); encoded = base64.b64encode(buffer.getvalue()).decode()
        for body in ({'inspectionTask': WEAR_CATEGORY}, {'inspectionTask': WEAR_CATEGORY, 'equipmentCategory': 'ad_screw'}, {'inspectionTask': 'internal_motor_failure'}, {'inspectionTask': 'ad_screw'}):
            try: http('/v1/analyze-hazard', {**body, 'image': encoded, 'promptVersion': 'invalid-input-control'})
            except HTTPError as error: assert error.code == 422
            else: raise AssertionError('Unknown, mismatched or unassessable input must not be marked healthy')
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        log.close()
    with zipfile.ZipFile(ARCHIVE) as z:
        hashes = json.loads(z.read('package-sha256.json')); assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in hashes.items())
    assert len(proofs) == 15 and sum(v['operations'] for v in proofs) == 45
    write(REPORTS/'industrial-night-v9-release.json', {'status': 'passed', 'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
        'model_sha256': digest, 'one_checkpoint_one_shared_backbone': True, 'actual_CPP_operations': 45, 'CPP_cases': proofs,
        'isolated_ZIP_CPU_API_checked': True, 'hazard_replies': replies, 'prior_reference_native_screw_and_metal_policies_exact': True,
        'new_wear_alias_and_explicit_category_positive_and_negative_checked': True, 'new_wear_no_fabricated_localization': True,
        'new_surface_original_coordinate_heatmap_sizes_checked': True, 'unknown_mismatched_and_unassessable_inputs_rejected422': True,
        'CPU_single_request_seconds_not_latency_benchmark': timings, 'ZIP_CRC_member_SHA_and_local_links_verified': True,
        'owned_server_stopped': process.poll() is not None, 'fresh_pip_install_tested': False, 'Android_APK_built': False,
        'independent_field_validated': False, 'app_default_promoted': False})
    print(json.dumps({'release_passed': True, 'CPP_operations': 45, 'archive_SHA': sha(ARCHIVE)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['docs', 'package', 'smoke']); args = parser.parse_args(); globals()[args.mode]()
