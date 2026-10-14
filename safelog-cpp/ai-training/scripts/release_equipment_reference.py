"""Package the selected ONE checkpoint; verify isolated CPU API and actual C++."""
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
from scripts.study_equipment_reference_budget import RUN, REPORTS, validate, read
from scripts.fetch_equipment_expansion import sha
from scripts.release_unified_main import MODULES, local_links
from scripts.release_metal_calibration import http, torch_state, write
REPO = ROOT.parents[1]; OUTPUTS = REPO.parents[1]/'outputs'
NAME = 'unified-equipment-reference-role4-handoff-2026-10-14-v2'
STAGE = RUN/'handoff-stage'; ISOLATED = RUN/'isolated-zip'; ARCHIVE = OUTPUTS/(NAME+'.zip')


def guide():
    write(REPO/'safelog-cpp/docs/EQUIPMENT_REFERENCE_HANDOFF_KO.md', '''# 역할4 · 케이블·너트 정상 참조 연결

## 전달받는 기능

사진 오탐·미탐을 줄인 연구 연결용 버전이다. 신경망 공통 분석부는 하나이며, 정상 케이블·너트의 특징 버퍼를 같은 모델 파일에 추가했다.

| 유형 | 사진 오탐 | 사진 미탐 | 공개 평가 사진 |
|---|---:|---:|---|
| 케이블 | 10.34% (6/58) | 7.61% (7/92) | 150장 |
| 너트 | 13.64% (3/22) | 11.83% (11/93) | 115장 |
| 금속판 | 3.85% | 18.31% | 이전 결과 유지 |
| 부품 표면 | 12.64% | 10.91% | 이전 결과 유지 |

신규11유형 중4개가 사진20% 이내이며 그중3개가15% 이내다. 나머지7개는 보강/평가 부족 상태다. 같은 공개 평가 사진을 반복 사용해 후보를 선택한 개발 결과이며, 새로운 산업현장 검증 결과가 아니다. 위치 기능은 별도 성능으로 표시한다.

## 1. 실행

새 ZIP을 새 폴더에 푼다. Python3.12에서 기존 시제품과 같은 의존성을 사용한다. 이번 확인에 새 pip 설치 검사는 포함되지 않았다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$referencePrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.equipment_reference_budget_server --checkpoint $referencePrototype.path --sha256 $referencePrototype.sha256 --evaluation evaluation.json --port 8775
```

모델 파일에는 정상 특징90,624개가 포함되어 이전 금속판 버전보다 크다. 이 서버 모듈과 ZIP의 model.json을 함께 사용한다. 이전 서버로 새 파일을 읽지 않는다. 모델 파일과 평가 JSON의 SHA가 일치해야 실행된다.

서버는127.0.0.1에 연결된다. 스마트폰 실연결 주소·HTTPS·로그인·인증은 기존 팀 앱 구조에서 연결한다. 이번 전달물은 APK가 아니다.

## 2. C++ 요청

기존 C++ 어댑터를 그대로 사용한다. 유형별로 SAFELOG_AI_INSPECTION_TASK를 지정한다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="ad_cable"
```

너트는ad_metal_nut, 금속판은mpdd_metal_plate, 부품 표면은ksdd2_surface다. 나머지 유형 ID는 [전체 틀 가이드](EQUIPMENT_EXPANSION_HANDOFF_KO.md)를 참고한다. 기존3항목과 신규11유형을 한 파일에서 처리한다.

새 UI는inspectionTask=equipment_surface와equipmentCategory=ad_cable 또는ad_metal_nut를 명시해도 된다. 사진은 기존처럼 원본 파일 bytes를 base64로 보낸다.

## 3. 화면과 DB에서 구분할 값

| 응답 필드 | 역할4 화면·역할2 저장 내용 |
|---|---|
| candidateDetected / classResults | 해당 부품의 사진 외관 이상 후보 |
| photoScoreSource=normal_reference | 사진 판별이 정상 특징 비교에서 나온 결과 |
| referenceHeatmapPngBase64 | 사진 점수의 참고 시각화. 확정 손상 영역으로 표시하지 않음 |
| heatmapPngBase64 / heatmapSource=original_reconstruction | 기존 위치 표시. 이번에 위치 성능을 개선한 출력이 아님 |
| modelSha256 / inferencePolicySha256 | 모델 파일과 적용 경고 정책을 함께 기록 |
| normalCALAlpha / normalCALRank | 케이블.05/rank44, 너트.10/rank41 |
| benchmarkStage | 해당 유형의 사진 목표 상태. 앱 전체 정확도로 표시하지 않음 |
| mechanicalFailureState=not_assessed | 사진만으로 내부 작동 고장을 확정하지 않음 |
| humanConfirmationRequired | 점검자의 채택·수정·거절 기록 필요 |

normalCALAlpha는 정상 보정 사진에서 경고 기준을 정하는 비율이며 실제 오탐률을 보장하는 수치가 아니다. confidence와raw score도 고장 확률로 표시하지 않는다.

케이블의 손상pixel 미탐76.03%, 너트87.50%는 이전과 같다. 사진 판별15% 통과를 위치 미탐15% 통과로 표시하면 안 된다. 참조 heatmap은 위치 정답으로 평가한 분할 결과가 아니다.

## 4. 확인 순서

1. GET /health에서ready, 모델1개·공통 분석부1개, model.json의SHA를 확인한다.
2. GET /v1/equipment-categories에서11유형과 목표 상태를 확인한다.
3. GET /v1/inference-policy의referencePhoto에서 케이블·너트의 정책SHA와alpha/rank를 확인한다. 기존metalPlate·legacyBelt 정책도 유지된다.
4. POST /v1/analyze-hazard로 사진을 넣고 두 heatmap을 구분해서 연결한다.
5. POST /v1/compare-action에 같은 사진을 넣으면 조치 완료로 표시되지 않아야 한다.
6. POST /v1/summarize의 요약을 보고서 초안에 연결하고 점검자 최종 확인을 받는다. 등급·조치·요약은 규칙 템플릿이다.

이번 역할4 전달물의 실제 검사 기록은 [GitHub release JSON](https://github.com/jungsahun43-commits/robo/blob/feature/ai-engine/safelog-cpp/ai-training/reports/equipment-reference-budget-v2-release.json)에 남긴다. 기록의 ZIP SHA와 전달받은 파일을 대조한다. 새 모델 연결 전에는 이전 전달 ZIP도 보관한다.

- [비교 결과·실패 후보·평가 한계](../ai-training/reports/EQUIPMENT_REFERENCE_RESULTS_KO.md)
- [자료 출처·연구 사용 조건](../ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md)
''')


def package():
    validate(); choice = read(REPORTS/'equipment-reference-budget-v2-selection.json')
    assert choice['reference_counts'] == {'ad_cable': 45568, 'ad_metal_nut': 45056}
    assert not STAGE.exists() and not ARCHIVE.exists(); STAGE.mkdir()
    checkpoint = ROOT/choice['checkpoint_path']; digest = choice['checkpoint_sha256']; assert sha(checkpoint) == digest
    shutil.copy2(checkpoint, STAGE/'model.pt'); shutil.copy2(RUN/'selected/evaluation.json', STAGE/'evaluation.json')
    state = torch_state(checkpoint)
    write(STAGE/'model.json', {'path': 'model.pt', 'sha256': digest, 'architecture': state['architecture'], 'checkpoint_count': 1, 'backbone_count': 1,
        'inspectionTasks': state['tasks'], 'equipmentCategories': state['equipment_categories'], 'referenceCounts': state['reference_counts'],
        'selectedPolicies': choice['selected_per_category'], 'baseHeadEpochs': 30, 'baseHeadOptimizerUpdates': 2640, 'newNeuralUpdatesThisRefinement': 0,
        'normalReferenceFITPhotos': 354, 'normalReferenceFITTensors': 90624, 'stage20Categories': choice['stage20_categories'],
        'researchPrototype': True, 'allCategoriesWithin20': False, 'adaptiveDevelopmentOnKnownTEST': True, 'fieldPerformanceValidated': False, 'appDefaultPromoted': False})
    write(STAGE/'safelog_ai/__init__.py', '')
    for name in (*MODULES, 'belt_tiles.py', 'unified_tiles_server.py', 'equipment_expansion.py', 'equipment_expansion_server.py',
                 'metal_reference.py', 'metal_calibration_server.py', 'equipment_reference_photo.py', 'equipment_reference_server.py', 'equipment_reference_budget_server.py'):
        shutil.copy2(ROOT/'safelog_ai'/name, STAGE/'safelog_ai'/name)
    shutil.copy2(ROOT/'runs/equipment-expansion-v1/handoff-stage-v1.1/requirements-api.txt', STAGE/'requirements-api.txt')
    write(STAGE/'README_KO.md', '''# SafeLog 케이블·너트 사진 판별 보강 시제품

같은 공통 신경망에 정상 특징을 추가한 모델 파일 하나다. 케이블 오탐10.34%·미탐7.61%, 너트 오탐13.64%·미탐11.83%는 알려진 공개 평가 자료를 반복 사용한 개발 결과다. 기존 금속판·부품 표면 결과를 유지했다. 위치 정확도는 별도로 보강해야 한다.

- [역할4 실행·C++·화면/DB 연결](safelog-cpp/docs/EQUIPMENT_REFERENCE_HANDOFF_KO.md)
- [후보 비교와 측정 한계](safelog-cpp/ai-training/reports/EQUIPMENT_REFERENCE_RESULTS_KO.md)
- [자료 출처·연구 사용 조건](safelog-cpp/ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md)
''')
    queue = [REPO/'safelog-cpp/docs/EQUIPMENT_REFERENCE_HANDOFF_KO.md', REPORTS/'EQUIPMENT_REFERENCE_RESULTS_KO.md',
             REPO/'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp', REPO/'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp']
    seen = set()
    while queue:
        source = queue.pop().resolve()
        if source in seen: continue
        assert source.is_relative_to(REPO) and source.is_file(), str(source); seen.add(source)
        destination = STAGE/source.relative_to(REPO); destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        if source.suffix == '.md': queue.extend(local_links(source))
    for doc in STAGE.rglob('*.md'):
        for link in local_links(doc): assert link.is_relative_to(STAGE) and link.is_file(), str(link)
    write(STAGE/'package-sha256.json', {p.relative_to(STAGE).as_posix(): sha(p) for p in STAGE.rglob('*') if p.is_file()})
    with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path in sorted(STAGE.rglob('*')):
            if path.is_file(): archive.write(path, path.relative_to(STAGE).as_posix())
    print(json.dumps({'archive': str(ARCHIVE), 'sha256': sha(ARCHIVE), 'bytes': ARCHIVE.stat().st_size}), flush=True)


def smoke():
    assert not ISOLATED.exists()
    with socket.socket() as s: assert s.connect_ex(('127.0.0.1', 8775)) != 0, 'Preserve unrelated server'
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None and sum(n.endswith('.pt') for n in archive.namelist()) == 1
        for name in archive.namelist(): assert (ISOLATED/name).resolve().is_relative_to(ISOLATED.resolve())
        archive.extractall(ISOLATED)
    meta = read(ISOLATED/'model.json'); digest = meta['sha256']; assert sha(ISOLATED/'model.pt') == digest
    original = read(ROOT/'data/unified-main-v1/manifest.json'); rows = read(ROOT/'data/equipment-expansion-v1/manifest.json')['rows']
    phases = {'facility_surface': 'FIT', 'belt_damage': 'TRAIN_embedded_sample', 'conveyor_foreign': 'FIT', 'mpdd_metal_plate': 'FIT', 'ksdd2_surface': 'FIT'}
    photos = {'facility_surface': ROOT/original['fit']['facility_surface'][0]['image'], 'belt_damage': ROOT/'runs/belt-author-embedded-sample.jpg',
              'conveyor_foreign': ROOT/original['fit']['conveyor_foreign'][0]['image']}
    photos.update({c: ROOT/next(r['image'] for r in rows if r['category'] == c and r['phase'] == 'fit') for c in ('mpdd_metal_plate', 'ksdd2_surface')})
    selected = read(REPORTS/'equipment-reference-budget-v2-selection.json')['selected_per_category']; expected = {}
    for category, name in selected.items():
        result = read(REPORTS/f'equipment-reference-budget-v2-{name}-validation.json')['categories'][category]
        row = next(r for r in result['observations'] if r['label'] == 1 and r['candidate_detected'] and r['score']-result['photo_threshold'] > .005)
        source = next(r for r in rows if r['id'] == row['id']); photos[category] = ROOT/source['image']; phases[category] = 'known_TEST_reused'; expected[category] = row
    env = dict(os.environ); env.pop('PYTHONPATH', None); env['PYTHONIOENCODING'] = 'utf-8'; env['QT_QPA_PLATFORM'] = 'offscreen'
    env['PATH'] = 'C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH', '')
    log = (ISOLATED/'server.log').open('x', encoding='utf-8')
    process = subprocess.Popen([sys.executable, '-m', 'safelog_ai.equipment_reference_budget_server', '--checkpoint', 'model.pt', '--sha256', digest,
                               '--evaluation', 'evaluation.json', '--port', '8775'], cwd=ISOLATED, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    cpp_proofs = []; replies = []; negative_proofs = []; durations = {}
    try:
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError('Packaged server exited: '+(ISOLATED/'server.log').read_text())
            try: health = http('/health'); break
            except (OSError, ValueError): time.sleep(.25)
        else: raise RuntimeError('Packaged CPU server timed out')
        assert health['ready'] and health['modelSha256'] == digest and health['neuralModelCount'] == health['sharedBackboneCount'] == 1
        categories = http('/v1/equipment-categories')['categories']; assert len(categories) == 11 and sum(c['benchmarkStage']['ready_for_next_stage'] for c in categories) == 4
        policies = http('/v1/inference-policy'); reference = policies['referencePhoto']; assert set(reference) == set(selected)
        assert reference['ad_cable']['normalCALAlpha'] == .05 and reference['ad_cable']['normalCALRank'] == 44
        assert reference['ad_metal_nut']['normalCALAlpha'] == .10 and reference['ad_metal_nut']['normalCALRank'] == 41
        assert policies['metalPlate']['sha256'] == read(REPORTS/'equipment-metal-reference-v1-release.json')['metal_policy_sha256']
        for task, photo in photos.items():
            start = time.monotonic()
            result = http('/v1/analyze-hazard', {'inspectionTask': task, 'promptVersion': 'reference-budget-v2-smoke', 'image': base64.b64encode(photo.read_bytes()).decode()})
            durations[task] = time.monotonic()-start
            assert result['modelSha256'] == digest and result['inputSha256'] == sha(photo) and result['neuralModelCount'] == result['sharedBackboneCount'] == 1
            assert not result['fieldPerformanceValidated'] and result['humanConfirmationRequired']
            if result.get('heatmapPngBase64'):
                with Image.open(io.BytesIO(base64.b64decode(result['heatmapPngBase64']))) as heat: assert heat.size == (result['inputWidth'], result['inputHeight'])
            elif task in reference:
                raise AssertionError('Reference categories require the original location map')
            result.pop('heatmapPngBase64', None)
            if task in reference:
                assert result['candidateDetected'] and abs(result['classResults'][0]['score']-expected[task]['score']) < 2e-4
                assert result['normalCALAlpha'] == reference[task]['normalCALAlpha'] and result['normalCALRank'] == reference[task]['normalCALRank']
                assert result['inferencePolicySha256'] == reference[task]['sha256'] and result['photoScoreSource'] == 'normal_reference'
                assert not result['originalLocalizationChanged'] and not result['referenceHeatmapIsEvaluatedLocalization'] and result['backbonePassesPerPhoto'] == 1
                with Image.open(io.BytesIO(base64.b64decode(result['referenceHeatmapPngBase64']))) as heat: assert heat.size == (result['inputWidth'], result['inputHeight'])
                result.pop('referenceHeatmapPngBase64')
            replies.append({'task': task, 'source_phase': phases[task], 'result': result})
            output = ISOLATED/(task+'-cpp.json')
            cp = subprocess.run([str(ROOT/'runs/equipment-expansion-v1/qt-probe/build/unified_probe.exe'), task, str(photo), digest, str(output)],
                cwd=ISOLATED, env=env, capture_output=True, timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
            assert cp.returncode == 0 and output.exists(), cp.stderr.decode(errors='replace')
            proof = read(output); assert proof['operations'] == 3 and proof['samePhotoDoesNotResolve']; cpp_proofs.append(proof)
            print(json.dumps({'CPU_CPP_passed': task, 'CPP_operations': 3, 'CPU_hazard_seconds': durations[task]}), flush=True)
        for category in reference:
            photo = ROOT/next(r['image'] for r in rows if r['category'] == category and r['phase'] == 'fit')
            result = http('/v1/analyze-hazard', {'inspectionTask': 'equipment_surface', 'equipmentCategory': category, 'promptVersion': 'explicit-normal-FIT', 'image': base64.b64encode(photo.read_bytes()).decode()})
            assert not result['candidateDetected'] and result['equipmentCategory'] == category
            negative_proofs.append({'category': category, 'source_phase': 'FIT', 'candidate_detected': False, 'score': result['classResults'][0]['score']})
        for body in [{'inspectionTask': 'equipment_surface'}, {'inspectionTask': 'internal_motor_fault'}, {'inspectionTask': 'ad_cable', 'equipmentCategory': 'ad_metal_nut'}]:
            try: http('/v1/analyze-hazard', {**body, 'image': 'x', 'promptVersion': 'invalid'})
            except HTTPError as error: assert error.code == 422
            else: raise AssertionError('Unknown/missing/conflicting category must fail')
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        log.close()
    with zipfile.ZipFile(ARCHIVE) as archive:
        hashes = json.loads(archive.read('package-sha256.json')); assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n, h in hashes.items())
    write(REPORTS/'equipment-reference-budget-v2-release.json', {'status': 'passed', 'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
        'model_sha256': digest, 'one_checkpoint_one_shared_backbone': True, 'isolated_ZIP_CPU_API_checked': True, 'reference_policies': reference,
        'metal_policy_SHA_preserved': True, 'actual_CPP_operations_this_release': 21, 'CPP_cases': cpp_proofs, 'CPU_hazard_replies': replies,
        'normal_FIT_explicit_category_proofs': negative_proofs, 'CPU_hazard_single_request_seconds_not_benchmark': durations,
        'three_invalid_requests_rejected422': True, 'known_TEST_examples_reused_for_integration': ['ad_cable', 'ad_metal_nut'],
        'same_photo_does_not_resolve': True, 'both_heatmap_sizes_checked': True, 'ZIP_CRC_member_SHA_links_verified': True,
        'owned_server_stopped': process.poll() is not None, 'fresh_pip_install_tested': False, 'Android_APK_built': False,
        'independent_field_validated': False, 'app_default_promoted': False})
    print(json.dumps({'release': 'passed', 'CPP_operations': 21, 'ZIP_SHA': sha(ARCHIVE)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['guide', 'package', 'smoke']); args = parser.parse_args(); globals()[args.mode]()
