"""Measured photo-only handoff, preserved maps and isolated runtime verification."""
import argparse
import base64
from datetime import datetime, timezone
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
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import study_ksdd_photo_only as study
from scripts import study_ksdd_supervised as trained
from scripts.fetch_equipment_expansion import sha
from scripts.release_unified_main import local_links
from scripts.release_metal_calibration import write, http
from safelog_ai.equipment_expansion import CATEGORIES
from safelog_ai.industrial_supervised import WEAR_CATEGORY

RUN, REPORTS = study.RUN, study.REPORTS; REPO = ROOT.parents[1]; OUTPUTS = REPO.parents[1]/'outputs'
STAGE = RUN/'handoff-stage'; ISOLATED = RUN/'isolated-zip'
ARCHIVE = OUTPUTS/'unified-ksdd-photo-role4-handoff-2026-10-15-v11.zip'


def checked():
    study.validate(); choice = study.read(REPORTS/'industrial-ksdd-photo-only-v11-selection.json')
    assert all(choice['guards'].values()) and sha(ROOT/choice['checkpoint_path']) == choice['checkpoint_sha256']
    return choice


def docs():
    choice = checked(); train = study.read(trained.RUN/'trained/training.json'); raw = study.read(study.VALIDATION)
    report = '''# 실제 위치 정답 학습 · 사진 판별만 채택

기준일: 2026-10-15

## 현재 선택 결과

KSDD2의 보관된 실제 손상 TRAIN 사진과 위치 마스크로 새 판별기를 학습했다. 알려진 공개 평가 1,004장(정상 894·손상 110)에서 사진 오탐 113/894=12.64% → 110/894=12.30%, 미탐 12/110=10.91% → 5/110=4.55%로 줄었다.

기존 위치 지도와 위치 경고 기준은 그대로 유지한다. 새 신경망의 사진 점수만 사용한다. 추가 설비 12유형 중 사진 20% 이내 8개·15% 이내 5개 상태는 같다. 이번 개선은 기존 통과 유형의 미탐을 더 줄인 것이다.

## 학습과 자료 분리

- 원래 보관해 둔 publisher TRAIN 실제 손상 사진 246장을 사용했다. 이번에 새로 촬영하거나 웹에서 더 받은 사진으로 세지 않는다.
- 정확 RGB, 회전·반전 8방향 근접 pHash 거리≤2, 파일 번호 인접 10개 묶음을 적용한 뒤 그룹별로 분리했다. TEST와 연결된 TRAIN 사진은 제외하도록 검사했으며 이번 자료에서는 0장이었다.
- 파일 번호 인접 묶음은 보수적인 대용 그룹이다. 물리적 부품·실제 촬영 프레임 그룹을 확인한 것으로 주장하지 않는다.
- 학습 정상 1,254·손상 146장, 검증 정상 389·손상 52장, 보정 정상 442·손상 48장이다. publisher TEST 1,004장은 학습·검증·보정에 넣지 않았다.
- 기존 공통 분석부·모든 기존 가중치·참조를 고정했다. 같은 분석부의 low40/high960 특징에 새 위치 판별기 하나를 붙여 40epoch·2,560번 실제 optimizer 업데이트를 수행했다.
- 검증 자료로 선택한 13epoch·832번 시점 가중치를 사용한다. 특징 캐시와 실행부 모두 FP16 반올림 뒤 FP32 판별을 적용한다.
- 비율을 보존한 256×704 입력에서 실제 마스크 BCE·Dice·사진 분류 손실을 학습했다. 특징 격자와 마스크를 함께 뒤집는 증강은 새로운 사진 수로 세지 않는다.
- 사진 경고 기준은 별도 라벨 보정 자료의 오탐·미탐 균형으로 정했다. 위치 기준도 사전 고정한 0.1~0.9 후보를 보정 자료에서만 비교했다. TEST로 회차나 기준을 고르지 않았다.

[KolektorSDD2 원출처](https://www.vicos.si/resources/kolektorsdd2/)와 기존 자료 출처 기록을 유지한다. 기존 다운로드는 고정 HuggingFace 미러와 그 LFS SHA를 사용했으며, 원배포 ZIP과의 동일성을 직접 검증한 체크섬으로 표시하지 않는다. 연구용 CC BY-NC-SA 4.0 조건을 보존한다.

## 전체 교체를 제외한 이유

아래 위치 수치는 두 방식 모두 **같은 원본 사진 좌표와 같은 원본 정답 마스크**에서 다시 계산했다. 과거 256×256 위치 수치와 분모가 다른 비교를 섞지 않는다.

| 항목 | 기존 위치 지도 | 새 지도 후보 |
|---|---:|---:|
| 위치 IoU | 23.09% | 48.09% |
| 위치 미탐 | 17.10% | 26.81% |
| 위치 오탐 | 0.8636% | 0.1740% |

새 지도는 IoU와 위치 오탐은 좋아졌지만 위치 미탐이 늘어 v10 전체 교체 조건을 충족하지 못했다. 해당 실패 결과와 학습 가중치를 보존했다.

v11은 새 사진 점수·이미 정한 사진 기준을 그대로 사용하면서, 기본 위치 출력과 위치 기준을 기존 것으로 보존하는 **단일 후속 조합**이다. 새 학습·경고 기준 변경은 0회다. 한 모델·한 공통 분석부를 사용하며 사진 점수와 원래 위치 보존을 위해 같은 분석부를 두 번 실행한다.

**v11 조합 설계와 채택에는 이미 본 v10 TEST 결과를 사용했다.** 독립 현장 검증이나 새로운 블라인드 평가 결과가 아니다. 사진 미탐 4.55%를 전체 산업현장·모든 고장·위치 미탐 5% 이내로 표시하지 않는다.

## 앱 연결

유형 ID `ksdd2_surface`와 기존 C++ 어댑터를 유지한다. 기본 `heatmapPngBase64`는 기존 지도다. `supervisedAuxiliaryHeatmapPngBase64`와 `supervisedAuxiliaryMaskPngBase64`는 미탐이 더 큰 새 지도의 참고 출력이며 기본 지도와 구분한다. `auxiliaryLocalizationWorseFNR=true`를 함께 전달한다.

이전 케이블·나사·너트·브래킷·금속판·볼스크루와 시설·벨트·이물질 가중치는 모두 그대로다. 볼스크루 피팅과 일반 체결나사를 같은 라벨로 섞지 않았다. 내부 기계 고장은 평가하지 않고 점검자가 최종 확인한다. 위험 등급·조치·요약은 기존 규칙 템플릿이다.

새 APK나 앱 기본 모델 자동 배포는 하지 않았다. ZIP의 CPU·C++ 실행 기록과 CPU 전체 사진 비교 기록을 확인한다.

- [실제 마스크 학습 조건](industrial-ksdd-supervised-v10-protocol.json)
- [v10 전체 결과와 제외 조건](industrial-ksdd-supervised-v10-validation.json)
- [기존 자료·가중치 보존](industrial-ksdd-supervised-v10-verification.json)
- [사진 전용 후속 조합 조건](industrial-ksdd-photo-only-v11-protocol.json)
- [사진 전용 선택](industrial-ksdd-photo-only-v11-selection.json)
- [역할4 실행 안내](../../docs/KSDD_PHOTO_HANDOFF_KO.md)
- [CPU·C++ 기록](https://github.com/jungsahun43-commits/robo/blob/feature/ai-engine/safelog-cpp/ai-training/reports/industrial-ksdd-photo-only-v11-release.json)
'''
    write(REPORTS/'KSDD_PHOTO_ONLY_RESULTS_KO.md', report)
    guide = '''# 역할4 · 부품 표면 사진 미탐 보강 모델

## 적용 범위

`ksdd2_surface` 사진 판별 미탐이 10.91%→4.55%, 오탐이 12.64%→12.30%로 줄었다. 같은 공개 TEST를 재사용한 개발 수치다. 기존 위치 지도는 그대로다. 새 위치 지도 후보는 미탐이 더 커 기본 출력으로 채택하지 않았다.

## CPU 서버 실행

새 ZIP을 새 폴더에 푼다. 모델 파일은 하나다. 이전 전달 ZIP과 의존성은 같고 새 서버 모듈과 model.json을 함께 쓴다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$surfacePrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.ksdd_photo_only_server --checkpoint $surfacePrototype.path --sha256 $surfacePrototype.sha256 --evaluation evaluation.json --port 8775
```

이번 검증은 기존 설치 환경에서 실행했으며 새 pip 설치 자체나 Android APK는 검증한 것이 아니다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="ksdd2_surface"
```

C++ 유형 ID와 요청 경로는 같다. 다른 설비 유형도 이전과 같이 사용한다. 스마트폰의 주소·HTTPS·로그인 연결은 팀 앱 서버 구조에 맞춘다.

## 출력·화면·DB

| 필드 | 의미 |
|---|---|
| classResults / candidateDetected | 새 실제 마스크 학습 판별기의 사진 이상 후보 |
| heatmapPngBase64 | 기존 위치 출력. 바이트가 같은 것을 검사하며 기본 지도로 유지 |
| originalPixelThreshold / originalLocalizationChanged | 기존 위치 기준 유지, changed=false |
| supervisedAuxiliaryHeatmapPngBase64 / supervisedAuxiliaryMaskPngBase64 | 새 지도 참고 출력. 기본 지도와 구분 |
| auxiliaryLocalizationWorseFNR | true. 새 위치 미탐26.81%로 기존17.10%보다 큼 |
| inferencePolicySha256 / modelSha256 | 사진·위치 정책과 모델을 함께 보관 |
| backbonePassesPerPhoto | 2. 같은 공통 분석부를 원래 위치와 새 사진 점수에 각각 사용 |

사진 점수는 실제 고장 확률로 보정된 값이 아니다. 내부 기계 고장은 `not_assessed`이고 점검자가 최종 확인한다. 64픽셀보다 작은 입력이나 판독 불가 오류를 정상으로 저장하지 않는다.

`GET /v1/inference-policy`의 `supervisedPhoto.ksdd2_surface`와 `/v1/equipment-categories`에서 정책과 평가 범위를 확인한다. 나사산 옆면의 기존 미탐 등 다른 유형의 한계도 유지된다.

- [전체 결과·자료 분리·제외 이유](../ai-training/reports/KSDD_PHOTO_ONLY_RESULTS_KO.md)
- [이전 설비별 안내](https://github.com/jungsahun43-commits/robo/blob/feature/ai-engine/safelog-cpp/docs/INDUSTRIAL_NIGHT_HANDOFF_KO.md)
'''
    write(REPO/'safelog-cpp/docs/KSDD_PHOTO_HANDOFF_KO.md', guide)
    header = '## 부품 표면 실제 마스크 학습 · 사진 미탐 보강 · 2026-10-15'
    block = '\n'.join((header, '', '실제 위치 정답으로 40epoch·2,560회 추가 학습했다. 사진 미탐10.91%→4.55%, 오탐12.64%→12.30%를 측정했다.',
        '새 위치 지도는 위치 미탐이 늘어 전체 교체에서 제외했다. 사진 점수만 채택하고 원래 위치 지도·기준은 유지한다. 기존12유형 중8개 사진20% 이내 상태는 같다.',
        '이미 본 공개TEST로 후속 조합을 설계한 개발 결과이며 독립 현장 검증은 아니다.', ''))
    for path, link in ((REPO/'README.md', 'safelog-cpp/ai-training/reports/KSDD_PHOTO_ONLY_RESULTS_KO.md'), (ROOT/'README_KO.md', 'reports/KSDD_PHOTO_ONLY_RESULTS_KO.md'),
        (REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md', '../ai-training/reports/KSDD_PHOTO_ONLY_RESULTS_KO.md'), (REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md', '../ai-training/reports/KSDD_PHOTO_ONLY_RESULTS_KO.md')):
        raw = path.read_bytes().decode('utf-8'); assert header not in raw; nl = '\r\n' if '\r\n' in raw else '\n'; title, body = raw.split(nl+nl, 1)
        path.write_bytes((title+nl+nl+(block+'\n- [실제 마스크 학습·사진 전용 결과]('+link+')\n').replace('\n', nl)+nl+body).encode('utf-8'))


def package():
    choice = checked(); assert not STAGE.exists() and not ARCHIVE.exists(); STAGE.mkdir()
    shutil.copy2(ROOT/choice['checkpoint_path'], STAGE/'model.pt'); shutil.copy2(RUN/'evaluation.json', STAGE/'evaluation.json')
    write(STAGE/'model.json', {'path': 'model.pt', 'sha256': choice['checkpoint_sha256'], 'checkpoint_count': 1, 'backbone_count': 1,
        'ksdd_photo_only': True, 'primary_original_location_preserved': True, 'auxiliary_localization_worse_FNR': True,
        'actual_prior_real_mask_optimizer_updates': 2560, 'selected_epoch_by_validation': 13, 'new_composition_updates': 0,
        'known_TEST_reused_for_followup_design': True, 'independent_field_validated': False, 'app_default_promoted': False})
    write(STAGE/'safelog_ai/__init__.py', '')
    previous = ROOT/'runs/industrial-surfaces-v9/handoff-stage'
    for name in [p.name for p in (previous/'safelog_ai').glob('*.py') if p.name != '__init__.py']+['ksdd_supervised.py', 'ksdd_photo_only.py', 'ksdd_photo_only_server.py']:
        shutil.copy2(ROOT/'safelog_ai'/name, STAGE/'safelog_ai'/name)
    shutil.copy2(previous/'requirements-api.txt', STAGE/'requirements-api.txt')
    write(STAGE/'README_KO.md', '# SafeLog 부품 표면 사진 미탐 보강\n\n모델 하나·공통 분석부 하나. 사진 점수만 보강하고 기본 위치 출력은 유지한다.\n\n'
        '- [역할4 실행·출력](safelog-cpp/docs/KSDD_PHOTO_HANDOFF_KO.md)\n- [실제 학습·전체 제외·사진 전용 채택](safelog-cpp/ai-training/reports/KSDD_PHOTO_ONLY_RESULTS_KO.md)\n')
    queue = [REPO/'safelog-cpp/docs/KSDD_PHOTO_HANDOFF_KO.md', REPORTS/'KSDD_PHOTO_ONLY_RESULTS_KO.md']; seen = set()
    while queue:
        source = queue.pop().resolve()
        if source in seen: continue
        assert source.is_file() and source.is_relative_to(REPO); seen.add(source)
        target = STAGE/source.relative_to(REPO); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        if source.suffix == '.md': queue.extend(local_links(source))
    for path in STAGE.rglob('*.md'):
        for link in local_links(path): assert link.is_file() and link.is_relative_to(STAGE)
    write(STAGE/'package-sha256.json', {p.relative_to(STAGE).as_posix(): sha(p) for p in STAGE.rglob('*') if p.is_file()})
    with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for path in sorted(STAGE.rglob('*')):
            if path.is_file(): z.write(path, path.relative_to(STAGE).as_posix())
    print({'archive': str(ARCHIVE), 'SHA': sha(ARCHIVE), 'bytes': ARCHIVE.stat().st_size}, flush=True)


def smoke():
    choice = checked(); assert not ISOLATED.exists()
    with socket.socket() as s: assert s.connect_ex(('127.0.0.1', 8775)) != 0
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist()) == 1
        for name in z.namelist(): assert (ISOLATED/name).resolve().is_relative_to(ISOLATED.resolve())
        z.extractall(ISOLATED)
    digest = choice['checkpoint_sha256']; assert sha(ISOLATED/'model.pt') == digest
    old = study.read(ROOT/'data/unified-main-v1/manifest.json'); rows = study.read(trained.previous.MANIFEST)['rows']
    photos = {'facility_surface': ROOT/old['fit']['facility_surface'][0]['image'], 'belt_damage': ROOT/'runs/belt-author-embedded-sample.jpg', 'conveyor_foreign': ROOT/old['fit']['conveyor_foreign'][0]['image']}
    photos.update({c: ROOT/next(r['image'] for r in rows if r['phase'] == 'fit' and r['category'] == c) for c in CATEGORIES})
    data = study.read(ROOT/'data/industrial-bsclass-v1/manifest.json')['rows']; photos[WEAR_CATEGORY] = ROOT/next(r['image'] for r in data if r['phase'] == 'fit' and not r['label'])
    point = study.read(trained.RUN/'calibration.json')['point']; expected = next(r for r in study.read(study.VALIDATION)['observations'] if r['label'] and r['score'] > point['photo_threshold']+.2)
    photos['ksdd2_surface'] = ROOT/next(r['image'] for r in rows if r['id'] == expected['id'])
    env = dict(os.environ); env.pop('PYTHONPATH', None); env.update(PYTHONIOENCODING='utf-8', QT_QPA_PLATFORM='offscreen')
    env['PATH'] = 'C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH', '')
    log = (ISOLATED/'server.log').open('x', encoding='utf-8')
    process = subprocess.Popen([sys.executable, '-m', 'safelog_ai.ksdd_photo_only_server', '--checkpoint', 'model.pt', '--sha256', digest, '--evaluation', 'evaluation.json', '--port', '8775'], cwd=ISOLATED, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    proofs = []; replies = []
    try:
        deadline = time.monotonic()+50
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError((ISOLATED/'server.log').read_text())
            try: health = http('/health'); break
            except (OSError, ValueError): time.sleep(.25)
        else: raise RuntimeError('Packaged CPU server timed out')
        assert health['modelSha256'] == digest and health['neuralModelCount'] == health['sharedBackboneCount'] == 1
        policies = http('/v1/inference-policy'); assert policies['supervisedPhoto']['ksdd2_surface']['originalLocalizationChanged'] is False
        categories = http('/v1/equipment-categories')['categories']; assert len(categories) == 12 and sum(c['benchmarkStage']['ready_for_next_stage'] for c in categories) == 8
        old_replies = {r['task']: r['result'] for r in study.read(REPORTS/'industrial-night-v9-release.json')['hazard_replies']}
        for task, photo in photos.items():
            encoded = base64.b64encode(photo.read_bytes()).decode(); reply = http('/v1/analyze-hazard', {'inspectionTask': task, 'image': encoded, 'promptVersion': 'ksdd-photo-only-v11-smoke'})
            assert reply['modelSha256'] == digest and reply['inputSha256'] == sha(photo) and reply['neuralModelCount'] == reply['sharedBackboneCount'] == 1
            assert reply['humanConfirmationRequired'] and not reply['fieldPerformanceValidated']
            if task == 'ksdd2_surface':
                assert reply['candidateDetected'] and abs(reply['classResults'][0]['score']-expected['score']) < 2e-4
                from safelog_ai.industrial_batch_server import IndustrialEngine
                parent = IndustrialEngine(trained.START, trained.EXPECTED, trained.PARENT_EVALUATION); prior = parent.analyze(task, encoded)
                assert reply['heatmapPngBase64'] == prior['heatmapPngBase64'] and not reply['originalLocalizationChanged']
                assert reply['auxiliaryLocalizationWorseFNR'] and not reply['auxiliaryIsPrimaryLocationOutput']
                for key in ('heatmapPngBase64', 'supervisedAuxiliaryHeatmapPngBase64', 'supervisedAuxiliaryMaskPngBase64'):
                    with Image.open(io.BytesIO(base64.b64decode(reply[key]))) as im: assert im.size == (reply['inputWidth'], reply['inputHeight'])
                del parent
            elif 'inferencePolicySha256' in old_replies[task]: assert reply['inferencePolicySha256'] == old_replies[task]['inferencePolicySha256'], task
            for key in [k for k in reply if k.endswith('PngBase64')]: reply.pop(key)
            replies.append({'task': task, 'result': reply})
            if task in ('ksdd2_surface', 'ball_screw_pitting', 'mpdd_bracket_white'):
                output = ISOLATED/(task+'-cpp.json')
                cp = subprocess.run([str(trained.previous.RUN/'qt-probe/build/unified_probe.exe'), task, str(photo), digest, str(output)], cwd=ISOLATED, env=env, capture_output=True, timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
                assert cp.returncode == 0 and output.exists(), cp.stderr.decode(errors='replace')
                proof = study.read(output); assert proof['operations'] == 3 and proof['samePhotoDoesNotResolve']; proofs.append(proof)
            print({'CPU_alias_checked': task}, flush=True)
        buffer = io.BytesIO(); Image.new('RGB', (32, 32), 'white').save(buffer, format='PNG')
        try: http('/v1/analyze-hazard', {'inspectionTask': 'ksdd2_surface', 'image': base64.b64encode(buffer.getvalue()).decode(), 'promptVersion': 'too-small'})
        except HTTPError as error: assert error.code == 422
        else: raise AssertionError('Too-small input must require review')
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        log.close()
    with zipfile.ZipFile(ARCHIVE) as z:
        hashes = json.loads(z.read('package-sha256.json')); assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in hashes.items())
    write(REPORTS/'industrial-ksdd-photo-only-v11-release.json', {'status': 'passed', 'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
        'model_sha256': digest, 'all15_CPU_task_aliases_checked': True, 'actual_current_CPP_operations': 9, 'CPP_cases': proofs,
        'prior_v9_CPP45_not_counted_as_current': True, 'primary_KSDD_heatmap_PNG_exact_to_prior_CPU': True,
        'all_other_exposed_policy_SHA_preserved': True, 'auxiliary_worse_localization_explicit': True, 'old_and_auxiliary_masks_original_sizes_checked': True,
        'too_small_input_rejected422': True, 'hazard_replies': replies, 'ZIP_CRC_members_SHA_links_verified': True,
        'owned_server_stopped': process.poll() is not None, 'one_checkpoint_one_backbone': True, 'fresh_pip_install_tested': False, 'APK_built': False, 'independent_field_validated': False})
    print({'release_passed': True, 'actual_CPP_operations': 9, 'CPU_aliases': 15}, flush=True)


def cpu():
    # Import the immutable extracted ZIP; dataset paths remain external read-only inputs.
    import importlib
    script = RUN/'cpu-whole-test.py'
    source = (ROOT/'scripts/verify_ksdd_photo_cpu.py').read_text(encoding='utf-8')
    with script.open('x', encoding='utf-8') as out: out.write(source)
    cp = subprocess.run([sys.executable, str(script), str(ROOT)], cwd=ISOLATED, capture_output=True, text=True, encoding='utf-8', timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    with (RUN/'cpu-whole-test.log').open('x', encoding='utf-8') as out: out.write(cp.stdout+'\n'+cp.stderr)
    assert cp.returncode == 0, cp.stdout[-3000:]+cp.stderr[-3000:]
    print(cp.stdout[-5000:], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['docs', 'package', 'smoke', 'cpu']); args = parser.parse_args(); globals()[args.mode]()
