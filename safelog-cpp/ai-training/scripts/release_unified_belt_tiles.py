"""Measured one-model native-view study; release the selected weights AND policy."""
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
from urllib.request import Request, urlopen
import zipfile
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.study_unified_belt_tiles import BASE, PROTOCOL, CONFIGS, VIEWS
from scripts.unified_main_data import DATA, read, sha
from scripts.release_unified_main import local_links, MODULES
REPO = ROOT.parents[1]; REPORTS = ROOT/'reports'; OUTPUTS = REPO.parents[1]/'outputs'
NAME = 'unified-main-belt-tiles-role4-handoff-2026-10-13-v1'
STAGE = BASE/'handoff-stage'; ARCHIVE = OUTPUTS/(NAME+'.zip'); ISOLATED = BASE/'isolated-zip'


def new(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('x', encoding='utf-8', newline='\n') as out: out.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def pct(v): return '미측정' if v is None else f'{100*v:.2f}%'


def report():
    choice = read(REPORTS/'unified-belt-tiles-selection.json'); verified = read(REPORTS/'unified-belt-tiles-verification.json')
    p = read(PROTOCOL); t = read(REPORTS/'unified-belt-tiles-training.json'); views = read(VIEWS/'manifest.json')
    results = {c: read(REPORTS/f'unified-belt-tiles-{c}-validation.json') for c in CONFIGS}
    lines = ['# 통합 메인 · 원본 구역 학습과 분석 비교', '', '기준일: 2026-10-13', '', '## 실제 수행', '',
        '직전 native_mix 통합 모델에서 벨트 출력만 원본 구역 사진으로 실제 추가 학습했다. 공통 분석부·시설·이물질 가중치와 모든 BN 통계는 정확히 고정했다. 전체 사진과 구역 사진에 같은 모델 하나를 반복 사용하며 여러 모델을 묶은 ensemble이 아니다.', '',
        f"- **실제 추가10epoch·1,280회·{t['elapsed_training_minutes']:.2f}분**. 세 분석 설정을 비교하지만 추가 신경망 학습을3회 했다는 뜻이 아니다.",
        f"- 작성자 TRAIN1,397장의 원본·라벨·기존 전처리를 확인했다. 미리 정한 순서에서 사용할 파생 구역은{views['derived_scheduled_views']:,}개이며 해당 구역의 작성자 mask가 비어 있는 경우는{views['empty_author_mask_views']:,}개다. 이는 그 구역의 작성자 pixel 정답이며 전체 사진 정상/설비 정상 정답이 아니다.",
        '- 새 독립 사진·새 정답은0개다. 배치8장 중4장은 기존 전체 사진,4장은 원본 구역이다. 추가 배경 penalty·교사를 넣지 않고 기존 BCE posweight20+Dice를 사용했다. 같은 원본 반복 추출을 새 사진으로 세지 않는다.',
        '- 구역은 사진 크기로만 정한 최대6개 정사각형이다. 긴 변3개·짧은 변2개로 균등 배치하며 전체 사진을 덮는다. 손상 도형·예측 점수로 구역을 고르지 않는다. 학습 마스크는 원래 작성자 도형을 같은 좌표로 변환한다.',
        '- 구역 분석은 각 구역 확률을 원본 좌표에서 Hann 가중평균으로 합친 뒤 기존640 유효 영역으로 옮긴다. 전체 확률50%와 구역 확률50%를 고정 평균하며 기준0.5는 그대로다. 작은 사진과3:1초과 비율은 전체 사진 분석으로 fallback한다.',
        '- 마지막10회차만 평가했다. 각64회마다 재개 상태와 journal을 저장했다. epoch별VAL튜닝·예약TEST사용은 없다.', '',
        '| 비교 설정 | 가중치 | 이번 실제 학습 | 벨트 분석 |', '|---|---|---:|---|',
        '| 기준 native_mix | 직전 가중치 | 0회 | 전체 사진만 |',
        '| accepted_tiles | 직전과 동일 | 0회 | 전체 + 최대6구역 |',
        '| trained_global | 이번 한 후보 | 1,280회 | 전체 사진만 |',
        '| trained_tiles | trained_global과 동일 | 위1,280회 공유 | 전체 + 최대6구역 |', '',
        '## 같은 알려진 VAL353장·4개 촬영 구간', '',
        '작성자 원본 입력을 사용하되 평가 마스크와 분모는 기존640 raster grid를 유지했다. 구역을 고르거나 합치는 함수에는 정답을 넘기지 않는다. 알려진VAL을 재사용하는 연구 비교이며 독립 현장 시험이 아니다.', '',
        '| 지표 | 기준 native_mix | accepted_tiles | trained_global | trained_tiles |', '|---|---:|---:|---:|---:|']
    combined = [p['baseline'], *(results[c]['summary'] for c in CONFIGS)]
    for label, get in [('사진 미탐', lambda r: pct(r['photo_rates']['fnr'])), ('사진 미탐 건수', lambda r: f"{r['photo_rates']['fn']}/{r['photo_rates']['positive_photos']}"),
                       ('손상 pixel 미탐', lambda r: pct(r['pixel_fnr'])), ('배경 pixel 오탐', lambda r: pct(r['pixel_fpr'])),
                       ('위치 IoU', lambda r: pct(r['pixel_iou'])), ('정답 위치와 겹침 없는 손상 사진', lambda r: str(r['positive_photos_without_any_gt_overlap'])),
                       ('정상 사진 오탐', lambda r: '미측정·정상0장')]: lines.append('| '+label+' | '+' | '.join(get(r) for r in combined)+' |')
    lines += ['', '픽셀 오탐과 정상 사진 오탐은 다른 지표다. 사진에서 후보를 찾더라도 정답 위치를 놓칠 수 있어서 위치IoU·pixel미탐·정답겹침실패도 함께 본다.', '',
        '## 사전 선택 조건', '',
        f"선택: **{choice['selected']}**. IoU 최소0.5%p 증가, 배경오탐·사진미탐·pixel미탐·정답겹침실패가 모두 기준보다 나빠지지 않아야 한다. 통과 설정은 IoU·낮은오탐·전체사진 방식 순으로 선택한다. 실패하면 직전 native_mix를 유지한다.", '',
        '| 설정 | IoU +0.5%p | 배경 오탐 유지 | 사진 미탐 유지 | pixel 미탐 유지 | 겹침 실패 유지 |', '|---|---|---|---|---|---|']
    for c in CONFIGS: lines.append('| '+c+' | '+' | '.join('통과' if v else '실패' for v in choice['guards'][c].values())+' |')
    lines += ['', '## 연결·보존·한계', '',
        f"기존 입력·모델/보고서{verified['protected_files']:,}개와 Python소스{verified['legacy_sources']}개를 SHA로 보존했다. 실제 journal은{verified['actual_journal_updates']:,}회이고 최종 가중치는 마지막progress와 동일하다. 모든 비벨트tensor는 직전과 정확히 같고 기존 시설·이물질 CPU 예측 및3기능 CPU/GPU, 원본 구역 CPU/GPU 일치를 확인했다.", '',
        '분석 설정은 가중치와 별개이므로 modelSha256과 inferencePolicySha256을 함께 기록한다. 동일한 가중치라도 구역 분석 방식이 다르면 정책SHA가 다르다. 기존 앱 기본 모델은 자동 교체하지 않는다.', '',
        '정상벨트사진정답0장으로 정상 사진 오탐은 미측정이며 산업현장 전체5% 미만을 검증하지 못했다. 시설·이물질 수치는 기존 모델에서 계승하며 새시험이 아니다. 내부 모터·롤러 고장·가동 여부는 not_assessed다. 위험 등급·조치·요약은 기존 규칙 템플릿이고 점검자 확인이 필요하다.', '',
        '![실제 구역 분석 비교](unified-belt-tiles-comparison.png)', '',
        '- [사전 조건](unified-belt-tiles-protocol.json)', '- [실제 학습 기록](unified-belt-tiles-training.json)',
        '- [선택 조건별 결과](unified-belt-tiles-selection.json)', '- [학습·재개·보존 검증](unified-belt-tiles-verification.json)',
        '- [같은 가중치의 구역 분석 결과](unified-belt-tiles-accepted_tiles-validation.json)',
        '- [학습 후 전체사진 결과](unified-belt-tiles-trained_global-validation.json)',
        '- [학습 후 구역 분석 결과](unified-belt-tiles-trained_tiles-validation.json)',
        '- [실행 파이프라인](../../docs/UNIFIED_BELT_TILES_STUDY_KO.md)',
        '- [기준 native_mix 성능](UNIFIED_BELT_DETAIL_RESULTS_KO.md)', '- [다른 기능 성능](UNIFIED_MAIN_RESULTS_KO.md)',
        '- [출처·사용 조건](UNIFIED_MAIN_ATTRIBUTION_KO.md)', '']
    new(REPORTS/'UNIFIED_BELT_TILES_RESULTS_KO.md', '\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 5)); labels = ['Prior global', 'Prior + tiles', 'Trained global', 'Trained + tiles']
    for axis, key, title in [(axes[0], 'pixel_iou', 'Pixel IoU (higher is better)'), (axes[1], 'pixel_fnr', 'Damage pixel FNR'),
                             (axes[2], 'pixel_fpr', 'Background pixel FPR'), (axes[3], 'photo', 'Photo FNR')]:
        ys = [100*(r['photo_rates']['fnr'] if key == 'photo' else r[key]) for r in combined]
        axis.bar(labels, ys, color=['#607684', '#a075b4', '#3e92bd', '#47864d']); axis.set_title(title, fontsize=10)
        axis.set_ylabel('%'); axis.set_ylim(0, max(ys)*1.25+.15); axis.tick_params(axis='x', labelrotation=18); axis.spines[['top', 'right']].set_visible(False)
        for x, y in enumerate(ys): axis.text(x, y+max(ys)*.025, f'{y:.2f}%', ha='center', fontsize=9)
    fig.suptitle('ONE model: native uniform views, real +10 epochs, known reused VAL353', fontsize=14)
    fig.text(.5, .02, 'Normal photo FPR unavailable; annotation-free inference. Selection: '+choice['selected'], ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, .92)); fig.savefig(REPORTS/'unified-belt-tiles-comparison.png', dpi=150); plt.close(fig)
    print(json.dumps({'report': 'created', 'selected': choice['selected']}), flush=True)

def package():
    choice = read(REPORTS/'unified-belt-tiles-selection.json')
    if choice['selected'] == 'prior_native_mix':
        new(REPORTS/'unified-belt-tiles-release.json', {'status': 'prior_native_mix_retained', 'new_package': None,
            'selection': choice, 'app_default_promoted': False}); print('Prior native mix retained; no new handoff recommended', flush=True); return
    assert not STAGE.exists() and not ARCHIVE.exists(); STAGE.mkdir()
    checkpoint = ROOT/choice['checkpoint_path']; assert sha(checkpoint) == choice['checkpoint_sha256']; shutil.copy2(checkpoint, STAGE/'model.pt')
    new(STAGE/'model.json', {'path': 'model.pt', 'sha256': sha(checkpoint), 'researchVariant': choice['selected'],
        'apiArchitectureFamily': 'unified-main-v1', 'checkpoint_count': 1, 'backbone_count': 1,
        'inspectionTasks': ['facility_surface', 'belt_damage', 'conveyor_foreign'], 'base_epochs': 20,
        'accepted_native_belt_epochs': 10, 'additional_uniform_belt_epochs': 10 if choice['selected_new_neural_updates'] else 0, 'additional_optimizer_updates': choice['selected_new_neural_updates'], 'inferenceMode': choice['mode'], 'inferencePolicySha256': choice['inference_policy']['sha256'], 'non_belt_parameters_exactly_v1': True,
        'app_default_promoted': False, 'fieldPerformanceValidated': False})
    new(STAGE/'requirements-api.txt', 'fastapi==0.142.1\nuvicorn==0.54.0\npydantic==2.13.5\npillow==12.3.0\nnumpy==2.5.2\n')
    new(STAGE/'safelog_ai/__init__.py', '')
    for name in (*MODULES, 'belt_tiles.py', 'unified_tiles_server.py'): shutil.copy2(ROOT/'safelog_ai'/name, STAGE/'safelog_ai'/name)
    new(STAGE/'README_KO.md', '''# SafeLog 벨트 부분 추가 학습 · 역할4 전달

모델 파일 하나로 시설·벨트 손상·이물질 항목을 처리한다. 선택한 벨트 연구 후보와 CPU API·기존 C++ 연결을 제공한다.

- [연결 방법](safelog-cpp/docs/UNIFIED_BELT_TILES_HANDOFF_KO.md)
- [개선·악화·실제 결과](safelog-cpp/ai-training/reports/UNIFIED_BELT_TILES_RESULTS_KO.md)
- [출처와 연구·교육 사용 조건](safelog-cpp/ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md)

정상 벨트 사진 오탐은 미측정이며 독립 현장 5% 검증과 전체 설비 내부 고장 진단을 제공하는 모델이 아니다. 앱 기본 모델은 자동 교체하지 않았다.
''')
    guide = f'''# 역할4 · 벨트 출력 추가 학습 통합 모델

연구 후보 `{choice['selected']}`. 파일 하나·공통 분석부 하나·3개 출력이며 기존 시설·이물질 가중치는 기준 native_mix 및 원래 v1과 정확히 동일하다. 모델 파일 하나를 전체 사진과 구역에 반복 사용한다. 분석모드는model.json과정책SHA로구분한다. 모델 SHA는 `{choice['checkpoint_sha256']}`다. API modelName은 구조 계열 unified-main-v1이며 실제 학습 버전은 modelSha256 및 model.json으로 구분한다.

## CPU 실행

`{NAME}.zip`을 새 폴더에 풀고 Python 3.12에서 실행한다. 제작 환경의 실제 실행과 새 환경의 pip 설치 시험은 구분한다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$beltTilesModelRecord = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.unified_tiles_server --checkpoint $beltTilesModelRecord.path --sha256 $beltTilesModelRecord.sha256 --mode $beltTilesModelRecord.inferenceMode --port 8775
```

서버는127.0.0.1에서 실행한다. 실제 스마트폰은 별도 서버 주소·인증·HTTPS를 팀에서 구성한다. Android APK를 만든 전달물이 아니다.

## C++ 앱 연결

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="belt_damage"
```

[기존 어댑터 설정](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp)과 [요청 코드](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp)를 사용한다. 점검 항목은 facility_surface·belt_damage·conveyor_foreign이다. 환경변수는 실행 모드를 선택하며 촬영 화면의 항목 선택 UI·DB 저장은 팀 앱 흐름에 연결한다.

- GET /health: ready, neuralModelCount=1, sharedBackboneCount=1, modelSha256 확인.
- GET /v1/inference-policy: 분석정책과sha256 확인. modelSha256과inferencePolicySha256을함께DB/보고서에기록한다.\n- POST /v1/analyze-hazard: image, inspectionTask, userMemo, promptVersion.
- POST /v1/compare-action: beforeImage, afterImage, inspectionTask, actionNote, promptVersion.
- POST /v1/summarize: inspectionContext, promptVersion.

점검 항목 누락·미지원은422다. 요청하지 않은 다른 항목은 정상으로 판단하지 않는다. 벨트 heatmap은 원본 크기 확인 자료이며 확정 결함 위치가 아니다. 점수는 보정된 고장 확률이 아니고 같은 사진으로 수리 완료가 나오면 안 된다. 내부 기계 고장은 not_assessed다. 위험 등급·조치·요약은 규칙 템플릿이며 점검자 최종 확인이 필요하다.

[실제 학습 성능](../ai-training/reports/UNIFIED_BELT_TILES_RESULTS_KO.md), [기존 다른 기능의 수치](../ai-training/reports/UNIFIED_MAIN_RESULTS_KO.md), [출처·사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md)을 함께 전달한다. 기존 앱 기본 모델은 자동 교체하지 않았다.
'''
    new(REPO/'safelog-cpp/docs/UNIFIED_BELT_TILES_HANDOFF_KO.md', guide)
    queue = [REPORTS/'UNIFIED_BELT_TILES_RESULTS_KO.md', REPO/'safelog-cpp/docs/UNIFIED_BELT_TILES_HANDOFF_KO.md']
    queue += [p for p in REPORTS.glob('unified-belt-tiles-*') if p.suffix in ('.json', '.png')]
    queue += [REPO/'safelog-cpp/apps/mobile-qt/tools/unified_main_probe.cpp', REPO/'safelog-cpp/src/ai/include/safelog/ai/ai_analyzer.hpp']
    seen = set()
    while queue:
        source = queue.pop().resolve()
        if source in seen: continue
        assert source.is_relative_to(REPO) and source.is_file(), source
        seen.add(source); target = STAGE/source.relative_to(REPO); target.parent.mkdir(exist_ok=True, parents=True); shutil.copy2(source, target)
        if source.suffix == '.md': queue.extend(local_links(source))
    for doc in STAGE.rglob('*.md'):
        for link in local_links(doc): assert link.is_relative_to(STAGE) and link.is_file(), link
    new(STAGE/'package-sha256.json', {p.relative_to(STAGE).as_posix(): sha(p) for p in STAGE.rglob('*') if p.is_file()})
    with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for file in sorted(STAGE.rglob('*')):
            if file.is_file(): archive.write(file, file.relative_to(STAGE).as_posix())
    print(json.dumps({'archive': str(ARCHIVE), 'sha256': sha(ARCHIVE), 'bytes': ARCHIVE.stat().st_size}), flush=True)


def http(path, body=None):
    request = Request('http://127.0.0.1:8775'+path, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=30) as response: return json.load(response)


def smoke():
    choice = read(REPORTS/'unified-belt-tiles-selection.json')
    if choice['selected'] == 'prior_native_mix': return
    assert not ISOLATED.exists()
    with socket.socket() as port: assert port.connect_ex(('127.0.0.1', 8775)) != 0, '8775 occupied; preserve unrelated server'
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None and sum(name.endswith('.pt') for name in archive.namelist()) == 1
        for name in archive.namelist(): assert (ISOLATED/name).resolve().is_relative_to(ISOLATED)
        archive.extractall(ISOLATED)
    digest = choice['checkpoint_sha256']; assert sha(ISOLATED/'model.pt') == digest
    manifest = read(DATA/'manifest.json'); belt = ROOT/'runs/belt-author-embedded-sample.jpg'
    assert any(r['embedded_image_sha256'] == sha(belt) for r in manifest['fit']['belt_damage'])
    photos = {'facility_surface': ROOT/next(r['image'] for r in manifest['fit']['facility_surface'] if r['targets'][0] == 1),
              'belt_damage': belt, 'conveyor_foreign': ROOT/next(r['image'] for r in manifest['fit']['conveyor_foreign'] if r['label'] == 1)}
    env = dict(os.environ); env.pop('PYTHONPATH', None); env['PYTHONIOENCODING'] = 'utf-8'; env['QT_QPA_PLATFORM'] = 'offscreen'
    env['PATH'] = 'C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH', '')
    log = (ISOLATED/'server.log').open('x', encoding='utf-8')
    process = subprocess.Popen([sys.executable, '-m', 'safelog_ai.unified_tiles_server', '--checkpoint', 'model.pt', '--sha256', digest, '--mode', choice['mode'], '--port', '8775'],
        cwd=ISOLATED, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    hazards, proofs = [], []
    try:
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError('Packaged CPU server exited')
            try: health = http('/health'); break
            except (OSError, ValueError): time.sleep(.25)
        else: raise RuntimeError('Packaged CPU server timeout')
        assert health['ready'] and health['neuralModelCount'] == health['sharedBackboneCount'] == 1 and health['modelSha256'] == digest
        policy_reply = http('/v1/inference-policy')
        assert policy_reply['sha256'] == choice['inference_policy']['sha256'] and policy_reply['beltMode'] == choice['mode']
        for task, path in photos.items():
            reply = http('/v1/analyze-hazard', {'image': base64.b64encode(path.read_bytes()).decode(), 'inspectionTask': task, 'promptVersion': 'belt-tiles-release-v1'})
            assert reply['inferencePolicySha256'] == choice['inference_policy']['sha256'] and reply['configuredBeltInferenceMode'] == choice['mode']
            assert reply['nativeDetailViewsEvaluated'] == (6 if task == 'belt_damage' and choice['mode'] == 'global_tile_mean' else 0)
            assert reply['modelSha256'] == digest and reply['inputSha256'] == sha(path) and reply['humanConfirmationRequired']
            assert not reply['fieldPerformanceValidated'] and not reply['otherTasksEvaluated'] and not reply['beltPhotoFprValidated']
            assert reply['mechanicalFailureState'] == 'not_assessed'
            if task == 'belt_damage':
                from PIL import Image
                with Image.open(io.BytesIO(base64.b64decode(reply['heatmapPngBase64']))) as image: assert image.size == (reply['inputWidth'], reply['inputHeight'])
            else: assert reply['heatmapPngBase64'] is None
            reply.pop('heatmapPngBase64'); hazards.append(reply)
            output = ISOLATED/(task+'-cpp.json')
            result = subprocess.run([str(ROOT/'runs/unified-main-qt-probe/build/unified_probe.exe'), task, str(path), digest, str(output)],
                cwd=ISOLATED, env=env, timeout=45, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            assert result.returncode == 0 and output.is_file(), result.stderr.decode(errors='replace')
            proof = read(output); assert proof['status'] == 'passed' and proof['modelSha256'] == digest and proof['operations'] == 3
            proofs.append(proof); print(task+': actual CPU/Qt HTTP passed', flush=True)
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        log.close()
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = json.loads(archive.read('package-sha256.json'))
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == digest for name, digest in members.items())
    for doc in ISOLATED.rglob('*.md'):
        for link in local_links(doc): assert link.is_relative_to(ISOLATED) and link.is_file()
    new(REPORTS/'unified-belt-tiles-release.json', {'status': 'passed', 'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE),
        'archive_bytes': ARCHIVE.stat().st_size, 'model_sha256': choice['checkpoint_sha256'], 'selected': choice['selected'],
        'mode': choice['mode'], 'inference_policy_sha256': choice['inference_policy']['sha256'], 'checkpoint_count': 1, 'backbone_count': 1, 'isolated_zip_CPU_HTTP': True, 'actual_CPP_http_operations': 9,
        'cpp_proofs': proofs, 'hazard_responses': hazards, 'all_smoke_photos_are_TRAIN': True,
        'zip_crc_member_SHA_and_local_links_verified': True, 'owned_server_stopped': process.poll() is not None,
        'fresh_pip_install_tested': False, 'all_industrial_tasks_5percent_verified': False, 'app_default_promoted': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['report', 'package', 'smoke'])
    globals()[parser.parse_args().mode]()
