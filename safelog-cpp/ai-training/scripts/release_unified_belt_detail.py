"""Honest measured report and optional one-model handoff for the native crop study."""
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
from scripts.study_unified_belt_detail import BASE, PROTOCOL, VARIANTS
from scripts.unified_main_data import DATA, read, sha
from scripts.release_unified_main import local_links, MODULES
REPO = ROOT.parents[1]; REPORTS = ROOT/'reports'; OUTPUTS = REPO.parents[1]/'outputs'
NAME = 'unified-main-belt-detail-role4-handoff-2026-10-13-v1'
STAGE = BASE/'handoff-stage'; ARCHIVE = OUTPUTS/(NAME+'.zip'); ISOLATED = BASE/'isolated-zip'


def new(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('x', encoding='utf-8', newline='\n') as out:
        out.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def pct(value): return '미측정' if value is None else f'{100*value:.2f}%'


def report():
    choice = read(REPORTS/'unified-belt-detail-selection.json'); verified = read(REPORTS/'unified-belt-detail-verification.json')
    protocol = read(PROTOCOL); baseline = protocol['baseline']
    results = {v: read(REPORTS/f'unified-belt-detail-{v}-validation.json')['summary'] for v in VARIANTS}
    training = {v: read(REPORTS/f'unified-belt-detail-{v}-training.json') for v in VARIANTS}
    lines = ['# 통합 메인 · 벨트 원본 손상 부위 추가 학습 결과', '', '기준일: 2026-10-13', '',
        '## 실제 수행', '',
        '동일한 완료 통합 v1에서 벨트 출력 부분만 실제 추가 학습했다. 공통 MobileNet 분석부·시설 출력·이물질 출력의 모든 tensor와 BN 통계는 정확히 유지했다. 시설·벨트·이물질을 처리하는 기존 모델 파일 하나의 구조를 그대로 사용한다.', '',
        f"- 후보마다 **10epoch·1,280회**, 합계 **20후보 epoch·2,560회** 실제 추가 업데이트. 두 후보의 journal은 합계{sum(v['actual_journal_updates'] for v in verified['models'].values()):,}회이며 완료 가중치에 반영된 업데이트와 재실행 기록을 구분한다.",
        f"- 작성자 TRAIN 1,397장에서 원본 crop 1,397개를 만들고 검증했다. 실제 학습에 예약한 고유 TRAIN은{protocol['unique_scheduled_TRAIN']:,}장이다. 새 독립 사진·새 AI 정답은 0개다.",
        '- full_control은 기존 전체 640 사진만 학습했다. native_mix는 같은 ID·순서에서 배치 8장 중 4장을 원본 손상 부위 crop으로 바꿨다. 손실·학습률·추출 수·좌우 반전·seed는 같다.',
        '- 원본 JSON·포함 사진 SHA와 디코딩 RGB를 검증하고 기존 전체 사진·마스크·유효 영역의 픽셀 일치를 모두 확인했다. native crop은 원본에서 자른 것으로 기존 640 사진 확대와 다르다.',
        '- 첫 작성자 손상 도형 중심 주변을 최소 720 native pixel 정사각형으로 잘라 640으로 바꿨다. 모든 원래 도형을 같은 좌표로 변환하고 crop 경계에서 잘랐다. 전체 사진과 crop을 섞어 같은 전체 사진 평가 조건을 유지했다.',
        '- 모든 회차에 다른 기능 가중치가 바뀌지 않았음을 확인했다. 64회마다 재개 상태를 저장했다. 마지막 10회차만 평가하고 epoch별 VAL 선택은 하지 않았다.', '',
        '| 후보 | 실제 추가 epoch | 업데이트 | 학습·가중치 검사 시간 | 원본 crop 추출 | 전체 사진 추출 |', '|---|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        t, proof = training[v], verified['models'][v]
        lines.append(f"| {v} | {t['epochs']} | {t['optimizer_updates']:,} | {t['elapsed_training_minutes']:.2f}분 | {proof['native_crop_draws']:,} | {proof['fullframe_draws']:,} |")
    lines += ['', '## 같은 알려진 VAL353장·4개 촬영 구간', '',
        '평가 입력은 crop이 아닌 기존 전체 사진 640이며 손상 기준은 모두 0.5다. 예약된 작성자 TEST는 사용하지 않았다. 재사용된 VAL 연구 비교이며 독립 산업현장 검증이 아니다.', '',
        '| 지표 | 통합 v1 | full_control | native_mix |', '|---|---:|---:|---:|']
    combined = [baseline, results['full_control'], results['native_mix']]
    for title, accessor in [('사진 미탐률', lambda r: pct(r['photo_rates']['fnr'])), ('사진 미탐 건수', lambda r: f"{r['photo_rates']['fn']}/{r['photo_rates']['positive_photos']}"),
                            ('손상 위치 pixel IoU', lambda r: pct(r['pixel_iou'])), ('손상 pixel 미탐률', lambda r: pct(r['pixel_fnr'])),
                            ('background pixel 오탐률', lambda r: pct(r['pixel_fpr'])),
                            ('정답 위치와 겹침 없는 손상 사진', lambda r: str(r['positive_photos_without_any_gt_overlap'])),
                            ('정상 사진 오탐률', lambda r: '미측정·정상 0장')]:
        lines.append('| '+title+' | '+' | '.join(accessor(r) for r in combined)+' |')
    lines += ['', '사진 미탐은 사진 전체에서 후보가 전혀 검출되지 않은 비율이다. 정답 위치와 겹침 없는 사진은 다른 위치만 표시하는 경우도 포함한다. 위치 IoU와 pixel 미탐을 별도로 제시하며 background pixel 오탐을 정상 사진 오탐으로 발표하지 않는다.', '',
        '## 사전 조건에 따른 연구 선택', '',
        f"선택: **{choice['selected']}**. 사전 기준은 위치 IoU 최소 0.5%p 개선, 사진 미탐·pixel 미탐·정답 겹침 없는 사진 수가 v1보다 나빠지지 않는 것이다. 모두 통과한 후보에서 IoU, pixel 미탐, control 순으로 선택했다. 두 후보가 실패하면 기존 v1을 유지한다.", '',
        '| 후보 | IoU +0.5%p 이상 | 사진 미탐 유지 | pixel 미탐 유지 | 위치 겹침 실패 유지 |', '|---|---|---|---|---|']
    for v in VARIANTS: lines.append('| '+v+' | '+' | '.join('통과' if ok else '실패' for ok in choice['guards'][v].values())+' |')
    lines += ['', '## 다른 기능과 앱 연결', '',
        '시설·이물질·공통 분석부의 모든 상태 tensor는 v1과 byte 동일하다. 각 기능 TRAIN 사진 2장의 CPU 예측도 정확히 같고 3개 기능 CPU/GPU 예측 일치를 확인했다. 시설·이물질의 기존 알려진 수치는 v1에서 계승하며 이번 회차에 다시 시험했다고 표시하지 않는다. 이물질 CAL·TEST를 다시 보정하거나 선택에 사용하지 않았다.', '',
        '선택 후보는 추가 연결 연구용이다. 기존 앱 기본 모델은 자동 교체하지 않는다. 정상 벨트 정답·독립 산업현장 검증이 없어 전체 오탐·미탐 5% 미만을 확인하지 못했다. 내부 모터·롤러 고장과 가동 여부는 not_assessed다. 위험 등급·조치·요약은 기존 규칙 템플릿이고 점검자의 최종 확인이 필요하다.', '',
        '![실제 벨트 추가 학습 비교](unified-belt-detail-comparison.png)', '',
        '- [사전 등록 조건](unified-belt-detail-protocol.json)',
        '- [원본·학습·재개·다른 기능 보존 검증](unified-belt-detail-verification.json)',
        '- [선택 조건별 통과·실패](unified-belt-detail-selection.json)',
        '- [전체 사진 control 계수·사진별 결과](unified-belt-detail-full_control-validation.json)',
        '- [원본 crop 혼합 계수·사진별 결과](unified-belt-detail-native_mix-validation.json)',
        '- [실행 파이프라인](../../docs/UNIFIED_BELT_DETAIL_STUDY_KO.md)',
        '- [기존 통합 v1 전체 기능 수치](UNIFIED_MAIN_RESULTS_KO.md)',
        '- [출처·사용 조건](UNIFIED_MAIN_ATTRIBUTION_KO.md)', '']
    new(REPORTS/'UNIFIED_BELT_DETAIL_RESULTS_KO.md', '\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8)); labels = ['Unified v1', 'Full control', 'Native mix']
    for axis, key, title in [(axes[0], 'pixel_iou', 'Belt pixel IoU (higher is better)'), (axes[1], 'pixel_fnr', 'Belt pixel FNR (lower is better)'), (axes[2], 'photo', 'Belt photo FNR (lower is better)')]:
        values = [100*(r['photo_rates']['fnr'] if key == 'photo' else r[key]) for r in combined]
        axis.bar(labels, values, color=['#607684', '#3e92bd', '#47864d']); axis.set_title(title, fontsize=11)
        axis.set_ylim(0, max(values)*1.25+.5); axis.set_ylabel('%'); axis.spines[['top', 'right']].set_visible(False)
        for x, y in enumerate(values): axis.text(x, y+.25, f'{y:.2f}%', ha='center')
    fig.suptitle('Belt head only: fixed +10 epochs, known reused fullframe VAL353', fontsize=14)
    fig.text(.5, .02, 'Normal photo FPR unavailable; other-task tensors unchanged. Selection: '+choice['selected'], ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .92)); fig.savefig(REPORTS/'unified-belt-detail-comparison.png', dpi=150); plt.close(fig)
    print(json.dumps({'report': 'created', 'selected': choice['selected']}), flush=True)


def package():
    choice = read(REPORTS/'unified-belt-detail-selection.json')
    if choice['selected'] == 'prior_v1':
        new(REPORTS/'unified-belt-detail-release.json', {'status': 'prior_v1_retained', 'new_package': None,
            'selection': choice, 'app_default_promoted': False}); print('Prior v1 retained; no new handoff recommended', flush=True); return
    assert not STAGE.exists() and not ARCHIVE.exists(); STAGE.mkdir()
    checkpoint = ROOT/choice['checkpoint_path']; assert sha(checkpoint) == choice['checkpoint_sha256']; shutil.copy2(checkpoint, STAGE/'model.pt')
    new(STAGE/'model.json', {'path': 'model.pt', 'sha256': sha(checkpoint), 'researchVariant': choice['selected'],
        'apiArchitectureFamily': 'unified-main-v1', 'checkpoint_count': 1, 'backbone_count': 1,
        'inspectionTasks': ['facility_surface', 'belt_damage', 'conveyor_foreign'], 'base_epochs': 20,
        'additional_belt_epochs': 10, 'additional_optimizer_updates': 1280, 'non_belt_parameters_exactly_v1': True,
        'app_default_promoted': False, 'fieldPerformanceValidated': False})
    new(STAGE/'requirements-api.txt', 'fastapi==0.142.1\nuvicorn==0.54.0\npydantic==2.13.5\npillow==12.3.0\nnumpy==2.5.2\n')
    new(STAGE/'safelog_ai/__init__.py', '')
    for name in MODULES: shutil.copy2(ROOT/'safelog_ai'/name, STAGE/'safelog_ai'/name)
    new(STAGE/'README_KO.md', '''# SafeLog 벨트 부분 추가 학습 · 역할4 전달

모델 파일 하나로 시설·벨트 손상·이물질 항목을 처리한다. 선택한 벨트 연구 후보와 CPU API·기존 C++ 연결을 제공한다.

- [연결 방법](safelog-cpp/docs/UNIFIED_BELT_DETAIL_HANDOFF_KO.md)
- [개선·악화·실제 결과](safelog-cpp/ai-training/reports/UNIFIED_BELT_DETAIL_RESULTS_KO.md)
- [출처와 연구·교육 사용 조건](safelog-cpp/ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md)

정상 벨트 사진 오탐은 미측정이며 독립 현장 5% 검증과 전체 설비 내부 고장 진단을 제공하는 모델이 아니다. 앱 기본 모델은 자동 교체하지 않았다.
''')
    guide = f'''# 역할4 · 벨트 출력 추가 학습 통합 모델

연구 후보 `{choice['selected']}`. 파일 하나·공통 분석부 하나·3개 출력이며 기존 시설·이물질 가중치는 v1과 정확히 동일하다. 모델 SHA는 `{choice['checkpoint_sha256']}`다. API modelName은 구조 계열 unified-main-v1이며 실제 학습 버전은 modelSha256 및 model.json으로 구분한다.

## CPU 실행

`{NAME}.zip`을 새 폴더에 풀고 Python 3.12에서 실행한다. 제작 환경의 실제 실행과 새 환경의 pip 설치 시험은 구분한다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$beltDetailModelRecord = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.unified_main_server --checkpoint $beltDetailModelRecord.path --sha256 $beltDetailModelRecord.sha256 --port 8775
```

서버는127.0.0.1에서 실행한다. 실제 스마트폰은 별도 서버 주소·인증·HTTPS를 팀에서 구성한다. Android APK를 만든 전달물이 아니다.

## C++ 앱 연결

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="belt_damage"
```

[기존 어댑터 설정](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp)과 [요청 코드](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp)를 사용한다. 점검 항목은 facility_surface·belt_damage·conveyor_foreign이다. 환경변수는 실행 모드를 선택하며 촬영 화면의 항목 선택 UI·DB 저장은 팀 앱 흐름에 연결한다.

- GET /health: ready, neuralModelCount=1, sharedBackboneCount=1, modelSha256 확인.
- POST /v1/analyze-hazard: image, inspectionTask, userMemo, promptVersion.
- POST /v1/compare-action: beforeImage, afterImage, inspectionTask, actionNote, promptVersion.
- POST /v1/summarize: inspectionContext, promptVersion.

점검 항목 누락·미지원은422다. 요청하지 않은 다른 항목은 정상으로 판단하지 않는다. 벨트 heatmap은 원본 크기 확인 자료이며 확정 결함 위치가 아니다. 점수는 보정된 고장 확률이 아니고 같은 사진으로 수리 완료가 나오면 안 된다. 내부 기계 고장은 not_assessed다. 위험 등급·조치·요약은 규칙 템플릿이며 점검자 최종 확인이 필요하다.

[실제 학습 성능](../ai-training/reports/UNIFIED_BELT_DETAIL_RESULTS_KO.md), [기존 다른 기능의 수치](../ai-training/reports/UNIFIED_MAIN_RESULTS_KO.md), [출처·사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md)을 함께 전달한다. 기존 앱 기본 모델은 자동 교체하지 않았다.
'''
    new(REPO/'safelog-cpp/docs/UNIFIED_BELT_DETAIL_HANDOFF_KO.md', guide)
    queue = [REPORTS/'UNIFIED_BELT_DETAIL_RESULTS_KO.md', REPO/'safelog-cpp/docs/UNIFIED_BELT_DETAIL_HANDOFF_KO.md']
    queue += [p for p in REPORTS.glob('unified-belt-detail-*') if p.suffix in ('.json', '.png')]
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
    choice = read(REPORTS/'unified-belt-detail-selection.json')
    if choice['selected'] == 'prior_v1': return
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
    process = subprocess.Popen([sys.executable, '-m', 'safelog_ai.unified_main_server', '--checkpoint', 'model.pt', '--sha256', digest, '--port', '8775'],
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
        for task, path in photos.items():
            reply = http('/v1/analyze-hazard', {'image': base64.b64encode(path.read_bytes()).decode(), 'inspectionTask': task, 'promptVersion': 'belt-detail-release-v1'})
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
    new(REPORTS/'unified-belt-detail-release.json', {'status': 'passed', 'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE),
        'archive_bytes': ARCHIVE.stat().st_size, 'model_sha256': choice['checkpoint_sha256'], 'selected': choice['selected'],
        'checkpoint_count': 1, 'backbone_count': 1, 'isolated_zip_CPU_HTTP': True, 'actual_CPP_http_operations': 9,
        'cpp_proofs': proofs, 'hazard_responses': hazards, 'all_smoke_photos_are_TRAIN': True,
        'zip_crc_member_SHA_and_local_links_verified': True, 'owned_server_stopped': process.poll() is not None,
        'fresh_pip_install_tested': False, 'all_industrial_tasks_5percent_verified': False, 'app_default_promoted': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['report', 'package', 'smoke'])
    globals()[parser.parse_args().mode]()
