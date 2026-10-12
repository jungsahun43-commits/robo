"""Report fixed PCB results and deliver a separately runnable, SHA-pinned CPU API."""
import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUTPUT = REPO.parents[1] / 'outputs'
RUN = ROOT / 'runs/pcb-reverse-v2'
REPORT = ROOT / 'reports'
DOC = ROOT.parent / 'docs/PCB_REVERSE_HANDOFF_KO.md'
NAME = 'pcb-ai-role4-handoff-2026-10-12-v1'
STAGE = ROOT / 'runs' / (NAME + '-standalone')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_json(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result = read(REPORT / 'pcb-reverse-results.json')
    training = read(REPORT / 'pcb-reverse-training.json')
    old = {r['category']: r for r in read(REPORT / 'equipment-visa-results.json')['results']}
    current = {r['category']: r for r in result['categories']}
    categories = sorted(current)
    changes = {}
    for category in categories:
        before = read(ROOT / f'runs/equipment-visa-stfpm-v1/{category}/test-predictions.json')
        after = read(RUN / category / 'test-predictions.json')
        assert before['image_ids'] == after['image_ids'] and before['labels'] == after['labels']
        flags = {'corrected_false_negative': [], 'new_false_negative': [], 'corrected_false_positive': [], 'new_false_positive': []}
        for image, label, a, b in zip(after['image_ids'], after['labels'], before['scores'], after['scores']):
            p, q = a > old[category]['threshold'], b > current[category]['threshold']
            if p == q:
                continue
            key = ('corrected_' if q == bool(label) else 'new_') + ('false_negative' if label else 'false_positive')
            flags[key].append(image)
        changes[category] = flags
    new_json(REPORT / 'pcb-reverse-paired-errors.json', {'same_802_ids_and_labels': True, 'categories': changes})
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, rate, title in zip(axes, ['fpr', 'fnr'], ['False positive / normal photos', 'False negative / anomaly photos']):
        for offset, rows, label, color in [(-.18, old, 'STFPM final20', '#94a3b8'), (.18, current, 'Reverse final40', '#2563eb')]:
            bars = ax.bar(x + offset, [rows[c][rate] * 100 for c in categories], width=.36, label=label, color=color)
            ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
        ax.axhline(5, linestyle='--', color='#dc2626', linewidth=1, label='5% target')
        ax.set_xticks(x, [c.upper() for c in categories]); ax.set_ylabel('Percent'); ax.set_title(title)
        ax.set_ylim(0, max(12, max(max(old[c][rate], current[c][rate]) for c in categories) * 100 + 6))
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.15)
    fig.suptitle('Same known VisA TEST: 402 normal + 400 anomaly; fixed normal-only CAL')
    plot = REPORT / 'pcb-reverse-comparison.png'
    assert not plot.exists(); fig.savefig(plot, dpi=160); plt.close(fig)
    lines = [
        '# PCB 회로기판 4종 실제 재학습 결과 · 2026-10-12', '',
        '## 실제로 진행한 작업', '',
        f"정상 학습 사진3,250장으로 종류별40epoch, 합계 **{training['actual_category_model_epochs']} category-model epoch·{training['actual_optimizer_updates']:,}회 가중치 업데이트**를 완료했다. 네 모델을 순차 학습한 전체 시간은 {training['elapsed_training_minutes']:.2f}분이다. 일회용 사전 시험1업데이트는 본학습에 포함하지 않는다.", '',
        'ImageNet ResNet18 교사는 고정하고, 정상 특징을 압축한 뒤 3개 해상도로 복원하는 별도 학생 모델을 학습했다. [Reverse Distillation 원저자 구현](https://github.com/hq-deng/RD4AD)의 개념을 참고한 자체 소형 구현이며 논문 재현 모델은 아니다. 이전 모델과 구조·지도 집계 방식·학습량이 함께 달라 순수 구조 변경의 효과로 해석하지 않는다.', '',
        '공식 [VisA PCB1~4](https://github.com/amazon-science/spot-diff) 자료(CC BY4.0)만 사용했다. 각 종류 정상 CAL91장으로 사진 기준과 위치 표시 기준을 먼저 고정했다. 불량 CAL, TEST, 새 AI/사람 라벨을 학습하거나 기준 조정에 쓰지 않았다. 마지막40epoch를 미리 선택했으며 TEST 성능에 따라 epoch나 기준을 고르지 않았다.', '',
        '## 같은 공식 시험 사진 비교', '',
        '| 유형 | 정상/불량 수 | 이전 오탐 | 새 오탐 | 이전 미탐 | 새 미탐 | 새 AUROC |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for c in categories:
        a, b = old[c], current[c]
        lines.append(f"| {c.upper()} | {b['negative_photos']}/{b['positive_photos']} | {a['fp']}건 ({a['fpr']*100:.2f}%) | {b['fp']}건 ({b['fpr']*100:.2f}%) | {a['fn']}건 ({a['fnr']*100:.2f}%) | {b['fn']}건 ({b['fnr']*100:.2f}%) | {b['auroc']:.4f} |")
    p = result['pooled']
    old_fp, old_fn = sum(r['fp'] for r in old.values()), sum(r['fn'] for r in old.values())
    lines += ['', f"합산 오탐: **{old_fp}/402 ({old_fp/402*100:.2f}%) → {p['fp']}/402 ({p['fpr']*100:.2f}%)**. 합산 미탐: **{old_fn}/400 ({old_fn/400*100:.2f}%) → {p['fn']}/400 ({p['fnr']*100:.2f}%)**.", '',
        f"종류별 오탐·미탐 8개 모두5% 미만: **{'충족' if result['all_eight_photo_rates_below5_percent'] else '미달'}**. 합산 수치와 종류별 수치를 함께 보며 악화된 항목도 표에서 확인한다.", '',
        '![같은 사진의 오탐·미탐 비교](pcb-reverse-comparison.png)', '',
        '## 위치 표시 평가', '', '| 유형 | 모델256 좌표 IoU | 리사이즈 후 정답이 사라진 사진 |', '|---|---:|---:|']
    for c in categories:
        b = current[c]
        lines.append(f"| {c.upper()} | {b['pixel_iou_model256']*100:.2f}% | {b['native_positive_mask_vanished_at256']} |")
    lines += ['', '원저자 불량 마스크를256×256으로 nearest 리사이즈한 정답과, 정상 CAL 픽셀 분포에서 고정한 기준을 비교했다. 정상 사진의 배경 오탐도 IoU 분모에 포함된다. 작은 결함은 리사이즈로 사라질 수 있어 원본 해상도 위치 정확도로 해석하지 않는다. 이전 모델에는 픽셀 평가가 없어 위치 개선률은 제시하지 않는다. heatmap은 특징 차이이며 불량 확률·확정 결함 위치가 아니다.', '',
        '## 해석과 다음 작업', '',
        '시험802장은 이전에도 사용한 알려진 공식 벤치마크다. 정확히 같은 사진 ID와 정답으로 비교했으나 새 공장·새 물리적 기판·촬영 세션의 독립성은 확인되지 않았다. 정상 CAL5% 목표는 실제 TEST 오탐5%의 보장이 아니다. 이 결과를 산업현장 전체 오차율로 소개하지 않는다.', '',
        '현재 대상은 등록된 PCB1~4 종류의 근접 외관이다. 전체 배전반, 다른 기판, 작동·절연·과열·내부 고장을 판단하지 않는다. 연구 후보를 별도 API로 제공하며 기존 앱 기본 모델은 자동 교체하지 않는다. 오류 사진은 다음 실험을 위한 분석 자료로 보존하되, 동일 TEST를 조정 자료로 쓰면 새 독립 검증 자료가 필요하다.', '',
        '## 재현·검증 파일', '',
        '- [학습 전 고정한 프로토콜](pcb-reverse-protocol.json)',
        '- [실제 학습 기록](pcb-reverse-training.json)',
        '- [사진·픽셀 결과](pcb-reverse-results.json)',
        '- [동일 사진별 새 오류·교정 목록](pcb-reverse-paired-errors.json)',
        '- [원본·기존 모델·소스 보존 검증](pcb-reverse-verification.json)',
        '- [이전 모델 고정 결과](equipment-visa-results.json)',
        '- [역할4 API 연결](../../docs/PCB_REVERSE_HANDOFF_KO.md)', '']
    destination = REPORT / 'PCB_REVERSE_RESULTS_KO.md'
    assert not destination.exists(); destination.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def package():
    assert read(REPORT / 'pcb-reverse-verification.json')['status'] == 'passed'
    assert DOC.exists() and not STAGE.exists()
    STAGE.mkdir(); OUTPUT.mkdir(exist_ok=True)
    registry = {}
    modules = ['pcb_reverse_server.py', 'pcb_reverse.py', 'equipment_anomaly.py', 'belt_inference.py', 'belt_segmentation.py']
    for name in modules:
        target = STAGE / 'safelog_ai' / name; target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'safelog_ai' / name, target)
    (STAGE / 'safelog_ai/__init__.py').write_text('', encoding='utf-8')
    training = read(REPORT / 'pcb-reverse-training.json')
    for record in training['categories']:
        category = record['equipment_category']; source = RUN / category / 'final.pt'
        target = STAGE / f'pcb-reverse-v2/{category}/final.pt'; target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        assert sha(target) == record['weights_sha256']
        registry[category] = {'path': target.relative_to(STAGE).as_posix(), 'sha256': sha(target)}
    new_json(STAGE / 'models.json', registry)
    # Preserve the repository-relative documentation layout and recursively close local links.
    pending = [REPORT / 'PCB_REVERSE_RESULTS_KO.md', DOC]; included = set()
    while pending:
        source = pending.pop().resolve()
        if source in included:
            continue
        assert source.is_relative_to(REPO) and source.is_file(), source
        included.add(source); target = STAGE / source.relative_to(REPO); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if source.suffix == '.md':
            for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', source.read_text(encoding='utf-8')):
                raw = raw.strip('<>').split('#')[0]
                if raw and not re.match(r'^[a-zA-Z]+:', raw):
                    pending.append(source.parent / raw)
    for name in ['torch', 'torchvision', 'fastapi', 'uvicorn', 'pillow', 'numpy', 'pydantic']:
        version = importlib.metadata.version(name)
        with (STAGE / 'requirements-host-tested.txt').open('a', encoding='utf-8') as output:
            output.write(f'{name}=={version}\n')
    (STAGE / 'requirements-api.txt').write_text('fastapi==0.142.1\nuvicorn==0.54.0\npillow==12.3.0\nnumpy==2.5.2\n', encoding='utf-8')
    (STAGE / 'README_KO.md').write_text('''# 역할4 전달용 PCB 외관 AI

1. 이 ZIP을 새 폴더에 압축 해제하고 PowerShell에서 그 폴더로 이동한다.
2. Python3.12 환경을 생성하고 의존성을 설치한다.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-api.txt
.\\.venv\\Scripts\\python.exe -m safelog_ai.pcb_reverse_server --registry models.json --port 8773
```

CPU 실행에는 GPU가 필요하지 않다. requirements-host-tested.txt는 제작자가 실제 사용한 CUDA 포함 환경 기록이다. 완전히 새 환경에서 pip 설치 자체를 재시험한 것은 아니다.

GET http://127.0.0.1:8773/health가 ready=true여야 한다. POST /v1/equipment/pcb/analyze 요청과 Qt 연결은 [역할4 설명서](safelog-cpp/docs/PCB_REVERSE_HANDOFF_KO.md), 성능과 적용 범위는 [전체 결과](safelog-cpp/ai-training/reports/PCB_REVERSE_RESULTS_KO.md)를 읽는다. 기판 종류와 촬영 조건을 사용자가 명시하고 점검자가 최종 확인한다.

모델4개는 models.json의 SHA로 고정된다. 기존 컨베이어 서버8771/8772와 별도로 실행한다. ZIP의 모델을 Git에 올리지 않는다. 원본 사진 데이터셋은 ZIP에 포함하지 않는다.

데이터: Amazon Science VisA PCB1~4, CC BY4.0, https://github.com/amazon-science/spot-diff . 변환: RGB256 리사이즈와 ImageNet 정규화. 원본 작성자 정답 유지. 모델 자체 구조는 SafeLog 구현이며 Reverse Distillation 논문의 아이디어를 참고했다: https://github.com/hq-deng/RD4AD .
''', encoding='utf-8', newline='\n')
    files = {p.relative_to(STAGE).as_posix(): {'sha256': sha(p), 'size': p.stat().st_size} for p in STAGE.rglob('*') if p.is_file()}
    new_json(STAGE / 'FILES_SHA256.json', files)
    archive = OUTPUT / (NAME + '.zip'); assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(STAGE.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(STAGE).as_posix())
    new_json(REPORT / 'pcb-reverse-release-v1.json', {'archive': str(archive), 'sha256': sha(archive), 'size_bytes': archive.stat().st_size,
        'model_count': 4, 'registry': registry, 'files': files, 'closed_local_document_links': True,
        'http_cpu_standalone_verification': 'Run release_pcb_reverse.py smoke next; see pcb-reverse-standalone-v1.json'})
    print(json.dumps({'archive': str(archive), 'sha256': sha(archive), 'size_bytes': archive.stat().st_size}), flush=True)


def smoke():
    import torch
    import numpy as np
    from PIL import Image, ImageOps
    sys.path.insert(0, str(ROOT))
    from safelog_ai.pcb_reverse_server import photo_tensor
    torch.set_num_threads(4)
    archive = OUTPUT / (NAME + '.zip'); standalone = ROOT / 'runs/pcb-reverse-zip-verification-v1'
    assert not standalone.exists(); standalone.mkdir()
    with zipfile.ZipFile(archive) as packet:
        packet.extractall(standalone)
    for name, record in read(standalone / 'FILES_SHA256.json').items():
        assert sha(standalone / name) == record['sha256']
    manifest = read(ROOT / 'data/equipment-visa/manifest.json'); registry = read(standalone / 'models.json')
    env = dict(os.environ); env.pop('PYTHONPATH', None); env['PYTHONIOENCODING'] = 'utf-8'
    samples = []; process = None
    with (standalone / 'server.log').open('w', encoding='utf-8') as log:
        try:
            process = subprocess.Popen([sys.executable, '-m', 'safelog_ai.pcb_reverse_server', '--registry', 'models.json', '--port', '8773'],
                cwd=standalone, env=env, stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = time.monotonic() + 45
            while True:
                assert process.poll() is None, 'Owned API failed: ' + (standalone / 'server.log').read_text(encoding='utf-8')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8773/health', timeout=2) as response:
                        health = json.load(response)
                    if health['ready']:
                        break
                except OSError:
                    pass
                assert time.monotonic() < deadline, 'API readiness timed out'
                time.sleep(.3)
            for category in sorted(registry):
                row = next(r for r in manifest['rows'] if r['object'] == category and r['phase'] == 'fit')
                source = ROOT / 'data/equipment-visa/originals' / row['image']; payload = source.read_bytes()
                rgb = ImageOps.exif_transpose(Image.open(io.BytesIO(payload))).convert('RGB')
                prepared = Image.open(ROOT / 'data/equipment-visa' / row['prepared']).convert('RGB')
                assert np.array_equal(np.array(rgb.resize((256, 256), Image.Resampling.BILINEAR)), np.array(prepared))
                assert torch.equal(photo_tensor(rgb), photo_tensor(prepared)), 'API preprocessing differs from original training preparation'
                body = {'image': base64.b64encode(payload).decode(), 'equipmentCategory': category,
                        'sceneType': 'registered_pcb_closeup', 'promptVersion': 'pcb-http-smoke-v1'}
                request = urllib.request.Request('http://127.0.0.1:8773/v1/equipment/pcb/analyze', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                started = time.monotonic()
                with urllib.request.urlopen(request, timeout=30) as response:
                    result = json.load(response)
                heat = Image.open(io.BytesIO(base64.b64decode(result['heatmapPngBase64'])))
                assert heat.size == rgb.size and (result['inputWidth'], result['inputHeight']) == rgb.size
                assert result['modelSha256'] == registry[category]['sha256'] == health['models'][category]['modelSha256']
                assert result['inputSha256'] == hashlib.sha256(payload).hexdigest()
                assert result['candidateDetected'] == (result['anomalyScore'] > result['threshold'])
                assert result['mechanicalFailureState'] == 'not_assessed' and result['humanConfirmationRequired']
                assert not result['fieldPerformanceValidated'] and not result['scoreIsProbability'] and not result['heatmapIsConfirmedDefectLocalization']
                samples.append({'category': category, 'source_train_image': row['image'], 'phase': 'fit',
                    'seconds': time.monotonic() - started, 'model_sha256': result['modelSha256'], 'input_sha256': result['inputSha256'],
                    'native_geometry': list(rgb.size), 'score': result['anomalyScore'], 'threshold': result['threshold'], 'candidateDetected': result['candidateDetected']})
        finally:
            if process is not None and process.poll() is None:
                process.terminate(); process.wait(timeout=10)
    new_json(REPORT / 'pcb-reverse-standalone-v1.json', {'status': 'passed', 'four_actual_cpu_http_requests': samples,
        'train_only_smoke_no_test_rescoring': True, 'exact_training_preprocessing_verified_all_four': True,
        'zip_sha256': sha(archive), 'isolated_extracted_runtime_cwd': str(standalone), 'workspace_pythonpath_removed': True,
        'owned_server_stopped': process.poll() is not None, 'archive_source_files_sha_verified': True})
    print('Four trained CPU models / native geometry / HTTP / exact preprocessing / stopped owned server passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['report', 'package', 'smoke'])
    {'report': report, 'package': package, 'smoke': smoke}[parser.parse_args().mode]()
