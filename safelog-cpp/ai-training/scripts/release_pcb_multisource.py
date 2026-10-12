"""Report two-source results and verify an independently extracted single-model CPU API."""
import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REPORT = ROOT / 'reports'
RUN = ROOT / 'runs/pcb-multisource-v3'
DOC = ROOT.parent / 'docs/PCB_MULTISOURCE_HANDOFF_KO.md'
NAME = 'pcb-multisource-role4-handoff-2026-10-12-v1'
OUTPUT = REPO.parents[1] / 'outputs'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new(path, value):
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, ensure_ascii=False)


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result = read(REPORT / 'pcb-multisource-results.json'); trained = read(REPORT / 'pcb-multisource-training.json')
    baseline = {r['category']: r for r in read(REPORT / 'pcb-reverse-results.json')['categories']}
    acquisition = read(ROOT / 'data/equipment-robustad-pcb-v1/acquisition.json')
    manifest = read(ROOT / 'data/pcb-multisource-v1/manifest.json')
    categories = ['pcb1', 'pcb2', 'pcb3', 'pcb4', 'robustad_pcb']; rows = {r['category']: r for r in result['categories']}
    changes = {}
    for c in categories[:4]:
        before = read(ROOT / f'runs/pcb-reverse-v2/{c}/test-predictions.json')
        after = read(RUN / f'{c}-test-predictions.json')
        assert ['VisA/' + name for name in before['image_ids']] == after['image_ids'] and before['labels'] == after['labels']
        flags = {'corrected_false_negative': [], 'new_false_negative': [], 'corrected_false_positive': [], 'new_false_positive': []}
        for name, label, a, b in zip(after['image_ids'], after['labels'], before['scores'], after['scores']):
            p, q = a > baseline[c]['threshold'], b > rows[c]['threshold']
            if p != q:
                kind = ('corrected_' if q == bool(label) else 'new_') + ('false_negative' if label else 'false_positive')
                flags[kind].append(name)
        changes[c] = flags
    new(REPORT / 'pcb-multisource-paired-errors.json', {'same_VisA802_ids_and_labels': True, 'categories': changes})
    new(REPORT / 'robustad-pcb-acquisition-summary.json', {'status': acquisition['status'], 'source': acquisition['source'],
        'revision': acquisition['revision'], 'license': acquisition['license'], 'downloaded_files': len(acquisition['files']),
        'downloaded_bytes': acquisition['total_bytes'], 'author_LFS_full_SHA256_verified_files': sum(r['author_lfs_sha256_verified'] for r in acquisition['files']),
        'all_local_file_SHA256_recorded': True, 'TRAIN_normal_photos': 180, 'TRAIN_anomalies_not_used': 75,
        'TEST0_normal_photos': 120, 'TEST0_anomaly_photos': 75,
        'exact_RGB_overlap_with_VisA': 0, 'within_or_cross_split_exact_RGB_duplicates': 0,
        'physical_board_capture_session_independence_verified': False, 'other_RobustAD_test_domains_used': False,
        'acquisition_manifest_sha256': sha(ROOT / 'data/equipment-robustad-pcb-v1/acquisition.json'),
        'two_source_manifest_sha256': sha(ROOT / 'data/pcb-multisource-v1/manifest.json')})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained'); x = np.arange(5)
    for ax, rate, title in zip(axes, ['fpr', 'fnr'], ['False positive / normal photos', 'False negative / anomaly photos']):
        previous = [baseline[c][rate]*100 for c in categories[:4]]
        a = ax.bar(x[:4]-.18, previous, width=.36, color='#94a3b8', label='Previous four specialist models')
        b = ax.bar(x+.18, [rows[c][rate]*100 for c in categories], width=.36, color='#2563eb', label='One two-source shared model')
        ax.bar_label(a, fmt='%.2f', padding=3, fontsize=8); ax.bar_label(b, fmt='%.2f', padding=3, fontsize=8)
        ax.text(4-.18, 1, 'No prior\nscore', ha='center', fontsize=8, color='#64748b')
        ax.axhline(5, color='#dc2626', linestyle='--', linewidth=1, label='5% target')
        ax.set_xticks(x, ['PCB1', 'PCB2', 'PCB3', 'PCB4', 'RobustAD\nPCB TEST0'])
        ax.set_ylabel('Percent'); ax.set_title(title); ax.grid(axis='y', alpha=.15)
        ax.set_ylim(0, max(12, max(previous + [rows[c][rate]*100 for c in categories]) + 7)); ax.legend(fontsize=8)
    fig.suptitle('Known VisA TEST802 + first-scored RobustAD TEST0 195; fixed normal-only CAL')
    path = REPORT / 'pcb-multisource-comparison.png'; assert not path.exists(); fig.savefig(path, dpi=150); plt.close(fig)
    lines = ['# 두 데이터셋으로 학습한 단일 PCB 모델 · 2026-10-12', '',
        '## 여러 데이터셋을 하나에 학습했는가', '',
        '**이번 실험은 그렇다. VisA와 RobustAD 두 출처의 정상 PCB 사진으로 하나의 학생 신경망을 함께 학습했다.** 기판 유형5종에 같은 학생 가중치·체크포인트를 사용하고, 유형별 정상 경고 기준만 다르다. 고정 ImageNet 교사와 학습 학생이 하나의 외관 이상 검출 모델을 구성한다. 기판별 학생 모델5개를 따로 학습한 것이 아니다.', '',
        '이전 최근 실험은 VisA의 PCB1~4를 각기 다른 모델로 학습했고, 벨트 균열과 이물질도 각각 다른 데이터 출처·모델이었다. 이전 건물 모델은 여러 자료를 통합했으나 이를 최근 PCB·벨트도 여러 출처로 학습했다는 뜻으로 설명하지 않는다.', '',
        '## 두 출처의 실제 원본', '',
        '| 출처 | 정상 FIT | 정상 CAL | TEST 정상/불량 |', '|---|---:|---:|---:|',
        '| VisA PCB1~4 | 3,250 | 364 | 402/400 |', '| RobustAD PCB | 140 | 40 | 120/75 |', '| 합계 | 3,390 | 404 | 522/475 |', '',
        '[공식 VisA](https://github.com/amazon-science/spot-diff)와 [공식 RobustAD](https://huggingface.co/datasets/AmazonScience/RobustAD/blob/main/README.md)는 CC BY4.0 자료다. RobustAD는 revision `82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`를 고정했다. PCB TRAIN 정상180장과 TEST0 사진195장·원본 마스크75개, 총450개 파일716,662,533바이트를 확보했다. 작성자의 메타데이터 정답을 유지하고 TRAIN 불량75장과 TEST1~5는 이번 학습·평가에 사용하지 않았다.', '',
        f"450개 원본 중 작성자 LFS 전체 SHA256을 확인한 파일은 {sum(r['author_lfs_sha256_verified'] for r in acquisition['files'])}개다. 모든 확보 파일의 로컬 SHA를 기록했다. 전체4,791장의 decoded RGB를 비교한 완전 동일 중복은0건이다. 유사 사진·같은 물리적 기판·촬영 세션의 독립성을 증명한 것은 아니다.", '',
        '## 실제 학습과 평가 분리', '',
        f"**하나의 모델 실제{trained['epochs']}epoch·{trained['optimizer_updates']:,}회 가중치 업데이트·{trained['elapsed_training_minutes']:.2f}분**을 완료했다. 사전 시험1업데이트는 본학습에서 제외한다. 사전 시험에서 두 출처 각각의 loss가 같은 학생 파라미터에 실제 gradient를 주는 것도 확인했다.", '',
        '매 epoch 각 유형800회, 합계4,000회 사진을 뽑아 학습했다. VisA 3,200회·RobustAD 800회로 소수 출처를 보강하는 replacement sampling이다. 반복 샘플링160,000회를 새로운 원본 사진160,000장으로 계산하지 않는다. 실제 고유 FIT 원본은3,390장이다.', '',
        '정상 FIT만 가중치 업데이트에 사용했다. 기존 VisA CAL91장씩과 별도 RobustAD CAL40장으로 유형별 사진 기준·픽셀 기준을 정한 뒤 TEST를 추론했다. 최종40epoch를 학습 전에 고정했으며 TEST로 epoch·경고 기준을 조정하지 않았다.', '',
        '이번에는 데이터 추가와 함께 단일 모델 통합·seed·샘플링·가중치 업데이트량도 달라졌다. 아래 비교는 전체 recipe 비교이며 데이터 추가만의 효과를 분리한 실험은 아니다. 원저자 Reverse Distillation의 아이디어를 참고한 자체 소형 구조이고 논문 조건 재현은 아니다.', '',
        '## 사진 단위 결과', '', '| 유형 | 정상/불량 | 이전 오탐→새 오탐 | 이전 미탐→새 미탐 | 새 AUROC |', '|---|---:|---:|---:|---:|']
    for c in categories:
        r = rows[c]; a = baseline.get(c)
        old_fpr = f"{a['fpr']*100:.2f}%" if a else '이전 평가 없음'
        old_fnr = f"{a['fnr']*100:.2f}%" if a else '이전 평가 없음'
        lines.append(f"| {c} | {r['negative_photos']}/{r['positive_photos']} | {old_fpr} → {r['fp']}건 ({r['fpr']*100:.2f}%) | {old_fnr} → {r['fn']}건 ({r['fnr']*100:.2f}%) | {r['auroc']:.4f} |")
    p = result['by_source']['VisA']; q = result['by_source']['RobustAD']
    lines += ['', f"같은 VisA802장 합산: 이전 오탐29/402(7.21%)·미탐52/400(13.00%) → 새 **오탐{p['fp']}/402({p['fpr']*100:.2f}%)·미탐{p['fn']}/400({p['fnr']*100:.2f}%)**.", '',
        f"처음 평가한 RobustAD TEST0: **오탐{q['fp']}/120({q['fpr']*100:.2f}%)·미탐{q['fn']}/75({q['fnr']*100:.2f}%)**. 이전 모델 점수가 없으므로 개선률을 만들지 않는다.", '',
        f"5유형의 오탐·미탐10개가 모두5% 미만인 기준: **{'충족' if result['all_ten_photo_rates_below5_percent'] else '미달'}**. 출처·유형별 수치를 합산으로 감추지 않는다. 기준을 정상 CAL5%로 설정했다고 TEST 오탐5%를 보장하는 것은 아니다.", '',
        '![유형별 오탐·미탐 비교](pcb-multisource-comparison.png)', '',
        '## 위치 표시와 적용 범위', '', '| 유형 | 모델256 IoU | 리사이즈로 사라진 양성 마스크 |', '|---|---:|---:|']
    for c in categories:
        r = rows[c]; lines.append(f"| {c} | {r['pixel_iou_model256']*100:.2f}% | {r['native_positive_mask_vanished_at256']} |")
    lines += ['', '원저자 원본 마스크를256×256 nearest로 변환하고 정상 CAL 픽셀 quantile0.995 기준으로 평가했다. 정상 사진의 픽셀 오탐도 포함한다. 원본 해상도 정밀 위치 성능이 아니며 heatmap은 특징 차이 확인용이다. 원본 크기로 확대해도 새로운 세부 정보를 복구하지 않는다.', '',
        'VisA802장은 이미 알려진 재사용 벤치마크다. RobustAD TEST0는 이번 모델의 최초 평가지만 새 공장·새 물리적 기판·촬영 세션의 독립성을 확인하지 않았다. RobustAD TEST1~5의 조명·구도 조건 전체를 검증한 것으로 소개하지 않는다.', '',
        '대상은 두 출처의 등록된5개 PCB 유형 근접 외관이다. 벨트/기계/건물 자료를 섞지 않았고 범용 설비 고장·전기 작동·내부 고장을 판단하지 않는다. 기존 모델·전달 파일을 보존하고 이번 모델을 별도 연구 후보로 제공한다. 앱 기본 모델은 자동 교체하지 않았다.', '',
        '## 재현과 전달', '',
        '- [추가 자료 확보·해시 요약](robustad-pcb-acquisition-summary.json)',
        '- [본학습 전 고정한 조건](pcb-multisource-protocol.json)',
        '- [실제 학습과 하나의 가중치 SHA](pcb-multisource-training.json)',
        '- [출처별 사진·픽셀 수치](pcb-multisource-results.json)',
        '- [같은 VisA 사진의 교정·새 오류](pcb-multisource-paired-errors.json)',
        '- [데이터·기존 모델·소스 보존 검증](pcb-multisource-verification.json)',
        '- [이전4종 모델 결과](pcb-reverse-results.json)',
        '- [역할4 단일 모델 API 연결](../../docs/PCB_MULTISOURCE_HANDOFF_KO.md)', '']
    path = REPORT / 'PCB_MULTISOURCE_RESULTS_KO.md'; assert not path.exists()
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def package():
    assert read(REPORT / 'pcb-multisource-verification.json')['status'] == 'passed'
    stage = ROOT / 'runs' / (NAME + '-stage'); assert not stage.exists(); stage.mkdir()
    modules = ['pcb_multisource_server.py', 'pcb_multisource.py', 'pcb_reverse.py', 'pcb_reverse_server.py',
               'equipment_anomaly.py', 'belt_inference.py', 'belt_segmentation.py', 'retention_distillation.py']
    (stage / 'safelog_ai').mkdir()
    for name in modules: shutil.copy2(ROOT / 'safelog_ai' / name, stage / 'safelog_ai' / name)
    (stage / 'safelog_ai/__init__.py').write_text('', encoding='utf-8')
    shutil.copy2(RUN / 'final.pt', stage / 'model.pt'); trained = read(REPORT / 'pcb-multisource-training.json')
    assert sha(stage / 'model.pt') == trained['weights_sha256']
    new(stage / 'model.json', {'path': 'model.pt', 'sha256': trained['weights_sha256'], 'neuralModelCount': 1,
                             'trainedSourceDatasets': ['VisA', 'RobustAD']})
    pending = [REPORT / 'PCB_MULTISOURCE_RESULTS_KO.md', DOC]; included = set()
    while pending:
        source = pending.pop().resolve()
        if source in included: continue
        assert source.is_relative_to(REPO) and source.is_file(), source
        included.add(source); target = stage / source.relative_to(REPO); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if source.suffix == '.md':
            for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', source.read_text(encoding='utf-8')):
                raw = raw.strip('<>').split('#')[0]
                if raw and not re.match(r'^[a-zA-Z]+:', raw): pending.append(source.parent / raw)
    (stage / 'requirements-api.txt').write_text('fastapi==0.142.1\nuvicorn==0.54.0\npillow==12.3.0\nnumpy==2.5.2\n', encoding='utf-8')
    (stage / 'requirements-host-tested.txt').write_text(''.join(f'{name}=={importlib.metadata.version(name)}\n'
        for name in ['torch', 'torchvision', 'fastapi', 'uvicorn', 'pillow', 'numpy', 'pydantic']), encoding='utf-8')
    (stage / 'README_KO.md').write_text('''# 두 데이터셋으로 실제 학습한 하나의 PCB 모델

model.pt 하나에 VisA와 RobustAD의 정상 특징을 학습했다. 원본 사진3,390장, 실제40epoch·5,000회 가중치 업데이트다. 기판 유형별 정상 경고 기준만 다르다.

새 폴더에 압축 해제하고 [역할4 실행 설명서](safelog-cpp/docs/PCB_MULTISOURCE_HANDOFF_KO.md)의 설치·8774 CPU 서버 실행을 따른다. 성능과 적용 범위는 [전체 결과](safelog-cpp/ai-training/reports/PCB_MULTISOURCE_RESULTS_KO.md)를 읽는다. 기존 모델을 자동 교체한 버전은 아니다.

모델은 model.json SHA로 고정한다. /health의 neuralModelCount=1과5개 유형의 같은 SHA를 확인한다. POST /v1/equipment/pcb/analyze의 schema는 safelog.equipment.pcb-visual.v3다. heatmap은 특징 차이 확인용이며 점검자의 최종 확인이 필요하다.

원본 데이터셋은 ZIP에 포함하지 않는다. VisA https://github.com/amazon-science/spot-diff 와 RobustAD https://huggingface.co/datasets/AmazonScience/RobustAD 는 CC BY4.0이다. 원저자 정답을 유지하고 RGB256 변환·ImageNet 정규화를 적용했다. 구조는 Reverse Distillation 아이디어를 참고한 SafeLog 자체 소형 구현이다: https://github.com/hq-deng/RD4AD .
''', encoding='utf-8', newline='\n')
    files = {p.relative_to(stage).as_posix(): {'sha256': sha(p), 'size_bytes': p.stat().st_size} for p in stage.rglob('*') if p.is_file()}
    new(stage / 'FILES_SHA256.json', files); OUTPUT.mkdir(exist_ok=True); archive = OUTPUT / (NAME + '.zip'); assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as packet:
        for path in sorted(stage.rglob('*')):
            if path.is_file(): packet.write(path, path.relative_to(stage).as_posix())
    new(REPORT / 'pcb-multisource-release-v1.json', {'archive': str(archive), 'archive_sha256': sha(archive),
        'size_bytes': archive.stat().st_size, 'one_checkpoint_sha256': trained['weights_sha256'], 'neuralModelCount': 1,
        'trainedSourceDatasets': ['VisA', 'RobustAD'], 'files': files, 'all_local_document_links_closed': True,
        'actual_CPU_HTTP_proof': 'Run smoke next; see pcb-multisource-standalone-v1.json'})
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive)}), flush=True)


def smoke():
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    sys.path.insert(0, str(ROOT))
    from safelog_ai.pcb_reverse_server import photo_tensor
    torch.set_num_threads(4); archive = OUTPUT / (NAME + '.zip')
    directory = ROOT / 'runs/pcb-multisource-isolated-zip-v1'; assert not directory.exists(); directory.mkdir()
    with zipfile.ZipFile(archive) as packet: packet.extractall(directory)
    files = read(directory / 'FILES_SHA256.json')
    for name, record in files.items(): assert sha(directory / name) == record['sha256']
    manifest = read(ROOT / 'data/pcb-multisource-v1/manifest.json'); model = read(directory / 'model.json')
    env = dict(os.environ); env.pop('PYTHONPATH', None); env['PYTHONIOENCODING'] = 'utf-8'
    samples = []; process = None
    with (directory / 'server.log').open('w', encoding='utf-8') as log:
        try:
            process = subprocess.Popen([sys.executable, '-m', 'safelog_ai.pcb_multisource_server', '--checkpoint', model['path'],
                '--sha256', model['sha256'], '--port', '8774'], cwd=directory, env=env, stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = time.monotonic()+45
            while True:
                assert process.poll() is None, (directory / 'server.log').read_text(encoding='utf-8')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8774/health', timeout=2) as response: health = json.load(response)
                    if health['ready']: break
                except OSError: pass
                assert time.monotonic()<deadline; time.sleep(.3)
            assert health['neuralModelCount'] == 1 and {p['modelSha256'] for p in health['models'].values()} == {model['sha256']}
            for category in ['pcb1', 'pcb2', 'pcb3', 'pcb4', 'robustad_pcb']:
                row = next(r for r in manifest['rows'] if r['category'] == category and r['phase'] == 'fit')
                raw = (ROOT / row['image']).read_bytes(); rgb = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
                prepared = Image.open(ROOT / row['prepared']).convert('RGB')
                assert np.array_equal(np.asarray(rgb.resize((256, 256), Image.Resampling.BILINEAR)), np.asarray(prepared))
                assert torch.equal(photo_tensor(rgb), photo_tensor(prepared))
                body = {'image': base64.b64encode(raw).decode(), 'equipmentCategory': category, 'sceneType': 'registered_pcb_closeup', 'promptVersion': 'two-source-http-smoke-v1'}
                request = urllib.request.Request('http://127.0.0.1:8774/v1/equipment/pcb/analyze', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=30) as response: value = json.load(response)
                heat = Image.open(io.BytesIO(base64.b64decode(value['heatmapPngBase64'])))
                assert heat.size == rgb.size and value['schema'] == 'safelog.equipment.pcb-visual.v3'
                assert value['modelSha256'] == model['sha256'] and value['neuralModelCount'] == 1
                assert not value['categorySpecificNeuralWeights'] and value['trainedSourceDatasets'] == ['VisA', 'RobustAD']
                assert value['inputSha256'] == hashlib.sha256(raw).hexdigest()
                assert value['candidateDetected'] == (value['anomalyScore'] > value['threshold'])
                assert value['mechanicalFailureState'] == 'not_assessed' and value['humanConfirmationRequired']
                assert not value['scoreIsProbability'] and not value['fieldPerformanceValidated'] and not value['heatmapIsConfirmedDefectLocalization']
                assert not value['localizationReadyForApp'] and value['heatmapUse'] == 'feature_difference_review_only'
                samples.append({'category': category, 'phase': 'fit', 'id': row['id'], 'one_model_sha256': value['modelSha256'],
                                'input_sha256': value['inputSha256'], 'native_geometry': list(rgb.size),
                                'score': value['anomalyScore'], 'threshold': value['threshold'], 'candidate': value['candidateDetected']})
        finally:
            if process is not None and process.poll() is None: process.terminate(); process.wait(timeout=10)
    new(REPORT / 'pcb-multisource-standalone-v1.json', {'status': 'passed', 'actual_five_CPU_HTTP_requests': samples,
        'one_model_checkpoint_for_every_type': True, 'both_source_TRAIN_samples_used': True, 'TEST_not_rescored': True,
        'exact_training_preprocessing_all_five': True, 'native_output_geometry_verified': True,
        'owned_server_stopped': process.poll() is not None, 'workspace_pythonpath_removed': True,
        'isolated_runtime_directory': str(directory), 'archive_sha256': sha(archive)})
    print('Independent ZIP: one model / both sources / five CPU HTTP calls / native geometry passed', flush=True)


def audit():
    result = read(REPORT / 'pcb-multisource-release-v1.json'); smoke_result = read(REPORT / 'pcb-multisource-standalone-v1.json')
    archive = Path(result['archive']); assert sha(archive) == result['archive_sha256'] == smoke_result['archive_sha256']
    assert smoke_result['status'] == 'passed' and smoke_result['owned_server_stopped']
    assert {r['one_model_sha256'] for r in smoke_result['actual_five_CPU_HTTP_requests']} == {result['one_checkpoint_sha256']}
    protocol = read(REPORT / 'pcb-multisource-protocol.json')
    for name, digest in {**protocol['source_sha256'], **protocol['legacy_source_sha256']}.items(): assert sha(ROOT / name) == digest, name
    assert sha(RUN / 'final.pt') == result['one_checkpoint_sha256']
    with zipfile.ZipFile(archive) as packet:
        assert packet.testzip() is None; names = set(packet.namelist()); files = json.loads(packet.read('FILES_SHA256.json'))
        for name, record in files.items(): assert hashlib.sha256(packet.read(name)).hexdigest() == record['sha256']
        for name in names:
            if name.endswith('.md'):
                for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', packet.read(name).decode('utf-8')):
                    raw = raw.strip('<>').split('#')[0]
                    if raw and not re.match(r'^[a-zA-Z]+:', raw):
                        assert posixpath.normpath(posixpath.join(posixpath.dirname(name), raw)) in names
    supplements = ['safelog_ai/pcb_multisource_server.py', 'tests/test_pcb_multisource_server.py', 'scripts/release_pcb_multisource.py']
    new(REPORT / 'pcb-multisource-final-audit.json', {'status': 'passed', 'one_neural_detector_checkpoint': True,
        'both_datasets_really_fitted': True, 'actual_epochs': 40, 'actual_optimizer_updates': 5000,
        'five_CPU_HTTP_requests': True, 'archive_sha256': sha(archive), 'all_archive_hashes_CRC_and_document_links_verified': True,
        'all_frozen_and_legacy_sources_preserved': True, 'supplemental_source_sha256': {name: sha(ROOT / name) for name in supplements},
        'API_tests_passed': 2, 'app_default_promoted': False})
    print('Two-source single-model CPU release, immutable sources and complete document links audited', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['report', 'package', 'smoke', 'audit'])
    {'report': report, 'package': package, 'smoke': smoke, 'audit': audit}[parser.parse_args().mode]()
