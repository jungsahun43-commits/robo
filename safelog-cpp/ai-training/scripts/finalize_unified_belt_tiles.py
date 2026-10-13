"""Current summaries and artifact/policy evidence for the native uniform views."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.unified_main_data import read, sha
from scripts.study_unified_belt_tiles import validate, SOURCES, CONFIGS
from scripts.release_unified_main import local_links
REPO = ROOT.parents[1]; REPORTS = ROOT/'reports'; OUTPUTS = REPO.parents[1]/'outputs'
HEADER = '## 원본 구역 학습·분석 대조 · 2026-10-13'
PREVIOUS_ZIPS = {'unified-main-belt-detail-role4-handoff-2026-10-13-v1.zip': 'b2b77bfb5f1b79a1d53a5af0f4ff79ef506d90e456e97deed4d94bbaec774cb0',
                 'unified-main-role4-handoff-2026-10-13-v1.1.zip': '9cf40237b0d795166bf90e2cfba69c17d8342f01bb992d597af583c88874bec1'}


def summary(report_link, guide_link):
    p = read(REPORTS/'unified-belt-tiles-protocol.json'); choice = read(REPORTS/'unified-belt-tiles-selection.json')
    results = {c: read(REPORTS/f'unified-belt-tiles-{c}-validation.json')['summary'] for c in CONFIGS}
    message = '모든 설정이 조건을 통과하지 못해 직전 native_mix와 기존 전달 ZIP을 유지했다.' if choice['selected'] == 'prior_native_mix' else f"사전5조건을 통과한 **{choice['selected']}**를 다음 연결 연구 설정으로 선택했다."
    lines = [HEADER, '', '직전 모델에서 원본 구역 사진으로 벨트 출력을 **실제10epoch·1,280회** 추가 학습했다. 작성자 TRAIN1,397장과 파생 구역3,957개를 사용했으며 새 독립 사진·정답은0개다. 구역은 사진 크기로만 고른다.', '',
        '| 같은 알려진 VAL353 | 직전 전체사진 | 직전 + 구역 | 학습 후 전체사진 | 학습 후 + 구역 |', '|---|---:|---:|---:|---:|']
    values = [p['baseline'], *(results[c] for c in CONFIGS)]
    for label, get in [('사진 미탐', lambda r: r['photo_rates']['fnr']), ('손상 pixel 미탐', lambda r: r['pixel_fnr']),
                       ('배경 pixel 오탐', lambda r: r['pixel_fpr']), ('위치 IoU', lambda r: r['pixel_iou'])]:
        lines.append('| '+label+' | '+' | '.join(f'{100*get(r):.2f}%' for r in values)+' |')
    lines += ['', message+' IoU 최소0.5%p 개선과 배경오탐·사진미탐·pixel미탐·정답겹침실패 모두 유지하는 조건을 적용했다.', '',
        '직전 + 구역은 가중치를 학습하지 않은 분석 방식 비교이고, 학습 후 두 설정은 이번에 한 번 학습한 같은 가중치다. 추론은 모델파일 하나를 반복 사용한다. 모델SHA와 분석정책SHA를 함께 기록한다.', '',
        '모든 비벨트tensor와 시설·이물질 CPU 예측은 정확히 유지했다. 정상벨트사진오탐은 미측정이며 독립 산업현장5% 검증은 없다. 예약TEST는 사용하지 않았고 기존 앱 기본 모델은 자동 교체하지 않았다.', '',
        f'- [실제 학습·분석 비교·선택]({report_link})', f'- [파이프라인]({guide_link})', '']
    return '\n'.join(lines)


def insert(path, block):
    content = path.read_bytes().decode('utf-8'); assert HEADER not in content
    nl = '\r\n' if '\r\n' in content else '\n'; title, body = content.split(nl+nl, 1)
    path.write_bytes((title+nl+nl+block.replace('\n', nl)+nl+body).encode('utf-8'))


def docs():
    validate()
    insert(REPO/'README.md', summary('safelog-cpp/ai-training/reports/UNIFIED_BELT_TILES_RESULTS_KO.md', 'safelog-cpp/docs/UNIFIED_BELT_TILES_STUDY_KO.md'))
    insert(ROOT/'README_KO.md', summary('reports/UNIFIED_BELT_TILES_RESULTS_KO.md', '../docs/UNIFIED_BELT_TILES_STUDY_KO.md'))
    for name in ['EQUIPMENT_SCOPE_KO.md', 'DATASET_MODEL_MAP_KO.md']:
        insert(REPO/'safelog-cpp/docs'/name, summary('../ai-training/reports/UNIFIED_BELT_TILES_RESULTS_KO.md', 'UNIFIED_BELT_TILES_STUDY_KO.md'))
    print('Current summaries updated with real training and inference counts', flush=True)


def audit():
    p = validate(); verified = read(REPORTS/'unified-belt-tiles-verification.json'); release = read(REPORTS/'unified-belt-tiles-release.json')
    choice = read(REPORTS/'unified-belt-tiles-selection.json'); assert verified['status'] == 'passed'
    for name, digest in PREVIOUS_ZIPS.items(): assert sha(OUTPUTS/name) == digest
    docs = [REPO/'README.md', ROOT/'README_KO.md', REPORTS/'UNIFIED_BELT_TILES_RESULTS_KO.md', REPO/'safelog-cpp/docs/UNIFIED_BELT_TILES_STUDY_KO.md',
            REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md', REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md']
    archive_proof = None
    if release['status'] == 'passed':
        path = Path(release['archive']); assert sha(path) == release['archive_sha256']
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None and sum(n.endswith('.pt') for n in archive.namelist()) == 1
            members = json.loads(archive.read('package-sha256.json'))
            assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n, h in members.items())
            assert hashlib.sha256(archive.read('model.pt')).hexdigest() == choice['checkpoint_sha256']
            model = json.loads(archive.read('model.json'))
            assert model['inferenceMode'] == choice['mode'] and model['inferencePolicySha256'] == choice['inference_policy']['sha256']
            assert all(not n.startswith(('data/', 'runs/')) and '/original-annotations/' not in n for n in archive.namelist())
        assert release['actual_CPP_http_operations'] == 9 and release['isolated_zip_CPU_HTTP'] and release['owned_server_stopped']
        assert release['inference_policy_sha256'] == choice['inference_policy']['sha256']
        docs.append(REPO/'safelog-cpp/docs/UNIFIED_BELT_TILES_HANDOFF_KO.md'); archive_proof = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    else: assert release['status'] == 'prior_native_mix_retained' and release['new_package'] is None
    links = 0
    for doc in docs:
        for link in local_links(doc): assert link.is_relative_to(REPO) and link.exists(), link; links += 1
    from PIL import Image
    with Image.open(REPORTS/'unified-belt-tiles-comparison.png') as figure: size = list(figure.size); figure.verify()
    import torch
    result = {'status': 'passed', 'selected': choice['selected'], 'real_neural_updates': 1280,
        'selected_new_neural_updates': choice['selected_new_neural_updates'], 'selected_mode': choice['mode'],
        'inference_policy_sha256': choice['inference_policy']['sha256'], 'protected_files': verified['protected_files'], 'legacy_sources': verified['legacy_sources'],
        'non_belt_tensors_exact_prior': True, 'final_equals_progress': True, 'native_tile_CPU_GPU_close': True,
        'previous_zip_SHA_preserved': PREVIOUS_ZIPS, 'new_archive': archive_proof, 'local_links_checked': links, 'figure_pixels': size,
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES}, 'audit_script_sha256': sha(Path(__file__)),
        'torch_version': str(torch.__version__), 'GPU': torch.cuda.get_device_name(0), 'new_independent_photos': 0, 'new_labels': 0,
        'reserved_TEST_used': False, 'new_foreign_CAL_TEST_evaluation': False, 'normal_photo_FPR_available': False,
        'industrial_all_tasks_5percent_verified': False, 'app_default_promoted': False}
    with (REPORTS/'unified-belt-tiles-final-audit.json').open('x', encoding='utf-8') as out: json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['docs', 'audit'])
    globals()[parser.parse_args().mode]()
