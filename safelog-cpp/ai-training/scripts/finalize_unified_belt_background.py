"""Current summaries and immutable handoff evidence for the background study."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.unified_main_data import read, sha
from scripts.study_unified_belt_background import validate, SOURCES
from scripts.release_unified_main import local_links
REPO = ROOT.parents[1]; REPORTS = ROOT/'reports'; OUTPUTS = REPO.parents[1]/'outputs'
PREVIOUS_ZIP = OUTPUTS/'unified-main-belt-detail-role4-handoff-2026-10-13-v1.zip'
PREVIOUS_SHA = 'b2b77bfb5f1b79a1d53a5af0f4ff79ef506d90e456e97deed4d94bbaec774cb0'
ORIGINAL_ZIP = OUTPUTS/'unified-main-role4-handoff-2026-10-13-v1.1.zip'
ORIGINAL_SHA = '9cf40237b0d795166bf90e2cfba69c17d8342f01bb992d597af583c88874bec1'
HEADER = '## 벨트 배경 오탐 감소 대조 · 2026-10-13'


def summary(report_link, guide_link):
    choice = read(REPORTS/'unified-belt-background-selection.json'); baseline = read(REPORTS/'unified-belt-background-protocol.json')['baseline']
    results = {v: read(REPORTS/f'unified-belt-background-{v}-validation.json')['summary'] for v in ['anchor_control', 'hard_background']}
    selection = '두 후보가 모든 조건을 통과하지 못해 직전 native_mix 모델과 전달 ZIP을 유지했다.' if choice['selected'] == 'prior_native_mix' else f"사전 5개 조건을 통과한 **{choice['selected']}**를 다음 연결 연구 후보로 선택했다."
    lines = [HEADER, '',
        '직전 개선 모델에서 손상 보존 대조군과 배경 집중 후보를 각각 **10epoch·1,280회**, 합계 **2,560회** 실제 추가 학습했다. 같은 TRAIN1,397장과 원본 crop을 사용했으며 새 독립 사진·새 정답은 0개다.', '',
        '| 같은 알려진 VAL353 | 직전 native_mix | 손상 보존 대조군 | 배경 집중 후보 |', '|---|---:|---:|---:|']
    combined = [baseline, results['anchor_control'], results['hard_background']]
    for name, get in [('배경 pixel 오탐', lambda r: r['pixel_fpr']), ('손상 pixel 미탐', lambda r: r['pixel_fnr']),
                      ('위치 IoU', lambda r: r['pixel_iou']), ('사진 미탐', lambda r: r['photo_rates']['fnr'])]:
        lines.append('| '+name+' | '+' | '.join(f'{100*get(r):.2f}%' for r in combined)+' |')
    lines += ['', selection+' 배경 pixel 오탐을 상대10% 이상 줄이면서 IoU·사진 미탐·pixel 미탐·정답 위치 겹침 실패를 모두 유지하는 조건을 적용했다.', '',
        '공통 분석부·시설·이물질 tensor와 기존 CPU 예측은 정확히 유지했다. 다른 기능 수치는 기존 모델에서 계승한 값이다. 학습 보존용 기준 모델은 추론에 추가하지 않는다.', '',
        '정상 벨트 사진 오탐은 미측정이며 전체 산업현장 오탐·미탐5% 미만은 검증하지 못했다. 예약 TEST는 사용하지 않았다. 기존 앱 기본 모델은 자동 교체하지 않았다.', '',
        f'- [실제 비교·실패 조건·전체 계수]({report_link})', f'- [학습 파이프라인]({guide_link})', '']
    return '\n'.join(lines)


def insert(path, block):
    original = path.read_bytes().decode('utf-8'); assert HEADER not in original
    newline = '\r\n' if '\r\n' in original else '\n'; title, body = original.split(newline+newline, 1)
    path.write_bytes((title+newline+newline+block.replace('\n', newline)+newline+body).encode('utf-8'))


def docs():
    validate()
    insert(REPO/'README.md', summary('safelog-cpp/ai-training/reports/UNIFIED_BELT_BACKGROUND_RESULTS_KO.md', 'safelog-cpp/docs/UNIFIED_BELT_BACKGROUND_STUDY_KO.md'))
    insert(ROOT/'README_KO.md', summary('reports/UNIFIED_BELT_BACKGROUND_RESULTS_KO.md', '../docs/UNIFIED_BELT_BACKGROUND_STUDY_KO.md'))
    for name in ['EQUIPMENT_SCOPE_KO.md', 'DATASET_MODEL_MAP_KO.md']:
        insert(REPO/'safelog-cpp/docs'/name, summary('../ai-training/reports/UNIFIED_BELT_BACKGROUND_RESULTS_KO.md', 'UNIFIED_BELT_BACKGROUND_STUDY_KO.md'))
    print('Current summaries updated from measured results; existing line endings retained', flush=True)


def audit():
    p = validate(); verified = read(REPORTS/'unified-belt-background-verification.json'); release = read(REPORTS/'unified-belt-background-release.json')
    assert verified['status'] == 'passed' and sha(PREVIOUS_ZIP) == PREVIOUS_SHA and sha(ORIGINAL_ZIP) == ORIGINAL_SHA
    docs = [REPO/'README.md', ROOT/'README_KO.md', REPORTS/'UNIFIED_BELT_BACKGROUND_RESULTS_KO.md',
            REPO/'safelog-cpp/docs/UNIFIED_BELT_BACKGROUND_STUDY_KO.md', REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md', REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md']
    archive_proof = None
    if release['status'] == 'passed':
        path = Path(release['archive']); assert sha(path) == release['archive_sha256']
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None and sum(n.endswith('.pt') for n in archive.namelist()) == 1
            members = json.loads(archive.read('package-sha256.json'))
            assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n, h in members.items())
            assert hashlib.sha256(archive.read('model.pt')).hexdigest() == release['model_sha256']
            assert all(not n.startswith(('data/', 'runs/')) and '/original-annotations/' not in n for n in archive.namelist())
        assert release['actual_CPP_http_operations'] == 9 and release['isolated_zip_CPU_HTTP'] and release['owned_server_stopped']
        docs.append(REPO/'safelog-cpp/docs/UNIFIED_BELT_BACKGROUND_HANDOFF_KO.md')
        archive_proof = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    else:
        assert release['status'] == 'prior_native_mix_retained' and release['new_package'] is None
    links = 0
    for doc in docs:
        for link in local_links(doc): assert link.is_relative_to(REPO) and link.exists(), link; links += 1
    from PIL import Image
    with Image.open(REPORTS/'unified-belt-background-comparison.png') as figure: size = list(figure.size); figure.verify()
    import torch
    result = {'status': 'passed', 'selected': verified['selected'], 'total_real_updates': 2560,
        'protected_inputs_and_artifacts': verified['protected_files'], 'protected_legacy_sources': verified['legacy_sources'],
        'final_equals_last_progress_for_both': True, 'all_non_belt_tensors_exact_prior': True, 'reference_unchanged_no_gradients': True,
        'reference_in_runtime': False, 'previous_native_zip_SHA_unchanged': PREVIOUS_SHA, 'original_v1_1_zip_SHA_unchanged': ORIGINAL_SHA,
        'new_archive': archive_proof, 'local_links_checked': links, 'figure_pixels': size,
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES}, 'audit_script_sha256': sha(Path(__file__)),
        'torch_version': str(torch.__version__), 'GPU': torch.cuda.get_device_name(0),
        'new_photos': 0, 'new_labels': 0, 'reserved_belt_TEST_used': False, 'new_foreign_CAL_TEST_evaluation': False,
        'normal_belt_photo_FPR_available': False, 'industrial_all_tasks_5percent_verified': False, 'app_default_promoted': False}
    with (REPORTS/'unified-belt-background-final-audit.json').open('x', encoding='utf-8') as out: json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['docs', 'audit'])
    globals()[parser.parse_args().mode]()
