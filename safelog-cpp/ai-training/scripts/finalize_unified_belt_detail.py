"""Update current summaries from measured results, then audit the immutable handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.unified_main_data import read, sha
from scripts.release_unified_main import local_links
from scripts.study_unified_belt_detail import validate, SOURCES
REPO = ROOT.parents[1]; REPORTS = ROOT/'reports'; OUTPUTS = REPO.parents[1]/'outputs'
OLD_ZIP = OUTPUTS/'unified-main-role4-handoff-2026-10-13-v1.1.zip'
OLD_SHA = '9cf40237b0d795166bf90e2cfba69c17d8342f01bb992d597af583c88874bec1'


def current_summary(report_link, guide_link):
    choice = read(REPORTS/'unified-belt-detail-selection.json')
    baseline = read(REPORTS/'unified-belt-detail-protocol.json')['baseline']
    results = {v: read(REPORTS/f'unified-belt-detail-{v}-validation.json')['summary'] for v in ['full_control', 'native_mix']}
    message = '기존 통합 v1을 유지한다.' if choice['selected'] == 'prior_v1' else f"벨트 추가 학습 후보 **{choice['selected']}**를 다음 연결 연구 모델로 선택했다. 같은 모델 파일 하나·공통 분석부 하나로3개 점검 항목을 처리한다."
    lines = ['## 벨트 출력만 추가 학습 · 2026-10-13', '',
        '같은 통합 v1에서 전체 사진 control과 원본 손상 부위 혼합을 각각 **10epoch·1,280회**, 합계 **2,560회** 실제 추가 학습했다. TRAIN1,397장의 작성자 원본·라벨을 검증하고 원본 crop을 생성했다. 새 독립 사진·새 정답은0개다.', '',
        '| 같은 알려진 VAL353 | 통합 v1 | 전체 사진 control | 원본 crop 혼합 |', '|---|---:|---:|---:|']
    values = [baseline, results['full_control'], results['native_mix']]
    for label, get in [('사진 미탐', lambda r: r['photo_rates']['fnr']), ('위치 IoU', lambda r: r['pixel_iou']),
                       ('손상 pixel 미탐', lambda r: r['pixel_fnr']), ('background pixel 오탐', lambda r: r['pixel_fpr'])]:
        lines.append('| '+label+' | '+' | '.join(f'{100*get(v):.2f}%' for v in values)+' |')
    lines += ['', message+' 공통 분석부·시설·이물질의 모든 tensor와 기존 CPU 예측은 정확히 유지됐다. 다른 기능 수치는 기존 v1에서 계승한 것이며 새 현장 시험이 아니다.', '',
        '원본 crop 혼합은 손상 pixel 미탐이 줄었지만 background pixel 오탐은 늘었다. 이 pixel 지표를 정상 사진 오탐률로 바꾸어 해석하지 않는다.', '',
        '정상 벨트 오탐률은 미측정이고 전체 산업현장 오탐·미탐5% 미만은 검증하지 못했다. 예약 TEST는 사용하지 않았다. 기존 앱 기본 모델은 자동 교체하지 않았다.', '',
        f'- [실제 비교·선택·한계]({report_link})', f'- [파이프라인]({guide_link})', '']
    return '\n'.join(lines)


def insert_after_title(path, block):
    content = path.read_bytes().decode('utf-8')
    newline = '\r\n' if '\r\n' in content else '\n'
    assert '## 벨트 출력만 추가 학습 · 2026-10-13' not in content, 'Already summarized; preserve completed docs'
    first, rest = content.split(newline+newline, 1)
    path.write_bytes((first+newline+newline+block.replace('\n', newline)+newline+rest).encode('utf-8'))


def docs():
    validate()
    insert_after_title(REPO/'README.md', current_summary('safelog-cpp/ai-training/reports/UNIFIED_BELT_DETAIL_RESULTS_KO.md', 'safelog-cpp/docs/UNIFIED_BELT_DETAIL_STUDY_KO.md'))
    insert_after_title(ROOT/'README_KO.md', current_summary('reports/UNIFIED_BELT_DETAIL_RESULTS_KO.md', '../docs/UNIFIED_BELT_DETAIL_STUDY_KO.md'))
    for name in ['EQUIPMENT_SCOPE_KO.md', 'DATASET_MODEL_MAP_KO.md']:
        path = REPO/'safelog-cpp/docs'/name
        content = path.read_bytes().decode('utf-8')
        content = content.replace('## 후속 대조 결과', '## 이전 보존 대조 결과')
        content = content.replace('현재 사용할 모델 파일은 이전 통합 v1 하나로 유지한다.', '해당 대조에서는 이전 통합 v1 하나를 유지했다.')
        content = content.replace('## 현재 메인 연구 후보 · 2026-10-13', '## 기준 통합 v1의 학습 · 2026-10-13')
        path.write_bytes(content.encode('utf-8'))
        insert_after_title(path, current_summary('../ai-training/reports/UNIFIED_BELT_DETAIL_RESULTS_KO.md', 'UNIFIED_BELT_DETAIL_STUDY_KO.md'))
    print('Current summaries updated from actual results; original line endings retained', flush=True)


def audit():
    protocol = validate(); verified = read(REPORTS/'unified-belt-detail-verification.json'); release = read(REPORTS/'unified-belt-detail-release.json')
    assert verified['status'] == 'passed' and sha(OLD_ZIP) == OLD_SHA
    docs = [REPO/'README.md', ROOT/'README_KO.md', REPORTS/'UNIFIED_BELT_DETAIL_RESULTS_KO.md',
            REPO/'safelog-cpp/docs/UNIFIED_BELT_DETAIL_STUDY_KO.md', REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md', REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md']
    new_archive = None
    if release['status'] == 'passed':
        archive_path = Path(release['archive']); assert sha(archive_path) == release['archive_sha256']
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.testzip() is None and sum(n.endswith('.pt') for n in archive.namelist()) == 1
            members = json.loads(archive.read('package-sha256.json'))
            assert all(hashlib.sha256(archive.read(n)).hexdigest() == digest for n, digest in members.items())
            assert hashlib.sha256(archive.read('model.pt')).hexdigest() == release['model_sha256']
            assert all(not n.startswith(('data/', 'runs/')) and '/original-annotations/' not in n for n in archive.namelist())
        assert release['actual_CPP_http_operations'] == 9 and release['isolated_zip_CPU_HTTP'] and release['owned_server_stopped']
        docs.append(REPO/'safelog-cpp/docs/UNIFIED_BELT_DETAIL_HANDOFF_KO.md')
        new_archive = {'path': str(archive_path), 'sha256': sha(archive_path), 'bytes': archive_path.stat().st_size}
    else: assert release['status'] == 'prior_v1_retained' and release['new_package'] is None
    links = 0
    for doc in docs:
        # Repository README also links to existing project directories.
        for link in local_links(doc): assert link.is_relative_to(REPO) and link.exists(), link; links += 1
    from PIL import Image
    with Image.open(REPORTS/'unified-belt-detail-comparison.png') as figure: size = list(figure.size); figure.verify()
    import torch
    result = {'status': 'passed', 'selected': verified['selected'], 'total_real_updates': 2560,
        'protected_inputs_artifacts': verified['protected_files'], 'protected_legacy_sources': verified['legacy_sources'],
        'final_equals_last_progress_for_both': True, 'all_non_belt_tensors_exact_v1': True,
        'original_v1_1_zip_SHA_unchanged': OLD_SHA, 'new_archive': new_archive,
        'local_markdown_links_checked': links, 'figure_pixels': size,
        'current_source_sha256': {name: sha(ROOT/name) for name in SOURCES}, 'audit_script_sha256': sha(Path(__file__)),
        'torch_version': str(torch.__version__), 'torchvision_version': __import__('torchvision').__version__,
        'GPU': torch.cuda.get_device_name(0), 'reserved_belt_TEST_used': False, 'new_foreign_CAL_TEST_evaluation': False,
        'normal_belt_photo_FPR_available': False, 'industrial_all_tasks_5percent_verified': False, 'app_default_promoted': False}
    with (REPORTS/'unified-belt-detail-final-audit.json').open('x', encoding='utf-8') as out: json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['docs', 'audit'])
    globals()[parser.parse_args().mode]()
