"""Verify the retained handoff and record this study's actual training scope."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.unified_main_data import read, sha
from scripts.study_unified_belt_photo_guard import validate, BASE, new, START_SHA
from scripts.release_unified_main import local_links
REPORTS = ROOT/'reports'; REPO = ROOT.parents[1]; OUTPUTS = REPO.parents[1]/'outputs'
ZIP_NAME = 'unified-main-belt-tiles-role4-handoff-2026-10-13-v1.1.zip'
ZIP_SHA = '3e8235e8d3cc626bf0476a05a6f8686872fd90d34ed517c6d081e707cfdbd7c9'


def main():
    p = validate(); v = read(REPORTS/'unified-belt-photo-guard-v1-verification.json')
    c = read(REPORTS/'unified-belt-photo-guard-v1-selection.json')
    t = read(REPORTS/'unified-belt-photo-guard-v1-training.json')
    r = read(REPORTS/'unified-belt-photo-guard-v1-validation.json')
    changes = read(REPORTS/'unified-belt-photo-guard-v1-photo-changes.json')
    assert c['selected'] == 'prior_accepted_tiles' and c['checkpoint_sha256'] == START_SHA
    assert v['actual_journal_updates'] == v['unique_committed_updates'] == t['optimizer_updates'] == 512
    assert v['photo_constraints_checked'] == 3072 and sha(BASE/'train/final.pt') == t['weights_sha256']
    archive_path = OUTPUTS/ZIP_NAME; assert sha(archive_path) == ZIP_SHA
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and sum(n.endswith('.pt') for n in archive.namelist()) == 1
        hashes = json.loads(archive.read('package-sha256.json'))
        assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n, h in hashes.items())
        model = json.loads(archive.read('model.json'))
        assert model['sha256'] == START_SHA and model['inferencePolicySha256'] == c['policy']['sha256']
        assert hashlib.sha256(archive.read('model.pt')).hexdigest() == START_SHA
    docs = [REPO/'README.md', ROOT/'README_KO.md', REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md',
            REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md', REPO/'safelog-cpp/docs/UNIFIED_BELT_PHOTO_GUARD_KO.md',
            REPORTS/'UNIFIED_BELT_PHOTO_GUARD_RESULTS_KO.md']
    links = 0
    for doc in docs:
        for link in local_links(doc):
            assert link.is_relative_to(REPO) and link.exists(), link
            if link.suffix:
                assert link.is_file(), link
            links += 1
    with Image.open(REPORTS/'unified-belt-photo-guard-v1-comparison.png') as fig:
        pixels = list(fig.size); fig.verify()
    result = {'status': 'passed', 'epochs_completed': 4, 'completed_unique_optimizer_steps': 512,
              'optimizer_calls_completed_this_turn': 512, 'excluded_failed_optimizer_calls_this_turn': 0,
              'photo_constraints_checked': 3072, 'projected_steps': v['projected_updates'],
              'reverted_parameter_steps': v['reverted_parameter_steps'], 'all_effective_step_protection_checks_passed': True,
              'protection_scope': 'Each positive TRAIN photo loss, first-order only; not validation recall guarantee',
              'matched_control_arrays_exact': p['matched_control_arrays_exact'], 'new_images_or_labels_this_turn': 0,
              'TRAIN_mask_diagnostic': p['train_mask_diagnostic'],
              'normal_ROI_CAL_false_alarms': r['normal_ROI_CAL']['photo_guard']['false_alarms'], 'normal_ROI_CAL_count': 3,
              'known_positive_VAL_misses': r['known_summary']['photo_rates']['fn'], 'known_positive_VAL_count': 353,
              'photo_changes': {k: len(changes[k]) for k in ('recovered_prior_misses', 'new_misses', 'persistent_misses')},
              'known_VAL_reused': True, 'whole_photo_normal_FPR': None, 'reserved_normal7_model_predictions_performed': False,
              'old_BeltCrack_TEST_used': False, 'old_files_preserved': v['protected_files'],
              'old_sources_preserved': v['legacy_sources'], 'final_equals_progress': True, 'non_belt_tensors_exact_prior': True,
              'new_selected_model': False, 'retained_model_sha256': START_SHA, 'retained_policy_sha256': c['policy']['sha256'],
              'unselected_trained_weights_sha256': t['weights_sha256'], 'retained_handoff': ZIP_NAME, 'retained_handoff_sha256': ZIP_SHA,
              'retained_ZIP_CRC_member_SHA_weights_and_policy_verified': True, 'new_CPP_HTTP_tests_this_turn': 0,
              'new_CPU_GPU_three_tasks_and_native_tiles_checked': True, 'local_links_checked': links, 'figure_pixels': pixels,
              'app_default_promoted': False, 'industrial_all_tasks_5percent_verified': False,
              'slice_score_scope': 'Belt scores are maximum fused global/tile probabilities over valid pixels; uncalibrated'}
    new(REPORTS/'unified-belt-photo-guard-v1-final-audit.json', result)
    new(OUTPUTS/'UNIFIED_BELT_PHOTO_GUARD_RESUME_2026-10-13.json', result)
    content = '\n'.join(['# 사진별 손상 보호 · 재개 기준 · 2026-10-13', '',
        '이번 실제 완료 학습: 4epoch·512 optimizer calls, 제외된 실패 호출0.6개 사진 조건을 총3,072번 검사했다.',
        f"보정 {v['projected_updates']}회, 정확한 파라미터 복원 {v['reverted_parameter_steps']}회. TRAIN 일차 근사 보호 조건이며 검증 recall을 보장하지 않는다.", '',
        f"이번 손상 사진 미탐 {r['known_summary']['photo_rates']['fn']}/353, 정상ROI 이상후보 {r['normal_ROI_CAL']['photo_guard']['false_alarms']}/3. 사전 조건을 충족하지 못해 기존 모델·정책·전달 ZIP 유지.",
        '현재 모델SHA: '+START_SHA, '정책: global_tile_mean / '+c['policy']['sha256'], '역할4 ZIP: '+ZIP_NAME, 'ZIP SHA: '+ZIP_SHA, '',
        '시설·이물질tensor와 CPU출력을 보존했다. 정상14 FIT·CAL3·보관7 유지. 보관7·예약BeltCrack TEST 모델점수는 미사용. 정상전체사진FPR은null이며 산업현장 전체5% 미만 검증은 없다.',
        '완료 실험을 같은namespace로 덮어쓰지 않는다. 이후 비교는 새run·학습전 조건 고정·commit을 사용한다. WindowsPID는 읽기전용process_exists로 확인한다.',
        '다음은 정상 전체사진과 손상 정답·촬영 구간이 확인된 추가 데이터 확보를 우선한다. 같은VAL로 반복 비교해 선택한 숫자는 독립 현장 성능이 아니다.', '',
        '결과: safelog-cpp/ai-training/reports/UNIFIED_BELT_PHOTO_GUARD_RESULTS_KO.md',
        '파이프라인: safelog-cpp/docs/UNIFIED_BELT_PHOTO_GUARD_KO.md', '새 학습 파일: runs/unified-belt-photo-guard-v1/train/final.pt (미채택)', ''])
    with (OUTPUTS/'UNIFIED_BELT_PHOTO_GUARD_RESUME_2026-10-13.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write(content)
    print(json.dumps({'final_audit': 'passed', 'completed_updates': 512, 'retained_model': True, 'checked_links': links}))


if __name__ == '__main__':
    main()
