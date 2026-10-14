"""Check measured selection, immutable packaged runtime and new-data provenance."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.study_native_screw_budget import validate, RUN, REPORTS, read
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new


def main():
    protocol = validate()
    verification = read(REPORTS/'equipment-native-screw-budget-v7-verification.json')
    release = read(REPORTS/'equipment-native-screw-budget-v7-release.json')
    choice = read(REPORTS/'equipment-native-screw-budget-v7-selection.json')
    result = read(REPORTS/'equipment-native-screw-budget-v7-validation.json')['candidates']['normalCAL15']
    evaluation = read(RUN/'evaluation.json')
    assert verification['status'] == release['status'] == 'passed'
    assert choice['selected'] == 'normalCAL15'
    assert choice['checkpoint_sha256'] == release['model_sha256'] == sha(ROOT/choice['checkpoint_path'])
    for name, digest in protocol['legacy_source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    rates = result['photo_rates']
    assert (rates['fp'], rates['fn'], rates['negative_photos'], rates['positive_photos']) == (2, 15, 41, 119)
    assert sum(not r['candidate_detected'] for r in result['observations'] if r['label']) == 15
    assert sum(r['candidate_detected'] for r in result['observations'] if not r['label']) == 2
    assert evaluation['categories']['ad_screw']['photo_rates'] == rates
    assert result['by_defect']['thread_side']['fn'] == 13
    stage20 = [c for c, v in evaluation['categories'].items() if v['stage_gate']['ready_for_next_stage']]
    assert stage20 == choice['stage20_categories'] and len(stage20) == 5
    stage15 = [c for c, v in evaluation['categories'].items() if v['stage_gate']['state'] == 'target15_met']
    assert len(stage15) == 4
    assert release['actual_CPP_operations'] == sum(v['operations'] for v in release['CPP_cases']) == 24
    archive = Path(release['archive'])
    assert sha(archive) == release['archive_sha256'] and archive.stat().st_size == release['archive_bytes']
    runtime_files = 0
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist()) == 1
        hashes = json.loads(z.read('package-sha256.json'))
        for name, digest in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
            if name.startswith('safelog_ai/') and name.endswith('.py') and name != 'safelog_ai/__init__.py':
                assert sha(ROOT/name) == digest, name
                runtime_files += 1
    data_dir = ROOT/'data/industrial-bsclass-v1'
    dataset = read(data_dir/'manifest.json')
    audit = read(REPORTS/'bsclass-v1-data-audit.json')
    assert dataset['audit'] == audit and len(dataset['rows']) == audit['source_images'] == 21835
    assert sha(data_dir/'publisher-bagit.archive') == audit['archive_sha256']
    for name, digest in dataset['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    groups = defaultdict(set)
    for row in dataset['rows']:
        groups[row['group_id']].add(row['phase'])
    assert all(len(v) == 1 for v in groups.values())
    counts = {phase: dict(Counter(str(r['label']) for r in dataset['rows'] if r['phase'] == phase)) for phase in audit['split_counts']}
    assert counts == audit['split_counts'] and not audit['new_training_performed'] and not audit['new_reserved_TEST_model_predictions_run']
    old_zip = ROOT.parents[3]/'outputs/unified-equipment-reference-role4-handoff-2026-10-14-v2.zip'
    assert sha(old_zip) == '416b0845554344f9707f391ae0c232434eec602d15072ccf562e376443bf5722'
    record = {
        'status': 'passed_native_screw_improvement_packaged', 'audited_utc': datetime.now(timezone.utc).isoformat(),
        'research_checkpoint_path': str(ROOT/choice['checkpoint_path']), 'research_checkpoint_sha256': choice['checkpoint_sha256'],
        'role4_ZIP': str(archive), 'role4_ZIP_sha256': release['archive_sha256'], 'role4_ZIP_bytes': release['archive_bytes'],
        'old_role4_ZIP_bytes_preserved': True, 'one_checkpoint_one_shared_backbone': True, 'native_screw_backbone_passes_per_photo': 2,
        'screw_known_TEST_photo_rates': rates, 'screw_known_TEST_defect_groups': result['by_defect'],
        'stage20_categories': stage20, 'stage15_categories': stage15, 'all_screw_defect_groups_within20': False,
        'screw_original_pixel_FNR_unchanged': .9720065776893954, 'native_reference_heatmap_evaluated_as_segmentation': False,
        'v6_v7_neural_optimizer_updates': 0, 'new_screw_normal_reference_FIT_photos': 96,
        'selected_screw_normal_reference_patches': 73728, 'normalCAL_count': 24, 'normalCAL_alpha': .15, 'normalCAL_rank': 22,
        'v5_actual40epoch2560update_candidate_remains_rejected': True,
        'protected_old_files_checked_in_v7_verification': verification['protected_files'],
        'protected_legacy_sources_checked_again': len(protocol['legacy_source_sha256']), 'packaged_runtime_sources_checked_again': runtime_files,
        'all_parent_tensors_and_existing_reference_buffers_exact': verification['all_parent_tensors_and_reference_buffers_exact'],
        'other10_equipment_photo_outputs_preserved': all(v['parent_photo_output_exact'] for c, v in verification['category_proofs'].items() if c != 'ad_screw'),
        'actual_current_CPU_CPP_operations': 24, 'isolated_ZIP_CPU_API_checked': True, 'unassessable_input_rejected422': True,
        'focused_unit_tests_passed_this_round': 8, 'focused_unit_test_modules': ['test_native_component_views:4', 'test_native_component_server:2', 'test_bsclass_data:2'],
        'new_BSClass_labelled_photos_prepared': 21835, 'new_BSClass_split_counts': counts,
        'new_BSClass_rotations_exact_and_near_groups_disjoint': True, 'new_BSClass_training_performed': False,
        'new_BSClass_reserved_TEST_model_predictions_run': False, 'physical_spindle_or_acquisition_groups_verified': False,
        'known_public_TEST_reused_for_development': True, 'independent_field_validated': False,
        'fresh_pip_install_tested': False, 'Android_APK_built': False, 'app_default_promoted': False,
        'next_priority': 'Preregister real labelled BSClass auxiliary wear training with frozen accepted backbone and untouched reserved_TEST; keep ball-screw domain distinct from fastener screw.'
    }
    new(REPORTS/'equipment-native-screw-v6-v7-final-audit.json', record)
    outputs = ROOT.parents[3]/'outputs'
    new(outputs/'NATIVE_SCREW_RESUME_2026-10-15.json', {**record, 'repo': str(ROOT.parents[1]),
        'report': str(REPORTS/'NATIVE_SCREW_RESULTS_KO.md'), 'handoff': str(ROOT.parent/'docs/NATIVE_SCREW_HANDOFF_KO.md')})
    with (outputs/'NATIVE_SCREW_RESUME_2026-10-15.md').open('x', encoding='utf-8', newline='\n') as out:
        out.write('# 나사 개선 재개 기록 · 2026-10-15\n\n'
            '연구 모델: '+choice['checkpoint_sha256']+'\n\n'
            '같은 공개 TEST160장의 사진 오탐 2/41=4.88%, 미탐 15/119=12.61%. '
            '나사산 옆면 미탐 13/23=56.52%는 남아 있다. 현장 독립 평가 결과가 아니다.\n\n'
            '한 모델·한 공통 분석부. 나사는 원본 위치 보존과 방향 보정 비교로 같은 분석부 2회 사용. '
            '이번 v6/v7 신경망 업데이트 0회, 정상 FIT96장의 참조 특징을 적합했다. '
            '앞선 v5 실제 학습 40epoch·2560회는 미채택으로 보존했다.\n\n'
            '새 ZIP CPU API 및 C++ 실제 24작업 통과. 새 볼스크루 분류 자료 21,835장은 준비만 했고 '
            '아직 학습/예약TEST 예측을 하지 않았다. 물리적 부품·촬영 프레임 그룹은 미검증.\n\n'
            '다음: 별도 볼스크루 도메인으로 실제 손상 보조 학습 조건을 사전 고정하고 통과한 다섯 유형을 보존한다.\n')
    print(json.dumps({'final_audit': 'passed', 'stage20': len(stage20), 'stage15': len(stage15), 'actual_CPP_operations': 24}), flush=True)


if __name__ == '__main__': main()
