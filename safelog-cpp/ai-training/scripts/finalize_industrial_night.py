"""Durable selected-model receipt and specific next steps for hourly continuation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import study_industrial_surfaces as study
from scripts import study_bsclass_supervised as wear
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new


def main():
    p = study.validate(); choice = study.read(study.REPORTS/'industrial-surfaces-v9-selection.json')
    verification = study.read(study.REPORTS/'industrial-surfaces-v9-verification.json')
    release = study.read(study.REPORTS/'industrial-night-v9-release.json')
    cpu = study.read(study.REPORTS/'industrial-night-v9-wear-CPU.json')
    training = study.read(wear.RUN/'chosen-head.json'); evaluation = study.read(study.RUN/'evaluation.json')
    assert verification['status'] == release['status'] == cpu['status'] == 'passed'
    assert choice['checkpoint_sha256'] == release['model_sha256'] == cpu['model_sha256'] == sha(ROOT/choice['checkpoint_path'])
    assert training['total_actual_optimizer_updates'] == 3600 and training['variant'] == 'mlp' and training['epoch'] == 23
    assert release['actual_CPP_operations'] == sum(v['operations'] for v in release['CPP_cases']) == 45
    assert cpu['photos'] == 3236
    # A device difference must be reported rather than relabelled as improved training.
    assert max(cpu['CPU_photo_rates']['fpr'], cpu['CPU_photo_rates']['fnr']) <= .20
    for name, digest in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == digest, name
    archive = Path(release['archive']); assert sha(archive) == release['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist()) == 1
        hashes = json.loads(z.read('package-sha256.json'))
        for name, digest in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
            if name.startswith('safelog_ai/') and name.endswith('.py') and name != 'safelog_ai/__init__.py': assert sha(ROOT/name) == digest, name
        adapter = 'safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp'
        assert hashes[adapter] == sha(ROOT.parents[1]/adapter)
    stages = {c: v['stage_gate']['state'] for c, v in evaluation['categories'].items()}
    assert sum(v in ('target20_met', 'target15_met') for v in stages.values()) == 8
    assert sum(v == 'target15_met' for v in stages.values()) == 5
    qt_log = (study.RUN/'qt-probe/adapter-tests-host.txt').read_text()
    assert 'Totals: 9 passed, 0 failed' in qt_log
    record = {
        'status': 'completed_v8_v9_night_batch_continuation_enabled', 'audited_utc': datetime.now(timezone.utc).isoformat(),
        'repo': str(ROOT.parents[1]), 'active_research_checkpoint': str(ROOT/choice['checkpoint_path']), 'active_checkpoint_sha256': choice['checkpoint_sha256'],
        'role4_ZIP': str(archive), 'role4_ZIP_sha256': release['archive_sha256'], 'role4_ZIP_bytes': release['archive_bytes'],
        'one_checkpoint_one_shared_backbone': True, 'real_label_training_candidates': 2, 'actual_epochs_each_candidate': 30,
        'actual_optimizer_updates_total_candidates': 3600, 'selected_wear_head': 'mlp', 'selected_wear_epoch': 23, 'selected_wear_epoch_updates': 1380,
        'real_label_FIT_photos': 15333, 'validation_photos': 1653, 'labelled_calibration_photos': 1613, 'reserved_TEST_photos': 3236,
        'wear_GPU_photo_rates': cpu['original_GPU_photo_rates'], 'wear_CPU_photo_rates': cpu['CPU_photo_rates'],
        'all_wear_CPU_GPU_decisions_equal': cpu['all_CPU_GPU_decisions_equal'], 'physical_spindle_acquisition_groups_verified': False,
        'wear_scope': 'cropped ball-screw visible pitting, no localization masks or internal failure claim',
        'new_reference_categories': choice['selected'], 'new_reference_normal_FIT_photos_selected': 164, 'new_reference_patches_selected': 41984,
        'new_reference_optimizer_updates': 0, 'original11_stage20_categories': 7, 'including_new_wear_stage20_categories': 8,
        'including_new_wear_stage15_categories': 5, 'stage_states': stages,
        'all_prior_neural_tensors_and_reference_buffers_preserved': True, 'original_localization_outputs_preserved': True,
        'protected_files_checked': verification['protected_files'], 'legacy_source_SHA_rechecked': len(p['legacy_source_sha256']),
        'new_python_unit_tests_passed': 8, 'Qt_reported_test_cases_passed': 9, 'actual_isolated_ZIP_CPU_CPP_operations_passed': 45,
        'wear_CPU_entire_TEST_cross_device_check_photos': 3236, 'checker_first_import_failure_preserved_runtime_ZIP_unchanged': True,
        'known_original11_TEST_reused_for_development': True, 'independent_field_validated': False,
        'fresh_pip_install_tested': False, 'Android_APK_built': False, 'app_default_promoted': False,
        'continuation_automation_id': 'safelog', 'continuation_interval_hours': 1,
        'remaining_weak_categories': ['loco_splicing_connectors', 'mpdd_bracket_black', 'mpdd_tubes'],
        'insufficient_evaluation_category': {'mpdd_connector': '14 defect TEST photos; cannot declare20percent pass despite0/14 candidate misses'},
        'retained_screw_thread_side_FNR': 13/23, 'retained_screw_original_pixel_FNR': .9720065776893954,
        'next_priorities': [
            'Use246 reserved_training_defect KSDD2 publisher TRAIN photos and true pixel masks for a preregistered supervised branch; audit cross-split exact/near duplicates and physical groups before any fit; never train on current TEST.',
            'Inspect normalFIT-only pose/background variability for black bracket and tubes; preregister new input/feature study instead of repeatedly tuning known TEST thresholds.',
            'For splicing-connector logical errors seek labelled relation/assembly data or an input-only component graph; do not equate generic surface scratches with incorrect wiring.',
            'Acquire genuinely additional defect evaluation for metal connectors and whole-photo healthy belts before claiming FPR/stage success there.'
        ],
        'next_operator_rules': ['Check operation locks/processes and do not duplicate GPU jobs', 'Preserve accepted checkpoints/data/reports; use new run namespace and frozen protocol',
            'Promote only joint guards; keep rejected candidates and honest subtype/localization limits', 'Commit code/reports and push feature/ai-engine; raw data, weights and ZIP remain local']
    }
    new(study.REPORTS/'industrial-night-v8-v9-final-audit.json', record)
    outputs = ROOT.parents[3]/'outputs'
    new(outputs/'INDUSTRIAL_NIGHT_RESUME_2026-10-15.json', record)
    with (outputs/'INDUSTRIAL_NIGHT_RESUME_2026-10-15.md').open('x', encoding='utf-8', newline='\n') as out:
        out.write('# 설비별 야간 학습 재개 기록\n\n'
            '현재 연구 모델 SHA: '+choice['checkpoint_sha256']+'\n\n'
            '볼스크루 실제 라벨 학습: 선형/MLP 각각30epoch·1800회, 총3600회. 검증에서 선택한 MLP23epoch 시점. '
            '예약 평가3236장 오탐14/1643=0.85%, 미탐10/1593=0.63%. 물리적 설비 그룹은 미검증, 부품을 잘라낸 사진만의 결과.\n\n'
            '갈색/흰색 브래킷 정상 참조를 추가했다. 원래11유형 중7개, 새 볼스크루 포함12유형 중8개가 사진20% 이내. '
            '위치 성능·내부 고장 진단과 구분한다. 검정 브래킷/튜브 후보는 제외, 금속 연결부는 결함TEST14장으로 부족.\n\n'
            '새 ZIP의 CPU API/C++45작업, Qt9항목, Python8항목, 전체볼스크루 CPU/GPU3236사진 비교를 기록했다. '
            '앱 기본값 배포·APK·새 pip 설치·현장 독립 검증은 하지 않았다.\n\n'
            '다음 우선순위는 JSON의 next_priorities를 읽는다. 새 실험 전에 프로토콜·실제 라벨·중복 분리를 고정하고 '
            '같은TEST 숫자를 맞추려고 경고 기준을 반복 조정하지 않는다. 현재 모델과 모든 기존 자료를 보존한다.\n')
    print(json.dumps({'final_audit': 'passed', 'stage20_categories': 8, 'actual_training_updates': 3600, 'CPP_operations': 45}), flush=True)


if __name__ == '__main__': main()
