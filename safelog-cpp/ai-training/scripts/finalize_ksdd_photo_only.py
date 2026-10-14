"""Finish the real-mask batch, retain failed replacement and advance night resume."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import study_ksdd_supervised as trained
from scripts import study_ksdd_photo_only as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new


def main():
    p = trained.validate(); study.validate()
    choice = study.read(study.REPORTS/'industrial-ksdd-photo-only-v11-selection.json')
    release = study.read(study.REPORTS/'industrial-ksdd-photo-only-v11-release.json')
    cpu = study.read(study.REPORTS/'industrial-ksdd-photo-only-v11-CPU.json')
    verification = study.read(study.REPORTS/'industrial-ksdd-supervised-v10-verification.json')
    training = study.read(trained.RUN/'trained/training.json'); data = study.read(trained.MANIFEST)['audit']
    assert release['status'] == cpu['status'] == verification['status'] == 'passed'
    assert release['model_sha256'] == cpu['model_sha256'] == choice['checkpoint_sha256'] == sha(ROOT/choice['checkpoint_path'])
    assert cpu['all_CPU_GPU_photo_decisions_equal'] and cpu['photos'] == 1004
    assert training['actual_optimizer_updates'] == 2560 and training['selected_epoch_by_validation'] == 13
    assert release['actual_current_CPP_operations'] == sum(v['operations'] for v in release['CPP_cases']) == 9
    assert study.read(study.REPORTS/'industrial-ksdd-supervised-v10-selection.json')['selected'] == 'prior_prototype'
    for name, digest in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == digest, name
    archive = Path(release['archive']); assert sha(archive) == release['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist()) == 1
        hashes = json.loads(z.read('package-sha256.json'))
        for name, digest in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
            if name.startswith('safelog_ai/') and name.endswith('.py') and name != 'safelog_ai/__init__.py': assert sha(ROOT/name) == digest, name
    old_archive = ROOT.parents[3]/'outputs/unified-industrial-night-role4-handoff-2026-10-15-v9.zip'
    assert sha(old_archive) == '1e865f5f2e79b42369bb92468e747249a45c311d83618b655a143a96c191a2c1'
    record = {'status': 'completed_real_mask_training_photo_only_selected', 'audited_utc': datetime.now(timezone.utc).isoformat(),
        'repo': str(ROOT.parents[1]), 'active_research_checkpoint': str(ROOT/choice['checkpoint_path']), 'active_checkpoint_sha256': choice['checkpoint_sha256'],
        'role4_ZIP': str(archive), 'role4_ZIP_sha256': release['archive_sha256'], 'role4_ZIP_bytes': archive.stat().st_size,
        'previous_v9_ZIP_preserved': True, 'one_model_one_shared_backbone': True, 'KSDD_backbone_passes_per_photo': 2,
        'real_mask_actual_training_epochs': 40, 'real_mask_actual_optimizer_updates': 2560, 'selected_epoch_by_validation': 13, 'selected_epoch_updates': 832,
        'v11_composition_updates': 0, 'previous_v8_training_updates': 3600, 'night_total_actual_updates_across_candidates': 6160,
        'source_data': data, 'full_v10_location_replacement_rejected': True, 'reason': 'new original-coordinate pixelFNR26.81percent worse than17.10percent parent',
        'v11_photo_score_and_threshold_exactly_from_frozen_v10_CAL': True, 'photo_rates': cpu['CPU_photo_rates'], 'prior_photo_rates': choice['prior_photo_rates'],
        'primary_original_pixel_metrics_unchanged': choice['primary_original_pixel_metrics_unchanged'],
        'auxiliary_supervised_pixel_metrics_not_primary': choice['auxiliary_new_mask_metrics_not_primary'],
        'all_old_neural_tensors_reference_buffers_and_operating_points_preserved': True, 'primary_KSDD_PNG_exact_to_parent_CPU': True,
        'known_TEST_used_for_followup_design_and_selection': True, 'physical_acquisition_groups_verified': False, 'independent_field_validated': False,
        'current_CPU_API_aliases_checked': 15, 'current_actual_CPP_operations': 9, 'prior_v9_CPP45_not_counted_as_current': True,
        'whole_CPU_GPU_photo_decisions_equal': True, 'CPU_GPU_compared_photos': 1004,
        'new_unit_tests_passed': 5, 'protected_files_checked': verification['protected_files'], 'legacy_source_SHA_rechecked': len(p['legacy_source_sha256']),
        'additional12_stage20_count_unchanged': 8, 'additional12_stage15_count_unchanged': 5,
        'app_default_promoted': False, 'Android_APK_built': False, 'fresh_pip_install_tested': False,
        'next_priorities': ['NormalFIT-only foreground/background audit for black bracket and tubes; preregister input-only masking before any new TEST scoring.',
            'Read official LOCO assembly constraints and consider relation-aware training; do not hardcode one normal image as a universal wiring rule.',
            'Acquire additional connector defect evaluation;14 photos remain insufficient. Whole-photo healthy belt FPR remains unestablished.'],
        'next_fit_review': str(study.REPORTS/'INDUSTRIAL_FOREGROUND_FIT_REVIEW_KO.md'),
        'completed_studies_do_not_rerun': ['industrial-bsclass-supervised-v8', 'industrial-surfaces-v9', 'industrial-ksdd-supervised-v10', 'industrial-ksdd-photo-only-v11'],
        'continuation_automation_id': 'safelog', 'keep_existing_bounded_awake_helper_no_duplicate': True}
    new(study.REPORTS/'industrial-ksdd-v10-v11-final-audit.json', record)
    outputs = ROOT.parents[3]/'outputs'
    new(outputs/'INDUSTRIAL_KSDD_RESUME_2026-10-15-v11.json', record)
    with (outputs/'INDUSTRIAL_KSDD_RESUME_2026-10-15-v11.md').open('x', encoding='utf-8', newline='\n') as out:
        out.write('# 실제 마스크 학습 후 재개 기록\n\n'+
            '활성 연구 모델 SHA: '+choice['checkpoint_sha256']+'\n\n'
            '40epoch·2560회 실제 학습, 검증에서 선택한13epoch·832회 시점. 사진 오탐12.64%→12.30%, 미탐10.91%→4.55%. '
            '전체1004장 CPU/GPU 판정 일치.\n\n'
            '새 위치 지도는 원본좌표 pixel미탐17.10%→26.81%로 악화돼 전체 교체에서 제외했다. '
            '사진 점수만 새로 쓰고 기본 위치·기준은 기존 것 그대로. 새 지도를 참고 출력으로 따로 표시한다. '
            '이 후속 조합은 이미 본TEST를 바탕으로 설계했고 새 학습/기준변경은0회다.\n\n'
            '새ZIP CPU15유형 API, 대표3유형 C++9작업 통과. 이전CPP45기록을 이번 검사로 세지 않는다. '
            '추가설비12유형 중8개 사진20% 이내 상태는 같다. 현장 독립성은 미검증.\n\n'
            '다음: 검정브래킷/튜브의 정상FIT 배경·부품영역을 검증하고 새 프로토콜로 진행한다. '
            'v8/v9/v10/v11 완료 작업은 다시 돌리지 않는다. 기존 절전방지 helper도 중복 실행하지 않는다.\n')
    print({'final_audit': 'passed', 'actual_new_updates': 2560, 'photo_rates': cpu['CPU_photo_rates']}, flush=True)


if __name__ == '__main__': main()
