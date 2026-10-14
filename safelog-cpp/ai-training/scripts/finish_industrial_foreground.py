"""Preserve the rejected foreground hypothesis and advance the resume record."""
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import study_industrial_foreground as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new


def main():
    p = study.validate(); report = study.read(study.REPORTS/'industrial-foreground-v12-validation.json')
    choice = study.read(study.REPORTS/'industrial-foreground-v12-selection.json')
    proof = study.read(study.REPORTS/'industrial-foreground-v12-verification.json')
    recovery = study.read(study.REPORTS/'industrial-foreground-v12-metric-recovery.json')
    assert proof['status'] == 'passed' and not choice['selected_categories']
    assert sha(study.START) == choice['checkpoint_sha256'] == study.EXPECTED
    assert sha(study.RUN/'normal-banks.pt') == recovery['normal_bank_sha256']
    assert sha(study.RUN/'normalCAL.json') == recovery['normal_CAL_sha256']
    assert not (study.RUN/'operation.lock').exists()
    for name, digest in recovery['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    outputs = ROOT.parents[3]/'outputs'
    previous = study.read(outputs/'INDUSTRIAL_KSDD_RESUME_2026-10-15-v11.json')
    archive = Path(previous['role4_ZIP']); assert sha(archive) == previous['role4_ZIP_sha256']
    rows = study.read(study.MANIFEST)['rows']; banks = __import__('torch').load(study.RUN/'normal-banks.pt', map_location='cpu', weights_only=True)
    results = {c: {k: report['categories'][c][k] for k in ('photo_rates_worst_case_manual_review_as_error', 'baseline_photo_rates', 'photo_threshold', 'CAL', 'manual_review_TEST_count', 'by_defect', 'guards')}
        for c in study.CATEGORIES}
    record = {'status': 'completed_foreground_experiment_rejected_prior_v11_retained', 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'repo': str(ROOT.parents[1]), 'active_research_checkpoint': str(study.START), 'active_checkpoint_sha256': study.EXPECTED,
        'role4_ZIP': str(archive), 'role4_ZIP_sha256': previous['role4_ZIP_sha256'], 'existing_ZIP_unchanged': True,
        'one_model_one_shared_backbone': True, 'selected_categories': [], 'new_neural_optimizer_updates': 0,
        'normal_reference_FIT_photos': {c: banks['records'][c]['normal_FIT_photos'] for c in study.CATEGORIES},
        'normal_reference_patches': {c: banks['records'][c]['patches'] for c in study.CATEGORIES},
        'raw_FIT_counts_after_new_group_audit': {c: sum(r['phase'] == 'fit' and r['category'] == c for r in rows) for c in study.CATEGORIES},
        'invalid_FIT_masks_excluded_not_called_healthy': 1, 'source_audit': p['source_data_audit'], 'candidate_results': results,
        'candidate_model_or_thresholds_retuned_after_TEST': False, 'known_TEST_reused_for_development': True,
        'inherited_parent_normal_FIT_had_newly_flagged_near_TEST_candidates': True,
        'near_pHash_candidates_are_not_verified_physical_duplicates': True,
        'new_reference_FIT_CAL_TEST_near_groups_disjoint': True,
        'partial_TEST_inference_repeated_only_after_metric_implementation_failure': True,
        'failure_and_calibration_bank_SHA_preserved': True, 'new_unit_tests_passed': 6,
        'parent_tensors_buffers_operating_points_and_old_CPU_locations_preserved': True,
        'protected_files_checked': proof['protected_files'], 'legacy_source_SHA_checked': proof['legacy_sources'],
        'KSDD_v11_photo_rates_unchanged': previous['photo_rates'], 'additional12_stage20_count_unchanged': 8,
        'new_device_or_CPP_package_verification': 'not needed: no candidate selected; prior verifiedZIP unchanged',
        'new_training_created_fault_labels': False, 'independent_field_validated': False, 'app_default_promoted': False,
        'next_priorities': ['Official LOCO relations are now documented in reports/LOCO_RELATION_NEXT_STUDY_KO.md. Audit data then preregister same-backbone global-context normal-FIT reconstruction training.',
            'Black bracket scratches remain difficult; tubes foreground removes background but increases false alerts. Do not retry thresholds until TEST rates match. Future studies require separate fixed protocols and appropriate data.',
            'Connector14 defect TEST photos are insufficient; whole-photo healthy belt FPR remains unknown.'],
        'completed_studies_do_not_rerun': previous['completed_studies_do_not_rerun']+['industrial-foreground-v12'],
        'continuation_automation_id': 'safelog', 'keep_existing_bounded_awake_helper_no_duplicate': True}
    new(study.REPORTS/'industrial-foreground-v12-final-audit.json', record)
    new(outputs/'INDUSTRIAL_FOREGROUND_RESUME_2026-10-15-v12.json', record)
    text = '''# 전경 비교 결과와 다음 재개 기록

기준 모델은 v11이며 기존 ZIP과 가중치를 유지한다. 새 후보는 채택하지 않았다.

| 유형 | 기존 사진 오탐/미탐 | 후보 사진 오탐/미탐 | 제외 이유 |
|---|---|---|---|
| 검정 브래킷 | 9.38% / 82.98% | 18.75% / 51.06% | CAL1장·TEST3장 판단 불가. 흠집 미탐23/35로 높음 |
| 튜브 | 3.12% / 68.12% | 46.88% / 17.39% | 정상32장 중15장 경고로 오탐 기준 초과 |

브래킷 후보 지표는 판단 불가 정상2장을 오탐, 결함1장을 미탐으로 세는 사전 고정 최악 조건이다. 판단 불가를 정상으로 바꾸지 않았다.

정상 FIT는 회전·반전 pHash근접 묶음 재감사 후88장/76장. 브래킷1장은 입력 영역 품질 실패로 새 특징 등록에서 제외했다. 실제 등록은87장/76장·9067개/8642개 특징. 이번 optimizer 추가 학습은0회다. 기존 학습된 공통 분석부에 정상 특징을 등록했다.

같은 공개 TEST를 재사용한 개발 비교다. 새 특징 등록에는 TEST근접 묶음을 제외했지만 부모 모델의 과거 정상 FIT에는 새 감사에서 근접 후보로 잡힌 사진이 있었다. pHash유사는 동일한 물리 부품의 증명이 아니고, 새 독립 성능으로 주장할 수 없다.

평가 집계 오류를 수정했으며 모델·전처리·보정 기준은 바꾸지 않았다. 오류 기록과 고정 조건·등록 특징·CAL 기준을 모두 보존한다. 실패 때문에 일부 동일 추론을 복구 실행한 것은 학습을 더 한 회차로 세지 않는다.

이전 출력·가중치·ZIP은 유지하며 추가설비12유형 중8개가 사진20% 이내라는 상태도 같다. 입력 전경은 실제 부품이나 결함의 정답 마스크가 아니다. 새 위치 지표를 검증한 것으로 쓰지 않는다.

다음은 공식 LOCO의 연결부 수·클램프 수·전선 색·연결 상대 위치 규칙을 참고해 전체 문맥 복원 분기를 같은 공통 분석부에 실제 학습하는 실험이다. 관련 내용은 LOCO_RELATION_NEXT_STUDY_KO.md에 있으며 아직 새 분기를 학습하거나 성능을 측정한 것은 아니다.
'''
    path = study.REPORTS/'INDUSTRIAL_FOREGROUND_RESULTS_KO.md'
    with path.open('x', encoding='utf-8', newline='\n') as out: out.write(text)
    with (outputs/'INDUSTRIAL_FOREGROUND_RESUME_2026-10-15-v12.md').open('x', encoding='utf-8', newline='\n') as out: out.write(text)
    print({'completed': True, 'selected_categories': [], 'active_v11_SHA': study.EXPECTED, 'new_optimizer_updates': 0}, flush=True)


if __name__ == '__main__': main()
