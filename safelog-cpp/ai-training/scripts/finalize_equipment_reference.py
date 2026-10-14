"""Audit selected reference prototype and write resumable local handoff receipts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.study_equipment_reference_budget import RUN, REPORTS, START, EXPECTED, validate, check
from scripts.study_equipment_expansion import read, MANIFEST
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.release_equipment_reference import ARCHIVE


def main():
    p = validate(); choice = read(REPORTS/'equipment-reference-budget-v2-selection.json')
    release = read(REPORTS/'equipment-reference-budget-v2-release.json')
    assert release['status'] == 'passed' and release['actual_CPP_operations_this_release'] == 21 and release['owned_server_stopped']
    assert sha(ARCHIVE) == release['archive_sha256']
    checkpoint = ROOT/choice['checkpoint_path']; digest = sha(checkpoint)
    assert digest == choice['checkpoint_sha256'] == release['model_sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True); prior = torch.load(START, map_location='cpu', weights_only=True)
    assert sha(START) == EXPECTED and all(torch.equal(v, state['state_dict'][n]) for n, v in prior['state_dict'].items())
    rows = read(MANIFEST)['rows']; ids = {r['id']: r for r in rows}
    for category, reference in state['normal_reference_fit'].items():
        source_rows = [ids[n] for n in reference['source_ids']]
        assert all(r['phase'] == 'fit' and r['label'] == 0 and r['category'] == category for r in source_rows)
        assert len(source_rows)*256 == state['reference_counts'][category]
    assert sum(len(v['source_ids']) for v in state['normal_reference_fit'].values()) == 354
    ledger = read(RUN/'protected-inputs.json')
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda item: check(*item), ledger.items()))
    for name, h in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == h, name
    evaluation = read(RUN/'selected/evaluation.json')
    assert len(evaluation['categories']) == 11
    assert sum(v['stage_gate']['ready_for_next_stage'] for v in evaluation['categories'].values()) == 4
    assert sum(v['stage_gate']['state'] == 'target15_met' for v in evaluation['categories'].values()) == 3
    rates = {c: evaluation['categories'][c]['photo_rates'] for c in choice['reference_counts']}
    audit = {'status': 'passed', 'audited_utc': datetime.now(timezone.utc).isoformat(),
        'model_sha256': digest, 'archive_sha256': release['archive_sha256'], 'archive_bytes': ARCHIVE.stat().st_size,
        'photo_rates': rates, 'stage20_categories': choice['stage20_categories'], 'stage15_category_count': 3,
        'all_11_categories_passed': False, 'normal_FIT_photos': 354, 'normal_FIT_patches': 90624, 'normal_CAL_counts': {'ad_cable': 45, 'ad_metal_nut': 44},
        'new_neural_optimizer_updates': 0, 'normal_reference_fit_is_not_gradient_training': True,
        'new_source_photos_or_labels': 0, 'all_prior_neural_tensors_exact': True, 'all_original_location_outputs_preserved': True,
        'protected_files_SHA_and_mtime_preserved': len(ledger), 'legacy_python_source_SHA_preserved': len(p['legacy_source_sha256']),
        'new_unit_tests': 3, 'retained_reference_unit_tests': 4, 'new_unit_test_optimizer_updates': 0,
        'actual_CPP_operations_this_release': 21, 'isolated_ZIP_CPU_API_verified': True,
        'first_smoke_checker_failure_excluded': True, 'first_smoke_CPP_operations': 0, 'retry_same_ZIP_SHA': True,
        'PNG_visually_inspected': True, 'researchPrototype': True, 'known_TEST_reused_for_development_and_selection': True,
        'independent_field_validated': False, 'Android_APK_built': False, 'fresh_pip_install_tested': False, 'app_default_promoted': False,
        'next_priority': 'Prespecified normal-reference photo presence on screw/connectors; keep location improvement and fresh field evaluation separate'}
    new(REPORTS/'equipment-reference-budget-v2-final-audit.json', audit)
    outputs = ROOT.parents[3]/'outputs'
    # ROOT is <workspace>/work/robo-ai-engine/safelog-cpp/ai-training.
    assert outputs.resolve() == ARCHIVE.parent.resolve()
    resume = {'completed': 'equipment-reference-budget-v2', 'repo': str(ROOT.parents[1]), 'selected_checkpoint': str(checkpoint),
        'checkpoint_sha256': digest, 'archive': str(ARCHIVE), 'archive_sha256': release['archive_sha256'], 'photo_rates': rates,
        'selected_policies': choice['selected_per_category'], 'stage20_categories': choice['stage20_categories'],
        'new_neural_updates': 0, 'normalFIT_reference_photos': 354, 'known_TEST_reused': True, 'location_outputs_preserved_not_improved': True,
        'unit_tests_passed': 7, 'CPP_operations_passed': 21, 'first_smoke_failure_excluded': True,
        'result_report': str(REPORTS/'EQUIPMENT_REFERENCE_RESULTS_KO.md'), 'role4_guide': str(ROOT.parents[1]/'safelog-cpp/docs/EQUIPMENT_REFERENCE_HANDOFF_KO.md'),
        'git_local_branch': 'codex/ai-engine', 'git_remote_branch': 'feature/ai-engine', 'app_default_promoted': False,
        'next': 'screw/connectors photo presence, with a new fixed protocol; preserve current passed categories and full location maps'}
    new(outputs/'EQUIPMENT_REFERENCE_RESUME_2026-10-14.json', resume)
    with (outputs/'EQUIPMENT_REFERENCE_RESUME_2026-10-14.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write('# 케이블·너트 보강 완료\n\n- 케이블 사진 오탐6/58=10.34%, 미탐7/92=7.61%.\n- 너트 사진 오탐3/22=13.64%, 미탐11/93=11.83%.\n'
            '- 신규11유형 중4개 사진20% 이내, 그중3개15% 이내.\n- 정상FIT354장의 특징 저장과 정상CAL 경고 기준 보정. 추가 신경망 업데이트0회.\n'
            '- 알려진 공개TEST를 반복 사용한 개발 결과. 새 산업현장 성능 아님. 위치 결과 유지, 위치 정확도 개선 아님.\n'
            '- 단위 검사7개, 분리 ZIP CPU API와 C++21개 요청 통과. 최초 검사 코드 오류는 제외하고 같은 ZIP으로 재검사.\n'
            '- 새 전달물: '+ARCHIVE.name+'\n- 모델SHA: '+digest+'\n- ZIP SHA: '+release['archive_sha256']+'\n'
            '- 다음 우선순위: 나사·연결부 사진 미탐 보강. 기존 통과 유형과 위치 출력 보존.\n')
    print(json.dumps({'audit': 'passed', 'protected_files': len(ledger), 'stage20': 4, 'stage15': 3, 'CPP_operations': 21}), flush=True)


if __name__ == '__main__': torch.set_num_threads(4); main()
