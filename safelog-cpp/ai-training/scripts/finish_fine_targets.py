"""Record completed v20 learning without promoting an unchecked release."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import study_fine_targets as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
OUTPUTS = ROOT.parents[3] / 'outputs'


def main():
    study.validate()
    s = study.read(study.RUN / 'selection.json')
    v = study.read(study.RUN / 'verification.json')
    t = s['training']
    assert v['status'] == 'passed' and t['actual_optimizer_updates'] >= 1920
    assert t['completed_epochs'] == 30
    checkpoint = study.RUN / 'trained/final.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    extra_weight = float(state['state_dict']['fine_target_head.fusion.0.weight'][:, 73:].abs().sum())
    # The fine projection is seeded before initialization exactly as in study.train.
    torch.manual_seed(study.SEED)
    initialized, _ = study.load(initialize=True)
    projection_changed = not torch.equal(initialized.state_dict()['fine_target_head.fine_projection.0.weight'],
                                         state['state_dict']['fine_target_head.fine_projection.0.weight'])
    del initialized
    status = 'v20_completed_candidate_pending_fresh_package_checks' if s['selected'] else 'v20_completed_all_candidates_rejected_v16_retained'
    result = {'status': status, 'UTC': datetime.now(timezone.utc).isoformat(), 'preregistered_commit': '0c8c709',
        'active_research_checkpoint': str(study.START), 'active_checkpoint_sha256': study.EXPECTED,
        'active_role4_ZIP': str(OUTPUTS / 'unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
        'active_role4_ZIP_sha256': '40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
        'v20_research_checkpoint': str(checkpoint), 'v20_research_checkpoint_sha256': sha(checkpoint),
        'actual_optimizer_updates': t['actual_optimizer_updates'], 'selected_epoch': t['selected_epoch'],
        'retained_selected_epoch_updates': t['selected_epoch_updates'], 'initial_VAL_loss': study.read(study.RUN / 'train/initial-VAL.json')['balanced_loss'],
        'best_VAL_loss': t['VAL_balanced_loss'], 'new_fine_projection_changed_from_seeded_initialization': projection_changed,
        'initially_zero_fine_fusion_weights_selected_L1': extra_weight,
        'training_labels': 'reused v19 publisher normal surfaces and procedural edit masks, not natural target defect GT',
        'data_manifest_sha256': study.MANIFEST_SHA, 'new_natural_target_fault_GT': 0,
        'FIT_normal': 131, 'FIT_synthetic': 478, 'VAL_normal': 33, 'VAL_synthetic': 129,
        'selected_methods': s['selected'], 'target_results': s['categories'], 'verification': v,
        'new_unit_tests_passed': 3, 'new_API_aliases_tested': 0, 'new_CPP_operations': 0, 'new_role4_ZIP_created': False,
        'all_parent_tensors_buffers_metadata_exact': True, 'primary_locations_and_thresholds_unchanged': True,
        'tracked_v8_v10_v13_v15_v16_v17_v18_v19_v20_actual_optimizer_updates_including_failed': 18640 + t['actual_optimizer_updates'],
        'known_public_TEST_reused_for_adoption': True, 'independent_field_validated': False,
        'natural_defect_VAL_available': False, 'physical_synthetic_fault_realism_verified': False,
        'sandbox_launch_without_output_or_optimizer': study.read(study.RUN / 'launch-observation-r1.json'),
        'all_original_data_failed_candidates_reports_preserved': True, 'awake_helper_restarted': False,
        'v14_human_relation_review_pending': True, 'automation_continues_user_at_school': True,
        'next': 'Fresh release CPU/GPU/C++ and preserved location checks before adopting selected methods.' if s['selected'] else 'Do not rerun v20 or tune current TEST thresholds; investigate FIT-only synthetic/normal separation and an appropriate next data or representation step before preregistering more training.'}
    for path in (study.RUN / 'final-audit.json', ROOT / 'reports/industrial-fine-targets-v20-final-audit.json', OUTPUTS / 'INDUSTRIAL_FINE_RESUME_2026-10-15-v20.json'):
        new(path, result)
    lines = ['# v20 세밀한 특징 학습 결과', '',
        '2026-10-15. 단일 후보 조건과 코드를0c8c709에 먼저 commit/push한 뒤 실제 학습·평가했다.', '',
        '## 실제 학습', '',
        f"30epoch / 실제 optimizer **{t['actual_optimizer_updates']}회**. 합성 VAL에서 고른 epoch는 **{t['selected_epoch']}**(선택 가중치에 반영된 예정 갱신{t['selected_epoch_updates']}회)다.",
        f"VAL loss {result['initial_VAL_loss']:.6f} → {result['best_VAL_loss']:.6f}. 새 fine projection 변경={projection_changed}, 처음0인 fine 입력 가중치 L1={extra_weight:.6f}.", '',
        '같은 공통 encoder의 stage3 fine24와 기존 low/high·설비 복원 잔차를 사용했다. 새 fine projection·fusion만 갱신했고 기존 backbone/분기/high projection/부모 메타데이터는 그대로다. FIT609/VAL162는 v19의 고정 정상·절차적 변형 자료이며 자연 고장 정답을 새로 추가한 것은 아니다.', '',
        '## 고정 CAL 이후 공개 TEST', '',
        '| 유형 | 방법 | 오탐 | 미탐 | 조건 통과 |', '|---|---|---|---|---|']
    for c, entry in s['categories'].items():
        for method, val in [('현재 v16', {'rates': entry['parent'], 'eligible': None}), *entry['methods'].items()]:
            r = val['rates']
            lines.append(f"| {c} | {method} | {100*r['fpr']:.2f}% ({r['fp']}/{r['negative_photos']}) | {100*r['fnr']:.2f}% ({r['fn']}/{r['positive_photos']}) | {val['eligible']} |")
    lines += ['', '선택 방법: `' + json.dumps(s['selected'], ensure_ascii=False) + '`.', '',
        '선택된 후보가 있어도 새 패키지 CPU/GPU·실제 C++와 원래 위치 PNG 보존을 검사하기 전까지 기존 v16-r1 전달물을 유지한다. 합성 위치 지도를 자연 결함의 기본 위치 정답으로 승격하지 않았다.', '',
        '## 검증과 한계', '',
        f"기존 Python {v['legacy_python_SHA_checked']}파일 SHA 검사, 보호 {v['protected_stat_unchanged']}파일 stat 동일·메타데이터 변경 {v['changed_metadata_SHA_rechecked']}개 재해시. 보호 원본 전체 SHA를 모두 다시 읽었다는 의미는 아니다. 실제 optimizer 장부·새/기존 캐시 SHA·모든 부모 tensor/buffer/metadata·고정 high projection을 확인했다. 새 unit3개 통과. 이 단계의 새 API 검사0, C++ 작업0, ZIP0이다.", '',
        '캐시별 사진 수, fresh FP16 low/high/복원 잔차 최대 차이와 bit 일치는 verification JSON의 fine_cache_receipts에 남겼다. 허용치는 학습 전에 atol5e-4/rtol1e-3으로 고정했고 기존 캐시를 재생성하지 않았다.', '',
        '첫 sandbox 백그라운드 실행은 stdout/stderr0byte, operation lock 없이 종료됐고 실제 학습0회였다. 종료된 PID를 확인한 뒤 같은 코드·조건을 host에서 실행했다. 원인은 확정하지 않았으며 원래 로그·관찰 기록을 보존했다.', '',
        '공개 TEST 반복 개발 결과이며 독립 현장 성능이 아니다. 합성 VAL은 실제 자연 결함 VAL을 대신하지 않는다. 물리 촬영 그룹 미검증과 부모의 과거 근접 자료 노출, KSDD1 결함18장 표본 한계, v14 관계 라벨 사람 검수 대기는 해소되지 않았다. 같은 TEST에 임계값을 맞추거나 통과 조건을 완화하지 않았다.', '',
        'KSDD1 CC BY-NC-SA4.0 비상업 연구/인용/동일조건 및 DOI10.1007/s10845-019-01476-x를 유지한다. 원자료·가중치·기존 실패 후보·ZIP은 모두 보존한다.']
    target = ROOT / 'reports/FINE_TARGET_RESULTS_KO.md'
    with target.open('x', encoding='utf-8', newline='\n') as out:
        out.write('\n'.join(lines) + '\n')
    print({k: val for k, val in result.items() if k not in ('verification', 'target_results')}, flush=True)


if __name__ == '__main__':
    main()
