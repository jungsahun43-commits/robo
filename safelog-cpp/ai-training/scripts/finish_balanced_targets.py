"""Seal v21 results; model adoption remains separate from research selection."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import study_balanced_targets as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
OUTPUTS = ROOT.parents[3] / 'outputs'


def main():
    study.validate()
    s = study.read(study.RUN / 'selection.json'); v = study.read(study.RUN / 'verification.json')
    t = s['training']; gradient = study.read(study.RUN / 'FIT-gradient-audit.json')
    assert v['status'] == 'passed' and t['actual_optimizer_updates'] >= 1920 and t['completed_epochs'] == 30
    path = study.RUN / 'trained/final.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    warm = torch.load(study.WARM, map_location='cpu', weights_only=True)
    fine_changed = not torch.equal(state['state_dict']['fine_target_head.fine_projection.0.weight'], warm['state_dict']['fine_target_head.fine_projection.0.weight'])
    extra_weight = float(state['state_dict']['fine_target_head.fusion.0.weight'][:, 73:].abs().sum())
    status = 'v21_completed_candidate_pending_fresh_package_checks' if s['selected'] else 'v21_completed_all_candidates_rejected_v16_retained'
    result = {'status': status, 'UTC': datetime.now(timezone.utc).isoformat(), 'preregistered_commit': 'de35fee',
        'active_research_checkpoint': str(study.START), 'active_checkpoint_sha256': study.EXPECTED,
        'active_role4_ZIP': str(OUTPUTS / 'unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
        'active_role4_ZIP_sha256': '40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
        'v21_research_checkpoint': str(path), 'v21_research_checkpoint_sha256': sha(path),
        'initial_v20_VAL_selected_checkpoint_sha256': study.WARM_SHA,
        'actual_optimizer_updates': t['actual_optimizer_updates'], 'selected_epoch': t['selected_epoch'],
        'retained_selected_epoch_updates': t['selected_epoch_updates'],
        'initial_VAL_loss_new_loss': study.read(study.RUN / 'train/initial-VAL.json')['balanced_loss'],
        'best_VAL_loss_new_loss': t['VAL_balanced_loss'], 'new_loss_not_numerically_comparable_to_v20_loss': True,
        'fine_projection_changed_from_v20_initialization': fine_changed, 'selected_fine_fusion_extra_channels_L1': extra_weight,
        'FIT_gradient_audit': gradient, 'training_labels': 'same publisher normal and procedural visual edits, not natural target fault GT',
        'data_manifest_sha256': study.MANIFEST_SHA, 'new_natural_target_fault_GT': 0,
        'FIT_normal': 131, 'FIT_synthetic': 478, 'VAL_normal': 33, 'VAL_synthetic': 129,
        'new_encoder_feature_extractions': 0, 'reused_v20_fine_cache_files': 6,
        'selected_methods': s['selected'], 'target_results': s['categories'], 'verification': v,
        'new_unit_tests_passed': 3, 'new_API_aliases_tested': 0, 'new_CPP_operations': 0, 'new_role4_ZIP_created': False,
        'all_parent_tensors_buffers_metadata_exact': True, 'primary_locations_and_thresholds_unchanged': True,
        'tracked_v8_v10_v13_v15_v16_v17_v18_v19_v20_v21_actual_optimizer_updates_including_failed': 20560 + t['actual_optimizer_updates'],
        'known_public_TEST_reused_for_adoption': True, 'independent_field_validated': False, 'natural_defect_VAL_available': False,
        'physical_synthetic_fault_realism_verified': False, 'all_original_failed_data_models_reports_preserved': True,
        'awake_helper_restarted': False, 'v14_human_relation_review_pending': True,
        'next': 'Selected candidate requires fresh packaged CPU/GPU/C++ and original location checks before release.' if s['selected'] else 'Do not rerun v20/v21 or tune current TEST. Study FIT-specific normal/edit separation and fixed-view representation limits before choosing further training or actual data enrichment.'}
    for target in (study.RUN / 'final-audit.json', ROOT / 'reports/industrial-balanced-targets-v21-final-audit.json', OUTPUTS / 'INDUSTRIAL_BALANCED_RESUME_2026-10-15-v21.json'):
        new(target, result)
    lines = ['# v21 小영역 학습 비중 보강 결과'.replace('小영역', '작은 영역'), '',
        '2026-10-15. 조건·코드를de35fee에 먼저 commit/push한 뒤 실제 학습했다.', '', '## 실제 학습', '',
        f"실제 optimizer **{t['actual_optimizer_updates']}회**,30epoch. 합성 VAL 선택 epoch **{t['selected_epoch']}**, 선택 가중치의 예정 갱신{t['selected_epoch_updates']}회.",
        f"새 loss의 VAL {result['initial_VAL_loss_new_loss']:.6f} → {result['best_VAL_loss_new_loss']:.6f}. 이전 v20과 loss 정의가 달라 loss 숫자 자체를 서로 비교하지 않는다.", '',
        '기존 v20의 세밀한 판별부 구조와 VAL 초기값을 유지하고 픽셀 BCE의 양성·배경 평균을 사진마다 동등하게 반영했다. 정상 사진은 배경 평균만 쓴다. 새 fine projection·fusion만 학습했고 기존 모든 부모 가중치·buffer·메타데이터와 high projection은 고정했다.', '',
        'FIT609/VAL162는 원래 정상·절차적 변형 자료를 그대로 썼다. 새 실제 자연 고장 라벨은0개다. 기존 coarse와 v20 fine 캐시6개를 SHA로 검증해 재사용했으며 새 encoder 특징 추출은0회다. TEST 점수는 새 CAL을 저장한 뒤 계산했다.', '',
        '## FIT 학습 신호 진단', '',
        '| 출력 logit gradient | 양성 절댓값 합 | 배경 절댓값 합 | 양성/배경 |', '|---|---|---|---|']
    for method, values in gradient['methods'].items():
        lines.append(f"| {method} | {values['positive_gradient_abs_mass']:.6f} | {values['negative_gradient_abs_mass']:.6f} | {values['positive_to_negative_mass']:.6f} |")
    lines += ['', '고정 FIT8입력의 학습 전 진단이며 모델 가중치 갱신0회다. 양성 신호가 커지는 것은 설계한 loss의 작동 확인이고 자연 결함 성능 개선의 증거가 아니다.', '',
        '## 고정 CAL 이후 공개 TEST', '', '| 유형 | 방법 | 오탐 | 미탐 | 조건 통과 |', '|---|---|---|---|---|']
    for category, entry in s['categories'].items():
        for method, values in [('현재 v16', {'rates': entry['parent'], 'eligible': None}), *entry['methods'].items()]:
            r = values['rates']
            lines.append(f"| {category} | {method} | {100*r['fpr']:.2f}% ({r['fp']}/{r['negative_photos']}) | {100*r['fnr']:.2f}% ({r['fn']}/{r['positive_photos']}) | {values['eligible']} |")
    lines += ['', '선택 방법: `' + json.dumps(s['selected'], ensure_ascii=False) + '`.', '',
        '통과 후보가 있어도 새 ZIP의 실제 CPU/GPU·C++ 연동과 원래 위치 PNG 보존을 검사한 뒤에만 반영한다. 이 결과 기록 시 새 API/C++/ZIP 검사는 각각0이며 현재 v16-r1을 유지한다.', '',
        '## 보존·한계', '',
        f"기존 Python {v['legacy_python_SHA_checked']}파일 SHA, 보호 {v['protected_stat_unchanged']}파일 stat 동일, 메타데이터 변경 재해시{v['changed_metadata_SHA_rechecked']}개를 확인했다. 보호 자료 전체 SHA를 다시 계산한 것은 아니다. 모든 부모 tensor/buffer/metadata·고정 high projection·실제 optimizer 장부·새 loss unit3개를 검증했다.", '',
        '원래 target 위치 지도와 pixel 기준을 유지하고 합성 지도를 자연 결함의 기본 지도로 승격하지 않았다. 공개 TEST 반복 개발로 얻은 수치이며 독립 현장 성능이 아니다. 합성 VAL, 자연 손상 실재성 미검증, 물리 촬영 그룹과 부모의 과거 근접 노출, KSDD1 TEST 결함18장 및 v14 사람 검수 대기의 한계를 그대로 남긴다.', '',
        'KSDD1 CC BY-NC-SA4.0·비상업 연구/인용/동일조건·DOI10.1007/s10845-019-01476-x를 유지한다. 원본·이전 가중치·실패 후보·보고서를 모두 보존했다.']
    with (ROOT / 'reports/BALANCED_TARGET_RESULTS_KO.md').open('x', encoding='utf-8', newline='\n') as out:
        out.write('\n'.join(lines) + '\n')
    print({k: val for k, val in result.items() if k not in ('verification', 'target_results', 'FIT_gradient_audit')}, flush=True)


if __name__ == '__main__':
    main()
