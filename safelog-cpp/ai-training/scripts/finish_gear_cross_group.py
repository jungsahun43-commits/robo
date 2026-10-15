"""Preserve the actual v33 trial and gate result without re-running inference."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import study_gear_cross_group as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    selected = read(study.RUN / 'selection.json')
    verified = read(study.RUN / 'verification.json')
    train = selected['training']
    assert verified['status'] == 'passed' and train['actual_optimizer_updates'] >= 1920
    assert not (study.RUN / 'operation.lock').exists()
    skipped = selected.get('CAL_TEST_skipped', False)
    eligible = selected['eligible_pending_package_checks']
    if skipped:
        assert train['selected_epoch'] == 0
        assert not (study.RUN / 'test-scores.json').exists()
        result_sentence = ('검증 BCE가 초기값보다 나아지지 않아 **epoch0을 선택**했다. '
                           '새 학습 개선으로 설명하지 않는다. 사전 조건에 따라 새 CAL/TEST 점수는0개이며 '
                           '기존 기어 TEST 숫자를 이번 학습 결과로 다시 표시하지 않는다.')
    else:
        rates = selected['rates']
        result_sentence = (f"선택 가중치에 정상 CAL85% 순서통계량 cut {selected['threshold']}을 먼저 저장한 뒤 "
                           f"기어 TEST{selected['TEST_patches']:,}패치를 평가했다. "
                           f"오탐 **{100*rates['fpr']:.2f}% ({rates['fp']}/{rates['negative_photos']})**, "
                           f"미탐 **{100*rates['fnr']:.2f}% ({rates['fn']}/{rates['positive_photos']})**, "
                           f"AUROC {selected['AUROC']:.5f}다. 이전 labeled minimax 보정과 달라 수치 차이의 "
                           '전부를 새 학습 효과로 설명하지 않는다.')
    result = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'v33_pending_release_checks' if eligible else 'v33_complete_rejected_v16_retained',
        'preregistered_commit': '05a68297e2d584ea57f6d7ddd13d3d43dd98748d',
        'active_model': str(study.START), 'active_model_sha256': study.EXPECTED,
        'active_role4_ZIP': str(ROOT.parents[3] / 'outputs/unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
        'active_role4_ZIP_sha256': '40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
        'research_delta': str(study.CHECKPOINT), 'research_delta_sha256': sha(study.CHECKPOINT),
        'research_delta_needs_parent_not_standalone_app': True,
        'delta_verification': read(study.RUN / 'trained/delta-verification.json'),
        'training': train, 'selection': selected, 'verification': verified,
        'this_turn_actual_optimizer': train['actual_optimizer_updates'],
        'tracked_partial_research_optimizer_including_failed': 37840 + train['actual_optimizer_updates'],
        'tracked_sum_not_entire_history': True,
        'new_CAL_scores': 0 if skipped else 1116,
        'new_TEST_scores': 0 if skipped else selected['TEST_patches'],
        'new_RGB_encoder_features': 0, 'new_API_CPP_ZIP_checks': 0, 'new_ZIPs': 0,
        'old_app_types_original_positions_KS_BS_LOCO_8of12_targets_unchanged': True,
        'new_independent_field_validation': False, 'original_labels_and_failed_weights_preserved': True,
        'awake_helper_restarted': False, 'free_disk_bytes': shutil.disk_usage(ROOT).free,
        'next': ('Complete independent package CPU/GPU/C++/old-alias/default-location preservation checks before any adoption.'
                 if eligible else 'Do not repeat old architectures, thresholds or loss coefficients. Review this FIT/VAL training trace and label/group support; prepare one evidence-based next study or a human-review handoff without TEST selection.')
    }
    for path in (study.RUN / 'final-audit.json', ROOT / 'reports/industrial-gear-cross-group-v33-final-audit.json',
                 ROOT.parents[3] / 'outputs/INDUSTRIAL_GEAR_CROSS_GROUP_RESUME_2026-10-15-v33.json'):
        new(path, result)
    decision = ('사진 판정 조건은 통과했으며 새 패키지 검사를 통과하기 전까지 앱에는 반영하지 않는다.'
                if eligible else '사전 채택 조건을 충족하지 못해 제외했으며 기존 앱용 v16-r1을 유지한다.')
    text = f'''# v33 실제 cross-group 보조 학습 결과

05a6829에 코드·자료·손실·선택·보정·저장 조건을 먼저 commit/push한 뒤 **실제 {train['actual_optimizer_updates']:,}회 / {train['completed_epochs']}epoch**를 학습했다. 선택 epoch는 **{train['selected_epoch']}**, 선택 가중치의 새 갱신 수는{train['selected_actual_updates']}회다. VAL macro label/group BCE는 초기{train['initial_VAL_macro_group_BCE']:.8f}에서 선택{train['VAL_macro_group_BCE']:.8f}다.

{result_sentence}

{decision}

원래 같은 encoder의 stage3 뒤 adapter220개와 v28 연구 선형부3,073개만 학습했다. 한 번의 후반 encoder에서 얻은 동일3,072차원 특징을 BCE와 보조 손실에 공유했다. 라벨별 다른4묶음×2패치의16배치를 쓰며 동일 묶음의 모든 쌍은 보조 손실의 분자/분모에서 제외했다. 원저자 라벨만 사용했고 그룹 ID는 샘플링/손실에만 썼다. 보조 손실(T=.2, 계수.1)과 배치를 함께 바꾼 연구라 각각의 인과 효과를 분리해 입증하지 않는다.

accepted v16의 모든 tensor/buffer/metadata와 v28 FIT 통계가 정확히 보존됐다. 새 delta {study.CHECKPOINT.stat().st_size:,}bytes를 부모 SHA에 결합해 저장한 뒤 다시 읽어 모든 state/metadata가 정확히 복원됨을 확인했다. 부모가 필요한 연구본이며 앱 전달파일이 아니다. 기존 원본·전체 가중치·실패모델·ZIP을 삭제하지 않았다.

기존 Python {verified['legacy_python_SHA_checked']}파일 SHA, 보호 {verified['protected_stat_unchanged']}파일 stat·변경 재해시{verified['changed_metadata_SHA_rechecked']}개, 기존 fine 캐시 SHA·실제 optimizer 장부·신규 unit3개와 기존 보조손실 unit3개를 확인했다. 보호 원본 전체 SHA를 모두 재계산한 것은 아니다. 새 RGB 특징0, 새 API/C++/ZIP 검사0, 새 ZIP0이다.

이번 실제 갱신 {train['actual_optimizer_updates']:,}회와 추적 일부 연구 합{result['tracked_partial_research_optimizer_including_failed']:,}회는 실패를 포함한다. 전체 역사 학습 총량이나 새 사진 수로 설명하지 않는다. epoch/배치 사이 재추출은 기존 실제 패치를 반복 사용한 것이다. FIT5,466/VAL1,440/CAL1,116/원래 gear TEST2,179의 분리와 모든 제외자료·원라벨은 유지했다.

범위는128×128 기어 표면 패치 마모 허용이며 전체 장비 건강·내부 고장·임의 현장 전체사진·검증된 결함 위치가 아니다. 공개 TEST 반복 개발, 과거 부모의 근접 노출/촬영 독립성 미검증, TEST4묶음·VAL3묶음의 한계가 남는다. 독립 현장 성능을 주장하지 않는다. 기존 브래킷·튜브·위치 지도·KS2·볼스크루·LOCO와 추가12유형 중8개가20% 이내인 상태는 그대로다.

Re-GBCv1 CC BY4.0의 출처·저자·버전·변경 표시와 DOI10.1109/ETFA65518.2025.11205777을 유지한다. 다른 기존 출처의 이용 조건도 각각 유지한다. 다음 단계는 실패 후보를 같은 TEST에 맞춰 반복 변경하기 전에 이번 FIT/VAL 장부와 데이터 지원 범위를 검토하는 것이다.
'''
    with (ROOT / 'reports/GEAR_CROSS_GROUP_V33_RESULTS_KO.md').open('x', encoding='utf-8', newline='\n') as out:
        out.write(text)
    print({'actual_optimizer': train['actual_optimizer_updates'], 'selected_epoch': train['selected_epoch'],
           'CAL_TEST_skipped': skipped, 'eligible_pending_package_checks': eligible,
           'research_delta_bytes': study.CHECKPOINT.stat().st_size}, flush=True)


if __name__ == '__main__': main()
