"""Conclude narrower v28 without turning its subset into a full-source pass."""
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_gear_only as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    selection=read(study.RUN/'selection.json');verification=read(study.RUN/'verification.json')
    assert verification['status']=='passed' and not (study.RUN/'operation.lock').exists()
    training=selection['training'];assert training['actual_optimizer_updates']==1920
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'v28_gear_only_complete_rejected_v16_retained' if not selection['eligible_pending_package_checks'] else 'v28_pending_package_checks',
       'preregistered_commit':'ec4dfd7','active_model':str(study.prior.START),'active_model_sha256':study.prior.EXPECTED,
       'active_role4_ZIP':str(ROOT.parents[3]/'outputs/unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
       'active_role4_ZIP_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
       'research_checkpoint':str(study.RUN/'trained/final.pt'),'research_checkpoint_sha256':sha(study.RUN/'trained/final.pt'),
       'training':training,'selection':selection,'verification':verification,
       'source_manifest_sha256':sha(study.MANIFEST),'data_audit':read(study.MANIFEST)['audit'],
       'v25_stopped_data_gate_optimizer0':True,'v26_v27_optimizer_completed_each':1920,
       'v26_v27_selected_epoch0_initial_not_learned_checkpoint':True,
       'this_turn_actual_optimizer_updates_v26_v27_v28':5760,
       'tracked_partial_research_optimizer_including_failed':32080,'tracked_updates_not_entire_history':True,
       'unique_photos_not_multiplied_across_reused_studies':True,
       'old_full_source_TEST3109_and_failures_preserved':True,'new_gear_subset_TEST2179_not_overall_source_improvement':True,
       'new_API_CPP_ZIP_checks':0,'new_ZIPs':0,'existing_additional_types_within20percent':8,'existing_additional_types_total':12,
       'old_bracket_tube_KS_BS_LOCO_and_primary_outputs_unchanged':True,
       'new_independent_field_validation':False,'awake_helper_restarted':False,
       'next':'FIT-only spatial fine-feature feasibility probe, separately preregistered before reading features; do not repeat pooling/threshold variants. Actual weak class labels are not pixel masks. Full new training protocol remains to be fixed after probe.'}
    new(study.RUN/'final-audit.json',r);new(ROOT/'reports/industrial-gear-only-v28-final-audit.json',r)
    new(ROOT.parents[3]/'outputs/INDUSTRIAL_GEAR_ONLY_RESUME_2026-10-15-v28.json',r)
    rates=selection['rates'];ci=selection['group_bootstrap_95percent']
    text=f'''# v28 기어 표면 패치 실제 학습 결과

실제 **1,920 optimizer회/30epoch**를 완료했다. VAL의 라벨·묶음 균등 BCE는 초기0.693147→**{training['VAL_macro_group_BCE']:.6f}**, epoch **{training['selected_epoch']}**/실제{training['selected_actual_updates']}회 가중치를 선택했다.

고정 CAL 임계값 `{selection['threshold']}`를 예약 기어 TEST2,179패치에 적용했다. 오탐 **{rates['fpr']*100:.2f}% ({rates['fp']}/{rates['negative_photos']})**, 미탐 **{rates['fnr']*100:.2f}% ({rates['fn']}/{rates['positive_photos']})**, AUROC **{selection['AUROC']:.5f}**다. 오탐·미탐 모두20% 이하 조건에 못 미쳐 **채택하지 않았다.** 오탐만 골라 반영하거나 TEST 임계값을 바꾸지 않았다.

물리·기능·촬영ID 및 근접 연결 TEST묶음은4개다. 묶음 bootstrap1,000회의 참고 구간은 오탐{ci[0][0]*100:.2f}~{ci[1][0]*100:.2f}%, 미탐{ci[0][1]*100:.2f}~{ci[1][1]*100:.2f}%로 넓다. 패치2,179개를 독립 부품2,179개로 설명하지 않는다.

자료는 FIT5,466/VAL1,440/CAL1,116과 원래 저자S3fold0의 기어 TEST2,179패치다. 이전 FIT/VAL/CAL 합집합에서 원래 TRAIN의 기어 묶음만 재분리했고, 이미 본 FIT예시 묶음1·19는 FIT로 보존했다. TEST·근접 제외 TRAIN·미주석 사진을 학습으로 가져오지 않았다.

이것은 자료 종류별 라벨 수 감사에 따라 먼저 선언한 **기어 표면만의 별도 과제**다. 전체 부품 v26/v27의 실패와 원래 TEST3,109개를 그대로 남기며, 이번 수치를 전체 기어박스 개선으로 합치지 않는다. v26/v27은 각1,920회를 실제 학습했지만 모든 학습 가중치의 VAL이 초기 기준을 개선하지 못해 epoch0 초기 새 head를 선택한 실패 연구다. 해당 평가를 학습으로 개선한 성능으로 표시하지 않는다.

기존 encoder 하나·고정3,072특징을 재사용하고 새0초기화 Linear head만 학습했다. 새 FIT만으로 정규화하고 모든 기존 부모 tensor/buffer/metadata가 정확히 같음을 확인했다. 보호 {verification['protected_stat_unchanged']}파일 stat·기존Python {verification['legacy_python_SHA_checked']}파일 SHA·실제 optimizer 장부·원래 특징 캐시 SHA를 검사했다. 보호 전체SHA를 재계산한 것은 아니다. 새unit2개가 통과했고 새RGB encoder특징 추출은0이다. 이 단계의 새 패키지 API/C++/ZIP 검사도0이다.

기존 앱용 v16-r1과 검정브래킷FP12.50/FN63.83,튜브FP3.12/FN68.12%,기존8/12유형 목표 상태는 그대로다. KS2·볼스크루·LOCO·원래 위치 지도·임계값도 유지한다. 원본·실패 모델·최종 학습 progress·매epoch 장부를 보존했다.

이 회차 v26/v27/v28 실제 optimizer합은 **5,760회**이며 같은 원본 재사용을 새 자연 사진 수로 더하지 않는다. 추적 일부연구합32,080회는 실패 포함이며 전체 역사 학습 총량은 아니다.

OK/nOK는 저자의128×128 기어 표면 마모 허용 라벨이다. 전체 장비 작동 건강·내부 고장·임의 현장 전체사진·검증된 위치 mask 정답이 아니다. 같은 공개TEST를 이전 개발에서 본 이력과 부모의 과거근접노출·촬영 그룹의 한계를 유지하며 독립 현장 검증을 주장하지 않는다. Re-GBC version1 CC BY4.0, DOI10.1109/ETFA65518.2025.11205777과 원배포본을 인용하고 기존 다른자료 조건도 유지한다.

다음은 FIT만의 fine24 공간 특징을 읽는 별도 사전 고정 조사다. 실제 학습 구조·loss·선택조건은 조사 후 다시 코드와 JSON으로 확정해야 한다. 클래스 라벨을 만든 pixel mask 정답으로 바꾸지 않는다.
'''
    with (ROOT/'reports/GEAR_ONLY_V28_RESULTS_KO.md').open('x',encoding='utf-8',newline='\n') as out:out.write(text)
    print({'actual_optimizer':1920,'selected_epoch':training['selected_epoch'],'rates':rates,'adopted':False},flush=True)


if __name__=='__main__':main()
