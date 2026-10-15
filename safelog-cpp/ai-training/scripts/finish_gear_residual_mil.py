"""Conclude v30 without calling its unchanged initialization an improvement."""
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_gear_residual_mil as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    s=read(study.RUN/'selection.json');v=read(study.RUN/'verification.json');t=s['training']
    assert v['status']=='passed' and t['actual_optimizer_updates']==1920 and not (study.RUN/'operation.lock').exists()
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'v30_complete_rejected_v16_retained' if not s['eligible_pending_package_checks'] else 'v30_pending_release_checks',
       'preregistered_commit':'5efc59f6b97e4edf722cb5fd51f03f20df490263','active_model':str(study.START),'active_model_sha256':study.EXPECTED,
       'active_role4_ZIP':str(ROOT.parents[3]/'outputs/unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
       'active_role4_ZIP_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
       'research_checkpoint':str(study.RUN/'trained/final.pt'),'research_checkpoint_sha256':sha(study.RUN/'trained/final.pt'),
       'training':t,'selection':s,'verification':v,'manifest_sha256':sha(study.MANIFEST),
       'selected_epoch0_means_unchanged_v28_global_not_new_training_improvement':t['selected_epoch']==0,
       'v29_completed_actual_optimizer':1920,'this_turn_v29_v30_actual_optimizer':3840,
       'tracked_partial_research_optimizer_including_failed':35920,'tracked_sum_not_entire_history':True,
       'new_RGB_encoder_features_in_v30':0,'new_API_CPP_ZIP_checks':0,'new_ZIPs':0,
       'old_tasks_locations_and_current8of12_targets_unchanged':True,'all_originals_failed_weights_and_reports_preserved':True,
       'independent_field_validated':False,'awake_helper_restarted':False,'free_disk_bytes':shutil.disk_usage(ROOT).free,
       'next':'Validate parent-bound research delta storage without deleting old files; FIT-only feasibility of a tiny zero-init task adapter feeding original frozen later backbone stages. No full new training protocol yet; do not rerun current threshold/head variants.'}
    new(study.RUN/'final-audit.json',r);new(ROOT/'reports/industrial-gear-residual-v30-final-audit.json',r)
    new(ROOT.parents[3]/'outputs/INDUSTRIAL_GEAR_RESIDUAL_RESUME_2026-10-15-v30.json',r)
    rates=s['rates']
    text=f'''# v30 제한된 공간 보정 학습 결과

5efc59f에 조건·코드를 먼저 commit/push한 뒤 **실제1,920 optimizer회/30epoch**를 완료했다. 초기 VAL BCE와 선택 VAL BCE는 모두{t['VAL_macro_group_BCE']:.6f}로 **epoch{t['selected_epoch']}**, 선택 새 가중치 갱신{t['selected_actual_updates']}회다. 모든 학습 epoch가 초기 기준을 개선하지 못했다.

초기 보정0에서 VAL 전체 logit이 고정 v28 global logit과 bitwise 같음을 확인했다. 선택 체크포인트의 보정도0이다. 아래 수치는 **이전 v28 초기값을 유지한 결과이며 새 학습의 개선으로 설명하지 않는다.** 최종 실제학습 progress/모든epoch와 실패 후보는 보존했다.

고정 CALcut{s['threshold']} 이후 기어 TEST2,179패치: 오탐 **{100*rates['fpr']:.2f}% ({rates['fp']}/{rates['negative_photos']})**, 미탐 **{100*rates['fnr']:.2f}% ({rates['fn']}/{rates['positive_photos']})**, AUROC{s['AUROC']:.5f}. 선택epoch>0 및 두지표20% 조건에 못 미쳐 제외했다. 오탐·미탐 일부만 반영하거나 TEST 혼합비·임계값을 반복 변경하지 않았다.

하나의 공통 encoder와 고정 global판별부에 .5*tanh(localMIL)의 학습 보정을 더하는 후보였다. v29 공간 Conv/GN과 FIT정규화를 초기값으로 쓰되 마지막층은0으로 초기화했다. 실제 patch OK/nOK로만 새 spatialhead를 학습했고 pixel정답을 만들지 않았다. 기존 v28 global전체 상태·정규화, 기존 acceptedv16 tensor/buffer/metadata와 원래 위치·다른 유형은 정확히 보존됐다.

기존Python{v['legacy_python_SHA_checked']}파일SHA·보호{v['protected_stat_unchanged']}파일stat, 변경 재해시{v['changed_metadata_SHA_rechecked']}개, 원fine/descriptor캐시SHA·실제step장부와 새unit2개를 확인했다. 보호 전체SHA를 재계산한 것은 아니다. 새RGB/encoder특징 추출0, 새API/C++/ZIP검사0이다. 기존앱 v16-r1과 추가12유형 중8개 목표 상태를 유지한다.

이번 v29/v30 실제 optimizer합3,840회, 추적 일부연구합35,920회는 실패 포함이며 전체 역사 학습 총량이 아니다. 같은 패치와 캐시 재사용을 새 자연 사진으로 더하지 않는다. 범위는 gear128표면의 마모 허용이며 전체 부품·기어박스 건강·내부 고장·임의 현장 사진·검증된 위치가 아니다. TEST연결묶음4개, 이전 공개TEST개발·과거부모근접노출·촬영 그룹 한계를 보존하며 독립 현장 검증을 주장하지 않는다.

Re-GBCv1 CC BY4.0, 원배포본과 DOI10.1109/ETFA65518.2025.11205777을 인용하고 기존 다른 자료 조건도 유지한다. 저장 여유가 약4GB이므로 원본과 실패모델을 유지하면서 향후 연구 체크포인트는 검증된 부모와 새 가중치를 연결하는 delta 보관을 우선 검증한다. 이후 FIT만에서 작은 task adapter가 기존 shared backbone의 후반층을 재사용하는 가능성을 조사한다. 아직 새 학습 조건·optimizer는0이다.
'''
    with (ROOT/'reports/GEAR_RESIDUAL_V30_RESULTS_KO.md').open('x',encoding='utf-8',newline='\n') as out:out.write(text)
    print({'actual_optimizer':1920,'selected_epoch':t['selected_epoch'],'adopted':False,'free_disk_bytes':r['free_disk_bytes']},flush=True)


if __name__=='__main__':main()
