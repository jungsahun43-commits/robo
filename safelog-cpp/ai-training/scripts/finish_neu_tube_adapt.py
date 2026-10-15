"""Seal v24 with exact photo metrics and immutable baseline retention."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_neu_tube_adapt as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
OUTPUTS=ROOT.parents[3]/'outputs'


def main():
    study.validate(data=True);s=study.read(study.RUN/'selection.json');v=study.read(study.RUN/'verification.json');t=s['training'];a=study.read(study.MANIFEST)['audit']
    assert v['status']=='passed' and t['actual_optimizer_updates']>=1920 and t['completed_epochs']==30
    checkpoint=study.RUN/'trained/final.pt'
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'v24_completed_candidate_pending_package_checks' if s['selected'] else 'v24_completed_all_candidates_rejected_v16_retained',
       'preregistered_commit':'c2be823','data_commit_before_features':'e651bc0',
       'active_model':str(study.START),'active_model_sha256':study.EXPECTED,
       'active_role4_ZIP':str(OUTPUTS/'unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
       'active_role4_ZIP_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
       'v24_checkpoint':str(checkpoint),'v24_checkpoint_sha256':sha(checkpoint),'warm_v23_VAL_checkpoint_sha256':study.WARM_SHA,
       'actual_optimizer_updates':t['actual_optimizer_updates'],'selected_epoch':t['selected_epoch'],'retained_selected_epoch_updates':t['selected_epoch_updates'],
       'initial_VAL_loss':study.read(study.RUN/'train/initial-VAL.json')['balanced_loss'],'best_VAL_loss':t['VAL_balanced_loss'],
       'FIT_actual_mask_original_photos':35,'VAL_actual_mask_original_photos':10,'FIT_normal_tube_original_photos':48,'VAL_normal_tube_original_photos':15,
       'actual_mask_source_reused_from_v23_not35_new_photos':True,'FIT_crops':t['FIT_crops'],'VAL_crops':t['VAL_crops'],
       'selected_methods':s['selected'],'target_results':s['categories'],'data_audit':a,'verification':v,
       'all_parent_tensors_buffers_metadata_exact':True,'black_and_other_branches_unchanged':True,
       'original_tube_primary_map_and_pixel_threshold_unchanged':True,
       'new_API_CPP_release_checks':0,'new_ZIPs':0,'new_unit_tests_passed':2,
       'v22_stopped_before_features_scores_optimizer':True,'v22_optimizer_updates':0,'v23_completed_optimizer_updates':1920,
       'this_turn_v23_plus_v24_actual_optimizer_updates':1920+t['actual_optimizer_updates'],
       'tracked_actual_optimizer_updates_v8_v10_v13_v15_through_v21_plus_v23_v24_including_failed':24400+t['actual_optimizer_updates'],
       'known_public_TEST_reused_for_design_and_adoption':True,'independent_field_validated':False,
       'physical_capture_groups_verified':False,'all_original_source_failed_models_reports_preserved':True,
       'awake_helper_restarted':False,
       'next':'Fresh packaged CPU/GPU/allalias/C++/primaryPNG checks required before adoption.' if s['selected'] else 'Stop repeating current small-mask transfer/threshold variants. Research same-equipment actual labeled data or component-specific source with publisher part IDs and normal/defect labels; audit license/splits/near groups before new training.'}
    for p in (study.RUN/'final-audit.json',ROOT/'reports/industrial-neu-tube-adapt-v24-final-audit.json',OUTPUTS/'INDUSTRIAL_NEU_TUBE_RESUME_2026-10-15-v24.json'):new(p,r)
    lines=['# v24 정상 튜브 적응 결과','','2026-10-15. c2be823에 조건·코드를, e651bc0에 자료 감사·manifest SHA를 새 특징 전에 commit/push했다.','',
       f"실제 optimizer **{t['actual_optimizer_updates']}회**,30epoch. VAL선택 epoch **{t['selected_epoch']}**(선택 가중치의 예정 갱신{t['selected_epoch_updates']}회). VAL {r['initial_VAL_loss']:.6f}→{r['best_VAL_loss']:.6f}.",'',
       '실제 mask source35FIT/10VAL과 정상 튜브48FIT/15VAL을 사용했다. crop은744/223개다. source는v23과 같은 자료이며 실제 결함 사진35장을 새로 확보한 것으로 더하지 않는다. 기존 공통 encoder·부모 전체 가중치/buffer/metadata·high projection을 고정하고 상세 판별부 fine projection/fusion만 갱신했다. 모든 source/추론에서 잔차는0이며 가짜 source/category embedding을 넣지 않았다.','',
       '| 튜브 원본 사진 평가 | 오탐 | 미탐 | 조건 통과 |','|---|---|---|---|']
    entry=s['categories']['mpdd_tubes']
    for name,item in [('현재v16',{'rates':entry['parent'],'eligible':None}),*entry['methods'].items()]:
        rate=item['rates'];lines.append(f"| {name} | {100*rate['fpr']:.2f}% ({rate['fp']}/{rate['negative_photos']}) | {100*rate['fnr']:.2f}% ({rate['fn']}/{rate['positive_photos']}) | {item['eligible']} |")
    lines+=['','선택 방법: `'+json.dumps(s['selected'],ensure_ascii=False)+'`.','',
       '정상 CAL23장의85% 순서 통계량을 먼저 저장하고 원본 TEST101장에 적용했다. 같은9영역 점수 최댓값을 사진 점수로 썼고 crop을 독립 표본으로 세지 않았다. TEST 임계값이나 오탐·미탐 반영 조건을 변경하지 않았다. 원래 튜브 위치 지도·pixel 기준과 브래킷·나머지 기존 분기를 유지한다.','',
       f"보존 검사: 기존 Python{v['legacy_python_SHA_checked']}파일 SHA, 보호{v['protected_stat_unchanged']}파일 stat 동일, 변경 재해시{v['changed_metadata_SHA_rechecked']}개. 보호 전체 SHA를 다시 계산한 것은 아니다. 모든 부모 tensor/buffer/metadata·고정 high projection·새 캐시 SHA·실제 optimizer 장부를 검증했고 새 unit2개가 통과했다. 이 단계의 새 패키지 API/C++/ZIP 검사는0이다.",'',
       '공개 TEST 반복 개발과 설계·선택 결과이며 독립 현장 성능이 아니다. 작은source·다른 촬영 영역·물리 부품 그룹 미검증·과거 부모 근접 노출의 한계를 유지한다. 주석상 음성 crop을 부품 전체 정상 작동으로 설명하지 않는다.','',
       'v22는 중복 제외 후 브래킷 정상29<사전30장으로 optimizer0에서 중단했고 조건을 완화하지 않았다. v23 실제mask source-only1920회에 이어 v24는 감사된 정상 튜브를 함께 학습하는 별도 조건으로1920회를 수행했다. 이 회차 실제 optimizer 합은3840회다. 실패 후보·원본·로그·보고서를 모두 보존했다.','',
       '다음에는 이 작은 다른표면mask 자료의 전이/TEST 임계값 변형을 반복하지 않고, 동일 설비의 실제 결함·정상 라벨과 부품 ID가 있는 원자료를 우선 검토한다. 이용조건·독립 분리·중복을 확인한 뒤에만 새로운 학습을 정한다.']
    with (ROOT/'reports/NEU_TUBE_ADAPT_RESULTS_KO.md').open('x',encoding='utf-8',newline='\n') as out:out.write('\n'.join(lines)+'\n')
    print({k:v for k,v in r.items() if k not in ('target_results','verification','data_audit')},flush=True)


if __name__=='__main__':main()
