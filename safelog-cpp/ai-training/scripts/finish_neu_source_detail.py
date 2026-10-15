"""Seal actual-mask v23 research separately from any later packaged release."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import study_neu_source_detail as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
OUTPUTS=ROOT.parents[3]/'outputs'


def main():
    study.validate(data=True)
    s=study.read(study.RUN/'selection.json');v=study.read(study.RUN/'verification.json');t=s['training'];a=study.read(study.MANIFEST)['audit']
    assert v['status']=='passed' and t['actual_optimizer_updates']>=1920 and t['completed_epochs']==30
    path=study.RUN/'trained/final.pt';state=torch.load(path,map_location='cpu',weights_only=True)
    warm=torch.load(study.WARM,map_location='cpu',weights_only=True)
    changed=not torch.equal(state['state_dict']['detail_surface_head.fine_projection.0.weight'],warm['state_dict']['fine_target_head.fine_projection.0.weight'])
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'v23_completed_candidate_pending_package_checks' if s['selected'] else 'v23_completed_all_candidates_rejected_v16_retained',
       'preregistered_commit':'e0a06f2','data_audit_commit_before_features':'18641c7',
       'active_model':str(study.START),'active_model_sha256':study.EXPECTED,
       'active_role4_ZIP':str(OUTPUTS/'unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
       'active_role4_ZIP_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
       'v23_research_checkpoint':str(path),'v23_checkpoint_sha256':sha(path),'warm_v21_VAL_checkpoint_sha256':study.WARM_SHA,
       'actual_optimizer_updates':t['actual_optimizer_updates'],'selected_epoch':t['selected_epoch'],'retained_selected_epoch_updates':t['selected_epoch_updates'],
       'initial_VAL_loss':study.read(study.RUN/'train/initial-VAL.json')['balanced_loss'],'best_VAL_loss':t['VAL_balanced_loss'],
       'FIT_actual_publisher_mask_original_photos':a['FIT_real_mask_original_photos'],'VAL_actual_mask_original_photos':a['VAL_real_mask_original_photos'],
       'FIT_crops':t['FIT_crops'],'VAL_crops':t['VAL_crops'],'new_synthetic_images':0,
       'new_source_normal_whole_photos':0,'annotation_negative_crops_are_not_whole_equipment_normal_labels':True,
       'fine_projection_changed_from_initialization':changed,'all_parent_tensors_buffers_metadata_exact':True,
       'selected_methods':s['selected'],'target_results':s['categories'],'data_audit':a,'verification':v,
       'new_release_API_aliases_tested':0,'new_release_CPP_operations':0,'new_release_ZIP_created':False,
       'source_only_unit_tests_passed':2,'same_turn_geometry_mask_dispatch_unit_tests_passed':4,
       'v22_stopped_before_features_scores_optimizer':True,'v22_actual_optimizer_updates':0,
       'tracked_actual_optimizer_updates_v8_v10_v13_v15_through_v21_plus_v23_including_failed':22480+t['actual_optimizer_updates'],
       'original_primary_target_locations_and_pixel_thresholds_unchanged':True,
       'known_public_TEST_reused_for_adoption':True,'independent_field_validated':False,'physical_capture_groups_verified':False,
       'new_source_independent_TEST_available':False,'all45_source_masks_team_exhaustively_verified':False,
       'all_source_failed_models_reports_preserved':True,'awake_helper_restarted':False,
       'next':'Package only selected methods; fresh CPU/GPU/all alias/C++/primary PNG preservation checks before adoption.' if s['selected'] else 'Preserve failure; use FIT-only optimization/generalization diagnosis and actual source access options. Do not replay v22/v23 or retune current TEST thresholds.'}
    for p in (study.RUN/'final-audit.json',ROOT/'reports/industrial-neu-source-detail-v23-final-audit.json',OUTPUTS/'INDUSTRIAL_NEU_SOURCE_RESUME_2026-10-15-v23.json'):new(p,r)
    lines=['# v23 실제 주석·상세 영역 학습 결과','','2026-10-15. e0a06f2에 조건·코드를,18641c7에 실제 분리·중복 감사와 manifest SHA를 먼저 commit/push한 뒤 새 특징과 학습을 실행했다.','','## 실제 학습 범위','',
      f"**실제 optimizer {t['actual_optimizer_updates']}회**,30epoch. 검증 선택 epoch **{t['selected_epoch']}**, 선택 가중치의 예정 갱신{t['selected_epoch_updates']}회.",
      f"VAL loss {r['initial_VAL_loss']:.6f} → {r['best_VAL_loss']:.6f}. v23의 두 집단·원본 사진 가중 VAL이므로 이전 실험의 loss와 직접 비교하지 않는다.",
      f"실제 배포 mask 원본 FIT{a['FIT_real_mask_original_photos']}장/VAL{a['VAL_real_mask_original_photos']}장, 영역{t['FIT_crops']}/{t['VAL_crops']}개다. 반복 추출이나 crop을 새 실제 사진 수로 세지 않았다. 새 합성은0개다.",
      '같은 사진의 결함 영역과 주석상 결함이 없는 영역을 구분하도록 학습했다. source 정상 전체 사진은0장이며 주석상 음성을 부품 전체의 정상 작동으로 바꾸지 않았다. target 정상은 gradient 학습·VAL 선택에 넣지 않았고 기존 CAL/TEST만 사용했다.',
      '기존 encoder와 모든 부모 분기·tensor·buffer·메타데이터·high projection은 고정했다. 새로운 fine projection/fusion만 갱신했고 잔차 입력은 모든 source/target에서0이다. 별도 encoder·가짜 category embedding을 추가하지 않았다.','','## 고정 CAL 이후 공개 TEST','',
      '| 유형 | 방법 | 사진 오탐 | 사진 미탐 | 채택 조건 |','|---|---|---|---|---|']
    for c,entry in s['categories'].items():
        for method,item in [('현재v16',{'rates':entry['parent'],'eligible':None}),*entry['methods'].items()]:
            rate=item['rates']
            lines.append(f"| {c} | {method} | {100*rate['fpr']:.2f}% ({rate['fp']}/{rate['negative_photos']}) | {100*rate['fnr']:.2f}% ({rate['fn']}/{rate['positive_photos']}) | {item['eligible']} |")
    lines+=['','선택 방법: `'+json.dumps(s['selected'],ensure_ascii=False)+'`。'.replace('。','.'),'',
      '사진마다 같은9개 영역의 점수 최댓값으로 평가했고 원본 사진을 한 번씩 집계했다. TEST 정답 위치로 crop을 고르거나 TEST에 맞게 임계값·채택 조건을 바꾸지 않았다.',
      '통과 후보가 있어도 새 패키지 CPU/GPU·실제 C++·모든 alias·원래 위치 PNG를 검사하기 전까지 기존v16-r1을 유지한다. 이 연구 결과 기록 시 새 패키지 API/C++/ZIP은0이다. 새 상세 지도를 target 기본 위치로 승격하지 않았다.','','## 보존과 자료 요건 중단','',
      f"기존 Python{v['legacy_python_SHA_checked']}파일 SHA 검사, 보호{v['protected_stat_unchanged']}파일 stat 동일, 메타데이터 변경 재해시{v['changed_metadata_SHA_rechecked']}개다. 보호 원본 전체 SHA를 모두 다시 계산한 것은 아니다. 실제 optimizer 장부·새 캐시 SHA·부모 전체 보존과 고정 high projection을 검증했다. source-only unit2개와 같은 회차의 geometry/mask/dispatch unit4개가 통과했다.",
      '앞선v22는 중복 제외 후 브래킷 정상 FIT29장으로 사전 최소30장을 충족하지 못해 features/점수/optimizer0에서 중단했다. 조건을 낮추지 않았고 자료·실패 기록을 보존했다. v23은 target 정상 전체 사진 학습을 제거하고 같은 실제 source 사진 안의 주석상 양성/음성을 구분하는 별도 학습 설계다.',
      '35FIT/10VAL의 파일 계열 예약, 흰 표시 Scratch6장 제외와 v22의5746 TEST 전체·8326전체/영역 참조 대조를 그대로 상속했다. 원래 photo 그룹의 근접 제외를 되돌리지 않았다.','','## 한계','',
      '공개 TEST를 반복 개발에 사용한 연구 결과이며 독립 현장 성능이 아니다. source VAL은 다른 파일 계열10장이며 물리 부품·촬영 세션 독립성이 검증되지 않았다. 팀이45장 모든 mask 픽셀이나 실제 기계·전기 고장 진실을 검수한 것은 아니다. 과거 부모 모델의 근접 노출, KSDD1 결함18장, 미검수 전선 관계 라벨의 한계도 유지한다.',
      'NEU 원 이미지 제공자와 abin24 주석 배포자를 인용하고, 명시된 학술 연구 전용/상업 금지 조건을 유지한다. 존재하지 않는 SPDX를 붙이지 않으며 원자료·가중치·실패·출력·보고서를 보존한다.']
    with (ROOT/'reports/NEU_SOURCE_DETAIL_RESULTS_KO.md').open('x',encoding='utf-8',newline='\n') as out:out.write('\n'.join(lines)+'\n')
    print({k:v for k,v in r.items() if k not in ('target_results','data_audit','verification')},flush=True)


if __name__=='__main__':main()
