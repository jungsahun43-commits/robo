"""Finalize the complete v29 study, separate from its earlier FIT-only probe."""
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_gear_spatial_mil as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    s=read(study.RUN/'selection.json');v=read(study.RUN/'verification.json');t=s['training']
    assert v['status']=='passed' and t['actual_optimizer_updates']==1920 and not (study.RUN/'operation.lock').exists()
    rates=s['rates'];ci=s['group_bootstrap_95percent']
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'v29_completed_rejected_v16_retained' if not s['eligible_pending_package_checks'] else 'v29_pending_release_checks',
       'preregistered_commit':'8d9db0836ec972c2e91a4dbbf00f1aed7143f26c',
       'active_model':str(study.START),'active_model_sha256':study.EXPECTED,
       'active_role4_ZIP':str(ROOT.parents[3]/'outputs/unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
       'active_role4_ZIP_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
       'research_checkpoint':str(study.RUN/'trained/final.pt'),'research_checkpoint_sha256':sha(study.RUN/'trained/final.pt'),
       'training':t,'selection':s,'verification':v,'manifest_sha256':sha(study.MANIFEST),
       'fresh_RGB_fine_cache_inputs':10201,'first_FIT_extra_ordinary_encoder_inputs_for_exact_check':32,
       'fine_cache_tensor_bytes':2005598208,'fine_cache_compatibility':{phase:read(study.RUN/(phase+'-cache.json')) for phase in ('fit','validation','cal','test')},
       'new_pixel_GT_masks_created':0,'same_actual_patch_labels_and_v28_split_preserved':True,
       'this_turn_actual_optimizer':1920,'tracked_partial_research_optimizer_including_failed':34000,'tracked_sum_not_entire_history':True,
       'new_API_CPP_ZIP_checks':0,'new_ZIPs':0,'old_app_v16_original_position_and_other_tasks_retained':True,
       'independent_field_validated':False,'awake_helper_restarted':False,
       'next':'Preregister one bounded spatial residual on the frozen v28 global patch classifier based on VAL context versus local losses, with unchanged v28 split and fixed existing caches. Do not sweep TEST thresholds/fusion weights or adopt photo metric halves separately.'}
    new(study.RUN/'final-audit.json',r);new(ROOT/'reports/industrial-gear-spatial-v29-final-audit.json',r)
    new(ROOT.parents[3]/'outputs/INDUSTRIAL_GEAR_SPATIAL_RESUME_2026-10-15-v29.json',r)
    text=f'''# v29 공간 특징 MIL 실제 학습 결과

8d9db08에 구조·회차·자료·cache·선택 조건을 commit/push한 뒤 **30epoch/실제1,920 optimizer회**를 완료했다. VAL 라벨·묶음 균등 BCE .693147→**{t['VAL_macro_group_BCE']:.6f}**, epoch **{t['selected_epoch']}**/실제{t['selected_actual_updates']}회 가중치를 선택했다.

고정 CAL 임계값 `{s['threshold']}`를 기어 TEST2,179패치에 적용했다. 오탐 **{100*rates['fpr']:.2f}% ({rates['fp']}/{rates['negative_photos']})**, 미탐 **{100*rates['fnr']:.2f}% ({rates['fn']}/{rates['positive_photos']})**, AUROC{s['AUROC']:.5f}. 두 지표20% 조건에 미달해 **제외했다**. v28의 오탐17.93%·미탐37.84%와 비교하면 미탐 감소와 오탐 증가가 함께 있으며 일부 지표만 반영하지 않는다.

연결 TEST묶음4개 bootstrap1,000회 참고 구간: 오탐{100*ci[0][0]:.2f}~{100*ci[1][0]:.2f}%, 미탐{100*ci[0][1]:.2f}~{100*ci[1][1]:.2f}%. 패치 수를 독립 부품·현장 수로 설명하지 않는다.

FIT5,466/VAL1,440/CAL1,116/TEST2,179의 원래 v28 분리와 마모 허용 라벨을 유지했다. 새 encoder나 pixel 정답 없이 같은 encoder의 fine24@64×64에 작은Conv/GN/GELU와 상위10% logit 평균을 학습했다. cachedfine H/Vflip은 실제RGB 증강과 구분한다. 출력 공간지도는 위치 정답으로 검증되지 않았고 기본 위치 지도를 바꾸지 않았다.

새 fine 특징은 원래 RGB10,201개에서 추출했고 첫FIT32개는 일반forward와의 bitwise 검사로 추가 통과했다. 모든 입력에서 원래 pooledfine72 캐시와 고정허용오차(atol.002/rtol.002)를 통과했다. 사진ID·원본SHA·geometry·encoder/protocol/manifestSHA·FP16경계를 보존했다. 약2.006GB tensor 캐시와 실제 optimizer 장부를 보존한다.

기존 모든 tensor/buffer/metadata가 정확히 같다. 기존Python{v['legacy_python_SHA_checked']}개SHA, 보호{v['protected_stat_unchanged']}파일stat와 변경SHA재검사{v['changed_metadata_SHA_rechecked']}개를 확인했다. 보호 전체SHA를 재계산한 것은 아니다. 새unit3개 통과. 이번 새 API/C++/ZIP 검사는0이며 기존v16-r1과 기존8/12유형의 목표 상태를 유지한다.

이것은 기어128패치의 표면 마모 허용 연구이며 전체 부품·기어박스 작동 건강·내부 고장·임의 현장 전체사진·검증된 결함위치 성능이 아니다. 전체부품 v26/v27의 실패와 S3fold0 TEST3,109개, 근접제외TRAIN1,317개·미주석103개, 원자료·실패모델·초기probe를 모두 보존했다. 현재 공개TEST를 반복 개발에서 본 이력과 부모의 과거근접노출·촬영그룹 한계가 남아 있고 독립 현장 검증은 없다.

추적 일부연구합34,000 optimizer회는 실패 포함이며 전체 역사 학습 총량이 아니다. Re-GBC version1 CC BY4.0, 저자 원배포본과 DOI10.1109/ETFA65518.2025.11205777을 인용하고 기존 다른자료 조건도 유지한다.

다음 연구는 VAL에서 전체 특징을 이용한 v28(.587897)이 공간 단독 v29(.653942)보다 나았다는 진단에 따라, 고정 v28 판별부에 학습되는 공간 보정을 제한해 더하는 후보 하나를 별도로 등록한다. TEST 숫자에 맞춘 가중치·임계값 조합은 돌리지 않는다.
'''
    with (ROOT/'reports/GEAR_SPATIAL_V29_RESULTS_KO.md').open('x',encoding='utf-8',newline='\n') as out:out.write(text)
    print({'actual_optimizer':1920,'selected_epoch':t['selected_epoch'],'rates':rates,'eligible':s['eligible_pending_package_checks']},flush=True)


if __name__=='__main__':main()
