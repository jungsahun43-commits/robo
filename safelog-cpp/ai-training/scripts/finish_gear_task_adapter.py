"""Complete v31: retain failed final weights and the exact parent-bound delta."""
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_gear_task_adapter as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    s=read(study.RUN/'selection.json');v=read(study.RUN/'verification.json');t=s['training']
    assert v['status']=='passed' and t['actual_optimizer_updates']==1920 and not (study.RUN/'operation.lock').exists()
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'v31_complete_rejected_v16_retained' if not s['eligible_pending_package_checks'] else 'v31_pending_release_checks',
       'preregistered_commit':'805ac1de165efa67c21f79fe2b99f76ffc759fe6','active_model':str(study.START),'active_model_sha256':study.EXPECTED,
       'active_role4_ZIP':str(ROOT.parents[3]/'outputs/unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
       'active_role4_ZIP_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
       'research_delta':str(study.CHECKPOINT),'research_delta_sha256':sha(study.CHECKPOINT),'delta_verification':read(study.RUN/'trained/delta-verification.json'),
       'research_delta_needs_parent_not_standalone_app':True,'training':t,'selection':s,'verification':v,
       'selected_epoch0_means_no_new_learned_improvement':t['selected_epoch']==0,
       'v29_actual_optimizer':1920,'v30_actual_optimizer':1920,'this_turn_v29_v30_v31_actual_optimizer':5760,
       'tracked_partial_research_optimizer_including_failed':37840,'tracked_sum_not_entire_history':True,
       'new_RGB_encoder_features_in_v31':0,'new_API_CPP_ZIP_checks':0,'new_ZIPs':0,'new_independent_field_validation':False,
       'old_app_tasks_positions_current8of12_targets_unchanged':True,'awake_helper_restarted':False,'free_disk_bytes':shutil.disk_usage(ROOT).free,
       'next':'Preregister VAL-only selected-versus-last-checkpoint ranking/calibration diagnostic and development-only label/group consistency audit; no new optimizer or TEST scoring, no repeated architecture/threshold variants until the source of stagnation is checked.'}
    targets=(study.RUN/'final-audit.json',ROOT/'reports/industrial-gear-task-adapter-v31-final-audit.json',ROOT.parents[3]/'outputs/INDUSTRIAL_GEAR_TASK_ADAPTER_RESUME_2026-10-15-v31.json')
    for target in targets:
        if target.exists():
            old=read(target)
            assert {k:v for k,v in old.items() if k not in ('UTC','free_disk_bytes')}=={k:v for k,v in r.items() if k not in ('UTC','free_disk_bytes')}
        else:new(target,r)
    rates=s['rates']
    text=f'''# v31 기어 특징 보정 모듈 실제 학습 결과

805ac1d에 구조·입력·저장·학습·선택 조건을 먼저 commit/push한 뒤 **실제1,920 optimizer회/30epoch**를 완료했다. adapter220개와 기어 head3,073개 parameter만 학습했고 기존 shared backbone 전체 parameter/buffer는 고정·eval로 유지했다.

선택epoch **{t['selected_epoch']}**/새 선택 가중치 갱신{t['selected_actual_updates']}회다. 초기 VAL .58790398보다 학습 후 VAL이 나아지지 않아 초기 모듈을 선택했다. 이전v28 VAL과의 정밀도경계 차이는.00000697로 사전.005 이내였다. 아래 값은 **새 학습으로 개선한 성능이 아니다.** 실제 마지막 학습가중치·optimizer·매epoch·장부도 보존했다.

고정 CALcut{s['threshold']} 이후 기어 TEST2,179패치 오탐 **{100*rates['fpr']:.2f}% ({rates['fp']}/{rates['negative_photos']})**, 미탐 **{100*rates['fnr']:.2f}% ({rates['fn']}/{rates['positive_photos']})**, AUROC{s['AUROC']:.5f}. 선택epoch>0과 두지표20% 기준을 통과하지 못해 제외했다. 기존앱용 v16-r1과 원래 위치·다른 유형은 유지한다.

같은 encoder stage3의 cached FP16fine에 작은 residual을 적용하고 원래 같은 후반 module 객체로 low/high를 계산했다. 새 encoder·복제 backbone·hook·return_layers 변경이 없다. 기존 v28 FIT 정규화는 고정했다. pixel mask/위치 정답을 만들지 않았으며 범위는128기어표면의 마모 허용이다.

새 연구 delta **{study.CHECKPOINT.stat().st_size:,}bytes**를 부모SHA에 결합해 저장하고 다시 읽어 모든 state·buffer·metadata가 정확히 같게 복원됨을 확인했다. 부모가 필요한 연구형식이며 앱 전달파일은 아니다. 기존 full모델·원자료·실패파일은 삭제·압축하지 않았다. 새RGB/encoder캐시0, 새API/C++/ZIP검사0이다.

accepted v16의 모든 tensor/buffer/metadata가 정확히 같다. 기존Python{v['legacy_python_SHA_checked']}파일SHA, 보호{v['protected_stat_unchanged']}파일stat·변경 재해시{v['changed_metadata_SHA_rechecked']}개, fine캐시SHA·실제step장부·새unit3개를 확인했다. 보호 전체SHA를 재계산한 것은 아니다.

이번v29/v30/v31 실제 optimizer합 **5,760회**이며 추적 일부연구합37,840회는 실패 포함이고 전체 역사 학습 총량이 아니다. 같은 원본 재사용을 새 자연사진으로 더하지 않는다. 전체부품 실패와 S3fold0 TEST3,109개·근접제외TRAIN1,317개·미주석103개를 보존한다. 적은 TEST묶음4개, 반복 공개TEST개발과 과거부모노출/촬영그룹의 한계가 남으며 독립현장검증을 주장하지 않는다.

Re-GBCv1 CC BY4.0, 저자 원배포본과 DOI10.1109/ETFA65518.2025.11205777을 인용하고 기존 다른자료 조건도 유지한다. 다음은 새로운 구조 학습을 반복하기 전에 VAL만에서 현재 선택가중치와 이미 보존한 마지막가중치의 순위/보정 지표 및 부품별 클래스 불균형을 조사한다. 이 진단은 새 TEST점수나 새 optimizer를 만들지 않는다.

최초 Markdown 생성에서 파일 크기 정수를 함수로 호출한 보고서 오류가 있었다. 원래 코드·오류 기록과 이미 저장된 JSON을 보존하고 해당 표기만 수정해 JSON 일치 검사 후 문서를 완성했다. 모델 학습·평가를 다시 실행하지 않았다.
'''
    document=ROOT/'reports/GEAR_TASK_ADAPTER_V31_RESULTS_KO.md'
    if document.exists():assert document.read_text(encoding='utf-8')==text
    else:
        with document.open('x',encoding='utf-8',newline='\n') as out:out.write(text)
    print({'actual_optimizer':1920,'selected_epoch':t['selected_epoch'],'rates':rates,'delta_bytes':study.CHECKPOINT.stat().st_size,'adopted':False},flush=True)


if __name__=='__main__':main()
