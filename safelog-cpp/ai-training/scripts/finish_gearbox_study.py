"""Conclude one immutable gearbox study; never promote epoch0 or failed candidates."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts import study_gearbox_supervised as v26


def main(version):
    run=v26.RUN if version==26 else ROOT/'runs/industrial-gearbox-linear-v27'
    selection=read(run/'selection.json');verification=read(run/'verification.json')
    assert verification['status']=='passed' and not (run/'operation.lock').exists()
    training=selection['training'];eligible=selection['eligible_pending_package_checks']
    assert training['actual_optimizer_updates']>=1920 and training['completed_epochs']==30
    result={'UTC':datetime.now(timezone.utc).isoformat(),'status':f'v{version}_completed_pending_package_checks' if eligible else f'v{version}_completed_rejected_active_v16_retained',
        'active_model':str(v26.START),'active_model_sha256':v26.EXPECTED,
        'active_role4_ZIP':str(ROOT.parents[3]/'outputs/unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
        'active_role4_ZIP_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
        'checkpoint':str(run/'trained/final.pt'),'checkpoint_sha256':sha(run/'trained/final.pt'),
        'training':training,'selection':selection,'verification':verification,
        'epoch0_means_initial_new_head_not_learned_head':training['selected_epoch']==0,
        'source_labels_not_whole_gearbox_health_or_verified_location':True,
        'source_split':read(v26.MANIFEST)['audit'],'source_manifest_sha256':sha(v26.MANIFEST),
        'same8220FIT_and1404VAL_reused_not_new_photos':version==27,
        'v25_stopped_before_model_features_optimizer':True,'v25_reason':'CALnormal10<fixed20, original failedmanifest preserved',
        'v26_actual_optimizer_updates':1920,
        'this_turn_actual_optimizer_updates':1920 if version==26 else 3840,
        'tracked_partial_research_optimizer_updates_including_failed':28240 if version==26 else 30160,
        'tracked_total_is_not_entire_training_history':True,
        'new_API_CPP_ZIP_checks':0,'new_ZIPs':0,'active_v16_unchanged':True,
        'new_independent_field_validation':False,'awake_helper_restarted':False,
        'next':'Complete fresh package CPU/GPU/C++/aliases/primary output checks before any adoption.' if eligible else 'Preserve failed candidate. Use FIT/VAL diagnosis and source-specific label/capture limitations to preregister the next independent candidate; no TEST threshold or split fishing.'}
    new(run/'final-audit.json',result)
    new(ROOT/f'reports/industrial-gearbox-v{version}-final-audit.json',result)
    new(ROOT.parents[3]/f'outputs/INDUSTRIAL_GEARBOX_RESUME_2026-10-15-v{version}.json',result)
    lines=[f'# v{version} 기어박스 실제 라벨 학습 결과','',
        f"실제 optimizer **{training['actual_optimizer_updates']}회**,30epoch. VAL로 선택한 epoch **{training['selected_epoch']}**, 선택 가중치의 실제 갱신 **{training['selected_actual_updates']}회**.",'']
    if training['selected_epoch']==0:
        lines+=['**학습을 1,920회 수행했지만 학습 후 가중치가 검증 기준을 개선하지 못했다. 저장된 선택 체크포인트는 epoch0 초기 새 판별부이며 아래 수치를 학습으로 개선한 성능이라고 설명하지 않는다. 배포 후보에서 제외한다.**','']
    rates=selection['rates']
    lines += [f"선택 체크포인트의 예약 TEST3,109패치: 오탐 **{rates['fpr']*100:.2f}% ({rates['fp']}/{rates['negative_photos']})**, 미탐 **{rates['fnr']*100:.2f}% ({rates['fn']}/{rates['positive_photos']})**, AUROC{selection['AUROC']:.5f}. 채택 가능(추가 패키지 검사 전): `{eligible}`.",'',
        '| 기능 부품 | OK 패치 | nOK 패치 | 오탐 | 미탐 |','|---|---:|---:|---:|---:|']
    def percent(x):return '표본 없음' if x is None else f'{100*x:.2f}%'
    for name,value in selection['by']['functional_component_category'].items():
        lines.append(f"| {name} | {value['negative_photos']} | {value['positive_photos']} | {percent(value['fpr'])} | {percent(value['fnr'])} |")
    ci=selection['group_bootstrap_95percent']
    lines+=['',f"연결 묶음7개 단위 bootstrap1,000회 참고 구간: 오탐{100*ci[0][0]:.2f}~{100*ci[1][0]:.2f}%, 미탐{100*ci[0][1]:.2f}~{100*ci[1][1]:.2f}%. 패치를 독립 부품으로 가정하지 않았지만 묶음 수가 적어 불확실하다.",'',
        '기존 공통 encoder 하나를 고정하고 새 기어 판별부만 학습했다. 기존 모든 tensor/buffer/metadata가 정확히 같음을 확인했다. 기존 브래킷·튜브를 평가·교체한 연구가 아니며 기존 목표8/12유형은 그대로다. 원래 위치 지도·임계값과 KS2·볼스크루·LOCO도 유지한다.','',
        f"보존 검사: 기존 Python{verification['legacy_python_SHA_checked']}파일 SHA, 보호{verification['protected_stat_unchanged']}파일 stat 동일, 변경 재해시{verification['changed_metadata_SHA_rechecked']}개, 실제 optimizer 장부·특징 캐시 SHA. 보호 원본 전체 SHA를 재계산한 것은 아니다. 새 API/C++/ZIP 검사는0회다.",'',
        'v25의 첫 자료 분리는 CAL 정상10<최소20으로 학습0회에서 중단·보존했다. v26은 저자 TRAIN 라벨만으로 부품 묶음을 균형 배정한 별도 사전 고정 설계다. FIT8220/VAL1404/CAL1538, 원래 저자S3fold0 TEST3109를 유지하고 근접 연결 TRAIN1317개·미주석103개를 제외했다. 기존 TEST5746장과 새 자료의 근접 후보는0개였다. 재분리나 TEST 임계값 반복은 하지 않았다.','',
        'OK/nOK는 저자의128×128 패치 마모 허용 정답이다. 작은 흔적도 OK일 수 있으며 전체 기어박스 작동 건강·내부 고장·임의 현장 전체 사진·결함 위치 정답이 아니다. 독립 현장 검증은 없고 부모의 과거 근접 노출·촬영 그룹 한계도 해소된 것으로 설명하지 않는다.','',
        '원자료·실패 모델·최종 학습 progress·매epoch 기록·선택 초기/학습 판별부·분리 실패를 모두 보존했다. CC BY4.0 Re-GBC version1, 저자 논문 DOI10.1109/ETFA65518.2025.11205777과 원배포본을 인용하고 기존 다른 자료의 이용조건도 유지한다.']
    if version==27:lines+=['','v27은 v26과 같은8,220 FIT 패치·1,404 VAL 패치와 특징을 재사용했다. 이를 새로 확보한 자연 사진으로 더하지 않는다. v26/v27 실제 optimizer 합은3,840회다.']
    with (ROOT/f'reports/GEARBOX_V{version}_RESULTS_KO.md').open('x',encoding='utf-8',newline='\n') as out:out.write('\n'.join(lines)+'\n')
    print({'version':version,'actual_optimizer':training['actual_optimizer_updates'],'selected_epoch':training['selected_epoch'],'rates':rates,'eligible':eligible},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',type=int,choices=(26,27));args=parser.parse_args();main(args.version)
