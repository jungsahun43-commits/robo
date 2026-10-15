"""Seal v22's data-gate stop, without pretending any optimizer was run."""
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import study_neu_detail as study
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha


def main():
    study.validate();a=study.read(study.MANIFEST)['audit']
    assert not a['ready'] and a['normal_original_photo_counts']['fit']['mpdd_bracket_black']==29
    assert a['FIT_real_mask_original_photos']==35 and a['VAL_real_mask_original_photos']==10
    assert not (study.RUN/'train').exists() and not (study.RUN/'fit-features.pt').exists()
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'v22_stopped_at_preregistered_data_gate_no_training',
       'preregistered_commit':'32a7559','manifest_sha256':sha(study.MANIFEST),'data_audit':a,
       'failed_condition':'black normal FIT original photos29 < preregistered30; no relaxation',
       'new_encoder_features':0,'new_model_scores':0,'actual_optimizer_updates':0,'new_API_CPP_tests':0,'new_ZIPs':0,
       'active_model':str(study.START),'active_model_sha256':study.EXPECTED,'v16_r1_kept':True,
       'all_preparation_and_failure_records_preserved':True,'current_TEST_masks_not_used':True,
       'next':'Separate source-only real annotated crop discrimination study, remove target normal training domain shortcut; preregister new conditions before features/gradients; unchanged target adoption guards.'}
    for p in (study.RUN/'final-audit.json',ROOT/'reports/industrial-neu-detail-v22-final-audit.json',ROOT.parents[3]/'outputs/INDUSTRIAL_NEU_DETAIL_RESUME_2026-10-15-v22-STOPPED.json'):new(p,r)
    text='''# v22 자료 요건 중단 기록

2026-10-15.32a7559에 학습·자료 조건을 먼저 commit/push했다. 새 모델 학습은 실행하지 않았다.

원본442장에 고정9영역을 적용한3978좌표를 준비했다. 기존 TEST전체5746장 및 VAL/CAL/TEST의 전체·상세 영역8326개를 raw/512 RGB SHA와8방향 pHash2로 검사했다. 연결된 FIT 원본 그룹54개를 제외했다. 기존 부모의 과거 근접 노출이나 물리 촬영 독립성을 해결한 것은 아니다.

제외 후 실제 주석 FIT35장/VAL10장은 유지됐고, 정상 FIT는 브래킷29장·튜브48장, 정상 VAL은18/15장이다. 브래킷 정상 FIT29장이 사전 최소30장에 못 미쳐 **features0·모델 점수0·optimizer0**에서 중단했다. 작은 양성 crop5개는 제외했고 정상으로 바꾸지 않았다. TEST mask를 분리에 사용하지 않았다. 수량 조건을29로 낮추거나 VAL/TEST·기존 제외 자료를 학습으로 이동하지 않았다.

1005개의 FIT 가능 crop과385개 VAL 가능 crop은 준비 기록이며 실제로 학습한 사진 수가 아니다. source만 보면 FIT양성130/주석상 음성182, VAL양성51/주석상 음성37이다. 같은 결함 사진 안의 주석상 음성 crop도 전체 기계 정상 정답이 아니다.

현재v16-r1 가중치·전달물·위치 기준은 유지한다. 새 API/C++/ZIP검사는0이다. 원본 조건·manifest·감사·prepare실패 기록은 보존한다.

다음은 별도 source-only 연구를 검토한다. 실제 금속 표면의 동일 원본 사진에서 양성/주석상 음성을 구분하게 해 ‘target 녹색 배경=정상,새 source 질감=결함’ 식의 학습 shortcut을 줄이려는 가설이다. 이는 실패한 수량 조건을 완화해v22를 통과시키는 것이 아니다. 새로운 데이터·학습·VAL 조건을 먼저 고정하고, 원래 target 오탐·미탐 채택 조건은 그대로 유지한다. 현재 TEST의 모델 점수로 새 구조를 선택하지 않는다.
'''
    with (ROOT/'reports/NEU_DETAIL_DATA_GATE_KO.md').open('x',encoding='utf-8',newline='\n') as out:out.write(text)
    print({k:v for k,v in r.items() if k!='data_audit'},flush=True)


if __name__=='__main__':main()
