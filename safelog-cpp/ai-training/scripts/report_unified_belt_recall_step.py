"""Measured shared-weight tradeoff and the attempted one-sided recall constraint."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.unified_main_data import read,sha
from scripts.study_unified_belt_recall_step_v2 import validate,BASE,new
REPO=ROOT.parents[1]; REPORTS=ROOT/'reports'


def pct(x): return '미측정' if x is None else f'{100*x:.2f}%'


def text_new(path,text):
    with path.open('x',encoding='utf-8',newline='\n') as f: f.write(text)


def main():
    p=validate(); t=read(REPORTS/'unified-belt-recall-step-v2-training.json'); r=read(REPORTS/'unified-belt-recall-step-v2-validation.json')
    c=read(REPORTS/'unified-belt-recall-step-v2-selection.json'); v=read(REPORTS/'unified-belt-recall-step-v2-verification.json')
    old=read(REPORTS/'unified-belt-normal-validation.json'); original=p['baseline']; control=old['known_summary']; result=r['known_summary']
    assert t['optimizer_updates']==v['actual_journal_updates']==v['unique_committed_updates']==512 and sha(BASE/'train/final.pt')==t['weights_sha256']
    kept=c['selected']=='prior_accepted_tiles'; decision='기존 모델·정책·역할4 전달 ZIP을 유지한다.' if kept else '사전 조건을 통과해 새 보정 후보를 연구 연결용으로 선택한다. 앱 기본 모델은 자동 교체하지 않는다.'
    lines=['# 오탐 개선만 부분 반영할 수 있는가 · 실제 보정 학습 비교','','기준일: 2026-10-13','','## 요청에 대한 답','','오탐과 미탐은 기능 스위치가 아니라 같은 가중치의 예측 결과다. 앞선 후보의 가중치를 그대로 적용하면 오탐 감소와 손상 미탐 증가가 함께 반영된다. **오탐을 줄이는 학습 방향 중 손상 탐지를 해치는 방향을 보정하는 방식**으로 부분 반영을 시도했고, 실제 검증 결과로 채택 여부를 결정했다.','',
        '## 같은 조건으로 실제 학습','','**실제4epoch·512회 추가 학습**. 원래 선택한 native_mix에서 새로 시작했다. 앞서 미탐이 늘어난 normal_pilot 가중치에서 이어 학습한 것이 아니다. 동일한 사진·순서·seed443·학습률·손실·학습 횟수를 사용하고 가중치 변화 보정만 추가했다.','',
        '- 배치8: 기존 전체 손상사진4개·기존 contextual crop2개·작성자 정상 ROI2개. FIT14 정상 구역을1,024회 반복 사용했고 양성 사진은3,072draw다. 새 독립 사진·새 정답은 추가하지 않았다.',
        '- 기존 양성 BCE20＋Dice＋교사 KL10＋정상 ROI 손실0.5는 그대로다. 별도로 작성자 정답의 양성 pixel 미탐 방향을 계산했다.',
        '- Adam의 momentum·weight decay까지 포함한 실제 가중치 변화가 학습용 양성 pixel 손실을 일차 근사로 늘리는 경우, 그 방향을 제거했다. 저장된FP32 값으로 다시 검사했다. optimizer의 raw moment는 유지한다.',
        f"- 실제{v['projected_updates']}/512업데이트를 보정했고, 전체 가중치를 되돌린 단계는{v['reverted_parameter_steps']}회다. 모든 저장된 변화에서 수치 조건을 확인했다. 실제 학습 시간{t['elapsed_training_minutes']:.2f}분.",
        '- 이 수학적 조건은 TRAIN에서의 평균 양성 pixel 손실에 대한 일차 근사다. 검증·현장·사진 미탐을 보장하는 조건이 아니다. 미탐률을 실제로 다시 측정했다.',
        '- [Yu 등, Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)의 충돌 방향 투영 아이디어를 참고했다. 원 논문의 대칭 PCGrad 구현과 같다고 주장하지 않는다. 여기서는 손상 재현을 우선한 한쪽 방향의 실제 parameter-step 보정이다.','',
        '## 실제 결과 · 같은 알려진 손상VAL353장','','| 항목 | 현재 선택 모델 | 앞선 정상 학습 | 이번 미탐 방향 보정 |','|---|---:|---:|---:|']
    for label,fn in [('위치 IoU',lambda x:x['pixel_iou']),('손상 pixel 미탐',lambda x:x['pixel_fnr']),('배경 pixel 오탐',lambda x:x['pixel_fpr']),('손상 사진 미탐',lambda x:x['photo_rates']['fnr'])]:
        lines.append('| '+label+' | '+' | '.join(pct(fn(x)) for x in (original,control,result))+' |')
    lines += ['| 놓친 손상 사진 | '+' | '.join(str(x['photo_rates']['fn'])+'/353' for x in (original,control,result))+' |',
        '| 정답과 겹친 예측이 전혀 없음 | '+' | '.join(str(x['positive_photos_without_any_gt_overlap']) for x in (original,control,result))+' |','',
        '알려진353장·4개 작성자 구간을 반복 사용한 비교다. 새 산업현장 검증이 아니다. 배경pixel 오탐은 손상사진 안의 mask 밖을 분모로 삼으며 정상 전체사진의 오탐과 다르다. 같은0.5 기준과 전체＋기존6구역 결합 정책을 유지했다.','',
        '## 정상 구역 비교 · CAL3개뿐','','| 항목 | 현재 선택 모델 | 앞선 정상 학습 | 이번 미탐 방향 보정 |','|---|---:|---:|---:|']
    rois=[old['normal_ROI_CAL']['accepted'],old['normal_ROI_CAL']['normal_pilot'],r['normal_ROI_CAL']['recall_projected']]
    lines += ['| 정상 구역에서 이상 후보 발생 | '+' | '.join(str(x['false_alarms'])+'/3' for x in rois)+' |',
        '| 정상 구역 오탐 비율 | '+' | '.join(pct(x['ROI_FPR']) for x in rois)+' |',
        '| 정상 전체사진 오탐 | 미측정 | 미측정 | 미측정 |','',
        'CAL3은 작성자normal bbox 내부 구역이다. 사진 전체 정상·기계 정상·독립 현장 정상의 정답이 아니다. 이3개에서0건이어도 산업현장 오탐0% 또는5% 미만을 검증한 것이 아니다. 보관 정상7개와 예약BeltCrack TEST는 모델 점수를 계산하지 않았다.','',
        '## 사전에 고정한 선택과 결과','',decision,'',
        '연구 연결용 후보의 선택은 현재 모델 대비 IoU0.5%p 이상 개선, 배경 오탐·pixel 미탐·사진 미탐·정답 겹침 실패 유지, 작은 손상 IoU·pixel 미탐 유지, 각 촬영 구간 pixel 미탐 악화1%p 이내, 정상CAL 후보 발생 감소를 함께 요구했다. 결과에 맞춰 조건을 완화하지 않았다. CAL과 원래 구간 정보 부족으로 독립 현장 성능을 주장하거나 앱 기본 모델을 자동 교체하지 않는다.','',
        '| 조건 | 결과 |','|---|---|']
    for name,passed in c['known_metric_guards'].items(): lines.append('| '+name+' | '+('통과' if passed else '실패')+' |')
    lines += ['| 정상CAL3 후보 발생 감소 | '+('통과' if c['normal_ROI_CAL_alarm_reduction'] else '실패')+' |','',
        f"현재 선택 모델SHA `{c['checkpoint_sha256']}`; 정책SHA `{c['policy']['sha256']}`. 이번 학습 파일SHA `{t['weights_sha256']}`.",'',
        '## 재개·보존·한계','',
        '- 처음의 보정 구현은 첫 optimizer 호출에서FP32 수치 오차 조건을 만족하지 못했다. 그 시도는 저장된 checkpoint·완료 journal0회이며 제외했다. 소스·protocol·실패 기록은 보존했다. 실제 완료된 새학습512회와 제외된 호출1회를 구분한다.',
        '- 별도v2에서FP32 쓰기 오차 경계에 따른 보정 여유와 검사 실패 시 정확한 이전 가중치 복원을 구현했다. 실제 학습에서는 복원 단계0회였고 처음 제안된 방향을 보정한275회는 완료된512회에 포함된다. 이 숫자는 실제journal과 비교한다.',
        f"- 기존 입력·모델·보고서{v['protected_files']:,}개의 SHA·수정시간과 기존Python소스{v['legacy_sources']}개의 SHA를 유지했다. 마지막progress와final이 같고 모든 비벨트tensor는 직전과 같다. 시설·이물질 CPU 출력의 정확 일치,3기능 CPU/GPU 및 원본 구역 CPU/GPU 일치를 확인했다.",
        '- 교사는 학습 제약용으로만 사용한다. 선택된 추론은 모델 하나·공통 분석부 하나다. 내부 모터 고장·가동 상태·수리 완료를 새로 진단하는 모델이 아니다.',
        '- 정상 구역 출처와 제안된frame 구간·중복·라이선스는 [앞선 실제 자료 검수](UNIFIED_BELT_NORMAL_RESULTS_KO.md)에 기록되어 있다. 원본 사진·정답은 Git이나 전달ZIP에 재배포하지 않는다.','',
        '![실제 보정 비교](unified-belt-recall-step-comparison.png)','',
        '- [사전 조건](unified-belt-recall-step-v2-protocol.json)','- [실제 학습 기록](unified-belt-recall-step-v2-training.json)','- [사진별 계수·CAL 구역 점수](unified-belt-recall-step-v2-validation.json)',
        '- [선택 조건](unified-belt-recall-step-v2-selection.json)','- [보정·보존 검증](unified-belt-recall-step-v2-verification.json)','- [제외된 초기 시도](unified-belt-recall-step-excluded-attempt.json)',
        '- [실행 파이프라인](../../docs/UNIFIED_BELT_RECALL_STEP_KO.md)','- [기존 출처·사용 조건](UNIFIED_MAIN_ATTRIBUTION_KO.md)','']
    text_new(REPORTS/'UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md','\n'.join(lines))
    text_new(REPO/'safelog-cpp/docs/UNIFIED_BELT_RECALL_STEP_KO.md','''# 오탐 개선을 유지하면서 미탐 방향을 보정하는 비교

이전 normal ROI 실험과 같은 데이터·draw 배열·seed·초기 가중치·optimizer·손실·4epoch512업데이트를 사용한다. 실제 Adam 가중치 변화가 학습용 양성pixel 손실을 일차 근사로 늘리는 방향을 제거한다. 실제 검증 recall이 보장되는 알고리즘이 아니므로, 알려진353장과 정상CAL3으로 다시 비교한다.

## 완료된 실행 기록

1. 최초study_unified_belt_recall_step.py는 첫 optimizer 호출에서 수치 조건 실패. checkpoint·완료journal0회이고 제외 기록을 보존했다. 이 버전으로 학습을 다시 시작하지 않는다.
2. belt_recall_step_v2.py는FP32 쓰기 오차 경계의 보정 여유와 정확한 파라미터 복원 fallback을 사용한다. optimizer의 raw moments는 유지한다. 모든 유효 변화의 signed dot를 다시 검사한다.
3. tests/test_belt_recall_step_v2.py의5개 테스트는 유해 방향 제거·직교 방향 유지·실제Adam momentum 변화·known양성만의손실·고차원FP32오차를 검증한다.
4. study_unified_belt_recall_step_v2.py --mode freeze: 기존 control draw 배열 exact 일치, 소스·입력·사전조건을 고정한다. 먼저commit한 뒤 train을 실행한다.
5. --mode train: 실제4epoch512회.64업데이트마다 파라미터·optimizer·scheduler·cursor·CPU/CUDA RNG를 저장한다. Windows PID는 읽기 전용process_exists로 확인한다. 완료된final은 재학습하지 않는다.
6. --mode eval: 같은 알려진 손상VAL353, 정상CAL3. 보관7개와BeltCrack예약TEST는 모델 점수를 계산하지 않는다.
7. --mode verify: journal의signed dot·보정/복원 횟수·실제완료 업데이트·progress·비벨트tensor·기존입력·소스·보고서 보존을 검사한다.
8. report_unified_belt_recall_step.py: 비교·선택·한계를 기록한다. 연구 선택 조건이 통과한 경우에만 별도CPU API·C++ 연결 전달물을 검증한다. 앱 기본 모델은 자동 교체하지 않는다.

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_belt_recall_step_v2.py
./.venv/Scripts/python.exe scripts/study_unified_belt_recall_step_v2.py --mode freeze
# 소스와 protocol commit 후
./.venv/Scripts/python.exe scripts/study_unified_belt_recall_step_v2.py --mode train
./.venv/Scripts/python.exe scripts/study_unified_belt_recall_step_v2.py --mode eval
./.venv/Scripts/python.exe scripts/study_unified_belt_recall_step_v2.py --mode verify
./.venv/Scripts/python.exe scripts/report_unified_belt_recall_step.py
```

완료된prepare·protocol·eval·report를 다시 실행해 기존 결과를 덮어쓰지 않는다. 이후 비교는 새run과사전조건을 사용한다. 원자료 normal ROI는 사진 전체 정상 정답이 아니고, CAL3 수치로 전체5% 미만을 선언하지 않는다.

[실제 결과](../ai-training/reports/UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md), [기존 정상 자료 범위](../ai-training/reports/UNIFIED_BELT_NORMAL_RESULTS_KO.md), [아이디어 참고](https://arxiv.org/abs/2001.06782).
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,5)); labels=['Accepted','Normal4','Protected4']
    metric_values=[(original['pixel_iou'],control['pixel_iou'],result['pixel_iou']),(original['pixel_fnr'],control['pixel_fnr'],result['pixel_fnr']),
        (original['photo_rates']['fnr'],control['photo_rates']['fnr'],result['photo_rates']['fnr']),tuple(x['ROI_FPR'] for x in rois)]
    for ax,values,title in zip(axes,metric_values,['Pixel IoU (higher better)','Damage pixel FNR','Damage photo FNR','Normal ROI FPR (ONLY CAL3)']):
        values=[100*x for x in values]; bars=ax.bar(labels,values,color=['#54738e','#caa34b','#519473'])
        for bar,value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.2f}%',ha='center',va='bottom',fontsize=8)
        ax.set_title(title,fontsize=11); ax.set_ylim(0,max(max(values)*1.22,.1)); ax.set_ylabel('%')
    fig.suptitle('Matched 4-epoch /512-update comparison: attempted first-order TRAIN recall protection',fontsize=13)
    fig.text(.5,.02,'Known VAL353 reused; CAL3 is ROI-only, no whole-photo normal FPR or field5% proof',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.93]); fig.savefig(REPORTS/'unified-belt-recall-step-comparison.png',dpi=160); plt.close(fig)
    header='## 오탐 개선·미탐 방향 보정 비교 · 2026-10-13'
    block='\n'.join([header,'',
        f"오탐과 미탐은 같은 가중치의 결과라 오탐 개선만 복사할 수 없다. 같은 데이터로 **실제4epoch512회** 학습하면서 손상 탐지를 약화하는 방향{v['projected_updates']}회를 보정했다. 검증 사진의 미탐 보장이 아니며 결과를 다시 비교했다.",'',
        f"현재/앞선정상학습/보정학습 손상사진 미탐: {original['photo_rates']['fn']}/{control['photo_rates']['fn']}/{result['photo_rates']['fn']}장(각353장). 정상CAL3의 이상 후보: {rois[0]['false_alarms']}/{rois[1]['false_alarms']}/{rois[2]['false_alarms']}개. "+decision,'',
        '정상전체사진오탐은 미측정이고 산업현장 전체5% 검증은 없다. 시설·이물질 가중치를 보존했고 보관7개·예약TEST 모델 점수는 계산하지 않았다.',''])
    for path,link in [(REPO/'README.md','safelog-cpp/ai-training/reports/UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md'),(ROOT/'README_KO.md','reports/UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md'),
        (REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md','../ai-training/reports/UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md'),(REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md','../ai-training/reports/UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md')]:
        content=path.read_bytes().decode('utf-8'); assert header not in content; nl='\r\n' if '\r\n' in content else '\n'; title,body=content.split(nl+nl,1)
        addition=block+'\n- [실제 오탐·미탐 비교]('+link+')\n'; path.write_bytes((title+nl+nl+addition.replace('\n',nl)+nl+body).encode('utf-8'))
    print(json.dumps({'report':'UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md','selected':c['selected'],'actual_completed_updates':512,'projected_steps':v['projected_updates']}),flush=True)


if __name__=='__main__': main()
