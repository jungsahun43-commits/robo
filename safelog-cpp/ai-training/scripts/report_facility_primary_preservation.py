"""Compare the real fixed primary preservation candidate with all controls."""
from pathlib import Path
import os
import sys
import json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_primary_preservation import NAME,CONTROL,REFERENCE,IDEA,PROTOCOL,read,sha,validate_protocol,require,WEIGHT,TEMPERATURE
from scripts.fetch_rc2119 import write_new
from scripts.report_facility_small_region import load_run,TARGETS
from scripts.report_facility_native_roi_results import normalize_ap_positive_support
from scripts.report_facility_native_roi import validate_measurements
from scripts.report_facility_discrimination import verify_validation_cache,DOMAINS
from scripts.report_facility_retention import retention_measurements
OUTPUT='reports/facility-primary-preservation-study-comparison.json'
MARKDOWN='reports/FACILITY_PRIMARY_PRESERVATION_STUDY_RESULTS_KO.md'

def main():
    require(not(ROOT/OUTPUT).exists()and not(ROOT/MARKDOWN).exists(),'Preserve existing report')
    protocol=validate_protocol(read(ROOT/PROTOCOL));proof=read(ROOT/'reports/facility-primary-preservation-study-verification.json')
    require(proof['status']=='passed'and proof['protocol_sha256']==sha(ROOT/PROTOCOL),'Verified actual training required')
    entries=[];truths=[]
    names=(REFERENCE,CONTROL,IDEA,NAME);titles=('추가 학습 전 원본','기존 저학습률 대조군','직전 IDEA 사진 보강군','균열·박락 기존 예측 보존 후보')
    for name,title in zip(names,titles):
        _,raw=load_run(name);entry,_=normalize_ap_positive_support(raw);entry['title']=title;truth={};require(not entry['test_executed'],'No sourceTEST in selection')
        for domain in DOMAINS:
            points,digest=verify_validation_cache(read(ROOT/f'runs/{name}/validation-{domain}.json'),read(ROOT/f'runs/{name}/target-validation-{domain}-grid1.json'),entry,domain)
            normalized,_=normalize_ap_positive_support({'ranking_ap':{domain:points}});entry['ranking_ap'][domain]=normalized['ranking_ap'][domain];truth[domain]=digest
        entries.append(entry);truths.append(truth)
    require(all(t==truths[0]for t in truths),'Original VAL truth/order differs')
    gate_entries=[entries[0],entries[1],entries[3]]
    measured=validate_measurements(gate_entries,{'reference':REFERENCE,'control':CONTROL,'treatment':NAME,'research_candidate_gate':protocol['research_candidate_gate']})
    measured['retention_gate']=retention_measurements(gate_entries,protocol['retention_candidate_gate'])
    rates=[]
    for task in TARGETS:
        for domain in DOMAINS:
            for metric in('fnr','fpr'):
                values=[e['per_class'][task]['domains'][domain][metric]for e in entries]
                rates.append({'class':task,'domain':domain,'metric':metric,'reference':values[0],'original_control':values[1],
                    'matched_idea_control':values[2],'treatment':values[3],'vs_original_pp':100*(values[3]-values[1]),'vs_matched_idea_pp':100*(values[3]-values[2])})
    training=read(ROOT/f'runs/{NAME}/TRAINING.json')
    result={'schema':'facility_primary_preservation_comparison_v1','experiments':entries,**measured,'rate_breakdown':rates,'technical_verification':proof,
        'actual_new_training_epochs':training['actual_epochs'],'protocol_sha256':sha(ROOT/PROTOCOL),
        'resources':{k:training[k]for k in('elapsed_training_minutes','peak_cuda_allocated_bytes','attempted_batches','actual_optimizer_steps','actual_primary_teacher_rows','actual_primary_teacher_forward_batches')},
        'comparison_scope':protocol['comparison_scope'],'teacher_probabilities_are_new_truth':False,'val_cases_used_in_training_or_sampling':False,
        'source_test_used':False,'local_noncommercial_research_only':True,'checkpoint_redistributed':False,'app_model_promoted':False,'independent_industrial_accuracy_verified':False}
    write_new(ROOT/OUTPUT,result)
    lines=['# 균열·박락 기존 예측 보존 학습 결과','','## 진단과 변경','',
        '직전 IDEA 보강은 DACL 박락의 오답 합계가154건으로 같았지만 기존 정답11사례를 새로 틀리고 이전 오답11사례를 해결했다. 기존 임계값으로도 새 미탐5/해결 미탐2가 남았다. 사진·native 주석8사례를 읽기 전용으로 대조했고 정답과 표본을 수정하지 않았다. 이 진단으로 망각의 인과관계를 확정한 것은 아니다.','',
        '이 후보는 기존 저학습률 대조 모델058e6의 soft 예측을 원래 TRAIN의 알려진 균열·박락에만 Bernoulli KL1/T2로 보존한다. 다른5종의 기존0773 교사 KL4/T2는 유지한다. 신규 IDEA 사진·미확인 항목은 새 보존 손실에서 제외하며 교사 확률을 새로운 정답으로 기록하지 않는다.','',
        'IDEA363장(박락163/작성자 정상200)·전체 추출 배열·기존 정답·640입력·324-state 학생·초기값·손실·BN 통계·학습률·6epoch budget을 직전 IDEA군과 동일하게 유지했다. 추가 교사 연산량은 같지 않다. 학생의 추론 구조는 유지한다.','',
        '## 동일 공개 VAL 결과','','| 모델 | 실제 epoch | 선택 epoch | 최대 오탐·미탐률 | 각 비율5% 미만 |','|---|---:|---:|---:|---|']
    for e in entries:lines.append(f"| {e['title']} | {e['actual_epochs']} | {e['best_epoch']} | {e['worst_error']*100:.4f}% | {'통과'if e['target_passed']else'미달'} |")
    lines +=['','균열·박락×DACL710/Dam424/CODEBRIM611×FNR/FPR의12개 비율이며 앱 전체 오차나 독립 산업현장 오류율은 아니다. 모델마다 같은 공개 VAL에서 선택한 임계값을 사용한다.','',
        '| 항목 | 자료 | 비율 | 원래 대조 | 직전 IDEA | 새 후보 | 원래 대비pp | IDEA 대비pp |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in rates:lines.append(f"| {'균열'if r['class']==TARGETS[0]else'박락'} | {r['domain']} | {'미탐'if r['metric']=='fnr'else'오탐'} | {r['original_control']*100:.2f}% | {r['matched_idea_control']*100:.2f}% | {r['treatment']*100:.2f}% | {r['vs_original_pp']:+.2f} | {r['vs_matched_idea_pp']:+.2f} |")
    lines +=['','## 작은 손상과 알려진 다른 항목','','| 작은 DACL 손상 | 원래 대조 FN/양성 | IDEA FN/양성 | 새 후보 FN/양성 |','|---|---:|---:|---:|']
    for task,title in zip(TARGETS,('균열','박락')):
        vals=[e['small_dacl_polygon_area_below_one_percent'][task]for e in entries[1:]]
        lines.append(f"| {title} | "+' | '.join(f"{v['false_negatives']}/{v['positive_photos']}"for v in vals)+' |')
    lines +=['','| 다른 알려진 항목 | 자료 | 원래 대조 AP | 새 후보 AP |','|---|---|---:|---:|']
    for row in result['retention_gate']['per_source_other_class_ap']:lines.append(f"| {row['class']} | {row['domain']} | {row['control_ap']:.4f} | {row['treatment_ap']:.4f} |")
    lines +=['','AP는 순위 지표이며 정확도 백분율이 아니다.',
        f"원래 대조와 원본을 비교자로 둔 연구 후보 기준 **{'통과'if result['research_gate']['research_candidate_nominated']else'미달'}**, 별도 항목 보존 후보 기준 **{'통과'if result['retention_gate']['retention_candidate_nominated']else'미달'}**. 최대 오류0.5pp 개선·개별 오류2pp 악화 제한·다른 항목 AP0.02 하락 제한·작은 손상 FN2건 감소 기준을 유지했다. 직전 IDEA 대비 변화는 별도 설명이며 출시 기준을 완화하지 않는다.",'',
        '## 실제 학습과 보존 검증','',f"신규 실제{training['actual_epochs']}epoch·{training['elapsed_training_minutes']:.2f}분. 시도{training['attempted_batches']:,}batch / 실제 업데이트{training['actual_optimizer_steps']:,}회. 사전 GPU 업데이트1회는 epoch에 포함하지 않는다. 원래 대조와 직전 IDEA는 재학습하지 않았다.",'',
        f"테스트{proof['tests']['tests_run']}개, 입력{proof['protected_file_count']:,}개의 SHA/크기/수정시각과 소스{proof['runtime_source_count']}개 Git blob 보존, 실제 동일 추출·원래 정답 개수·두 교사 동결·BN·324-state strict CPU 재로드를 검증했다. 추가 교사는 원래 TRAIN 표본{training['actual_primary_teacher_rows']:,}행을 처리하고 IDEA 행0개를 처리했다.",'',
        '같은 공개 VAL과 한 seed의 반복 탐색이며 실제 장면 독립성·현장 정확도는 검증하지 않았다. VAL·TEST 사진을 TRAIN에 넣거나 사례를 정답으로 고쳐 쓰지 않았다. 앱 기본 모델·프로필을 유지한다. IDEA의 CC BY-NC-ND4.0 사용 제한에 맞춰 후보·사진은 로컬 비상업적 연구 범위로 보관하고 배포하지 않는다.','',
        '## 근거','',
        '[오답 전이·사진 주석 진단](FACILITY_IDEA_TRANSITIONS_KO.md), [지식 증류 원 논문](https://arxiv.org/abs/1503.02531), [Learning without Forgetting](https://arxiv.org/abs/1606.09282). 원본 TRAIN을 사용하는 known Bernoulli 응용이며 LwF의 신규 데이터만 사용하는 원 실험을 재현한 것은 아니다.','']
    with(ROOT/MARKDOWN).open('x',encoding='utf-8',newline='\n')as handle:handle.write('\n'.join(lines))
    plot(result);print(json.dumps({'status':'reported','worst_error':entries[-1]['worst_error'],'research_gate':result['research_gate']['research_candidate_nominated']}),flush=True)

def plot(result):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.config/matplotlib'))
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    rates=result['rate_breakdown'];xs=np.arange(len(rates));fig,axes=plt.subplots(1,2,figsize=(13.6,6.3))
    for shift,key,title,color in((-.25,'original_control','Original low-LR','#536a80'),(0.,'matched_idea_control','IDEA data arm','#138b72'),(.25,'treatment','Added primary KL','#a45897')):
        axes[0].bar(xs+shift,[r[key]*100 for r in rates],.25,label=title,color=color)
    axes[0].set_xticks(xs,[('Crack'if r['class']==TARGETS[0]else'Spall')+'\n'+r['domain']+'\n'+r['metric'].upper()for r in rates],rotation=65,ha='right',fontsize=8)
    axes[0].set_ylabel('Selected public VAL error rate (%)');axes[0].set_title('All 12 unchanged validation rates');axes[0].legend(fontsize=8)
    for name,title,color in((IDEA,'IDEA data arm','#138b72'),(NAME,'Added primary KL','#a45897')):
        h=read(ROOT/f'runs/{name}/history.json');axes[1].plot([r['epoch']for r in h],[100*r['worst_target_error']for r in h],marker='o',label=title,color=color)
    axes[1].axhline(result['experiments'][1]['worst_error']*100,label='Original low-LR',color='#536a80');axes[1].set_ylim(0,max(30,max(r['treatment']*100 for r in rates)+2));axes[1].set_xlabel('Actual training epoch');axes[1].set_ylabel('Largest of 12 rates (%)');axes[1].set_title('Same data and draw budget; extra teacher compute');axes[1].legend(fontsize=8)
    for ax in axes:ax.axhline(5,color='#c33',linestyle='--');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Primary preservation | repeated public VAL | no independent factory test',fontsize=12);fig.tight_layout()
    path=ROOT/'reports/facility-primary-preservation-study-comparison.png';require(not path.exists(),'Preserve plot');fig.savefig(path,dpi=150);plt.close(fig)
    write_new(ROOT/'reports/facility-primary-preservation-study-comparison.plot.json',{'comparison_sha256':sha(ROOT/OUTPUT),'png_sha256':sha(path),'actual_epochs':result['actual_new_training_epochs']})

if __name__=='__main__':main()
