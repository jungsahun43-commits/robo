"""Report real IDEA photo-data results using unchanged public validation gates."""
from pathlib import Path
import os
import sys
import json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_idea_research import NAME,CONTROL,REFERENCE,PROTOCOL,read,sha,validate_protocol,require,REPLACEMENTS,DRAW_COUNT
from scripts.fetch_rc2119 import write_new
from scripts.report_facility_small_region import load_run,TARGETS
from scripts.report_facility_native_roi_results import normalize_ap_positive_support
from scripts.report_facility_native_roi import validate_measurements
from scripts.report_facility_discrimination import verify_validation_cache,DOMAINS
from scripts.report_facility_retention import retention_measurements
OUTPUT='reports/facility-idea-study-comparison.json'
MARKDOWN='reports/FACILITY_IDEA_STUDY_RESULTS_KO.md'

def main():
    require(not(ROOT/OUTPUT).exists()and not(ROOT/MARKDOWN).exists(),'Preserve completed report')
    protocol=validate_protocol(read(ROOT/PROTOCOL));proof=read(ROOT/'reports/facility-idea-study-verification.json')
    require(proof['status']=='passed'and proof['protocol_sha256']==sha(ROOT/PROTOCOL),'Verified actual training required')
    entries=[];truths=[]
    for name,title in zip((REFERENCE,CONTROL,NAME),('추가 학습 전 원본','기존 저학습률 대조군','IDEA 건물 박락·정상 사진 보강')):
        training,raw=load_run(name);entry,_=normalize_ap_positive_support(raw);entry['title']=title
        require(not entry['test_executed'],'SourceTEST excluded from repeated selection');truth={}
        for domain in DOMAINS:
            points,digest=verify_validation_cache(read(ROOT/f'runs/{name}/validation-{domain}.json'),read(ROOT/f'runs/{name}/target-validation-{domain}-grid1.json'),entry,domain)
            normalized,_=normalize_ap_positive_support({'ranking_ap':{domain:points}});entry['ranking_ap'][domain]=normalized['ranking_ap'][domain];truth[domain]=digest
        entries.append(entry);truths.append(truth)
    require(truths[0]==truths[1]==truths[2],'Original validation truth/order changed')
    measured=validate_measurements(entries,{'reference':REFERENCE,'control':CONTROL,'treatment':NAME,'research_candidate_gate':protocol['research_candidate_gate']})
    measured['retention_gate']=retention_measurements(entries,protocol['retention_candidate_gate'])
    training=read(ROOT/f'runs/{NAME}/TRAINING.json');data=read(ROOT/'reports/facility-idea-data.json');rates=[]
    for task in TARGETS:
        for domain in DOMAINS:
            old=entries[1]['per_class'][task]['domains'][domain];new=entries[2]['per_class'][task]['domains'][domain]
            for metric in('fnr','fpr'):
                rates.append({'class':task,'domain':domain,'metric':metric,'control':old[metric],'treatment':new[metric],
                    'change_pp':(new[metric]-old[metric])*100,'control_errors':old['fn'if metric=='fnr'else'fp'],
                    'treatment_errors':new['fn'if metric=='fnr'else'fp'],'denominator':new['positive_photos'if metric=='fnr'else'negative_photos']})
    result={'schema':'facility_idea_research_comparison_v1','protocol_sha256':sha(ROOT/PROTOCOL),'experiments':entries,**measured,'technical_verification':proof,
        'actual_new_training_epochs':training['actual_epochs'],'new_data':data,'rate_breakdown':rates,'resources':{k:training[k]for k in('elapsed_training_minutes','peak_cuda_allocated_bytes','attempted_batches','actual_optimizer_steps')},
        'comparison_scope':protocol['comparison_scope'],'local_noncommercial_research_only':True,'checkpoint_redistributed':False,'human_feedback_used':False,
        'ai_pseudo_labels_created':0,'source_test_used':False,'app_model_promoted':False,'independent_industrial_accuracy_verified':False}
    write_new(ROOT/OUTPUT,result)
    lines=['# IDEA 건물 박락·정상 사진 보강 결과','','## 이번에 바꾼 것','',
        f"구조기술자가 주석을 작성한 IDEA 공개 자료에서 중복·기하 검사를 통과한 원본 그룹 대표 사진 **{data['photos']}장**을 사용했다. 콘크리트 박락 양성 {data['strata']['rc_spalling']}장, 작성자가 사진 전체를 ‘손상 없음’으로 표시한 건물 사진 {data['strata']['author_no_damage']}장이다. 정상 사진의 재질은 확인되지 않았다.",'',
        '건물 유형 AE_BUILDING/GL-AEDES에서 RC column/beam/wall/Deck-sl ab/joint 부재의 Concrete spalling 명시 주석만 박락 양성으로 옮겼다. 같은 콘크리트 부재의 균열이 명시되면 균열 양성도 사용한다. 교량·댐·교회·미장 탈락·콘크리트 파쇄를 신규 박락 사진에 섞지 않았다. 전체 박락 후보189장을 조사한 뒤 사진 그룹 및 겹침을 제외한 수가 위 양성 장수다.','',
        '작성자의 no damage 사진만 균열·박락 음성0/0으로 사용한다. 손상 사진에서 표시가 없으면 음성으로 간주하지 않는다. 다른5종 항목·19종 보조 항목은 미확인으로 두며, 모든 새 위치 채널도 미확인이다. 사각형 주석을 픽셀 정답으로 채우지 않았다. 기존 정답을 수정하지 않았고 AI·사람 추가 정답0건이다.','',
        f'원래14248draw 중 회차마다{REPLACEMENTS}개({REPLACEMENTS/DRAW_COUNT*100:.2f}%)를 신규 사진으로 대체하고 박락 양성/정상 각{REPLACEMENTS//2}draw를 사용했다. 나머지 원본 추출 순서를 유지했다. 새 사진 노출 비율과 데이터가 함께 바뀐 패키지 비교이며 순수 데이터 한 요소의 인과효과는 아니다.','',
        '원본 MobileNet0773 초기값·324개 state 구조·640 입력·원래 사진 focal/위치 focal+Dice/보조19 손실·교사 증류4/T2·BN 통계·학습률을 유지하고6epoch를 실제 학습했다. 기존 완료 대조군을 재사용하며 재학습 회수로 중복 계산하지 않는다.','',
        '## 동일 공개 VAL 결과','','| 모델 | 실제 epoch | 선택 epoch | 최대 오탐·미탐률 | 각 비율5% 미만 |','|---|---:|---:|---:|---|']
    for entry in entries:lines.append(f"| {entry['title']} | {entry['actual_epochs']} | {entry['best_epoch']} | {entry['worst_error']*100:.4f}% | {'통과'if entry['target_passed']else'미달'} |")
    lines +=['','균열·박락×DACL710/Dam424/CODEBRIM611×오탐·미탐의12개 비율을 그대로 비교한다. 최대값은 앱 전체 오류율이 아니며, 실제 공장 현장의 정확도를 확인한 지표도 아니다.','',
        '| 항목 | 자료 | 오류 | 기존 대조 | 신규 후보 | 변화 pp |','|---|---|---|---:|---:|---:|']
    for row in rates:lines.append(f"| {'균열'if row['class']==TARGETS[0]else'박락'} | {row['domain']} | {'미탐'if row['metric']=='fnr'else'오탐'} | {row['control']*100:.2f}% | {row['treatment']*100:.2f}% | {row['change_pp']:+.2f} |")
    lines +=['','음수 변화가 개선이다. 검증 사진에 맞춰 선택한 threshold가 모델마다 달라서 직접적인 같은 threshold 효과로 해석하지 않는다.','',
        '## 작은 손상과 다른 항목 보존','','| 작은 DACL 손상 | 원본 FN/양성 | 기존 대조 FN/양성 | 신규 FN/양성 |','|---|---:|---:|---:|']
    for task,title in zip(TARGETS,('균열','박락')):
        values=[e['small_dacl_polygon_area_below_one_percent'][task]for e in entries]
        lines.append(f"| {title} | "+' | '.join(f"{v['false_negatives']}/{v['positive_photos']}"for v in values)+' |')
    lines +=['',f"기존 연구 후보 기준 **{'통과'if result['research_gate']['research_candidate_nominated']else'미달'}**, 별도 항목 보존 기준 **{'통과'if result['retention_gate']['retention_candidate_nominated']else'미달'}**. 최대 오류0.5pp 개선·개별 오류2pp 악화 제한·다른 알려진 항목 AP0.02 하락 제한·작은 손상 FN2건 감소 기준을 유지했다.",'',
        '## 검증 근거','',f"신규 실제6epoch, 시도{training['attempted_batches']:,}batch / 실제 업데이트{training['actual_optimizer_steps']:,}회, 학습·회차 검증{training['elapsed_training_minutes']:.2f}분. 별도 사전 GPU 업데이트1회는 epoch로 세지 않았다.",'',
        f"집중 테스트{proof['tests']['tests_run']}개, 기존·신규 입력{proof['protected_file_count']:,}개 SHA·크기·수정시각 보존, 동결 소스{proof['runtime_source_count']}개 Git blob, 실제 추출 순서와 알려진 정답·미확인 개수, 교사·BN 및 선택 모델의 strict CPU 재로드를 확인했다.",'',
        '같은 공개 VAL과 한 seed의 반복 탐색으로 독립 현장 정확도를 입증한 것은 아니다. 정상 건물 사진은 재질과 현장 종류가 확인되지 않았고 지진 피해 건물도 포함된다. 작성자의 손상 없음 기준을 앱의 가시적 표면 손상 기준으로 독립 재검수하지 않았다. 사진 dHash 그룹 및 기존 모든 자료의 local-view screen으로 중복 후보를 제외했지만 실제 장면 독립성은 확인하지 못했다. sourceTEST 추론·앱 기본 모델 교체는 하지 않았다.','',
        '## 출처와 사용 범위','',
        '[IDEA 원 데이터](https://zenodo.org/records/15120522), [EUCENTRE 작성자 소개](https://www.eucentre.it/en/image-database-for-earthquake-damage-annotation/). CC BY-NC-ND4.0이며 로컬 비상업적 연구에 제한한다. 원본·변형 사진과 이 연구 모델을 배포하지 않는다. GitHub에는 소프트웨어와 통계 보고서를 올린다. 앱 배포 모델의 데이터로 자동 채택하지 않는다.','',
        '용량20.2GB 전체 ZIP을 내려받은 것이 아니라 공개 byte-range로 주석 및 선택 사진만 확보했다. ZIP member CRC32/크기와 로컬 SHA256은 확인했고 전체 archive MD5는 확인하지 않았다고 명시했다.','']
    text='\n'.join(lines).replace('Deck-sl ab','Deck/slab')
    with(ROOT/MARKDOWN).open('x',encoding='utf-8',newline='\n')as handle:handle.write(text)
    plot(result)
    print(json.dumps({'status':'reported','worst_error':entries[-1]['worst_error'],'research_gate':result['research_gate']['research_candidate_nominated']}),flush=True)

def plot(result):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.config/matplotlib'))
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    history=read(ROOT/f'runs/{NAME}/history.json');rates=result['rate_breakdown']
    fig,axes=plt.subplots(1,2,figsize=(13,6));xs=np.arange(len(rates));width=.38
    axes[0].bar(xs-width/2,[r['control']*100 for r in rates],width,label='Original low-LR control',color='#536a80')
    axes[0].bar(xs+width/2,[r['treatment']*100 for r in rates],width,label='IDEA building supplement',color='#138b72')
    axes[0].set_xticks(xs,[('Crack'if r['class']==TARGETS[0]else'Spall')+'\n'+r['domain']+'\n'+r['metric'].upper()for r in rates],rotation=65,ha='right',fontsize=8)
    axes[0].set_ylabel('Selected validation error rate (%)');axes[0].set_title('All 12 original validation rates');axes[0].axhline(5,color='#c33',linestyle='--');axes[0].legend(fontsize=8)
    axes[1].plot([h['epoch']for h in history],[h['worst_target_error']*100 for h in history],marker='o',color='#138b72',label='IDEA actual epochs')
    axes[1].axhline(result['experiments'][1]['worst_error']*100,label='Original low-LR control',color='#536a80');axes[1].axhline(5,label='Strict target <5%',color='#c33',linestyle='--')
    axes[1].set_ylim(0,max(30,max(h['worst_target_error']*100 for h in history)+2));axes[1].set_xlabel('Actual training epoch');axes[1].set_ylabel('Largest of 12 validation rates (%)');axes[1].set_title('Fixed six-epoch research trial');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Local noncommercial IDEA research | repeated public VAL | no independent factory test',fontsize=11)
    fig.tight_layout();path=ROOT/'reports/facility-idea-study-comparison.png';require(not path.exists(),'Preserve existing plot');fig.savefig(path,dpi=150);plt.close(fig)
    write_new(ROOT/'reports/facility-idea-study-comparison.plot.json',{'comparison_sha256':sha(ROOT/OUTPUT),'png_sha256':sha(path),'actual_epochs':len(history),'independent_industrial_accuracy_verified':False})

if __name__=='__main__':main()
