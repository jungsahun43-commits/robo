"""Explain validated per-domain error tradeoffs without new inference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_spalling_ohem import NAME,read,sha,require,PROTOCOL
from scripts.fetch_rc2119 import write_new
def main():
    path=ROOT/'reports/facility-spalling-ohem-study-comparison.json';result=read(path);proof=result['technical_verification']
    require(proof['status']=='passed'and proof['weights_sha256']==sha(ROOT/'runs'/NAME/'best.pt'),'Verified selected model required')
    control,candidate=result['experiments'][1:];rows=[]
    for label,title in(('concrete_crack','균열'),('concrete_spalling','박락')):
        for domain in('dacl','damsegment','codebrim'):
            old=control['per_class'][label]['domains'][domain];new=candidate['per_class'][label]['domains'][domain]
            for metric,key,support_key in(('fnr','fn','tp'),('fpr','fp','tn')):
                require(old[key]+old[support_key]==new[key]+new[support_key],'Rate denominator differs')
                rows.append({'class':label,'title':title,'domain':domain,'metric':metric,'control_rate':old[metric],
                    'candidate_rate':new[metric],'change_pp':(new[metric]-old[metric])*100,
                    'control_error_cases':old[key],'candidate_error_cases':new[key],'denominator':new[key]+new[support_key]})
    require(len(rows)==12,'Twelve primary rate measurements required')
    worst=max(rows,key=lambda r:r['candidate_rate']);mining=proof['actual_pixel_mining']
    for row in mining:
        require(.1*row['known_background_cells']-1e-6<=row['selected_background_cells']
            <.1*row['known_background_cells']+14248+1e-6,'Aggregated per-photo ceiling bound differs')
    report={'schema':'facility_spalling_ohem_rate_breakdown_v1','comparison_sha256':sha(path),'protocol_sha256':sha(ROOT/PROTOCOL),
        'weights_sha256':proof['weights_sha256'],'source_script_sha256':sha(Path(__file__)),'primary_rate_comparisons':rows,
        'worst_candidate_rate':worst,'actual_mining_totals':{k:sum(r[k]for r in mining)for k in mining[0]},
        'per_photo_ceiling_aggregate_bound_verified':True,'new_inference':False,'new_training_epochs':0,'source_test_used':False,
        'app_model_promoted':False,'independent_industrial_accuracy_verified':False}
    write_new(ROOT/'reports/facility-spalling-ohem-rate-breakdown.json',report)
    lines=['# 박락 보강 결과의 세부 판단','',
        f"동일 공개 VAL의 최대 오류: 기존{control['worst_error']*100:.4f}% → 후보{candidate['worst_error']*100:.4f}%. 선택 회차{candidate['best_epoch']}.",'',
        '오탐은 FP/(FP+TN), 미탐은 FN/(TP+FN)이다. 모든 출처에서 두 비율을 같이 비교하며 한 항목 개선을 전체 오류 개선으로 표시하지 않는다.','',
        '| 항목 | 출처 | 비율 | 기존 오류/분모 | 신규 오류/분모 | 기존 | 신규 | 변화 |','|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['title']} | {row['domain']} | {'미탐'if row['metric']=='fnr'else'오탐'} | {row['control_error_cases']}/{row['denominator']} | {row['candidate_error_cases']}/{row['denominator']} | {row['control_rate']*100:.2f}% | {row['candidate_rate']*100:.2f}% | {row['change_pp']:+.2f}pp |")
    lines +=['',f"후보의 최대 오류는 {worst['domain']} {worst['title']} {'미탐'if worst['metric']=='fnr'else'오탐'} {worst['candidate_error_cases']}/{worst['denominator']} = {worst['candidate_rate']*100:.4f}%다.",'',
        '| 작은 DACL 손상 | 기존 FN/양성 | 후보 FN/양성 |','|---|---:|---:|']
    for task,title in(('concrete_crack','균열'),('concrete_spalling','박락')):
        a=control['small_dacl_polygon_area_below_one_percent'][task];b=candidate['small_dacl_polygon_area_below_one_percent'][task]
        lines.append(f"| {title} | {a['false_negatives']}/{a['positive_photos']} | {b['false_negatives']}/{b['positive_photos']} |")
    lines +=['','## 다른 시설 항목 AP','',
        '| 출처·항목 | 기존 대조군 AP | 후보 AP | 차이 |','|---|---:|---:|---:|']
    for row in result['retention_gate']['per_source_other_class_ap']:
        lines.append(f"| {row['domain']} · {row['class']} | {row['control_ap']:.4f} | {row['treatment_ap']:.4f} | {row['treatment_minus_control_ap']:+.4f} |")
    lines +=['', 'AP는 순위 지표이며 정확도 백분율이 아니다. 보존 기준은 추가 학습 전 원본 대비 하락 제한 등을 적용하는 별도 기준이며, 연구 기준은 원본과 기존 대조군 모두와 비교한다.','',
        f"연구 후보 기준: {'통과'if result['research_gate']['research_candidate_nominated']else'미달'}. 별도 항목 보존 기준: {'통과'if result['retention_gate']['retention_candidate_nominated']else'미달'}. 개별 비율5% 미만: {'통과'if candidate['target_passed']else'미달'}.",'',
        '## 실제 학습 적용','',
        f"6회 전체에서 알려진 박락 배경{report['actual_mining_totals']['known_background_cells']:,}개 중{report['actual_mining_totals']['selected_background_cells']:,}개를 선별했다. 사진마다 ceil을 적용하므로 합계 비율은 정확히10%보다 조금 높다. 양성 위치{report['actual_mining_totals']['foreground_cells_retained']:,}개는 모두 기존 학습에 유지했다. 이 숫자는 반복 노출한 픽셀 수이며 새 사진·독립 정답 개수가 아니다.",'',
        '새 객체 층을 추가하지 않았으며 기존324개 state 구조와7종 사진·위치·19종 보조 출력 계약을 유지한다. 선택된 모델은 기존 PresenceClassifier로 재로딩할 수 있다. 연구 기록으로 보관하며 앱 기본 모델·프로필을 교체하지 않았다. 같은 공개 VAL의 반복 탐색으로 독립 산업현장 성능을 측정하지 않았다.','']
    doc=ROOT/'reports/FACILITY_SPALLING_OHEM_BREAKDOWN_KO.md';require(not doc.exists(),'Preserve report');doc.write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('Verified spalling OHEM rate tradeoffs and actual mining totals documented',flush=True)
if __name__=='__main__':main()
