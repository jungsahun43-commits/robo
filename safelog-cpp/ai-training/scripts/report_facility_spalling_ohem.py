"""Report real spalling background-mining training and unchanged gates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_spalling_ohem import NAME,CONTROL,REFERENCE,PROTOCOL,RECIPE,read,sha,validate_protocol,require
from scripts.fetch_rc2119 import write_new
from scripts.report_facility_small_region import load_run,TARGETS
from scripts.report_facility_native_roi_results import normalize_ap_positive_support
from scripts.report_facility_native_roi import validate_measurements
from scripts.report_facility_discrimination import verify_validation_cache,DOMAINS
from scripts.report_facility_retention import retention_measurements

OUTPUT="reports/facility-spalling-ohem-study-comparison.json"
MARKDOWN="reports/FACILITY_SPALLING_OHEM_STUDY_RESULTS_KO.md"

def main():
    require(not(ROOT/OUTPUT).exists()and not(ROOT/MARKDOWN).exists(),"Preserve existing comparison")
    protocol=validate_protocol(read(ROOT/PROTOCOL));proof=read(ROOT/"reports/facility-spalling-ohem-study-verification.json")
    require(proof["status"]=="passed"and proof["protocol_sha256"]==sha(ROOT/PROTOCOL),"Verified actual training required")
    entries=[];truths=[]
    for name,title in zip((REFERENCE,CONTROL,NAME),("추가 학습 전 원본","기존 저학습률 대조군","박락 배경 픽셀 집중 학습 후보")):
        training,raw=load_run(name);entry,_=normalize_ap_positive_support(raw);entry["title"]=title
        require(not entry["test_executed"],"SourceTEST excluded from repeated selection");truth={}
        for domain in DOMAINS:
            points,digest=verify_validation_cache(read(ROOT/f"runs/{name}/validation-{domain}.json"),read(ROOT/f"runs/{name}/target-validation-{domain}-grid1.json"),entry,domain)
            normalized,_=normalize_ap_positive_support({"ranking_ap":{domain:points}});entry["ranking_ap"][domain]=normalized["ranking_ap"][domain];truth[domain]=digest
        entries.append(entry);truths.append(truth)
    require(truths[0]==truths[1]==truths[2],"Validation truth/order changed")
    measured=validate_measurements(entries,{"reference":REFERENCE,"control":CONTROL,"treatment":NAME,"research_candidate_gate":protocol["research_candidate_gate"]})
    measured["retention_gate"]=retention_measurements(entries,protocol["retention_candidate_gate"])
    training=read(ROOT/f"runs/{NAME}/TRAINING.json")
    result={"schema":"facility_spalling_ohem_comparison_v1","protocol_sha256":sha(ROOT/PROTOCOL),"experiments":entries,**measured,
        "technical_verification":proof,"actual_new_training_epochs":training["actual_epochs"],"primary_photo_loss_recipe":RECIPE,
        "actual_pixel_mining":proof["actual_pixel_mining"],"human_feedback_used":False,"ai_pseudo_labels_created":0,
        "resources":{k:training[k]for k in("elapsed_training_minutes","peak_cuda_allocated_bytes","attempted_batches","actual_optimizer_steps")},
        "comparison_scope":protocol["comparison_scope"],"source_test_used":False,"app_model_promoted":False,"independent_industrial_accuracy_verified":False}
    write_new(ROOT/OUTPUT,result)
    lines=['# 박락 배경 픽셀 집중 학습 결과','',
        '박락으로 잘못 인식하는 배경을 학습하는 공간 손실 후보를 원래 MobileNet0773 초기값에서 실제6epoch 학습했다. 원본 사진·정답·추출 순서·위치 지도·모델 구조324개 state와 교사·BN·학습률은 유지한다. 이전 ConvNeXt 후보와의 모델 크기 차이가 효과로 섞이지 않게 기존 저학습률 대조군을 재사용한다.','',
        '각 사진에서 위치 정답이 명시된 박락 배경 중 현재 negative focal 손실이 큰 상위10%를 ceil로 선택한다. 선택 손실 합에 전체 배경 수/선택 수를 곱하고 기존7종 전체의 분모를 유지한다. 어려운 꼬리의 평균을 강조하는 목적 함수이며 원래 배경 손실의 불편 추정량이 아니다.','',
        '박락 양성·0보다 큰 soft target, 다른6종 위치 focal,7종 Dice, 사진 gamma1 focal,19종 보조 손실과 알려진5종 교사 증류4/T2를 유지한다. 위치 미확인 채널은 선별하지 않는다. 선택되지 않은 배경의 focal 기여는0이지만 양성 사진에서는 기존 Dice gradient가 남는다. 원본 마스크를 새로운 정답으로 수정하지 않았고 AI·사람 추가 정답0건이다.','',
        '기존 태그 기반 사진 표본 변경·사진 순위 손실·ASL·pooling 실험과 다른 픽셀 단위 방법이다. 새 독립 사진을 추가하지 않았다. 같은 공개 VAL과 한 seed의 반복 탐색으로 독립 산업현장 정확도를 확인한 것은 아니다.','',
        "## 동일 공개 VAL 비교","","| 모델 | 실제 epoch | 선택 epoch | 최대 오탐·미탐률 | 각 비율5% 미만 |","|---|---:|---:|---:|---|"]
    for e in entries:lines.append(f"| {e['title']} | {e['actual_epochs']} | {e['best_epoch']} | {e['worst_error']*100:.4f}% | {'통과'if e['target_passed']else'미달'} |")
    lines +=["","최대값은 균열·박락×DACL710/Dam424/CODEBRIM611×오탐·미탐의12개 비율 중 최대다. 앱 전체 오류율과 독립 산업체 현장 오류율은 이 수치로 계산하지 않는다.","",
        "| 작은 DACL 손상 | 원본 FN/양성 | 기존 대조 FN/양성 | 신규 FN/양성 |","|---|---:|---:|---:|"]
    for task,title in zip(TARGETS,("균열","박락")):
        values=[e["small_dacl_polygon_area_below_one_percent"][task]for e in entries]
        lines.append(f"| {title} | "+" | ".join(f"{v['false_negatives']}/{v['positive_photos']}"for v in values)+" |")
    lines +=["",f"기존 연구 후보 기준 **{'통과'if result['research_gate']['research_candidate_nominated']else'미달'}**, 기존 항목 보존 기준 **{'통과'if result['retention_gate']['retention_candidate_nominated']else'미달'}**. 최대 오류0.5pp 개선·개별 오류2pp 악화 제한·다른 알려진 항목 AP0.02 하락 제한·작은 손상 FN2건 감소 기준을 유지한다.","",
        f"실제 신규 학습6epoch, 시도{training['attempted_batches']:,}batch, 실제 optimizer 업데이트{training['actual_optimizer_steps']:,}회. 학습·회차별 검증{training['elapsed_training_minutes']:.2f}분. 일회성 사전 GPU 업데이트1회는 epoch로 세지 않았다.","",
        f"집중 테스트{proof['tests']['tests_run']}개, 원본과 신규 입력{proof['protected_file_count']:,}개의 SHA·크기·수정시각 보존, 실제 추출 순서와 원래 정답 개수, 교사·BN 및 선택 모델의 strict CPU 재로드를 검증했다.","",
        "같은 공개 VAL과 한 seed의 반복 탐색이다. sourceTEST 추론과 앱 기본 모델 교체는 수행하지 않았다. 원래 OHEM은 detector region 선별 방법이며 본 후보는 알려진 박락 배경 픽셀에 독립 적용했다. 다른 자료의 성과가 본 앱의5% 달성을 의미하지 않는다.","",
        "## 방법 출처","",
        "[Shrivastava 외, CVPR2016 원 논문](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html). 원 논문의 region detector를 그대로 재현한 코드가 아닌 픽셀 선별 응용이다.",""]
    with(ROOT/MARKDOWN).open("x",encoding="utf-8",newline="\n")as stream:stream.write("\n".join(lines))
    print("Verified spalling pixel OHEM study report written",flush=True)

if __name__=="__main__":main()
