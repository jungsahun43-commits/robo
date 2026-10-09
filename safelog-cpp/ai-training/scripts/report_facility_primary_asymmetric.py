"""Report real primary-ASL training with unchanged source-validation gates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_primary_asymmetric import NAME,CONTROL,REFERENCE,PROTOCOL,RECIPE,read,sha,validate_protocol,require
from scripts.fetch_rc2119 import write_new
from scripts.report_facility_small_region import load_run,TARGETS
from scripts.report_facility_native_roi_results import normalize_ap_positive_support
from scripts.report_facility_native_roi import validate_measurements
from scripts.report_facility_discrimination import verify_validation_cache,DOMAINS
from scripts.report_facility_retention import retention_measurements

OUTPUT="reports/facility-primary-asymmetric-study-comparison.json"
MARKDOWN="reports/FACILITY_PRIMARY_ASYMMETRIC_STUDY_RESULTS_KO.md"

def main():
    require(not(ROOT/OUTPUT).exists()and not(ROOT/MARKDOWN).exists(),"Preserve existing comparison")
    protocol=validate_protocol(read(ROOT/PROTOCOL));proof=read(ROOT/"reports/facility-primary-asymmetric-study-verification.json")
    require(proof["status"]=="passed"and proof["protocol_sha256"]==sha(ROOT/PROTOCOL),"Verified actual training required")
    entries=[];truths=[]
    for name,title in zip((REFERENCE,CONTROL,NAME),("추가 학습 전 원본","기존 저학습률 대조군","균열·박락 비대칭 손실 후보")):
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
    result={"schema":"facility_primary_asymmetric_comparison_v1","protocol_sha256":sha(ROOT/PROTOCOL),"experiments":entries,**measured,
        "technical_verification":proof,"actual_new_training_epochs":training["actual_epochs"],"primary_photo_loss_recipe":RECIPE,
        "native_diagnostic_review":protocol["native_diagnostic_review"],"human_feedback_used":False,"ai_pseudo_labels_created":0,
        "resources":{k:training[k]for k in("elapsed_training_minutes","peak_cuda_allocated_bytes","attempted_batches","actual_optimizer_steps")},
        "comparison_scope":protocol["comparison_scope"],"source_test_used":False,"app_model_promoted":False,"independent_industrial_accuracy_verified":False}
    write_new(ROOT/OUTPUT,result)
    lines=["# 균열·박락 비대칭 손실 학습 결과","",
        "이전 AI 검토 선별은14,248회 중62회(약0.44%)만 표본이 바뀌어 최대 오류 개선이 없었다. 이번에는 모든 원래 TRAIN 추출 순서를 유지하고 균열·박락 사진 손실 계산을 바꾼 후보를 실제6epoch 학습했다.","",
        "## 진단과 방법","",
        "앞서 판단을 보류한 TRAIN 오류 후보12장을 원래 출처 사진으로 다시 열었다. 3장은 두 항목 모두 시각 판단이 명확해졌고9장은 한 항목 이상 판단 불가가 남았다. 이것은 출처 태그를 알고 진행한 AI 진단이며 사람·전문가의 정답이 아니다. 화면에 표시할 때 원본이 축소될 수 있고, 원본 자체가640픽셀인 사례도 있다. 기존200장 검토 기록은 보존했다.","",
        "관찰 결과를 새 학습 정답이나 선별 표본으로 쓰지 않았다. 원래 사진·위치 지도·세부19종 태그와 검증 정답을 유지한다. 공개 교량·댐·콘크리트 표면 자료이며 실제 공장 현장 정확도를 확인한 자료가 아니다.","",
        "균열·박락의 사진 손실에 ASL 방식을 적용했다. 양성 focal 지수0, 음성 지수4, 음성 확률 이동0.05, focusing 가중치의 gradient 분리를 사전 고정했다. 쉬운 음성의 손실과 gradient를 줄이고 양성을 상대적으로 더 학습한다. 매우 잘못 예측한 양성에서 gradient를 유지하도록 FP32 logsigmoid를 사용한다.","",
        "출처별 양성 가중치와 균열·박락 강조2, 사진별 알려진 정답 수로 나누는 계산은 유지한다. 나머지5개 사진 손실은 기존gamma1 focal이다. 미확인 항목은 손실·gradient가0이고 분모에도 포함되지 않는다. 실제 Torch autograd에서 이 조건과 다른5개 항목의 계산·gradient 일치를 검사했다.","",
        "학습 초기값0773, 공개7종·324개 state 구조,47개 BN 통계, 교사 증류4/T2, 학습률, 원본 사진 추출 순서, 위치 손실과19종 보조 손실을 유지했다. 과거 대조군6epoch는 재학습하지 않았다. 바뀐 조건은 두 주항목의 사진 손실 계산이다.","",
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
        "같은 공개 VAL과 한 seed의 반복 탐색이다. sourceTEST 추론과 앱 기본 모델 교체는 수행하지 않았다. 공식 ASL의 다른 자료 성과가 이 산업안전 앱의5% 달성을 뜻하지 않으며 이번 후보의 결과로 별도 판단한다.","",
        "## 방법 출처","",
        "[Ridnik 외, ICCV2021 원 논문](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html), [저자 공식 학습 예제](https://github.com/Alibaba-MIIL/ASL/blob/main/train.py). 예제의gamma_pos0/gamma_neg4/clip0.05를 참고하고, 본 프로젝트의 부분 정답·출처 가중치·정규화·다른5종 손실 계약에 맞춰 독립 구현했다.",""]
    with(ROOT/MARKDOWN).open("x",encoding="utf-8",newline="\n")as stream:stream.write("\n".join(lines))
    print("Verified primary ASL study report written",flush=True)

if __name__=="__main__":main()
