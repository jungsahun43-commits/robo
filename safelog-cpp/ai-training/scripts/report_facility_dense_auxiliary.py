"""Report verified native19 dense study without relaxing existing gates."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_dense_auxiliary import NAME,CONTROL,REFERENCE,PROTOCOL,read,sha,validate_protocol,require
from scripts.fetch_rc2119 import write_new
from scripts.report_facility_small_region import load_run,TARGETS
from scripts.report_facility_native_roi_results import normalize_ap_positive_support
from scripts.report_facility_native_roi import validate_measurements
from scripts.report_facility_discrimination import verify_validation_cache,DOMAINS
from scripts.report_facility_retention import retention_measurements

OUTPUT="reports/facility-dense-auxiliary-study-comparison.json"
MARKDOWN="reports/FACILITY_DENSE_AUXILIARY_STUDY_RESULTS_KO.md"
LABELS={"concrete_crack":"균열","concrete_spalling":"박락"}


def main():
    require(not(ROOT/OUTPUT).exists() and not(ROOT/MARKDOWN).exists(),"Preserve completed comparison")
    protocol=validate_protocol(read(ROOT/PROTOCOL)); proof=read(ROOT/"reports/facility-dense-auxiliary-study-verification.json")
    require(proof["status"]=="passed" and proof["protocol_sha256"]==sha(ROOT/PROTOCOL),"Verified training required")
    entries=[];truths=[]
    for name,title in zip((REFERENCE,CONTROL,NAME),("추가 학습 전 원본","기존 저학습률 대조군","원본19종 위치 보조 학습")):
        training,raw=load_run(name);entry,_=normalize_ap_positive_support(raw);entry["title"]=title
        require(not entry["test_executed"],"SourceTEST excluded from repeated selection")
        truth={}
        for domain in DOMAINS:
            points,digest=verify_validation_cache(read(ROOT/f"runs/{name}/validation-{domain}.json"),read(ROOT/f"runs/{name}/target-validation-{domain}-grid1.json"),entry,domain)
            normalized,_=normalize_ap_positive_support({"ranking_ap":{domain:points}})
            entry["ranking_ap"][domain]=normalized["ranking_ap"][domain];truth[domain]=digest
        entries.append(entry);truths.append(truth)
    require(truths[0]==truths[1]==truths[2],"Validation truth/order changed")
    measured=validate_measurements(entries,{"reference":REFERENCE,"control":CONTROL,"treatment":NAME,"research_candidate_gate":protocol["research_candidate_gate"]})
    measured["retention_gate"]=retention_measurements(entries,protocol["retention_candidate_gate"])
    training=read(ROOT/f"runs/{NAME}/TRAINING.json")
    result={"schema":"facility_dense_auxiliary_study_comparison_v1","protocol_sha256":sha(ROOT/PROTOCOL),
        "technical_verification_sha256":sha(ROOT/"reports/facility-dense-auxiliary-study-verification.json"),
        "experiments":entries,**measured,"technical_verification":proof,"actual_new_training_epochs":training["actual_epochs"],
        "dense_training":{k:protocol[k] for k in("dense_spatial_loss_weight","eligible_full_train_photos","eligible_draws_by_epoch","dense_train_pixel_counts")},
        "resources":{k:training[k] for k in("elapsed_training_minutes","peak_cuda_allocated_bytes","attempted_batches","actual_optimizer_steps")},
        "comparison_scope":protocol["package_comparison_scope"],"source_test_used":False,"app_model_promoted":False}
    write_new(ROOT/OUTPUT,result)
    lines=["# 원본19종 손상 위치 보조 학습 결과","",
        "원래 사진 전체 태그만 알려 주던19종 손상·객체에 위치 정답을 추가한 후보를 실제6epoch 학습했다. 새 사진을 추가하거나 기존7종 정답을 바꾸지 않았다.","",
        "## 학습 방식","",
        "DACL 원본 TRAIN 전체 사진6,225장에만 클래스별 독립80×80 마스크를 생성했다. 같은 위치에 여러 손상 표시가 공존할 수 있다. 박락·골재 드러남·공극·침식·풍화 등 원본 라벨의 차이를 유지한다.",
        "폴리곤 밖은 해당 원본 주석의 클래스 음성으로만 사용한다. 다른 출처·부분 사진·잘못된 주석 채널은 미확인이다. 현장이 안전하다거나 내부 손상이 없다는 정답으로 해석하지 않는다. Hollowareas는 검사 후 분필 표시를 뜻하며 사진만으로 내부 결함을 진단하는 기능이 아니다.",
        "기존 세 출력과 교사 증류·47개 BN 통계·학습률·표본 순서·증강 난수를 유지했다. RGB와 두 종류 위치 지도에는 같은 좌우 반전을 적용했다. 새 위치 손실 가중치는 사전 선언한0.1이다.",
        "학습 때만 쓰는 헤드는4개 state와63,499파라미터를 추가한다. 선택된328개 state 체크포인트를 보존하고 공개 출력이 정확히 같은324개 state 추론 파일을 별도 생성했다. 앱의 기존 공개7종 출력 계약을 유지한다.","",
        "## 실제 검증 결과","","| 모델 | 실제 epoch | 선택 epoch | 최대 FNR/FPR | 엄격한5% |","|---|---:|---:|---:|---|"]
    for entry in entries:lines.append(f"| {entry['title']} | {entry['actual_epochs']} | {entry['best_epoch']} | {entry['worst_error']*100:.2f}% | {'통과' if entry['target_passed'] else '미달'} |")
    lines +=["","최대값은 균열·박락×DACL710/Dam424/CODEBRIM611×오탐·미탐의12개 비율 중 최대다. 앱 전체 오류율이나 독립 공장 현장 정확도를 뜻하지 않는다.","",
        "| 작은 DACL 손상 | 원본 FN/양성 | 기존 대조 FN/양성 | 신규 FN/양성 |","|---|---:|---:|---:|"]
    for task in TARGETS:
        values=[e["small_dacl_polygon_area_below_one_percent"][task] for e in entries]
        lines.append(f"| {LABELS[task]} | "+" | ".join(f"{v['false_negatives']}/{v['positive_photos']}" for v in values)+" |")
    lines +=["","작은 손상은 주석 면적1% 미만인 항목·사진 사례198개(균열93·박락105)다. 동일 사진이 두 항목에 포함될 수 있다.",
        f"기존 연구 후보 기준은 **{'통과' if result['research_gate']['research_candidate_nominated'] else '미달'}**이다. 최대 오류0.5pp 개선·개별 비율2pp 이내 악화·알려진 다른 항목 AP0.02 이내 하락·작은 손상 FN2건 이상 감소 기준을 유지했다.","",
        "## 실제 실행과 적용 범위","",
        f"신규 학습{training['actual_epochs']}epoch, 시도{training['attempted_batches']:,}batch, 실제 optimizer 업데이트{training['actual_optimizer_steps']:,}회. 학습·epoch검증{training['elapsed_training_minutes']:.2f}분, allocated peak{training['peak_cuda_allocated_bytes']/2**30:.3f}GiB.",
        "TRAIN 전용 일회성 preflight1회는 학습 epoch로 세지 않았다. 과거 대조군6epoch도 다시 세지 않았다. 원본 입력·소스·교사·BN 보존, 실제 추출 순서와 선택된 모델의 strict CPU 재로드를 검사했다.",
        "같은 공개 VAL에서 반복한 탐색 결과이며 새 head와 위치 손실을 묶은 과거 대조군 비교다. source-TEST 추론과 앱 기본 모델 교체는 수행하지 않았다.","",
        "## 원출처","",
        "[DACL 저자 툴킷](https://github.com/phiyodr/dacl10k-toolkit), [저자 논문·부록](https://arxiv.org/pdf/2309.00460). 현재 로컬 v2의19종 계약을 사용하며 초기 논문의18종 설명과 구분한다. 원본 라벨과 라이선스를 유지하고 본 실험에는 TRAIN만 새 위치 감독에 사용했다.",""]
    with(ROOT/MARKDOWN).open("x",encoding="utf-8",newline="\n")as stream:stream.write("\n".join(lines))
    print("Verified dense study report written",flush=True)


if __name__=="__main__":main()
