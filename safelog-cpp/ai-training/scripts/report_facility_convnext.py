"""Report real primary-ASL training with unchanged source-validation gates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_convnext import NAME,CONTROL,REFERENCE,PROTOCOL,RECIPE,read,sha,validate_protocol,require
from scripts.fetch_rc2119 import write_new
from scripts.report_facility_small_region import load_run,TARGETS
from scripts.report_facility_native_roi_results import normalize_ap_positive_support
from scripts.report_facility_native_roi import validate_measurements
from scripts.report_facility_discrimination import verify_validation_cache,DOMAINS
from scripts.report_facility_retention import retention_measurements

OUTPUT="reports/facility-convnext-study-comparison.json"
MARKDOWN="reports/FACILITY_CONVNEXT_STUDY_RESULTS_KO.md"

def main():
    require(not(ROOT/OUTPUT).exists()and not(ROOT/MARKDOWN).exists(),"Preserve existing comparison")
    protocol=validate_protocol(read(ROOT/PROTOCOL));proof=read(ROOT/"reports/facility-convnext-study-verification.json")
    require(proof["status"]=="passed"and proof["protocol_sha256"]==sha(ROOT/PROTOCOL),"Verified actual training required")
    entries=[];truths=[]
    for name,title in zip((REFERENCE,CONTROL,NAME),("추가 학습 전 원본","기존 저학습률 대조군","ConvNeXt 전체 학습 후보")):
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
    result={"schema":"facility_convnext_comparison_v1","protocol_sha256":sha(ROOT/PROTOCOL),"experiments":entries,**measured,
        "technical_verification":proof,"actual_new_training_epochs":training["actual_epochs"],"primary_photo_loss_recipe":RECIPE,
        "model_inventory":training["model_inventory"],"human_feedback_used":False,"ai_pseudo_labels_created":0,
        "resources":{k:training[k]for k in("elapsed_training_minutes","peak_cuda_allocated_bytes","attempted_batches","actual_optimizer_steps")},
        "comparison_scope":protocol["comparison_scope"],"source_test_used":False,"app_model_promoted":False,"independent_industrial_accuracy_verified":False}
    write_new(ROOT/OUTPUT,result)

    lines=['# ConvNeXt 시설 모델 전체 학습 결과','',
        '기존 MobileNet 대신 ImageNet으로 초기화한 ConvNeXt-Tiny 특징 추출기 전체를 시설 TRAIN 사진으로 학습했다. 새 FPN 위치 지도와 사진7종·보조19종 head를 함께 학습한다. 이전 ConvNeXt 실험은 특징 추출기가 고정되어 있었으며 이번 후보는 해당 모델의 연속 학습이 아니다.','',
        '## 고정한 비교 조건','',
        '원본 full 사진14,248장과 파생 영역12,041개, 원래6회 추출 순서와 정답, 출처 가중치,640픽셀 입력, gamma1 사진 focal·위치 focal/Dice·19종 보조 손실을 유지했다. 교사 MobileNet과 알려진 다른5개 항목의 증류4/T2를 유지한다. AI 관찰을 정답으로 쓰지 않았고 사람 피드백0건이다.','',
        '새 학생 모델의 초기값·구조·정규화·stochastic depth가 함께 바뀌는 구성 비교다. 원래 MobileNet 가중치를 학생에게 이식하지 않았다. ImageNet 공식 encoder180개 tensor만 정확히 이식하고7종/19종/위치 head는 seed56으로 초기화했다. LayerNorm/GroupNorm을 학습하며 학생 BN 통계는 없다. 교사 BN 통계는 고정한다.','',
        'encoder와 pool normalization 학습률4e-5, 나머지 head1e-4, AdamW decay0.0002, cosine6회, batch8을 사전 고정했다. 메모리 절약을 위한 checkpointing은 stochastic-depth RNG를 보존한다. 기존 대조군6회는 재학습하지 않았다.','',
        '## 같은 공개 VAL 비교','',
        '| 모델 | 실제 epoch | 선택 epoch | 최대 오탐·미탐률 | 각 비율5% 미만 |','|---|---:|---:|---:|---|']
    for e in entries:lines.append(f"| {e['title']} | {e['actual_epochs']} | {e['best_epoch']} | {e['worst_error']*100:.4f}% | {'통과'if e['target_passed']else'미달'} |")
    lines +=['','최대값은 균열·박락×DACL710/Dam424/CODEBRIM611×오탐·미탐의12개 비율 중 최대다. 앱 전체 또는 독립 산업현장의 오류율이 아니다.','',
        '| 작은 DACL 손상 | 원본 FN/양성 | 대조군 FN/양성 | 신규 FN/양성 |','|---|---:|---:|---:|']
    for task,title in zip(TARGETS,('균열','박락')):
        values=[e['small_dacl_polygon_area_below_one_percent'][task]for e in entries]
        lines.append(f'| {title} | '+' | '.join(f"{v['false_negatives']}/{v['positive_photos']}"for v in values)+' |')
    lines +=['',f"연구 후보 기준 **{'통과'if result['research_gate']['research_candidate_nominated']else'미달'}**, 항목 보존 기준 **{'통과'if result['retention_gate']['retention_candidate_nominated']else'미달'}**. 이전 기준의 최대 오류0.5pp 개선·개별 오류2pp 악화 제한·알려진 다른 항목 AP0.02 하락 제한·작은 손상 FN2건 감소를 유지한다.",
        '',f"실제 신규 학습{training['actual_epochs']}epoch, 시도{training['attempted_batches']:,}batch, 실제 업데이트{training['actual_optimizer_steps']:,}회. 학습·회차 검증{training['elapsed_training_minutes']:.2f}분, GPU 최대 할당{training['peak_cuda_allocated_bytes']/1024**3:.2f}GiB.",
        '',f"학습 파라미터{training['model_inventory']['parameter_count']:,}개, state tensor{training['model_inventory']['state_tensor_count']}개. 기존3,244,151개 MobileNet보다 크므로 앱 연동에는 추가 비용 평가가 필요하다. 기존324개 state checkpoint와 호환되지 않으며 별도 strict 연구용 adapter를 제공한다.",
        '',f"테스트{proof['tests']['tests_run']}개, 실제 TRAIN-only GPU 업데이트와 CPU/GPU 저장·재로딩, encoder 실제 변경·고정 교사 보존, 원본{proof['protected_file_count']:,}개 파일의 SHA·크기·수정시각 보존을 확인했다.",
        '', '같은 공개 VAL과 한 seed의 반복 탐색이다. sourceTEST 추론과 앱 기본 모델 교체는 수행하지 않았다. 교량·댐·콘크리트 표면 공개 자료의 결과이며 독립 산업체 현장 성능을 확인한 것은 아니다.','',
        '## 방법 출처','',
        '[Torchvision 공식 모델](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.convnext_tiny.html), [CVPR2022 원 논문](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html). 공식 ImageNet 결과를 이 앱의 안전 판단 성능으로 대체하지 않는다.','']
    with(ROOT/MARKDOWN).open('x',encoding='utf-8',newline='\n')as stream:stream.write('\n'.join(lines))
    print('Verified full-encoder ConvNeXt report written',flush=True)
if __name__=='__main__':main()
