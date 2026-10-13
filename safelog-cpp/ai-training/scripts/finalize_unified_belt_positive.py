"""Final evidence for the retained model, two rejected policies and one real fit."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.unified_main_data import read,sha
from scripts.study_unified_belt_positive import validate,BASE,START,START_SHA,new
from scripts.release_unified_main import local_links
REPORTS=ROOT/'reports'; REPO=ROOT.parents[1]; OUTPUTS=REPO.parents[1]/'outputs'
ZIP_NAME='unified-main-belt-tiles-role4-handoff-2026-10-13-v1.1.zip'
ZIP_SHA='3e8235e8d3cc626bf0476a05a6f8686872fd90d34ed517c6d081e707cfdbd7c9'


def main():
    p=validate(); choice=read(REPORTS/'unified-belt-positive-selection.json'); v=read(REPORTS/'unified-belt-positive-verification.json')
    fine=read(REPORTS/'unified-belt-fine-selection.json'); training=read(REPORTS/'unified-belt-positive-training.json')
    assert choice['selected']=='prior_accepted_tiles' and fine['selected']=='global_tile_mean'
    assert choice['checkpoint_sha256']==sha(START)==START_SHA and v['status']=='passed'
    assert training['optimizer_updates']==v['actual_journal_updates']==v['committed_unique_updates']==768
    assert sha(BASE/'train/final.pt')==training['weights_sha256']
    archive_path=OUTPUTS/ZIP_NAME; assert sha(archive_path)==ZIP_SHA
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and sum(n.endswith('.pt') for n in archive.namelist())==1
        hashes=json.loads(archive.read('package-sha256.json'))
        assert all(hashlib.sha256(archive.read(n)).hexdigest()==h for n,h in hashes.items())
        model=json.loads(archive.read('model.json'))
        assert model['sha256']==START_SHA and model['inferencePolicySha256']==choice['inference_policy']['sha256']
        assert hashlib.sha256(archive.read('model.pt')).hexdigest()==START_SHA
    docs=[REPO/'README.md',ROOT/'README_KO.md',REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md',REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md',
        REPO/'safelog-cpp/docs/UNIFIED_BELT_POSITIVE_STUDY_KO.md',REPORTS/'UNIFIED_BELT_POSITIVE_RESULTS_KO.md',REPORTS/'UNIFIED_BELT_FINE_RESULTS_KO.md']
    links=0
    for doc in docs:
        for link in local_links(doc):
            assert link.is_relative_to(REPO) and link.exists(),link
            if link.suffix: assert link.is_file(),link
            links+=1
    figures={}
    for name in ['unified-belt-fine-comparison.png','unified-belt-positive-comparison.png']:
        with Image.open(REPORTS/name) as figure: figures[name]=list(figure.size); figure.verify()
    result={'status':'passed','performed_neural_epochs':6,'performed_unique_optimizer_updates':768,'actual_journal_optimizer_updates':768,
        'performed_inference_only_candidates':2,'inference_comparison_neural_updates':0,'new_selected_checkpoint':False,
        'research_main_checkpoint_sha256':START_SHA,'research_main_policy':choice['inference_policy'],
        'unselected_trained_checkpoint_sha256':training['weights_sha256'],'retained_handoff_archive':ZIP_NAME,'retained_archive_sha256':ZIP_SHA,
        'archive_weights_policy_member_SHA_and_CRC_verified':True,'new_handoff_created':False,'new_CPP_HTTP_tests_this_turn':0,
        'new_training_CPU_GPU_three_task_and_native_tile_checks':training['strict_CPU_GPU_three_tasks'],
        'old_inputs_models_reports_protected_files':v['protected_files'],'old_source_SHA_checked':v['legacy_sources'],
        'known_VAL_reused':True,'known_positive_belt_photos':353,'normal_whole_photo_labels':0,'normal_photo_FPR':None,
        'reserved_TEST_used':False,'industrial_all_tasks_5percent_verified':False,'new_independent_photos':0,'new_labels':0,
        'raw_summarizer_slice_score_definition_clarification':'Historical helper shorthand maximum sigmoid; actual scores in BOTH new studies are maximum fused global/tile probabilities over valid pixels, not calibrated. All raw counts, thresholds and scores preserved.',
        'local_links_checked':links,'figures':figures,'app_default_promoted':False}
    new(REPORTS/'unified-belt-positive-final-audit.json',result)
    new(OUTPUTS/'UNIFIED_BELT_POSITIVE_RESUME_2026-10-13.json',result)
    text='''# 다음 작업 재개 기준 · 2026-10-13

## 이번 완료 작업

- 작은 손상55장과 촬영 구간별 오류를 분리했다.
- 세밀 구역 추론2개 고정 후보는 성능이 악화해 제외했다. 이 비교의 추가 가중치 학습은0회다.
- 양성 구역 sampling＋예측 KL 보존으로 실제6epoch·768업데이트를 수행했다. 새 독립 사진·정답0개다.
- 현재 선택 모델보다 IoU와 배경 오탐은 소폭 개선했지만 손상사진 미탐2→3과 작은 손상pixel 미탐 유지 실패로 적용하지 않았다.

## 현재 유지할 모델·설정

- 모델: safelog-cpp/ai-training/runs/unified-belt-native-detail-v1/native_mix/final.pt
- modelSHA: '''+START_SHA+'''
- mode: global_tile_mean
- policySHA: '''+choice['inference_policy']['sha256']+'''
- 역할4 전달: '''+ZIP_NAME+'''
- 전달ZIP SHA: '''+ZIP_SHA+'''
- 같은 알려진 VAL353 기준: IoU47.6082%, 손상pixel 미탐32.9588%, 배경pixel 오탐0.69383%, 손상사진 미탐2/353=0.56657%.
- 정상 전체사진 오탐은 null. 독립 산업현장 전체5% 미만 검증은 없다. 앱 기본 모델은 자동 교체하지 않았다.

## 다음 우선순위

정상 벨트 전체사진의 crack 없음 정답과 촬영 구간을 확인할 수 있는 추가 자료를 먼저 확보·감사한다. 정답 범위·출처 사용 조건·촬영 구간·중복/증강 유출을 확인한 후 TRAIN/CAL/독립시험을 분리한다. local normal ROI나 이물질 없음 표기를 벨트 전체 정상 정답으로 확대하지 않는다. 같은 알려진 VAL만 반복 학습·조정해5%를 선언하지 않는다.

이번 fine와 positive 실험은 완료되었으므로 같은 이름으로 새 학습을 다시 실행하지 않는다. 새 작업은 별도 run·protocol·보고서를 작성하며 소스와 조건을 학습 전에 commit한다. Windows PID 확인에는 읽기 전용process_exists를 사용한다. 예약 BeltCrack TEST는 미사용 상태로 유지했다.

## 결과

- safelog-cpp/ai-training/reports/UNIFIED_BELT_POSITIVE_RESULTS_KO.md
- safelog-cpp/ai-training/reports/UNIFIED_BELT_FINE_RESULTS_KO.md
- safelog-cpp/ai-training/reports/unified-belt-positive-final-audit.json
- safelog-cpp/docs/UNIFIED_BELT_POSITIVE_STUDY_KO.md

새 CPU/GPU 확인은 학습 후보를 대상으로 수행했고, 이번 회차의 새 Qt/C++ HTTP 시험은0회다. 기존 전달ZIP을 변경하지 않았으므로 기존 연결 시험 결과를 계승한다.
'''
    with (OUTPUTS/'UNIFIED_BELT_POSITIVE_RESUME_2026-10-13.md').open('x',encoding='utf-8',newline='\n') as f: f.write(text)
    print(json.dumps({'status':'passed','updates':768,'retained_zip':ZIP_NAME,'new_model_adopted':False,'checked_links':links}),flush=True)


if __name__=='__main__': main()
