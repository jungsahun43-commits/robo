"""Retained-artifact and actual completed/failed optimizer-call evidence."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.unified_main_data import read,sha
from scripts.study_unified_belt_recall_step_v2 import validate,BASE,new,START_SHA
from scripts.release_unified_main import local_links
REPORTS=ROOT/'reports'; REPO=ROOT.parents[1]; OUTPUTS=REPO.parents[1]/'outputs'
ZIP_NAME='unified-main-belt-tiles-role4-handoff-2026-10-13-v1.1.zip'
ZIP_SHA='3e8235e8d3cc626bf0476a05a6f8686872fd90d34ed517c6d081e707cfdbd7c9'


def main():
    p=validate(); v=read(REPORTS/'unified-belt-recall-step-v2-verification.json'); c=read(REPORTS/'unified-belt-recall-step-v2-selection.json')
    t=read(REPORTS/'unified-belt-recall-step-v2-training.json'); r=read(REPORTS/'unified-belt-recall-step-v2-validation.json')
    excluded=read(REPORTS/'unified-belt-recall-step-excluded-attempt.json')
    assert c['selected']=='prior_accepted_tiles' and c['checkpoint_sha256']==START_SHA
    assert v['actual_journal_updates']==v['unique_committed_updates']==t['optimizer_updates']==512
    assert sha(BASE/'train/final.pt')==t['weights_sha256']
    assert excluded['optimizer_calls_attempted']==1 and excluded['journal_committed_updates']==0 and not excluded['saved_checkpoint']
    archive_path=OUTPUTS/ZIP_NAME; assert sha(archive_path)==ZIP_SHA
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and sum(n.endswith('.pt') for n in archive.namelist())==1
        hashes=json.loads(archive.read('package-sha256.json')); assert all(hashlib.sha256(archive.read(n)).hexdigest()==h for n,h in hashes.items())
        model=json.loads(archive.read('model.json')); assert model['sha256']==START_SHA and model['inferencePolicySha256']==c['policy']['sha256']
        assert hashlib.sha256(archive.read('model.pt')).hexdigest()==START_SHA
    docs=[REPO/'README.md',ROOT/'README_KO.md',REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md',REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md',
        REPO/'safelog-cpp/docs/UNIFIED_BELT_RECALL_STEP_KO.md',REPORTS/'UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md']
    links=0
    for doc in docs:
        for link in local_links(doc):
            assert link.is_relative_to(REPO) and link.exists(),link
            if link.suffix: assert link.is_file(),link
            links+=1
    with Image.open(REPORTS/'unified-belt-recall-step-comparison.png') as fig: pixels=list(fig.size); fig.verify()
    result={'status':'passed','epochs_completed':4,'completed_unique_optimizer_steps':512,'journal_optimizer_calls_completed':512,
        'excluded_failed_optimizer_calls':1,'total_optimizer_calls_attempted_this_turn':513,'saved_checkpoint_from_excluded_attempt':False,
        'projected_steps':v['projected_updates'],'reverted_parameter_steps':v['reverted_parameter_steps'],'all_effective_step_protection_checks_passed':True,
        'protection_scope':'First-order mean known-positive TRAIN pixel loss only; not validation recall guarantee',
        'matched_control_arrays_exact':p['matched_control_arrays_exact'],'no_new_images_or_labels_this_turn':True,
        'normal_ROI_CAL_false_alarms':r['normal_ROI_CAL']['recall_projected']['false_alarms'],'normal_ROI_CAL_count':3,
        'known_positive_VAL_misses':r['known_summary']['photo_rates']['fn'],'known_positive_VAL_count':353,'known_VAL_reused':True,
        'whole_photo_normal_FPR':None,'reserved_normal7_model_predictions_performed':False,'old_BeltCrack_TEST_used':False,
        'old_files_preserved':v['protected_files'],'old_sources_preserved':v['legacy_sources'],'final_equals_progress':True,'non_belt_tensors_exact_prior':True,
        'new_selected_model':False,'retained_model_sha256':START_SHA,'retained_policy_sha256':c['policy']['sha256'],
        'unselected_trained_weights_sha256':t['weights_sha256'],'retained_handoff':ZIP_NAME,'retained_handoff_sha256':ZIP_SHA,
        'retained_ZIP_CRC_member_SHA_weights_and_policy_verified':True,'new_CPP_HTTP_tests_this_turn':0,
        'new_CPU_GPU_three_tasks_and_native_tiles_checked':True,'local_links_checked':links,'figure_pixels':pixels,
        'app_default_promoted':False,'industrial_all_tasks_5percent_verified':False,
        'slice_score_scope':'Helper shorthand maximum sigmoid; actual belt scores are maximum fused global/tile probabilities over valid pixels, not calibrated'}
    new(REPORTS/'unified-belt-recall-step-final-audit.json',result); new(OUTPUTS/'UNIFIED_BELT_RECALL_STEP_RESUME_2026-10-13.json',result)
    content='\n'.join(['# 오탐·미탐 방향 보정 · 재개 기준 · 2026-10-13','',
        '이번 실제 완료 학습:4epoch512 optimizer calls. 첫 수치 조건 실패1call은checkpoint/journal완료0으로 제외. 총시도513과완료512를구분한다.',
        f"512단계 중보정{v['projected_updates']}·전체가중치복원{v['reverted_parameter_steps']}. signed-dot 수치검사 통과는TRAIN양성pixel일차근사만 의미하며 검증recall을 보장하지 않는다.",'',
        '기존/앞선normal/보정 후보: 손상사진 미탐2/6/5(각353장), 정상ROI CAL오탐2/0/0(각3개). 보정 후보도기존미탐유지조건을 충족하지 못해 적용하지 않았다.',
        '현재 선택 모델SHA: '+START_SHA,'정책: global_tile_mean / '+c['policy']['sha256'],'역할4 ZIP: '+ZIP_NAME,'ZIP SHA: '+ZIP_SHA,'',
        'normal14 FIT·CAL3·보관7 원자료는 이전대로다. 보관7과예약BeltCrack TEST 모델점수는 미사용. 정상전체사진FPR은null, 독립 산업현장 전체5% 미만검증은 없다.',
        '현재실험과초기실패namespace는 완료/제외상태로보존한다. 같은prepare/freeze/eval/report 이름으로반복해 덮어쓰지 않는다. 새 실험은별도run·사전조건·학습전commit을사용한다. WindowsPID는읽기전용process_exists로확인한다.',
        '다음은새로 확인된정상전체사진/손상사진과촬영구간 정보 확보를우선한다. 오탐과미탐은 같은가중치의결과이며오탐개선만직접복사할수없다. 부분반영방법도 실제미탐·위치와함께검증한다.','',
        '결과: safelog-cpp/ai-training/reports/UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md','파이프라인: safelog-cpp/docs/UNIFIED_BELT_RECALL_STEP_KO.md',
        '새후보: runs/unified-belt-recall-step-v2/train/final.pt (미채택)', ''])
    with (OUTPUTS/'UNIFIED_BELT_RECALL_STEP_RESUME_2026-10-13.md').open('x',encoding='utf-8',newline='\n') as f: f.write(content)
    print(json.dumps({'final_audit':'passed','completed_optimizer_steps':512,'excluded_calls':1,'retained_model':True,'checked_links':links}),flush=True)


if __name__=='__main__': main()
