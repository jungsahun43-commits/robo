"""Final receipt of the broadened prototype and retained accepted handoff."""
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.study_equipment_expansion import read,validate,RUN,new
from scripts.fetch_equipment_expansion import sha
from scripts.release_equipment_expansion import ARCHIVE,OUTPUTS,REPO
from scripts.release_unified_main import local_links


def main():
    validate();reports=ROOT/'reports';t=read(reports/'equipment-expansion-v1-training.json');v=read(reports/'equipment-expansion-v1-verification.json')
    results=read(reports/'equipment-expansion-v1-results.json');release=read(reports/'equipment-expansion-v1-release.json')
    assert release['status']=='passed' and release['actual_CPP_operations']==42
    assert sha(ARCHIVE)==release['archive_sha256'] and sha(RUN/'train/final.pt')==t['weights_sha256']
    previous=OUTPUTS/'unified-main-belt-tiles-role4-handoff-2026-10-13-v1.1.zip';previous_sha='3e8235e8d3cc626bf0476a05a6f8686872fd90d34ed517c6d081e707cfdbd7c9'
    assert sha(previous)==previous_sha
    with zipfile.ZipFile(previous) as z:assert z.testzip() is None
    docs=[REPO/'README.md',ROOT/'README_KO.md',REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md',REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md',
        REPO/'safelog-cpp/docs/EQUIPMENT_EXPANSION_HANDOFF_KO.md',reports/'EQUIPMENT_EXPANSION_RESULTS_KO.md',reports/'EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md']
    links=0
    for doc in docs:
        for link in local_links(doc):assert link.exists() and link.is_relative_to(REPO),link;links+=1
    result={'status':'passed','new_dataset_sources':4,'selected_source_photos':6602,'new_equipment_categories':11,
        'unique_normal_FIT':3056,'normal_CAL':858,'publisher_TEST':2199,'near_TEST_excluded':343,'reserved_TRAIN_defects':246,
        'actual_neural_epochs':30,'actual_fit_optimizer_updates':2640,'recovery_optimizer_updates':0,'discarded_unit_test_optimizer_updates':3,
        'normal_FIT_draws':42240,'one_checkpoint_one_shared_backbone':True,'architecture_extended_from_current_accepted':True,
        'source_TEST_not_used_for_fit_or_calibration':True,'normal_thresholds_fixed_before_TEST':True,'all_previous_tensors_exact':True,
        'all_old_CPU_outputs_exact':True,'old_and_new_CPU_GPU_outputs_close':True,'old_bytes_and_mtimes_preserved':v['old_files_bytes_mtimes_preserved'],
        'old_python_sources_preserved':v['old_sources_preserved'],'all_new_categories_within20percent':False,
        'stage20_categories':results['stage20_categories'],'ksdd2_surface_FPR':results['categories']['ksdd2_surface']['photo_rates']['fpr'],
        'ksdd2_surface_FNR':results['categories']['ksdd2_surface']['photo_rates']['fnr'],'photo_and_location_errors_reported_separately':True,
        'isolated_ZIP_CPU_API_checked':True,'actual_CPP_HTTP_operations':42,'owned_server_stopped':release['owned_server_stopped'],
        'prototype_model_sha256':t['weights_sha256'],'prototype_archive':str(ARCHIVE),'prototype_archive_sha256':release['archive_sha256'],
        'previous_accepted_handoff_retained':previous.name,'previous_accepted_handoff_sha256':previous_sha,
        'Android_APK_built':False,'fresh_pip_install_tested':False,'independent_industrial_field_performance_validated':False,
        'app_default_promoted':False,'new_normal_whole_photo_belt_labels':0,'local_document_links_checked':links,
        'next_priority':['preserve_ksdd2_surface_target15','improve_metal_plate_FNR_to20','cable_and_nut','normal_whole_belt_photo_FPR_and_small_damage_location','connector_defect_test_count14_needs_more_samples']}
    new(reports/'equipment-expansion-v1-final-audit.json',result);new(OUTPUTS/'EQUIPMENT_EXPANSION_RESUME_2026-10-14.json',result)
    text='\n'.join(['# 산업 설비 범위 확장 재개 기준 · 2026-10-14','',
        '사용자 목표: 각점검유형 사진오탐/미탐20%이하로우선틀확장, 다음15%이하로하나씩보강. 사진과위치지표를분리하고unknown/표본부족을통과로표시하지않는다.',
        '완료: 새4자료·11유형·사진6602장 검수. 정상FIT3056/CAL858/TEST2199. 실제30epoch2640회, 반복draw42240은새사진이아니다.',
        '같은공통MobileNet과모델파일하나에새외관분석head를추가했다. 기존시설·벨트·이물질tensor와CPU출력은정확히보존했다.',
        '첫목표통과: ksdd2_surface 오탐12.64%(113/894), 미탐10.91%(12/110). 다른10유형은보강또는평가부족. 모든유형20%이하나독립현장성능을달성한것이아니다.',
        '다음새설비보강은금속판(오탐0/26,미탐18/71=25.35%)부터. 케이블/너트로확장. TRAIN결함246장은추후지도학습용으로보관하며MVTec/MPDD의TEST결함을학습정답으로옮기지않는다.',
        '정상CALpixel분위수torch입력제한과CPU/GPU일치검사를별도finish스크립트로복구. 저장된2640회에서추가학습0회. 원frozen소스·protocol·progress·journal을보존했다.',
        'API: safelog_ai.equipment_expansion_server. 기존C++어댑터의inspectionTask에유형별칭(ad_cable,ksdd2_surface등)을전달. equipment_surface+equipmentCategory도지원한다.',
        'ZIP 실제분리CPU API와기존3+새11유형의C++42개동작통과. 같은사진비교는수리완료가아니다. AndroidAPK/새pip설치시험은미수행.',
        '새시제품: '+ARCHIVE.name,'시제품ZIP SHA: '+release['archive_sha256'],'시제품모델 SHA: '+t['weights_sha256'],
        '기존선택전달물은유지: '+previous.name,'기존ZIP SHA: '+previous_sha,'',
        '결과: safelog-cpp/ai-training/reports/EQUIPMENT_EXPANSION_RESULTS_KO.md','역할4: safelog-cpp/docs/EQUIPMENT_EXPANSION_HANDOFF_KO.md',
        '기존일반앱기본값은자동변경하지않았다. 부분성능개선은개별유형과위치에서검증한다. 이후비교는새run·학습전조건고정·commit으로진행한다.',''])
    with (OUTPUTS/'EQUIPMENT_EXPANSION_RESUME_2026-10-14.md').open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps({'final_audit':'passed','CPP_operations':42,'stage20_categories':results['stage20_categories'],'retained_old_handoff':True}),flush=True)


if __name__=='__main__':main()
