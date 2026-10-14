"""Audit selected context release and record the next data priority."""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_loco_context as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from safelog_ai.loco_context_server import ContextEngine
from safelog_ai.ksdd_photo_only_server import KSDDPhotoEngine


def main():
    p=study.validate();choice=study.read(study.REPORTS/'industrial-loco-context-v13-selection.json')
    result=study.read(study.REPORTS/'industrial-loco-context-v13-validation.json')['candidates'][choice['selected']]
    verification=study.read(study.REPORTS/'industrial-loco-context-v13-verification.json')
    release=study.read(study.REPORTS/'industrial-loco-context-v13-release.json')
    cpu=study.read(study.REPORTS/'industrial-loco-context-v13-CPU.json')
    training=study.read(study.RUN/'trained/training.json')
    assert cpu['status']==release['status']==verification['status']=='passed'
    assert cpu['all_CPU_GPU_photo_decisions_equal'] and cpu['photos']==312
    checkpoint=ROOT/choice['checkpoint_path']
    assert sha(checkpoint)==choice['checkpoint_sha256']==cpu['model_sha256']==release['model_sha256']
    assert release['actual_current_CPP_operations']==sum(r['operations'] for r in release['CPP_cases'])==9
    assert training['actual_optimizer_updates']==1920 and training['selected_epoch_by_normal_validation']==29
    for name,digest in p['legacy_source_sha256'].items():assert sha(ROOT/name)==digest,name
    archive=Path(release['archive']);assert sha(archive)==release['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist())==1
        for name,digest in json.loads(z.read('package-sha256.json')).items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest
            if name.startswith('safelog_ai/') and name.endswith('.py') and name!='safelog_ai/__init__.py':assert sha(ROOT/name)==digest,name
    outputs=ROOT.parents[3]/'outputs';previous=study.read(outputs/'INDUSTRIAL_FOREGROUND_RESUME_2026-10-15-v12.json')
    assert sha(Path(previous['role4_ZIP']))==previous['role4_ZIP_sha256']
    row=next(r for r in study.read(study.MANIFEST)['rows'] if r['phase']=='fit')
    encoded=base64.b64encode((ROOT/row['image']).read_bytes()).decode()
    new_engine=ContextEngine(checkpoint,choice['checkpoint_sha256'],study.RUN/'evaluation.json')
    old_engine=KSDDPhotoEngine(study.START,study.EXPECTED,study.PARENT_EVALUATION)
    a=new_engine.analyze(study.CATEGORY,encoded);b=old_engine.analyze(study.CATEGORY,encoded)
    assert a['heatmapPngBase64']==b['heatmapPngBase64'] and a['originalPixelThreshold']==old_engine.points[study.CATEGORY]['pixel_threshold']
    assert not a['benchmarkStage']['ready_for_next_stage'] and not a['semanticClampWireRelationsVerified']
    record={'status':'completed_context_training_selected_incremental_improvement','finished_utc':datetime.now(timezone.utc).isoformat(),
        'repo':str(ROOT.parents[1]),'active_research_checkpoint':str(checkpoint),'active_checkpoint_sha256':choice['checkpoint_sha256'],
        'role4_ZIP':str(archive),'role4_ZIP_sha256':release['archive_sha256'],'role4_ZIP_bytes':archive.stat().st_size,
        'previous_v11_ZIP_preserved':True,'one_model_one_shared_backbone':True,'context_same_backbone_passes':2,
        'new_actual_optimizer_updates':1920,'training_epochs':30,'selected_epoch_by_normal_validation':29,'selected_epoch_updates':1856,
        'night_optimizer_total_across_v8_v10_v13_candidates':8080,'prior_real_label_updates_v8_v10':6160,
        'new_training_kind':'normal-only feature reconstruction, not real defect or explicit relation-label supervision',
        'normal_FIT_photos':199,'normal_validation_photos':55,'normal_CAL_photos':44,'known_TEST_photos':312,
        'photo_method':choice['selected'],'photo_rates':result['photo_rates'],'prior_photo_rates':p['baseline']['photo_rates'],
        'by_defect':result['by_defect'],'prior_by_defect':p['baseline_by_defect'],'stage20_pass':False,
        'additional12_stage20_count_unchanged':8,'original_LOCO_location_PNG_exact_to_v11_CPU':True,
        'all_parent_tensors_buffers_and_operating_points_preserved':True,'protected_files_checked':verification['protected_files'],
        'current_CPU_API_aliases_checked':15,'current_actual_CPP_operations':9,'prior_CPP_checks_not_counted_as_current':True,
        'CPU_GPU_all312_photo_decisions_equal':True,'new_unit_tests_passed':3,'semantic_clamp_wire_relations_verified':False,
        'context_is_verified_pixel_localization':False,'known_public_TEST_reused_for_selection':True,
        'inherited_parent_near_training_exposure_not_undone':True,'physical_acquisition_groups_locally_verified':False,
        'mirror_original_equivalence_verified':False,'independent_field_validated':False,'app_default_promoted':False,'Android_APK_built':False,
        'fresh_pip_install_tested':False,'sandbox_server_startup_failures_preserved':2,'host_server_recovery_passed_without_model_threshold_changes':True,
        'exact_server_startup_root_cause_confirmed':False,'owned_server_stopped':True,
        'next_priorities':['Search publisher datasets with non-TEST connector defect TRAIN or component boxes/counts/wire-endpoint labels; verify license, split and near groups before downloading/fitting.',
            'Use only normal publisher TRAIN to prepare object/relation annotation review material if adequate public labels cannot be found. Provisional automatic labels are not verified ground truth; do not use TEST masks.',
            'No repeat of v8 through v13 or same TEST threshold fishing. Black bracket scratches and tubes false alerts still need appropriate labels/data. Connector14 defect evaluation and whole-photo healthy belt remain insufficient.'],
        'completed_studies_do_not_rerun':previous['completed_studies_do_not_rerun']+['industrial-loco-context-v13'],
        'continuation_automation_id':'safelog','keep_existing_bounded_awake_helper_no_duplicate':True}
    new(study.REPORTS/'industrial-loco-context-v13-final-audit.json',record)
    new(outputs/'INDUSTRIAL_LOCO_RESUME_2026-10-15-v13.json',record)
    text='''# 전선 연결부 실제 학습 재개 기록

정상 FIT199장·검증55장·CAL44장으로 전체 문맥 복원 분기를30epoch·1920회 실제 학습했다. 검증으로 선택한29epoch·1856회 가중치를 사용한다. 같은 공개 TEST312장에서 미탐66.84%→39.90%, 오탐19.33% 동일. 논리 미탐66.67%→33.33%, 표면·구조67.06%→48.24%다.

20% 목표 미달이지만 고정한 점진적 개선 조건을 충족해 문맥/표면 결합 후보를 연구용으로 선택했다. 12유형 중8개20% 이내 상태는 같다. 한 모델·한 공통 분석부이며 다른 유형과 원래 위치 지도·경고 기준은 보존했다. 실제 클램프 수·전선 끝점 정답으로 지도학습한 것은 아니다.

새 ZIP의 CPU15유형 및 실제 C++9작업을 검사했다. CPU/GPU312장 판정이 모두 같다. 샌드박스 서버 시작 실패2회와 로그를 보존했으며 같은 ZIP을 호스트에서 실행해 복구했다. 정확한 원인은 확정하지 않았고 모델이나 임계값은 바꾸지 않았다. 이전 ZIP도 보존한다.

다음은 TEST와 구분된 실제 TRAIN 결함·물체/관계 라벨 자료를 찾아 검증하는 일이다. 없으면 정상 TRAIN의 구성 라벨 검수 자료부터 준비한다. 자동 추정 라벨을 검증된 정답으로 취급하지 않고, 같은 TEST 기준을 반복 바꾸지 않는다. v8~v13 완료 작업과 기존 절전 방지 helper를 중복 실행하지 않는다.
'''
    with (outputs/'INDUSTRIAL_LOCO_RESUME_2026-10-15-v13.md').open('x',encoding='utf-8',newline='\n') as out:out.write(text)
    print({'final_audit':'passed','active_model_SHA':choice['checkpoint_sha256'],'photo_rates':result['photo_rates'],'new_optimizer_updates':1920},flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
