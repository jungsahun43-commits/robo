"""Seal completed limited-sample source research without replacing the app model."""
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_commutator_surface as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
OUTPUTS=ROOT.parents[3]/'outputs'


def main():
    study.validate();s=study.read(study.RUN/'selection.json');v=study.read(study.RUN/'verification.json');cpu=study.read(study.RUN/'CPU-verification.json')
    assert v['status']==cpu['status']=='passed' and s['selected']=={} and cpu['actual_current_photos']==135
    assert s['training']['actual_optimizer_updates']==1920 and s['training']['selected_epoch']==27
    candidate=study.RUN/'trained/final.pt';assert sha(candidate)==cpu['checkpoint_sha256']
    source=s['categories'][study.CATEGORY];assert source['candidate']['rates']['fn']==0 and source['candidate']['rates']['positive_photos']==18
    result={'status':'completed_v18_source_improved_limited_sample_targets_rejected_v16_retained','UTC':datetime.now(timezone.utc).isoformat(),
        'active_research_checkpoint':str(study.START),'active_checkpoint_sha256':study.EXPECTED,
        'role4_ZIP':str(OUTPUTS/'unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),'role4_ZIP_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
        'v18_research_checkpoint':str(candidate),'v18_research_checkpoint_sha256':cpu['checkpoint_sha256'],
        'v18_actual_optimizer_updates':1920,'selected_epoch_by_VAL':27,'retained_selected_epoch_updates':1728,
        'initial_VAL_loss':study.read(study.RUN/'train/initial-VAL.json')['balanced_loss'],'best_VAL_loss':s['training']['VAL_balanced_loss'],
        'source_research_results':source,'target_results':{c:s['categories'][c] for c in study.TARGETS},'target_methods_selected':{},
        'source_not_registered_as_app_category':True,'source_TEST_positive_views':18,'source_TEST_positive_views_below20':True,'source_photo20percent_PASS_claimed':False,
        'source_photo_FNR_zero_not_pixel_FNR_zero':True,'Wilson_photo_intervals_assume_independent_views_not_capture_cluster_adjusted':True,
        'source_negative_means_surface_view_not_whole_healthy_commutator':True,'all_parent_tensors_buffers_metadata_exact':True,
        'CPU_fresh_GPU_cached_GPU_photos_checked':135,'max_CPU_GPU_photo_difference':cpu['max_CPU_GPU_photo_difference'],
        'CPU_GPU_decisions_all_equal':True,'new_API_aliases_tested':0,'new_CPP_operations':0,'new_role4_ZIP_created':False,
        'verification':v,'new_unit_tests_passed':3,'actual_current_training_runtime_errors':0,
        'tracked_v8_v10_v13_v15_v16_v17_v18_actual_optimizer_updates_including_failed':16720,
        'known_MPDD_TEST_reused_for_target_selection':True,'source_TEST_first_scoring_by_this_study':True,'independent_field_validated':False,
        'all_original_data_failed_candidates_reports_preserved':True,'automation_continues_user_at_school':True,'awake_helper_restarted':False,
        'v14_human_relations_still_pending':True,
        'next':'FIT-only target-domain synthetic-defect feasibility review using already licensed normal MPDD FIT and real-mask FIT assets. Do not treat synthetic masks as natural defect GT, do not rerun completed TEST calibration or v18, preregister before any later learning.'}
    for path in (study.RUN/'final-audit.json',ROOT/'reports/industrial-commutator-v18-final-audit.json',OUTPUTS/'INDUSTRIAL_KSDD1_RESUME_2026-10-15-v18.json'):new(path,result)
    print({k:v for k,v in result.items() if k not in ('source_research_results','target_results','verification')},flush=True)


if __name__=='__main__':main()
