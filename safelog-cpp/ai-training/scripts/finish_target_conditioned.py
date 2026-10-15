"""Seal rejected v19; all original and failed research artifacts stay intact."""
from datetime import datetime,timezone
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_target_conditioned as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
OUTPUTS=ROOT.parents[3]/'outputs'


def main():
    study.validate();s=study.read(study.RUN/'selection.json');v=study.read(study.RUN/'verification.json');data=study.read(study.RUN/'data-ready.json')['audit']
    assert v['status']=='passed' and s['selected']=={} and s['training']['actual_optimizer_updates']==1920
    checkpoint=study.RUN/'trained/final.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    residual_weight=float(state['state_dict']['target_conditioned_head.fusion.0.weight'][:,72:].abs().sum())
    result={'status':'completed_v19_all_candidates_rejected_v16_retained','UTC':datetime.now(timezone.utc).isoformat(),
        'preregistered_commit':'b1284b8','active_research_checkpoint':str(study.START),'active_checkpoint_sha256':study.EXPECTED,
        'role4_ZIP':str(OUTPUTS/'unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),'role4_ZIP_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
        'v19_rejected_research_checkpoint':str(checkpoint),'v19_rejected_checkpoint_sha256':sha(checkpoint),
        'v19_actual_optimizer_updates':1920,'selected_epoch_by_synthetic_VAL':24,'retained_selected_epoch_updates':1536,
        'initial_VAL_loss':study.read(study.RUN/'train/initial-VAL.json')['balanced_loss'],'best_VAL_loss':s['training']['VAL_balanced_loss'],
        'training_labels':'publisher normal surfaces plus procedural RGB edit support, not natural target defect GT','new_natural_target_fault_GT':0,
        'data_counts':data['counts'],'synthetic_failure_attempts':data['synthetic_failure_attempts'],'source_asset_failures':data['source_asset_failures'],
        'target_results':s['categories'],'selected_methods':{},'initially_zero_residual_channel_weight_L1_after_training':residual_weight,
        'verification':v,'new_unit_tests_passed':3,'new_API_aliases_tested':0,'new_CPP_operations':0,'new_role4_ZIP_created':False,
        'all_parent_tensors_buffers_metadata_exact':True,'old_primary_locations_and_thresholds_unchanged':True,
        'tracked_v8_v10_v13_v15_v16_v17_v18_v19_actual_optimizer_updates_including_failed':18640,
        'known_public_TEST_reused_for_adoption':True,'independent_field_validated':False,'physical_synthetic_fault_realism_verified':False,
        'training_runtime_errors':0,'all_original_data_failed_candidates_reports_preserved':True,'awake_helper_restarted':False,
        'v14_human_relation_review_still_pending':True,'automation_continues_user_at_school':True,
        'next':'FIT-only fine-feature tap equivalence and paired procedural-edit response study. Do not change thresholds or rerun v19. Use v19 FIT/VAL only for later design; preregister any new decoder before new learning/CAL/TEST scores.'}
    for path in (study.RUN/'final-audit.json',ROOT/'reports/industrial-target-conditioned-v19-final-audit.json',OUTPUTS/'INDUSTRIAL_TARGET_CONDITIONED_RESUME_2026-10-15-v19.json'):new(path,result)
    print({k:v for k,v in result.items() if k not in ('target_results','verification','data_counts')},flush=True)


if __name__=='__main__':main()
