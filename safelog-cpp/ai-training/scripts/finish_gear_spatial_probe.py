"""Resume after three completed studies and a zero-optimizer FIT-only probe."""
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import probe_gear_spatial as probe
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    p=read(probe.POLICY);a=read(probe.RUN/'probe.json');previous=read(probe.prior.RUN/'final-audit.json')
    assert a['actual_new_optimizer_updates']==0 and a['maximum_fine_descriptor_abs_delta']==0
    assert a['fresh_low_high_bitwiseexact'] and a['all_parent_tensor_buffer_before_after_exact']
    assert not previous['selection']['eligible_pending_package_checks']
    locks={n:(ROOT/'runs'/n/'operation.lock').exists() for n in ('industrial-gearbox-v25','industrial-gearbox-v26','industrial-gearbox-linear-v27','industrial-gear-only-v28')}
    assert not any(locks.values())
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'v26_v27_v28_completed_rejected_v29_FIT_probe_only_prepared',
       'active_model':previous['active_model'],'active_model_sha256':previous['active_model_sha256'],
       'active_role4_ZIP':previous['active_role4_ZIP'],'active_role4_ZIP_sha256':previous['active_role4_ZIP_sha256'],
       'v25_data_gate_stopped_optimizer0':True,'v25_CAL_normal10_below_fixed20':True,
       'v26_actual_optimizer':1920,'v26_selected_epoch':0,'v26_selected_initial_head_not_learned':True,
       'v27_actual_optimizer':1920,'v27_selected_epoch':0,'v27_selected_initial_zero_head_not_learned':True,
       'v28_actual_optimizer':1920,'v28_selected_epoch':1,'v28_selected_actual_updates':64,
       'v28_checkpoint_sha256':previous['research_checkpoint_sha256'],'v28_VAL_initial':.6931471824645996,
       'v28_VAL_selected':previous['training']['VAL_macro_group_BCE'],'v28_gear_TEST_rates':previous['selection']['rates'],
       'v28_gear_TEST_AUROC':previous['selection']['AUROC'],'v28_not_overall_gearbox_or_oldtarget_improvement':True,
       'this_turn_actual_optimizer_v26_v27_v28':5760,'tracked_partial_research_optimizer_including_failed':32080,
       'tracked_partial_sum_not_entire_history':True,'new_API_CPP_ZIP_checks':0,'new_ZIPs':0,
       'source_manifest_v26_SHA256':sha(probe.prior.prior.MANIFEST),
       'gear_only_manifest_v28_SHA256':sha(probe.prior.MANIFEST),'v28_data_audit':previous['data_audit'],
       'v29_probe_policy_sha256':sha(probe.POLICY),'v29_probe':a,
       'v29_probe_declared_atol':.002,'v29_probe_declared_rtol':.002,
       'v29_training_protocol_fixed':False,'v29_new_optimizer':0,'v29_new_head_or_VAL_CAL_TEST_scores':0,
       'next_plan':str(ROOT/'reports/GEAR_SPATIAL_NEXT_KO.md'),
       'next':'Fix exactly one spatial fine24 MIL-classifier protocol/code with unchanged v28 split, actual patch class labels only. Freeze initial state/trainables/cache geometry/loss/sampling/epochs/VAL/CAL/guards and commit before full caches/gradients. Preserve current default positions and all old tasks. No threshold or split sweep.',
       'source_category_coverage_limits':'spline OK3 across retained nonTEST; collar absent in v26/v27FIT; ring/shaft no reservedTEST. v28 is separately declared gear-only task, not fullsource pass.',
       'original_fullTEST3109_gear_subset2179_originals_excludedTRAIN1317_unannotated103_preserved':True,
       'new_independent_field_validation':False,'same_public_TEST_previously_used_for_development':True,
       'original_location_thresholds_and_old_KS_BS_LOCO_unchanged':True,'additional12_types_pass20_percent':8,
       'operation_locks':locks,'free_disk_bytes':shutil.disk_usage(ROOT).free,'awake_helper_restarted':False}
    new(ROOT.parents[3]/'outputs/INDUSTRIAL_GEAR_SPATIAL_RESUME_2026-10-15-v29-PROBE.json',r)
    new(ROOT/'reports/industrial-gear-spatial-v29-preparation.json',r)
    print({k:v for k,v in r.items() if k not in ('v28_data_audit','v29_probe')},flush=True)


if __name__=='__main__':main()
