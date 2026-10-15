"""Seal verified r1; keep failed ZIP, source data, previous accepted checkpoint."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_real_mask_transfer as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
RUN=study.RUN;OUTPUTS=ROOT.parents[3]/'outputs'


def main():
    study.validate();package=study.read(RUN/'package-r1.json');release=study.read(ROOT/'reports/industrial-real-mask-transfer-v16-release.json');cpu=study.read(ROOT/'reports/industrial-real-mask-transfer-v16-CPU.json')
    assert release['status']=='passed' and release['actual_current_CPP_operations']==12 and len(release['CPP_cases'])==4
    assert cpu['status']=='passed' and cpu['actual_photos']==79 and cpu['all79_original_location_PNG_exact_to_parent_CPU']
    assert release['archive_sha256']==package['archive_sha256']==sha(Path(package['archive']))
    current=ROOT/package['checkpoint_path'];assert sha(current)==package['checkpoint_sha256']
    prior=torch.load(study.START,map_location='cpu',weights_only=True);current_state=torch.load(current,map_location='cpu',weights_only=True)
    assert all(torch.equal(t,current_state['state_dict'][n]) for n,t in prior['state_dict'].items())
    oldzip=OUTPUTS/'unified-loco-context-role4-handoff-2026-10-15-v13.zip'
    assert sha(oldzip)=='dab8309e816b6fec10a9c94393bef482f1d6d481e85b94ae2f2bf9a66ff98205'
    result={'status':'completed_verified_partial_black_bracket_improvement',
        'UTC':datetime.now(timezone.utc).isoformat(),'repo':str(ROOT.parents[1]),'active_research_checkpoint':str(current),
        'active_checkpoint_sha256':package['checkpoint_sha256'],'role4_ZIP':package['archive'],'role4_ZIP_sha256':package['archive_sha256'],'role4_ZIP_bytes':package['archive_bytes'],
        'runtime_revision':'r1','previous_v13_checkpoint_and_ZIP_preserved':True,'initial_failed_v16_ZIP_and_logs_preserved':True,
        'runtime_recovery':'restore original forward_surface(original,view,category), new forward_real_mask(native) alias; no weights, cut or evaluation changes',
        'black_bracket_photo_before':{'fpr':3/32,'fnr':39/47},'black_bracket_photo_after':{'fpr':4/32,'fnr':30/47},
        'black_bracket_by_defect':{'hole':{'fn':5,'photos':12},'scratches':{'fn':25,'photos':35}},'black_bracket_stage20_pass':False,
        'additional12_photo_stage20_count':8,'one_model_one_backbone':True,'target_backbone_passes':2,
        'v15_actual_optimizer_updates':2880,'v15_candidates_all_rejected':True,
        'v16_actual_optimizer_updates':1920,'this_work_actual_updates_total':4800,'tracked_v8_v10_v13_v15_v16_updates_including_failed_candidates':12880,
        'v16_selected_epoch_by_labelled_VAL':1,'v16_selected_epoch_retained_updates':64,
        'new_actual_defect_mask_FIT_photos':217,'new_normal_FIT_photos':499,'publisher_source_total_photos':1344,
        'source_masks_empty_or_vanishing':5,'entire_source_groups_excluded_photos':17,'empty_defect_masks_relabelled_normal':False,
        'source_research_permission':'author permits academic research only, no commercial use, citation required; no invented SPDX license',
        'new_source_self_heldout_photo_rates':{'fpr':40/123,'fnr':15/53},'new_source_self_heldout_stage20_pass':False,
        'new_source_pixel_fnr_at_half':.8393472790013606,'new_source_registered_as_app_task':False,
        'original_black_location_PNG_exact_current79':True,'original_parent_all_tensors_buffers_and_points_preserved':True,
        'current_CPU_GPU_photos_checked':79,'fresh_GPU_cached_classifications_equal':True,
        'max_CPU_GPU_score_difference':cpu['max_CPU_GPU_score_difference'],
        'current_CPU_API_aliases':15,'current_actual_CPP_operations':12,'previous_CPP_or_initial_failed_partial_checks_counted_as_current':False,
        'new_unit_tests':{'neighbor':3,'real_mask_geometry_and_group_index':3,'runtime_method_contract':2},
        'owned_server_stopped':release['owned_server_stopped'],'fresh_pip_install_tested':False,'Android_APK_built':False,
        'known_MPDD_TEST_used_for_selection':True,'source_split_fixed_before_model_scores':True,'physical_acquisition_groups_verified':False,
        'independent_field_validated':False,'app_default_promoted':False,'v14_human_relation_labels_still_pending':True,
        'v17_cross_source_preparation_complete':True,'v17_KS_FIT_VAL_vs_source_TEST_near_matches':0,'v17_optimizer_updates':0,
        'next':'Preregister FIT/VAL-derived small-learning-rate regularized mixed real-mask candidate using KS FIT1254normal146defect and magnetic FIT499normal217defect; protect all current points and outputs, do not rerun completed studies.',
        'automation_id':'safelog','automation_continues_for_user_authorized_work':True,'old_awake_helper_restored_not_restarted':True}
    for path in (RUN/'final-audit.json',ROOT/'reports/industrial-real-mask-transfer-v16-final-audit.json',OUTPUTS/'INDUSTRIAL_REALMASK_RESUME_2026-10-15-v16-r1.json'):new(path,result)
    print(result,flush=True)


if __name__=='__main__':main()
