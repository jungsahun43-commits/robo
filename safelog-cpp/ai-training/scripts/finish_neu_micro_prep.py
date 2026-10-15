"""Record FIT-only visual caveat without mutating source manifests/reservations."""
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_neu_micro_candidate import DATA
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    source = read(DATA / 'source.json'); receipt = read(DATA / 'download-receipt.json')
    manifest = read(DATA / 'candidate-manifest.json'); reserve = read(DATA / 'QA-reservation.json')
    review = {'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'v22_real_mask_candidate_prepared_not_trained',
        'candidate_manifest_sha256': sha(DATA / 'candidate-manifest.json'), 'QA_reservation_sha256': sha(DATA / 'QA-reservation.json'),
        'contact_sheet_sha256': sha(DATA / 'reserved-FIT-3-QA.png'),
        'agent_viewed_only_predesignated_FIT_images': reserve['visually_inspected_FIT_candidate_ids'],
        'visual_observations': ['Bright mask broadly follows dark marks on three sampled source views; not an exhaustive pixel QA.',
            'Oil Scratch_1 RGB includes conspicuous white marks outside target mask; physical paint/chalk vs burned-in graphic unknown.',
            'Exclude whole already-viewed Oil Scratch filename family before training; do not clean it using unverified inpainting or relabel white marks.',
            'SDI_1 is particularly small and may be diluted by whole-image resize; examine FIT resolution before choosing a training protocol.'],
        'source_image_mask_pairs_downloaded': 51,
        'eligible_FIT_candidate_ids': [r['id'] for r in manifest['rows'] if r['source_folder'] in ('NEU_SDI','NEU_SPDI')],
        'unviewed_VAL_candidate_ids': reserve['unviewed_validation_candidate_ids'],
        'excluded_white_marker_family_ids': [r['id'] for r in manifest['rows'] if r['source_folder']=='NEU_Oil' and Path(r['image']).stem.startswith('Scratch_')],
        'candidate_counts': {'FIT_real_mask':35,'VAL_real_mask':10,'excluded_FIT_family':6},
        'no_new_source_normal_whole_photos': True, 'source_whole_photo_FPR_not_available': True,
        'not_new_app_category_or_independent_TEST': True,
        'retained_TEST_images_checked': manifest['audit']['retained_TEST_images_checked'],
        'cross_exact_or_orientation_near_matches': manifest['audit']['exact_RGB_or_any8orientation_pHash2_TEST_matches'],
        'within_source_near_links': 0, 'physical_capture_groups_verified': False,
        'full_training_protocol_or_final_manifest_fixed': False, 'new_encoder_features': 0, 'new_model_scores': 0, 'new_optimizer_updates': 0,
        'publisher_commit': source['publisher_commit'], 'archive_sha256_local': receipt['archive_sha256_local'],
        'archive_git_blob_and_CRC_verified': True, 'source_use_conditions': source['usage'],
        'new_natural_fault_labels_are_publisher_masks_not_agent_synthetic_labels': True,
        'all_source_images_or_masks_human_verified_by_team': False,
        'all_original_source_failed_models_reports_preserved': True}
    assert len(review['eligible_FIT_candidate_ids'])==35 and len(review['unviewed_VAL_candidate_ids'])==10
    assert len(review['excluded_white_marker_family_ids'])==6
    new(DATA / 'FIT-review-r1.json',review)
    new(ROOT / 'reports/industrial-neu-micro-v22-FIT-review-r1.json',review)
    outputs=ROOT.parents[3]/'outputs'
    result={'UTC':review['UTC'],'status':review['status'], 'repo':str(ROOT.parents[1]),
        'v8_through_v21_training_evaluation_complete_do_not_rerun':True,
        'v20_final_resume':str(outputs/'INDUSTRIAL_FINE_RESUME_2026-10-15-v20.json'),
        'v21_final_resume':str(outputs/'INDUSTRIAL_BALANCED_RESUME_2026-10-15-v21.json'),
        'v20_actual_optimizer_updates':1920,'v21_actual_optimizer_updates':1920,
        'v20_and_v21_selected_methods':{},'new_API_CPP_operations_this_turn':0,'new_ZIPs_created':0,
        'active_model':'runs/industrial-magnetic-tiles-v16/released/final.pt',
        'active_model_sha256':'5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e',
        'active_role4_ZIP':str(outputs/'unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip'),
        'active_role4_ZIP_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
        'v22_candidate_manifest':str(DATA/'candidate-manifest.json'),'v22_FIT_review':str(DATA/'FIT-review-r1.json'),
        'v22_candidate_counts':review['candidate_counts'],'v22_new_optimizer_updates':0,
        'v22_full_training_protocol_fixed':False,'v22_candidate_manifest_sha256':sha(DATA/'candidate-manifest.json'),
        'v22_FIT_review_sha256':sha(DATA/'FIT-review-r1.json'),
        'retained_TEST_images_checked':5746,'cross_exact_or_near_matches':0,
        'next_plan':'reports/NEU_MICRO_NEXT_REAL_MASK_KO.md',
        'tracked_actual_optimizer_updates_v8_v10_v13_v15_to_v21_including_failed':22480,
        'all_current_20percent_additional_equipment_pass_count':8,'additional_equipment_types':12,
        'independent_field_validated':False,'awake_helper_restarted':False,
        'automation_id':'safelog','automation_continues_user_at_school':True,
        'next':'Read new real-mask candidate plan and FIT review. Preserve family reservations/exclusion. Fix source split, normal/patch label scope, one architecture/training/VAL/CAL/guards and commit before v22 features or gradients. Do not loop failed synthetic experiments or current TEST thresholds.'}
    new(outputs/'INDUSTRIAL_NEU_MICRO_RESUME_2026-10-15-v22-PREPARED.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
