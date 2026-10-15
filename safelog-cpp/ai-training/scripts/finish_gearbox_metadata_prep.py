"""Record completed v24 and metadata-only v25 preparation without scoring images."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

OUTPUTS = ROOT.parents[3] / 'outputs'
DATA = ROOT / 'data/industrial-gearbox-source-v25'


def main():
    metadata = read(DATA / 'metadata-labels-audit.json')
    receipt = read(DATA / 'download-receipt.json')
    previous = read(OUTPUTS / 'INDUSTRIAL_NEU_TUBE_RESUME_2026-10-15-v24.json')
    assert not previous['selected_methods'] and previous['actual_optimizer_updates'] == 1920
    assert metadata['annotated_unique_patches'] == 15588
    assert metadata['reserved_author_TEST_patches'] == 3109
    assert metadata['model_features_scores_optimizer_updates'] == 0
    assert not metadata['image_content_viewed_by_agent']
    assert not metadata['final_FIT_VAL_CAL_manifest_fixed']
    assert sha(DATA / 'publisher-metadata.json') == receipt['publisher_metadata_sha256']
    for name, audit in metadata['author_json_consistency'].items():
        assert sha(DATA / 'publisher-annotations' / name) == audit['sha256']
    locks = {name: (ROOT / 'runs' / name / 'operation.lock').exists() for name in (
        'industrial-neu-detail-v22', 'industrial-neu-source-detail-v23',
        'industrial-neu-tube-adapt-v24')}
    assert not any(locks.values())
    result = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'v24_completed_rejected_v25_download_and_metadata_only_prepared',
        'active_model': previous['active_model'],
        'active_model_sha256': previous['active_model_sha256'],
        'active_role4_ZIP': previous['active_role4_ZIP'],
        'active_role4_ZIP_sha256': previous['active_role4_ZIP_sha256'],
        'v22_stopped_before_features_scores_optimizer': True,
        'v22_reason': 'duplicate audit left 29 black normal FIT photos, below preregistered minimum30',
        'v23_v24_completed_actual_optimizer_updates': [1920, 1920],
        'this_turn_actual_optimizer_updates': 3840,
        'v23_v24_selected_methods': {},
        'v24_checkpoint_sha256': previous['v24_checkpoint_sha256'],
        'tracked_partial_research_optimizer_updates_including_failed': 26320,
        'tracked_updates_are_not_entire_training_history': True,
        'new_package_API_CPP_checks': 0,
        'new_ZIPs': 0,
        'operation_locks_present': locks,
        'v25_metadata_audit': metadata,
        'v25_archive_path': str(DATA / 'publisher-v1.zip'),
        'v25_archive_sha256_local': receipt['archive_sha256_local'],
        'v25_archive_bytes': receipt['downloaded_archive_bytes'],
        'v25_download_CRC_safe_paths_checked': True,
        'v25_JSON_only_extracted_and_read_as_data': True,
        'v25_PNGs_decoded_or_viewed': False,
        'v25_split_intent_sha256': sha(DATA / 'split-intent-before-image-viewing.json'),
        'v25_reserved_ids_sha256': sha(DATA / 'reserved-author-S3-fold0.json'),
        'v25_final_FIT_VAL_CAL_manifest_fixed': False,
        'v25_pixel_duplicate_audit_done': False,
        'v25_features_scores_optimizer_updates': 0,
        'v25_training_protocol_fixed': False,
        'next_plan': str(ROOT / 'reports/GEARBOX_NEXT_STUDY_KO.md'),
        'next': 'Audit annotated RGB exact/orientation-near links internally and against retained5746 TEST; merge author physical/functional/acquisition groups with duplicates; exclude TRAIN groups linked to reserved S3 fold0 TEST; fix grouped FIT/VAL/CAL and one shared-encoder training protocol before features/gradients.',
        'source_patch_acceptance_is_not_whole_gearbox_health_or_pixel_GT': True,
        'new_independent_field_validation': False,
        'free_disk_bytes': shutil.disk_usage(ROOT).free,
        'awake_helper_restarted': False,
        'all_originals_failed_candidates_reports_preserved': True,
    }
    new(OUTPUTS / 'INDUSTRIAL_GEARBOX_RESUME_2026-10-15-v25-METADATA.json', result)
    new(ROOT / 'reports/industrial-gearbox-v25-preparation.json', result)
    print({k: v for k, v in result.items() if k != 'v25_metadata_audit'}, flush=True)


if __name__ == '__main__':
    main()
