"""Seal a review-only handoff and preserve the active research checkpoint."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.loco_relation_review import check_package, strict_json
RUN = ROOT / 'runs/industrial-relation-review-v14'
OUTPUTS = ROOT.parents[3] / 'outputs'
CHECKPOINT = ROOT / 'runs/industrial-loco-context-v13/selected/final.pt'
CHECKPOINT_SHA = 'b3e0f1cc8c487f45795b60d4bd67e2a8a03d7095d3844c6e2f42ba14875b65b0'
MODEL_ZIP = OUTPUTS / 'unified-loco-context-role4-handoff-2026-10-15-v13.zip'
MODEL_ZIP_SHA = 'dab8309e816b6fec10a9c94393bef482f1d6d481e85b94ae2f2bf9a66ff98205'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()


def new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    ui = RUN / 'review'; package = strict_json((ui/'package.json').read_text(encoding='utf-8'))
    check_package(package)
    assert sha(CHECKPOINT)==CHECKPOINT_SHA and sha(MODEL_ZIP)==MODEL_ZIP_SHA
    for filename in ('index.html','review.css','review.js'):
        assert (ROOT/'relation-review-ui'/filename).read_bytes()==(ui/filename).read_bytes()
    for row in package['cases']:
        assert sha(ui/row['image'])==row['image_sha256']
        assert sha(ui/'publisher-annotations'/Path(row['image']).with_suffix('.json').name)==row['annotation_sha256']
    shutil.copyfile(ROOT.parent/'docs/LOCO_RELATION_REVIEW_KO.md',ui/'START_HERE_KO.md')
    shutil.copyfile(ROOT/'reports/LOCO_RELATION_SOURCE_REVIEW_KO.md',ui/'SOURCE_REVIEW_KO.md')
    output = OUTPUTS/'safelog-connector-relation-review-24-2026-10-15-v14.zip'
    assert not output.exists()
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(ui.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(ui).as_posix())
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.startswith('images/')])==24
        assert len([n for n in z.namelist() if n.startswith('publisher-annotations/')])==24
        assert not any('QA_' in n or n.endswith('.pt') or '/test/' in n for n in z.namelist())
    result = {
        'status':'review_deliverable_ready_waiting_for_review_and_annotation_rights',
        'finished_utc':datetime.now(timezone.utc).isoformat(), 'repo':str(ROOT.parents[1]),
        'active_research_checkpoint':str(CHECKPOINT),'active_checkpoint_sha256':CHECKPOINT_SHA,
        'current_role4_model_ZIP':str(MODEL_ZIP),'current_role4_model_ZIP_sha256':MODEL_ZIP_SHA,
        'accepted_checkpoint_and_role4_ZIP_unchanged':True,
        'review_ZIP':str(output),'review_ZIP_sha256':sha(output),'review_ZIP_bytes':output.stat().st_size,
        'review_index':str(ui/'index.html'),'review_package_sha256':package['package_sha256'],
        'normal_publisher_TRAIN_clean_FIT_photos':24,'source_archive_normal_entries_downloaded':50,
        'all24_author_source_png_file_and_RGB_match_local_v13':True,
        'appearance_group_proxy_count':24,'TEST_VAL_CAL_annotations_extracted_or_used':0,
        'new_real_TRAIN_defect_photos':0,'new_optimizer_updates':0,
        'night_optimizer_updates_remain':8080,'human_verified_relation_labels_collected':0,
        'training_labels_created':0,'training_ready':False,
        'current_photo_metrics_recomputed':False,'current_CPU_CPP_model_tests_this_turn':0,
        'prior_v13_photo_rates':{'fpr':0.19327731092436976,'fnr':0.39896373056994816},
        'prior_v13_stage20_pass':False,'prior_additional12_stage20_count':8,
        'python_integrity_tests_passed':5,'node_validator_tests_passed':5,
        'isolated_DOM_stub_event_tests_passed':1,'simulated_export_Python_import_passed':True,
        'simulated_QA_observations_are_real_labels':False,'simulated_QA_files_in_handoff':False,
        'actual_browser_rendering_verified':False,
        'browser_file_protocol_block_preserved':True,
        'browser_block_reason':'Browser Use URL policy permits http/https and rejected file protocol; no alternate-surface workaround used.',
        'publisher_code_executed':False,'new_publisher_annotation_license_explicitly_confirmed':False,
        'source_image_license':'CC BY-NC-SA4.0, bundled MVTec attribution and full license',
        'next_gates':['24 normal FIT observations reviewed independently; fix invalid polygons or flag uncertainty',
                      'Verify explicit reuse scope of newly added author annotations',
                      'Prepare validation object/relation labels separately, then preregister actual training'],
        'physical_acquisition_groups_verified':False,'inherited_parent_near_exposure_undone':False,
        'independent_field_validated':False,'automation_id':'safelog',
        'requested_automation_cleanup':'pause after saving final Git receipt; no more duplicate hourly training',
    }
    new(RUN/'final-audit.json',result)
    new(ROOT/'reports/industrial-loco-relation-review-v14-final-audit.json',result)
    new(OUTPUTS/'INDUSTRIAL_RELATION_REVIEW_RESUME_2026-10-15-v14.json',result)
    print(result)


if __name__=='__main__':main()
