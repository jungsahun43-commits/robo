"""Inspect author JSON as data only; reserve part holdout before any image viewing/model."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path,PurePosixPath
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_gearbox_candidate import DATA
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def shape(value,depth=0):
    if isinstance(value,dict):
        return {'type':'dict','length':len(value),'keys':list(value)[:15],
                'first':{k:shape(v,depth+1) for k,v in list(value.items())[:2]} if depth<3 else None}
    if isinstance(value,list):
        return {'type':'list','length':len(value),'first':[shape(v,depth+1) for v in value[:1]] if depth<3 else None}
    return value if isinstance(value,(str,int,float,bool,type(None))) and len(str(value))<300 else {'type':type(value).__name__}


def main():
    receipt=read(DATA/'download-receipt.json');assert sha(DATA/'publisher-v1.zip')==receipt['archive_sha256_local']
    assert not (DATA/'metadata-inspection.json').exists()
    new(DATA/'split-intent-before-image-viewing.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'scope':'data reservation intent, not a finalized training protocol',
        'official_file':'gbc_annotations_v20_functional_component_id.json',
        'holdout':'official functional-component-id split first declared CV fold (S3), never random image split for the primary result',
        'remaining_groups':'verify author group/part identity fields and TRAIN role; predeclare disjoint FIT/VAL/CAL from remaining groups before model scores',
        'current_model_features_scores_optimizer_updates':0,'image_content_viewed':False,
        'current_existing_TEST': 'no current TEST/near groups may enter FIT',
        'unannotated_or_discarded_images':'exclude, never label normal','masks_or_boxes':'not asserted; patch class labels are not location masks or whole gearbox functional health'})
    docs=DATA/'publisher-annotations';docs.mkdir(exist_ok=True)
    result={}
    with zipfile.ZipFile(DATA/'publisher-v1.zip') as archive:
        for info in archive.infolist():
            name=PurePosixPath(info.filename)
            if name.suffix.lower()!='.json':continue
            assert len(name.parts)==1 and info.file_size<40*1024**2
            raw=archive.read(info);obj=json.loads(raw)
            with (docs/name.name).open('xb') as out:out.write(raw)
            result[name.name]={'sha256':sha(docs/name.name),'bytes':info.file_size,'shape':shape(obj)}
            print({'file':name.name,**result[name.name]},flush=True)
    audit={'UTC':datetime.now(timezone.utc).isoformat(),'status':'author_JSON_inspected_split_still_pending',
        'archive_sha256_local':receipt['archive_sha256_local'],'author_json_files':result,
        'split_intent_sha256':sha(DATA/'split-intent-before-image-viewing.json'),
        'archive_or_author_code_executed':False,'PNG_image_content_viewed':False,
        'raw_image_entries_in_archive':receipt['suffix_counts']['.png'],
        'author_description_labeled_patches':15588,'image_count_discrepancy_not_resolved':True,
        'final_phase_manifest_fixed':False,'current_TEST_cross_duplicate_audit_done':False,
        'new_features_model_scores_optimizer_updates':0}
    new(DATA/'metadata-inspection.json',audit)


if __name__=='__main__':main()
