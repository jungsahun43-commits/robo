"""Reuse the audited disjoint normal groups, then split FIT/VAL before training."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.neighbor_surface import CATEGORIES
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

DATA=ROOT/'data/industrial-neighbor-surfaces-v15'
RUN=ROOT/'runs/industrial-neighbor-surfaces-v15'
PRIOR=ROOT/'data/industrial-foreground-v12/manifest.json'
REPORT=ROOT/'reports/industrial-neighbor-surfaces-v15-data-audit.json'


def main():
    assert not DATA.exists() and not RUN.exists()
    previous=read(PRIOR); rows=previous['rows']
    for category in CATEGORIES:
        local=[r for r in rows if r['category']==category]
        for r in local:
            assert sha(ROOT/r['image'])==r['image_sha256'],r['id']
            r['v12_phase']=r['phase']
        normal=[r for r in local if r['phase']=='fit']
        ordered=sorted(normal,key=lambda r:hashlib.sha256(('neighbor15:'+r['appearance_group_proxy']).encode()).hexdigest())
        for r in ordered[:round(.2*len(ordered))]:r['phase']='validation'
        phases={p:{r['appearance_group_proxy'] for r in local if r['phase']==p} for p in ('fit','validation','cal','test')}
        assert all(not phases[a]&phases[b] for i,a in enumerate(phases) for b in list(phases)[i+1:])
        for r in local:
            if r['phase'] in ('fit','validation','cal'):
                assert r['label']==0 and r['publisher_split']=='train' and not r['masks']
    counts={c:dict(Counter(r['phase'] for r in rows if r['category']==c)) for c in CATEGORIES}
    assert [(counts[c]['fit'],counts[c]['validation'],counts[c]['cal']) for c in CATEGORIES]==[(70,18,30),(61,15,23)]
    audit={'utc':datetime.now(timezone.utc).isoformat(),'source_manifest_sha256':sha(PRIOR),'counts':counts,
        'inherited_group_audit':previous['audit'],'source_image_bytes_rechecked':len(rows),
        'group_split':'First round20percent by SHA256(neighbor15:appearance_group_proxy) reserved VAL; remaining normal FIT',
        'original_v12_foreground_masks_used':False,'known_TEST_features_or_scores_newly_computed':False,
        'real_defect_or_object_relation_labels_for_training':0,'physical_acquisition_groups_verified':False,
        'inherited_parent_near_training_exposure_undone':False,'independent_field_validated':False}
    DATA.mkdir();RUN.mkdir();new(DATA/'manifest.json',{'rows':rows,'audit':audit});new(REPORT,audit)
    print({'counts':counts,'new_optimizer_updates':0},flush=True)


if __name__=='__main__':main()
