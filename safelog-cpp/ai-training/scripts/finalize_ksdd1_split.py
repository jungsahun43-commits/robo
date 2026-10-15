"""Before any model scores: enforce five whole validation and calibration groups."""
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_ksdd1_masks import DATA
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    original=DATA/'manifest.json';final=DATA/'manifest-r1.json';assert not final.exists()
    parent=read(original);assert parent['audit']['status']=='insufficient_grouped_source_do_not_train'
    # No feature/model/CAL/TEST scoring is allowed before the sample-count repair.
    assert not (ROOT/'runs/industrial-ksdd1-v18').exists()
    rows=deepcopy(parent['rows']);groups=defaultdict(list)
    for r in rows:
        if r['phase'] in ('fit','validation','cal'):groups[r['appearance_object_group']].append(r)
    assert len(groups)==30 and all(any(r['label'] for r in g) and any(not r['label'] for r in g) for g in groups.values())
    ordered=sorted(groups,key=lambda k:(hashlib.sha256(('ksdd1-v18-minimum-groups:'+k).encode()).hexdigest(),k))
    allocation={'validation':ordered[:5],'cal':ordered[5:10],'fit':ordered[10:]}
    for phase,keys in allocation.items():
        for key in keys:
            for r in groups[key]:r['initial_phase_before_count_repair']=r['phase'];r['phase']=phase
    assert [(r['id'],r['phase']) for r in rows if r['phase']=='test']==[(r['id'],r['phase']) for r in parent['rows'] if r['phase']=='test']
    counts={p:dict(Counter('defect' if r['label'] else 'normal' for r in rows if r['phase']==p)) for p in sorted({r['phase'] for r in rows})}
    assert all(min(counts[p].get('normal',0),counts[p].get('defect',0))>=5 for p in ('fit','validation','cal','test'))
    audit=deepcopy(parent['audit']);audit.update(UTC=datetime.now(timezone.utc).isoformat(),status='ready_for_small_sample_research_protocol',
        counts=counts,parent_manifest_sha256=sha(original),parent_audit_sha256=sha(ROOT/'reports/industrial-ksdd1-v18-data-audit.json'),
        split_repair_reason='Initial SHA70/15/15 gives only one validation defect. Before all model scores, reserve five complete groups for VAL and five for CAL by a single fixed SHA order; rest FIT. Do not reroll seeds or alter publisher TEST.',
        FIT_VAL_CAL_rule='NonTEST active groups sortedSHA(ksdd1-v18-minimum-groups:+group); first5VAL next5CAL restFIT',
        fixed_source_group_allocation=allocation,source_holdout_TEST_unchanged=True,source_TEST_defect_count=counts['test']['defect'],
        insufficient_TEST_defect_count_for_20percent_PASS=counts['test']['defect']<20,
        source_image_content_viewed_by_agent=False,new_model_scores_or_optimizer_updates=0,current_eligible_for_app_category=False)
    new(final,{'schema':'safelog-ksdd1-object-groups-v18-r1','rows':rows,'audit':audit})
    new(ROOT/'reports/industrial-ksdd1-v18-data-audit-r1.json',audit)
    print({'final_manifest':str(final),'counts':counts,'status':audit['status'],'TEST_defects_less_than20':counts['test']['defect']<20},flush=True)


if __name__=='__main__':main()
