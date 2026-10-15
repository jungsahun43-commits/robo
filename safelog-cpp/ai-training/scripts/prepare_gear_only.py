"""Metadata-led narrower gear task, not relabeling broad gearbox failures as success."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.gear_only_groups import allocate_gear_groups
from scripts.prepare_gearbox_stratified import MANIFEST as SOURCE
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

RUN=ROOT/'runs/industrial-gear-only-v28'
DATA=ROOT/'data/industrial-gear-only-v28'
POLICY=ROOT/'reports/industrial-gear-only-v28-data-policy.json'
MANIFEST=DATA/'manifest.json'


def freeze():
    assert not RUN.exists() and not DATA.exists() and not POLICY.exists()
    coverage=ROOT/'reports/industrial-gearbox-category-coverage-v26-v27.json'
    assert read(coverage)['remaining_nonTEST_spline_OK_total']==3
    RUN.mkdir();DATA.mkdir()
    p={'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_new_partition_or_model_features_scores_optimizer':True,
        'source_manifest_sha256':sha(SOURCE),'coverage_report_sha256':sha(coverage),
        'FIT_example_reservation_sha256':sha(ROOT/'runs/industrial-gearbox-v26/FIT-example-reservation.json'),
        'source_sha256':{n:sha(ROOT/n) for n in ('safelog_ai/gear_only_groups.py','scripts/prepare_gear_only.py','tests/test_gear_only_groups.py')},
        'reason':'original eligible TRAIN has only3 splineOK and no ring/shaftTEST; full-source label balance did not ensure supported-category coverage. Select gear task by TRAIN label/group availability, not a new full-source pass claim.',
        'task':'ONLY publisher128x128 gear visible-surface acceptance OK/nOK. Exclude other functionalcategories from this separately declared task; preserve full-source v26/v27 failures unchanged.',
        'TEST':'fixed original S3fold0 gear subset2179, no moving TEST/near or excluded1317TRAIN back into development; keep original fullTEST3109 elsewhere',
        'partition':'same original TRAIN eligible gear-only groups; force previously AI-viewed FIT-example groups toFIT; rest decreasing size then SHA256(gear28:+group), greedy normalized binarylabel .70/.15/.15 objective with groupcounts ceil15percentVAL/CAL/restFIT. No retry/seed sweep.',
        'data_gate':'FIT labels>=100/groups>=8, VAL/CAL labels>=20/groups>=3, TEST labels>=20/groups>=3; all groups disjoint, viewed examples remain FIT',
        'development_reuse':'previous FIT/VAL/CAL union regrouped from original TRAIN; raw image features may be reused byID only after fixing future learning protocol. New training must start from unchanged acceptedv16 and a fresh head, not old failed fitted heads.',
        'training_protocol_fixed':False,'new_features_gradients_optimizer_updates':0,'new_unit_tests_passed':2}
    new(POLICY,p)


def prepare():
    p=read(POLICY);assert sha(SOURCE)==p['source_manifest_sha256'] and not MANIFEST.exists()
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h
    rows=read(SOURCE)['rows'];examples=read(ROOT/'runs/industrial-gearbox-v26/FIT-example-reservation.json')
    viewed={r['group_id'] for r in rows if r['id'] in examples['ids'] and r['functional_component_category']=='gear'}
    eligible=[r for r in rows if r['phase'] in ('fit','validation','cal') and r['functional_component_category']=='gear']
    split=allocate_gear_groups(eligible,viewed)
    for r in rows:
        r['source_v26_phase']=r['phase']
        if r['functional_component_category']!='gear':r['phase']='outside_predeclared_gear_task'
        elif r['phase'] in ('fit','validation','cal'):r['phase']=split[r['group_id']]
    phases=('fit','validation','cal','test')
    counts={p:dict(Counter(str(r['label']) for r in rows if r['phase']==p)) for p in phases}
    groups={p:sorted({r['group_id'] for r in rows if r['phase']==p}) for p in phases}
    assert all(set(groups[a]).isdisjoint(groups[b]) for i,a in enumerate(phases) for b in phases[i+1:])
    assert viewed<=set(groups['fit']) and sum(counts['test'].values())==2179
    ready=(min(counts['fit'].get(str(y),0) for y in (0,1))>=100 and len(groups['fit'])>=8 and all(min(counts[k].get(str(y),0) for y in (0,1))>=20 and len(groups[k])>=3 for k in phases[1:]))
    audit={'UTC':datetime.now(timezone.utc).isoformat(),'ready':ready,'policy_sha256':sha(POLICY),
        'source_manifest_sha256':sha(SOURCE),'counts':counts,'groups_by_phase':groups,'previously_viewed_groups_preserved_FIT':sorted(viewed),
        'original_full_source_TEST_preserved_elsewhere':3109,'new_task_TEST_gear_subset':2179,
        'task_subset_predeclared_not_full_source_improvement':True,'new_RGB_features_model_scores_optimizer_updates':0,
        'same_publisher_TRAIN_development_repartitioned':True,'new_training_protocol_fixed':False,'new_unit_tests_passed':2,
        'independent_field_validation':False,'source_patch_label_not_whole_gearbox_health_or_verified_localization':True}
    new(MANIFEST,{'schema':'safelog-gear-only-v28','rows':rows,'audit':audit});new(RUN/'data-ready.json',{'ready':ready,'manifest_sha256':sha(MANIFEST),'audit':audit})
    new(ROOT/'reports/industrial-gear-only-v28-data-audit.json',audit);print(audit,flush=True)
    assert ready,'Predeclared data gate failed; preserve state and no learning'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','prepare'));args=parser.parse_args();globals()[args.stage]()
