"""Replay the completed pixel graph to count usable actual TRAIN condition labels.

No new pixel decoding, distances, split, image viewing, features or training.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.insplad_pixel_groups import Groups
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

OLD=ROOT/'data/industrial-insplad-grouped-v44/grouped-all-source.json'
META=ROOT/'data/industrial-insplad-unsupervised-v56-metadata/all-rows.json'
HELD=ROOT/'data/industrial-insplad-grouped-v44/retained-TEST-fingerprints.json'
V57=ROOT/'data/industrial-insplad-lower-shackle-v57'
EDGE=V57/'pixel-joining-edges.jsonl'
RUN=ROOT/'runs/industrial-insplad-remaining-labels-v59'
POLICY=ROOT/'reports/industrial-insplad-remaining-labels-v59-policy.json'
SCRIPT='scripts/audit_insplad_remaining_labels.py'
PRIOR=('data/industrial-insplad-grouped-v44/manifest.json','data/industrial-insplad-glass-v47/manifest.json',
       'data/industrial-insplad-lightning-v50/manifest.json','data/industrial-insplad-varigrip-v53/manifest.json')


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not RUN.exists() and not POLICY.exists()
    assert read(ROOT/'runs/industrial-insplad-lower-flow-v58/verification.json')['status']=='passed'
    assert not read(ROOT/'runs/industrial-insplad-lower-flow-v58/selection.json')['source_candidate_eligible_pending_package_checks']
    paths=[OLD,META,HELD,EDGE,V57/'manifest.json',ROOT/'reports/industrial-insplad-lower-shackle-v57-data-audit.json']+[ROOT/n for n in PRIOR]
    p={'UTC':utc(),'fixed_before_new_coverage_computation':True,'script_sha256':sha(ROOT/SCRIPT),
       'group_code_sha256':sha(ROOT/'safelog_ai/insplad_pixel_groups.py'),
       'files_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in paths},
       'graph_replay':'same ordered27620metadata+11525oldsource+8995oldTEST; seed oldgroup/v56group/oldsource links then94saved pixeljoiningedges. Reuse connectivity, no decode or pHash distance rerun.',
       'checks':'canonical labels must match all783 completedv57manifest rows; sourceindices/edge ranges fixed; compare held allnew6481TEST/old6417publisherVAL/old8995TEST',
       'actual_label_candidates':'all six oldsupervised assets, original filename-base publisherTRAIN only. Exclude any group connected to current heldout sets. Keep original good/rust/birdnest/missingcap labels; not automatically relabel.',
       'normal_support_candidates':'v56rawnovel_unassigned_TRAIN normal good rows only, exclude held connections. Counts are potential support, not fixedFIT or verified independent photos.',
       'prior_roles':'old approvedandfailed v45/v48/v51/v54 developmentrole single retained; conflictinggroups report excluded. No existing weights/splits changed retroactively.',
       'report':'counts perasset/condition/photo/group/developmentrole, datecoverage and existingnewTEST-linked exclusions; no modelperformance or finalFIT assignment',
       'minimum_before_future_split':'FITnormal30+VALnormal20+CALnormal20 andFITactualdefect10/VALactualdefect2, TESTeachcondition20/groups3; if pool insufficient no attempt to move TEST or weaken counts',
       'new_pixels_features_scores_gradient_optimizer':0,'new_content_viewing':0,'training_protocol_fixed':False,
       'license':'InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1 CC BY-NC3.0 noncommercialacademic withauthor/source/version/change attribution',
       'independent_field_validation':False}
    RUN.mkdir();new(POLICY,p);print({'policy_sha256':sha(POLICY)},flush=True)


def run():
    p=read(POLICY);assert sha(ROOT/SCRIPT)==p['script_sha256'] and sha(ROOT/'safelog_ai/insplad_pixel_groups.py')==p['group_code_sha256']
    for n,h in p['files_sha256'].items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    assert not (RUN/'audit.json').exists()
    rows=read(META)['rows'];old=read(OLD)['rows'];held=read(HELD)['rows'];n=len(rows);h=n+len(old)
    assert (n,len(old),len(held))==(27620,11525,8995)
    allrows=rows+old+held;g=Groups(len(allrows));oldgroups={};newgroups={}
    for j,r in enumerate(old):
        key=r['group_id']
        if key in oldgroups:g.union(n+j,oldgroups[key])
        else:oldgroups[key]=n+j
    for i,r in enumerate(rows):
        key=r['group_proxy']
        if key in newgroups:g.union(i,newgroups[key])
        else:newgroups[key]=i
        for key in r['old_source_group_links']:g.union(i,oldgroups[key])
    edges=[json.loads(v) for v in EDGE.read_text(encoding='utf-8').splitlines()];assert len(edges)==94
    for edge in edges:
        i,j,kind,*distance=edge;assert 0<=i<n and 0<=j<len(allrows) and kind in ('RGB_exact','pHash')
        g.union(i,j)
    keys=['unsupervised:'+r['member'] for r in rows]+['supervised:'+r['member'] for r in old]+['held:'+r['image'] for r in held]
    canonical={}
    for i,key in enumerate(keys):
        root=g.find(i);canonical[root]=min(canonical.get(root,key),key)
    bymember={r['member']:i for i,r in enumerate(rows)}
    reference=read(V57/'manifest.json')['rows']
    for r in reference:assert r['pixel_group_id']==canonical[g.find(bymember[r['member']])]
    heldroots={g.find(i) for i,r in enumerate(rows) if r['publisher_role']=='test'}
    heldroots.update(g.find(n+j) for j,r in enumerate(old) if r['publisher_role']=='val')
    heldroots.update(g.find(i) for i in range(h,len(allrows)))
    roles=defaultdict(set)
    for name in PRIOR:
        for r in read(ROOT/name)['rows']:
            if r['phase'] in ('fit','validation','cal'):roles[g.find(oldgroups[r['group_id']])].add(r['phase'])
    actual=[];excluded=[];normal=[]
    for j,r in enumerate(old):
        if r['publisher_role']!='train' or not r['is_filename_base']:continue
        root=g.find(n+j);reason='heldout_connected' if root in heldroots else 'prior_roles_conflict' if len(roles[root])>1 else None
        value={**r,'current_pixel_group_id':canonical[root],'prior_role':next(iter(roles[root])) if len(roles[root])==1 else None}
        if reason:excluded.append({**value,'new_exclusion':reason})
        else:actual.append(value)
    for i,r in enumerate(rows):
        if not r['rawnovel_unassigned_TRAIN']:continue
        root=g.find(i)
        if root in heldroots or len(roles[root])>1:continue
        assert r['publisher_role']=='train' and r['condition']=='good'
        normal.append({**r,'current_pixel_group_id':canonical[root],'prior_role':next(iter(roles[root])) if len(roles[root])==1 else None})
    summary={}
    for asset in sorted({r['asset'] for r in old+rows}):
        a=[r for r in actual if r['asset']==asset];b=[r for r in normal if r['asset']==asset]
        summary[asset]={'retained_actual_supervised_original_TRAIN':dict(Counter(r['condition'] for r in a)),
            'supervised_groups_per_condition':{c:len({r['current_pixel_group_id'] for r in a if r['condition']==c}) for c in sorted({r['condition'] for r in a})},
            'supervised_prior_roles':dict(Counter(str(r['prior_role']) for r in a)),
            'retained_rawnovel_normal_support_candidates':len(b),'normal_support_groups':len({r['current_pixel_group_id'] for r in b}),
            'current_held_or_role_conflict_excluded_original_supervised':dict(Counter(r['condition'] for r in excluded if r['asset']==asset))}
    audit={'UTC':utc(),'policy_sha256':sha(POLICY),'v57_manifest783_canonical_groups_exact':True,'replayed_pixel_edges':len(edges),
           'by_asset':summary,'retained_actual_supervised_photos':len(actual),'normal_support_candidates_not_readyFIT':len(normal),
           'new_pixel_decode_pHash_comparison_features_scores_gradient_optimizer':0,'new_visual_review':0,
           'old_models_and_splits_unchanged':True,'new_data_split_or_training_protocol':False,'physical_identity_verified':False,'independent_field_validation':False}
    new(RUN/'retained-supervised-originals.json',{'rows':actual});new(RUN/'excluded-supervised-originals.json',{'rows':excluded})
    new(RUN/'normal-support-candidates.json',{'rows':normal});new(RUN/'audit.json',audit)
    new(ROOT/'reports/industrial-insplad-remaining-labels-v59-audit.json',audit);print(audit,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args();globals()[args.stage]()
