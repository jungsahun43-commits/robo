"""Reuse completed global v44 fingerprint groups, reserve publisher VAL unchanged."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.fetch_equipment_expansion import sha
from safelog_ai.insplad_split import allocate

SOURCE=ROOT/'data/industrial-insplad-grouped-v44/grouped-all-source.json'
ARCHIVE=ROOT/'data/industrial-insplad-v43-source/publisher-supervised-fault-classification.zip'
OUT=ROOT/'data/industrial-insplad-lightning-v50'
POLICY=ROOT/'reports/industrial-insplad-lightning-v50-data-policy.json'
SCRIPT='scripts/prepare_insplad_lightning.py'


def freeze():
    assert not OUT.exists() and not POLICY.exists()
    assert sha(SOURCE)=='0371dc6a1b4b04195d951ece90a6152f09e4a89d2625d17b125fcd4890402119'
    p={'UTC':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(SOURCE),'script_sha256':sha(ROOT/SCRIPT),
       'allocation_code_sha256':sha(ROOT/'safelog_ai/insplad_split.py'),
       'archive_sha256':'562e33ca768aac532569f9976c534640327d082402c0efbf140d93c8638fb4a8',
       'preserved_development_manifests':{n:sha(ROOT/n) for n in ('data/industrial-insplad-grouped-v44/manifest.json','data/industrial-insplad-glass-v47/manifest.json')},
       'scope':'publisher cropped lightning-rod suspension good vsrust; not electrical health/whole scene/defect pixelmask',
       'reuse':'fixed all11525 source frame/exactRGB/8orientationpHash2 groups and retained8995TEST audit fromv44; no refingerprinting/redecode/redownload/relabeling',
       'reserved':'all251original publisherVAL lightning photographs evaluation only; no FIT/internalVAL/CAL use',
       'FIT_candidates':'original filename-base lightning TRAIN with phase unselected_asset_TRAIN only; all globalTEST exclusions/variants unchanged',
       'split':'lock groups with exactlyone prior v45/v48 developmentrole to thatrole, exclude groups with conflicting priorroles; allocate remaininggroups once by insplad_split labelsetstrata SHA15/15/70. Original publisherTEST remains reserved, no relabel/score-driven retries.',
       'minimums':{'fit':{'0':30,'1':20},'validation':{'0':20,'1':5},'cal':{'0':20},'test':{'0':20,'1':20}},'groups_per_required_label':3,
       'disk':'export unchanged selected memberbytes only <=128MiB and>=1GiBreserve, numeric paths; sourceZIPpreserved',
       'selection_reason':'original label support337normal/31rust and originalVAL231/20; date-label confounding remains, publisher evaluation has no both-label dates; metadata only, no new photos viewed/model scores',
       'date_caution':'bothlabeldates do not prove same physicalpart or site/flight independence; report actualdate support, not fieldvalidation',
       'QA':'after finalsplit first3FITperlabel only, no TESTcontentviewing or GTmodification; no agent QA performed by this dataexport script',
       'failure':'preserve and stop before features/optimizer; no relaxed minimum or repartition',
       'training_protocol_fixed':False,'new_model_features_scores_optimizer':0,
       'next_model_constraint':'preserve v16 allstate plus approved v45corrosion andv48glass heads exactly; one sharedencoder, separate new suspension rust classification branch only, fixedtrainingprotocol before features/gradient; no discarded head reuse',
       'license':'InsPLAD MendeleyV1 CC BY-NC3.0 DOI10.17632/5n3fjgvfyz.1 paper10.1080/01431161.2023.2283900, noncommercial academic citation/source/version/change, inherited licenses unchanged'}
    OUT.mkdir();new(POLICY,p);print({'policy_sha256':sha(POLICY),'new_decodes_features_scores_optimizer':0},flush=True)


def prepare():
    p=read(POLICY);assert sha(ROOT/SCRIPT)==p['script_sha256'] and sha(SOURCE)==p['source_sha256']
    assert sha(ROOT/'safelog_ai/insplad_split.py')==p['allocation_code_sha256'] and sha(ARCHIVE)==p['archive_sha256']
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    assert not (OUT/'manifest.json').exists()
    source=read(SOURCE)['rows'];fit=[r.copy() for r in source if r['asset']=='lightning-rod-suspension' and r['phase']=='unselected_asset_TRAIN']
    test=[{**r,'phase':'test'} for r in source if r['asset']=='lightning-rod-suspension' and r['publisher_role']=='val']
    assert len(fit)==368 and len(test)==251
    priorroles={}
    for name,h in p['preserved_development_manifests'].items():
        assert sha(ROOT/name)==h
        for row in read(ROOT/name)['rows']:
            if row['phase'] in ('fit','validation','cal'):
                priorroles.setdefault(row['group_id'],set()).add(row['phase'])
    conflicting={g for g,v in priorroles.items() if len(v)>1}
    excluded=[{**r,'phase':'excluded_prior_development_role_conflict'} for r in fit if r['group_id'] in conflicting]
    fit=[r for r in fit if r['group_id'] not in conflicting]
    unlocked=[r for r in fit if r['group_id'] not in priorroles]
    allocation=allocate(unlocked)
    for r in fit:r['phase']=next(iter(priorroles[r['group_id']])) if r['group_id'] in priorroles else allocation[r['group_id']]
    new(OUT/'prior-role-exclusions.json',{'excluded_rows':excluded,'prior_groups_with_conflicting_roles':sorted(conflicting),
        'caution':'prior head-specific development roles can differ; sharedencoder fixed, no old data/weights edited, no TRAIN/TEST conflict allowed'})
    rows=sorted(fit+test,key=lambda r:r['member'])
    counts={phase:dict(Counter(str(r['label']) for r in rows if r['phase']==phase)) for phase in p['minimums']}
    groups={phase:{str(y):len({r['group_id'] for r in rows if r['phase']==phase and r['label']==y}) for y in (0,1)} for phase in counts}
    failures=[]
    for phase,mins in p['minimums'].items():
        for label,n in mins.items():
            if counts[phase].get(label,0)<n or groups[phase][label]<3:failures.append(phase+'/'+label)
    for a in counts:
        for b in counts:
            if a<b:assert {r['group_id'] for r in rows if r['phase']==a}.isdisjoint({r['group_id'] for r in rows if r['phase']==b})
    size=sum(r['bytes'] for r in rows);assert size<=128*1024**2 and shutil.disk_usage(ROOT).free-size>1024**3
    if not failures:
        (OUT/'originals').mkdir()
        with zipfile.ZipFile(ARCHIVE) as archive:
            for i,r in enumerate(rows):
                raw=archive.read(r['member']);assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
                path=OUT/'originals'/f'{i:05d}{PurePosixPath(r["member"]).suffix.lower()}'
                with path.open('xb') as out:out.write(raw)
                r['image']=path.relative_to(ROOT).as_posix();r['category']='insplad_lightning_suspension_rust'
    result={'UTC':datetime.now(timezone.utc).isoformat(),'ready':not failures,'counts':counts,'groups_each_label':groups,
            'prior_role_conflict_excluded':len(excluded),'prior_locked_candidate_photos':sum(r['group_id'] in priorroles for r in fit),'gate_failures':failures,'original_evaluation251_preserved':len(test)==251,'exported_original_bytes':size if not failures else 0,
            'policy_sha256':sha(POLICY),'reused_global_audit_sha256':sha(SOURCE),'new_pixel_decodes_or_fingerprints':0,
            'new_neural_features_scores_gradient_optimizer':0,'new_visual_review_performed':False,'training_protocol_fixed':False,
            'physical_part_acquisition_independence_unverified':True,'independent_field_validation':False}
    new(OUT/'manifest.json',{'rows':rows,'audit':result});result['manifest_sha256']=sha(OUT/'manifest.json')
    new(ROOT/'reports/industrial-insplad-lightning-v50-data-audit.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','prepare'));args=parser.parse_args();globals()[args.stage]()
