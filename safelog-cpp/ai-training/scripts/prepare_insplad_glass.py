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
OUT=ROOT/'data/industrial-insplad-glass-v47'
POLICY=ROOT/'reports/industrial-insplad-glass-v47-data-policy.json'
SCRIPT='scripts/prepare_insplad_glass.py'


def freeze():
    assert not OUT.exists() and not POLICY.exists()
    assert sha(SOURCE)=='0371dc6a1b4b04195d951ece90a6152f09e4a89d2625d17b125fcd4890402119'
    p={'UTC':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(SOURCE),'script_sha256':sha(ROOT/SCRIPT),
       'allocation_code_sha256':sha(ROOT/'safelog_ai/insplad_split.py'),
       'archive_sha256':'562e33ca768aac532569f9976c534640327d082402c0efbf140d93c8638fb4a8',
       'scope':'publisher cropped glass insulator good vsmissing-cap; not electrical health/whole scene/defect pixelmask',
       'reuse':'fixed all11525 source frame/exactRGB/8orientationpHash2 groups and retained8995TEST audit fromv44; no refingerprinting/redecode/redownload/relabeling',
       'reserved':'all59original publisherVAL glass photographs evaluation only; no FIT/internalVAL/CAL use',
       'FIT_candidates':'original filename-base glass TRAIN with phase unselected_asset_TRAIN only; all globalTEST exclusions/variants unchanged',
       'split':'same insplad_split.allocate rule once: labelsetstrata, SHA(insplad44:+groupminmember),ceil15percent VAL,nextceil15percentCAL,restFIT, strata<3allFIT; no score-based retries',
       'minimums':{'fit':{'0':30,'1':20},'validation':{'0':20,'1':5},'cal':{'0':20},'test':{'0':20,'1':20}},'groups_per_required_label':3,
       'disk':'export unchanged selected memberbytes only <=128MiB and>=1GiBreserve, numeric paths; sourceZIPpreserved',
       'selection_reason':'original label support547normal/56missingcap and originalVAL29/30, TRAIN10/VAL5 dates contain bothlabels vs upper-shackle3/1; metadata only, no new photos viewed/model scores',
       'date_caution':'bothlabeldates do not prove same physicalpart or site/flight independence; report actualdate support, not fieldvalidation',
       'QA':'after finalsplit first3FITperlabel only, no TESTcontentviewing or GTmodification; no agent QA performed by this dataexport script',
       'failure':'preserve and stop before features/optimizer; no relaxed minimum or repartition',
       'training_protocol_fixed':False,'new_model_features_scores_optimizer':0,
       'next_model_constraint':'preserve v16 allstate plus approved v45corrosionhead exactly; one sharedencoder, separate new glass classification branch only, fixedtrainingprotocol before features/gradient; no discarded head reuse',
       'license':'InsPLAD MendeleyV1 CC BY-NC3.0 DOI10.17632/5n3fjgvfyz.1 paper10.1080/01431161.2023.2283900, noncommercial academic citation/source/version/change, inherited licenses unchanged'}
    OUT.mkdir();new(POLICY,p);print({'policy_sha256':sha(POLICY),'new_decodes_features_scores_optimizer':0},flush=True)


def prepare():
    p=read(POLICY);assert sha(ROOT/SCRIPT)==p['script_sha256'] and sha(SOURCE)==p['source_sha256']
    assert sha(ROOT/'safelog_ai/insplad_split.py')==p['allocation_code_sha256'] and sha(ARCHIVE)==p['archive_sha256']
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    assert not (OUT/'manifest.json').exists()
    source=read(SOURCE)['rows'];fit=[r.copy() for r in source if r['asset']=='glass-insulator' and r['phase']=='unselected_asset_TRAIN']
    test=[{**r,'phase':'test'} for r in source if r['asset']=='glass-insulator' and r['publisher_role']=='val']
    assert len(fit)==603 and len(test)==59
    allocation=allocate(fit)
    for r in fit:r['phase']=allocation[r['group_id']]
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
                r['image']=path.relative_to(ROOT).as_posix();r['category']='insplad_glass_missing_cap'
    result={'UTC':datetime.now(timezone.utc).isoformat(),'ready':not failures,'counts':counts,'groups_each_label':groups,
            'gate_failures':failures,'original_evaluation59_preserved':len(test)==59,'exported_original_bytes':size if not failures else 0,
            'policy_sha256':sha(POLICY),'reused_global_audit_sha256':sha(SOURCE),'new_pixel_decodes_or_fingerprints':0,
            'new_neural_features_scores_gradient_optimizer':0,'new_visual_review_performed':False,'training_protocol_fixed':False,
            'physical_part_acquisition_independence_unverified':True,'independent_field_validation':False}
    new(OUT/'manifest.json',{'rows':rows,'audit':result});result['manifest_sha256']=sha(OUT/'manifest.json')
    new(ROOT/'reports/industrial-insplad-glass-v47-data-audit.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','prepare'));args=parser.parse_args();globals()[args.stage]()
