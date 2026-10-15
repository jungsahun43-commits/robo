"""One fixed vari-grip data split using the already audited global groups."""
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
from safelog_ai.insplad_split import allocate
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

SOURCE=ROOT/'data/industrial-insplad-grouped-v44/grouped-all-source.json'
ARCHIVE=ROOT/'data/industrial-insplad-v43-source/publisher-supervised-fault-classification.zip'
OUT=ROOT/'data/industrial-insplad-varigrip-v53'
POLICY=ROOT/'reports/industrial-insplad-varigrip-v53-data-policy.json'
SCRIPT='scripts/prepare_insplad_varigrip.py'
CONDITIONS=('good','bird-nest','rust')
PRIOR=('data/industrial-insplad-grouped-v44/manifest.json',
       'data/industrial-insplad-glass-v47/manifest.json',
       'data/industrial-insplad-lightning-v50/manifest.json')


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not OUT.exists() and not POLICY.exists()
    assert sha(SOURCE)=='0371dc6a1b4b04195d951ece90a6152f09e4a89d2625d17b125fcd4890402119'
    assert read(ROOT/'runs/industrial-insplad-suspension-release-v52/release.json')['status']=='passed'
    p={'UTC':utc(),'fixed_before_new_split_or_content_view':True,'script_sha256':sha(ROOT/SCRIPT),
       'source_sha256':sha(SOURCE),'allocation_code_sha256':sha(ROOT/'safelog_ai/insplad_split.py'),
       'archive_sha256':'562e33ca768aac532569f9976c534640327d082402c0efbf140d93c8638fb4a8',
       'prior_development_manifests':{n:sha(ROOT/n) for n in PRIOR},
       'scope':'publisher vari-grip cropped photographs with original good/bird-nest/rust condition names; no pixelGT/internal or whole equipment health labels',
       'label_contract':'source binary label0 meansgood and1 meansbird-nest ORrust. Keep original condition separate; original multiclass numeric1=good, notdamage. No automatic relabeling.',
       'source_audit_reused':'all11525 source frame/exactRGB/eightorientationpHash2 groups, global publisherVAL6417 reservation, TEST-linkedTRAIN1045 exclusions and original filename-base policy; no new fingerprint/decode/download',
       'FIT_candidates':'vari-grip phase unselected_asset_TRAIN only, publisher train and filename-base required, original good165/bird-nest22/rust19 before prior-role exclusions',
       'TEST':'every original publisherVAL vari-grip member unchanged, good238/bird-nest23/rust20=281; report filename-base/variant counts and do not count known variants as independent natural parts',
       'allocation':'collect prior v45/v48/v51 group developmentroles. One priorrole locks thatrole, conflictingroles exclude. Remaining groups once with unchanged insplad_split.allocate labelsetstrata SHA15/15/70, no score/content-driven retries.',
       'minimum_photos':{'fit':{'good':30,'bird-nest':10,'rust':10},'validation':{'good':20,'bird-nest':2,'rust':2},'cal':{'good':20},'test':{'good':20,'bird-nest':20,'rust':20}},
       'minimum_groups':{'fit':{'good':3,'bird-nest':3,'rust':3},'validation':{'bird-nest':2,'rust':2},'test':{'good':3,'bird-nest':3,'rust':3}},
       'failure':'preserve manifest/exclusions/audit and stop before imageexport/features/gradient if anyminimum fails. Do not lower conditions, repartition, add CAL/TEST toFIT, or report a trained model.',
       'export':'only successful rows unchanged ZIPmemberbytes to numeric safe filenames, <=128MiB; perfile originalSHA/ZIPCRC checked, >=1GiBdiskreserve; originalZIP preserved',
       'QA':'after successful finalsplit, fixedfirst3FIT sorted bymember percondition only. No selectedVAL_CAL_TEST contentview, originalGT edits or humanreview claim.',
       'next_model':'NOT fixed. New singlebranch only after QA and exactcode/protocol preregistration; preserve v16+approvedv45/v48/v51 alltensor/buffer/meta; one encoder, no newcategory/source/date embedding, no classmask conversion.',
       'license':'InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1 CC BY-NC3.0 noncommercialacademic author/source/version/change attribution, paper10.1080/01431161.2023.2283900, other data licenses unchanged',
       'physical_part_capture_independence_verified':False,'new_neural_features_scores_gradient_optimizer':0}
    OUT.mkdir();new(POLICY,p);print({'policy_sha256':sha(POLICY),'new_features_gradient_optimizer':0},flush=True)


def prepare():
    p=read(POLICY)
    for path,h in ((ROOT/SCRIPT,p['script_sha256']),(SOURCE,p['source_sha256']),
                   (ROOT/'safelog_ai/insplad_split.py',p['allocation_code_sha256']),(ARCHIVE,p['archive_sha256'])):
        assert sha(path)==h,path
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for name in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])).hexdigest()==sha(ROOT/name),name
    assert not (OUT/'manifest.json').exists()
    source=read(SOURCE)['rows']
    fit=[r.copy() for r in source if r['asset']=='vari-grip' and r['phase']=='unselected_asset_TRAIN']
    test=[{**r,'phase':'test'} for r in source if r['asset']=='vari-grip' and r['publisher_role']=='val']
    assert Counter(r['condition'] for r in fit)=={'good':165,'bird-nest':22,'rust':19}
    assert Counter(r['condition'] for r in test)=={'good':238,'bird-nest':23,'rust':20}
    assert all(r['publisher_role']=='train' and r['is_filename_base'] for r in fit)
    assert all(r['condition'] in CONDITIONS and r['label']==int(r['condition']!='good') for r in fit+test)
    roles={}
    for name,h in p['prior_development_manifests'].items():
        assert sha(ROOT/name)==h,name
        for r in read(ROOT/name)['rows']:
            if r['phase'] in ('fit','validation','cal'):roles.setdefault(r['group_id'],set()).add(r['phase'])
    conflict={g for g,v in roles.items() if len(v)>1}
    excluded=[{**r,'phase':'excluded_prior_development_role_conflict'} for r in fit if r['group_id'] in conflict]
    fit=[r for r in fit if r['group_id'] not in conflict]
    allocation=allocate([r for r in fit if r['group_id'] not in roles])
    for r in fit:r['phase']=next(iter(roles[r['group_id']])) if r['group_id'] in roles else allocation[r['group_id']]
    rows=sorted(fit+test,key=lambda r:r['member'])
    for r in rows:r['category']='insplad_vari_grip_condition'
    counts={phase:{c:sum(r['condition']==c and r['phase']==phase for r in rows) for c in CONDITIONS} for phase in ('fit','validation','cal','test')}
    groups={phase:{c:len({r['group_id'] for r in rows if r['phase']==phase and r['condition']==c}) for c in CONDITIONS} for phase in counts}
    for a in counts:
        for b in counts:
            if a<b:assert {r['group_id'] for r in rows if r['phase']==a}.isdisjoint({r['group_id'] for r in rows if r['phase']==b}),(a,b)
    assert len({r['id'] for r in rows})==len(rows)
    failures=[]
    for kind,values in (('photos',p['minimum_photos']),('groups',p['minimum_groups'])):
        actual=counts if kind=='photos' else groups
        for phase,minimums in values.items():
            for condition,n in minimums.items():
                if actual[phase][condition]<n:failures.append({'kind':kind,'phase':phase,'condition':condition,'actual':actual[phase][condition],'minimum':n})
    new(OUT/'prior-role-exclusions.json',{'excluded_rows':excluded,'prior_conflicting_group_ids':sorted(conflict),'prior_rolelocked_photos':sum(r['group_id'] in roles for r in fit)})
    size=sum(r['bytes'] for r in rows)
    assert size<=128*1024**2 and shutil.disk_usage(ROOT).free-size>1024**3
    if not failures:
        (OUT/'originals').mkdir()
        with zipfile.ZipFile(ARCHIVE) as archive:
            for i,r in enumerate(rows):
                raw=archive.read(r['member']);assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
                suffix=PurePosixPath(r['member']).suffix.lower();assert suffix in ('.jpg','.jpeg','.png')
                path=OUT/'originals'/f'{i:05d}{suffix}'
                with path.open('xb') as out:out.write(raw)
                r['image']=path.relative_to(ROOT).as_posix()
    audit={'UTC':utc(),'ready':not failures,'policy_sha256':sha(POLICY),'counts':counts,'groups_each_condition':groups,
           'gate_failures':failures,'prior_role_conflict_excluded':len(excluded),'prior_locked_candidate_photos':sum(r['group_id'] in roles for r in fit),
           'original_evaluation281_preserved':len(test)==281,'evaluation_filename_base_photos':sum(r['is_filename_base'] for r in test),
           'evaluation_filename_variant_photos':sum(not r['is_filename_base'] for r in test),'exported_original_bytes':size if not failures else 0,
           'reused_global_audit_sha256':sha(SOURCE),'new_pixel_decodes_or_fingerprints':0,'new_neural_features_scores_gradient_optimizer':0,
           'new_visual_review_performed':False,'training_protocol_fixed':False,'physical_part_capture_independence_verified':False,'independent_field_validation':False}
    new(OUT/'manifest.json',{'rows':rows,'audit':audit});audit['manifest_sha256']=sha(OUT/'manifest.json')
    new(ROOT/'reports/industrial-insplad-varigrip-v53-data-audit.json',audit);print(audit,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','prepare'));args=parser.parse_args();globals()[args.stage]()
