"""Reserve all new TEST first; hash encoded bytes without decoding pixels."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.fetch_equipment_expansion import sha

SOURCE=ROOT/'data/industrial-insplad-unsupervised-v55-source'
ARCHIVE=SOURCE/'publisher-unsupervised-anomaly-detection.zip'
GROUPED=ROOT/'data/industrial-insplad-grouped-v44/grouped-all-source.json'
OUT=ROOT/'data/industrial-insplad-unsupervised-v56-metadata'
POLICY=ROOT/'reports/industrial-insplad-unsupervised-v56-metadata-policy.json'
SCRIPT='scripts/audit_insplad_unsupervised_metadata.py'
PRIOR={'v45_approved':'data/industrial-insplad-grouped-v44/manifest.json',
       'v48_approved':'data/industrial-insplad-glass-v47/manifest.json',
       'v51_approved':'data/industrial-insplad-lightning-v50/manifest.json',
       'v54_rejected':'data/industrial-insplad-varigrip-v53/manifest.json'}


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not OUT.exists() and not POLICY.exists()
    download=read(SOURCE/'download.json')
    assert download['inner_zip_local_SHA256']=='baab1d227b408ec6bbe5831cbcae5ea049a9b5132a74bf36f4e3d6051b0cb7e3'
    assert read(ROOT/'reports/industrial-insplad-unsupervised-v55-archive-audit.json')['all_member_CRCS_verified']
    oldpolicy=ROOT/'reports/industrial-insplad-v44-data-policy.json'
    held=read(oldpolicy)['old_TEST_manifests']
    OUT.mkdir()
    new(POLICY,{'UTC':utc(),'fixed_before_new_metadata_reservation_and_encoded_byte_hashes':True,
        'script_sha256':sha(ROOT/SCRIPT),'archive_sha256':download['inner_zip_local_SHA256'],
        'members_sha256':sha(SOURCE/'unsupervised-archive-members.json'),'old_source_groups_sha256':sha(GROUPED),
        'old_TEST_manifests':held,'prior_development_manifests':{name:{'path':path,'sha256':sha(ROOT/path)} for name,path in PRIOR.items()},
        'scope':'metadata/encodedbyteSHA/frameproxy only. No JPEG/PNG pixeldecode, visualview, RGB/phash recompute, neuralfeature/score/gradient oroptimizer.',
        'role_contract':'strict fourpathparts asset/(train|test)/condition/photo, TRAINcondition mustgood. OriginalTEST remainsreserved regardless priorroles. Unexpected path/date/role stops rather than inventsGT.',
        'reservation':'save allnew publisherTEST memberlist before reading anyencodedphoto bytes. Connect newrows by same date+DJIframe or rawSHA, and to oldsource connectedgroups through oldrawSHA/frame. Groups connectedto any newTEST, oldoriginalpublisherVALgroup or old8995TESTrawSHA exclude newTRAIN.',
        'prior_exposure':'report newTEST connectionto each priorFIT/VAL/CAL group; do not call these unseen. Existing approved weights/data/splits are preserved, no retroactive relabel or retrain.',
        'novelty':'rawexact oldsource/duplicates are not new naturalphotos; counts onlydiagnostic. Remaining rawnovel/frame-unlinked rows are NOTreadyFIT until separate fullRGB/eightorientationpHash2/physicalgroupaudit.',
        'no_training_split':'no FIT/VAL/CAL allocation/export/feature/modelprotocol fixed here. Allolddevelopmentroles held for anylaterstudy, no currentTEST inFIT.',
        'limits':'frame/group linkage notverifiedphysicalpart identity. ByteSHA nonmatch does not establish pixelindependence. No classmask/wholeequipmenthealthlabels.',
        'storage':'stream oneencodedmember<=32MiB intoRAM; save onlysmallJSONmetadata<=64MiB, no newphotos/fullZIPcopies. >=1GiBreserve, alloriginals/failedmodels/cache/ZIP preserved.',
        'license':'InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1 CC BY-NC3.0 noncommercialacademic author/source/version/change attribution paper10.1080/01431161.2023.2283900; inherited licenses unchanged',
        'new_pixel_features_scores_optimizer':0})
    print({'policy_sha256':sha(POLICY),'new_pixel_features_optimizer':0},flush=True)


def audit():
    p=read(POLICY)
    assert sha(ROOT/SCRIPT)==p['script_sha256'] and sha(ARCHIVE)==p['archive_sha256']
    assert sha(SOURCE/'unsupervised-archive-members.json')==p['members_sha256'] and sha(GROUPED)==p['old_source_groups_sha256']
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for path in (ROOT/SCRIPT,POLICY):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+path.relative_to(ROOT).as_posix()])).hexdigest()==sha(path)
    assert not (OUT/'all-rows.json').exists() and shutil.disk_usage(ROOT).free>1024**3+64*1024**2
    members=[r for r in read(SOURCE/'unsupervised-archive-members.json')['rows'] if not r['directory']]
    rows=[]
    for r in sorted(members,key=lambda r:r['member']):
        path=PurePosixPath(r['member']);assert len(path.parts)==4 and path.suffix.lower() in ('.jpg','.jpeg','.png')
        asset,role,condition,_=path.parts;assert role in ('train','test') and (role!='train' or condition=='good')
        found=re.findall(r'(\d{2}-\d{2}-\d{4})_DJI_(\d{4})',path.name);assert len(found)==1,r['member']
        date,number=found[0];datetime.strptime(date,'%d-%m-%Y')
        rows.append({**r,'asset':asset,'publisher_role':role,'condition':condition,'frame_proxy':date+'_DJI_'+number})
    new(OUT/'reserved-publisher-test-members.json',{'UTC':utc(),'member_metadata_sha256':p['members_sha256'],
        'reserved_members':[r['member'] for r in rows if r['publisher_role']=='test'],'photo_bytes_hashed_before_reservation':0,'pixel_decodes':0})
    old=read(GROUPED)['rows'];oldgroups=sorted({r['group_id'] for r in old});oldnodes={g:len(rows)+i for i,g in enumerate(oldgroups)}
    parents=list(range(len(rows)+len(oldgroups)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        a,b=find(i),find(j);parents[max(a,b)]=min(a,b)
    oldraw=defaultdict(set);oldframes=defaultdict(set)
    for r in old:oldraw[r['image_sha256']].add(r['group_id']);oldframes[r['frame_proxy']].add(r['group_id'])
    oldheld={}
    for name,h in p['old_TEST_manifests'].items():
        assert sha(ROOT/name)==h,name
        for r in read(ROOT/name)['rows']:
            if r['phase'] in ('test','reserved_test'):
                assert r['image'] not in oldheld or oldheld[r['image']]==r['image_sha256']
                oldheld[r['image']]=r['image_sha256']
    assert len(oldheld)==8995
    heldraw=set(oldheld.values())
    seenraw={};seenframe={};newoldheld=[]
    with zipfile.ZipFile(ARCHIVE) as archive:
        for i,r in enumerate(rows):
            assert r['bytes']<=32*1024**2
            raw=archive.read(r['member']);assert len(raw)==r['bytes']
            digest=hashlib.sha256(raw).hexdigest();r['image_sha256']=digest
            if digest in seenraw:union(i,seenraw[digest])
            else:seenraw[digest]=i
            frame=r['frame_proxy']
            if frame in seenframe:union(i,seenframe[frame])
            else:seenframe[frame]=i
            linked=oldraw.get(digest,set())|oldframes.get(frame,set())
            for group in linked:union(i,oldnodes[group])
            r['old_source_raw_exact']=digest in oldraw;r['old_source_group_links']=sorted(linked)
            if digest in heldraw:newoldheld.append(i)
            if i%5000==0:print({'encoded_members_hashed':i+1,'total':len(rows),'pixel_decodes':0},flush=True)
    heldroots={find(i) for i,r in enumerate(rows) if r['publisher_role']=='test'}
    heldroots.update(find(oldnodes[r['group_id']]) for r in old if r['publisher_role']=='val')
    heldroots.update(find(i) for i in newoldheld)
    exposures={}
    for task,entry in p['prior_development_manifests'].items():
        assert sha(ROOT/entry['path'])==entry['sha256'],task
        by=defaultdict(set)
        for r in read(ROOT/entry['path'])['rows']:
            if r['phase'] in ('fit','validation','cal'):by[r['phase']].add(find(oldnodes[r['group_id']]))
        exposures[task]={phase:sum(r['publisher_role']=='test' and find(i) in roots for i,r in enumerate(rows)) for phase,roots in by.items()}
    for i,r in enumerate(rows):
        r['group_proxy']=find(i)
        if r['publisher_role']=='test':r['phase']='reserved_publisher_TEST'
        elif find(i) in heldroots:r['phase']='excluded_TEST_or_oldVAL_connected'
        else:r['phase']='unassigned_TRAIN_needs_pixel_audit'
        r['rawnovel_unassigned_TRAIN']=r['phase']=='unassigned_TRAIN_needs_pixel_audit' and not r['old_source_raw_exact']
    byasset={}
    for asset in sorted({r['asset'] for r in rows}):
        subset=[r for r in rows if r['asset']==asset]
        byasset[asset]={'TRAIN':sum(r['publisher_role']=='train' for r in subset),'TEST':sum(r['publisher_role']=='test' for r in subset),
            'TRAIN_TEST_connected_excluded':sum(r['phase']=='excluded_TEST_or_oldVAL_connected' for r in subset),
            'unassigned_TRAIN_needs_pixel_audit':sum(r['phase']=='unassigned_TRAIN_needs_pixel_audit' for r in subset),
            'rawnovel_unassigned_TRAIN_needs_pixel_audit':sum(r['rawnovel_unassigned_TRAIN'] for r in subset),
            'TEST_conditions':dict(Counter(r['condition'] for r in subset if r['publisher_role']=='test'))}
    report={'UTC':utc(),'scope':'encodedbytes and frame metadata only','all_photos':len(rows),'all_raw_unique':len(seenraw),'all_proxy_groups':len({find(i) for i in range(len(rows))}),
        'publisher_TEST_reserved':sum(r['publisher_role']=='test' for r in rows),'all_source_raw_exact':sum(r['old_source_raw_exact'] for r in rows),
        'old8995TEST_raw_exact_matches':len(newoldheld),'by_asset':byasset,'newTEST_prior_development_connected_photos':exposures,
        'TRAIN_excluded':sum(r['phase']=='excluded_TEST_or_oldVAL_connected' for r in rows),
        'unassigned_TRAIN_needs_pixel_audit':sum(r['phase']=='unassigned_TRAIN_needs_pixel_audit' for r in rows),
        'rawnovel_unassigned_TRAIN_needs_pixel_audit':sum(r['rawnovel_unassigned_TRAIN'] for r in rows),
        'rawexact_and_frame_match_do_not_prove_physicalidentity':True,'pixel_near_independence_audit_completed':False,
        'source_and_previous_TEST_not_used_for_new_training':True,'new_pixel_decodes_features_scores_optimizer':0,
        'existing_weights_splits_and_GT_unchanged':True,'independent_field_validation':False,'policy_sha256':sha(POLICY),'archive_sha256':p['archive_sha256']}
    new(OUT/'all-rows.json',{'rows':rows,'audit':report});report['rows_sha256']=sha(OUT/'all-rows.json')
    new(ROOT/'reports/industrial-insplad-unsupervised-v56-metadata-audit.json',report)
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','audit'));args=parser.parse_args();globals()[args.stage]()
