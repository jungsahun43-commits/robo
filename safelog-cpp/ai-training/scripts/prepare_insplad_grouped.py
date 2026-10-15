"""ZIP-to-RAM fingerprints; original publisher VAL is never development data."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import threading
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.prepare_glinks_grouped import HELDOUT
from scripts.prepare_magnetic_tiles import Bands
from scripts.study_equipment_expansion import read
from safelog_ai.belt_normal_roi import fingerprints, distance
from safelog_ai.insplad_metadata import parse_member, filename_base_proxy
from safelog_ai.insplad_split import allocate
from safelog_ai.process_probe import process_exists

SOURCE = ROOT/'data/industrial-insplad-v43-source'
ARCHIVE = SOURCE/'publisher-supervised-fault-classification.zip'
METADATA = SOURCE/'supervised-metadata-frame-audit.json'
OUT = ROOT/'data/industrial-insplad-grouped-v44'
RUN = ROOT/'runs/industrial-insplad-grouped-v44'
POLICY = ROOT/'reports/industrial-insplad-v44-data-policy.json'
TARGET = 'polymer-insulator-upper-shackle'
SOURCES = ('scripts/prepare_insplad_grouped.py','safelog_ai/insplad_split.py','tests/test_insplad_split.py')
OLD = (*HELDOUT, 'data/industrial-mpdd2-white-v39/manifest.json')


def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    assert sha(METADATA) == '61868d388c5a83738a875a1f9f11a320b843733591e62d5fcd82b07501fc52f2'
    assert sha(ARCHIVE) == '562e33ca768aac532569f9976c534640327d082402c0efbf140d93c8638fb4a8'
    p = {'UTC':datetime.now(timezone.utc).isoformat(), 'frozen_before_source_pixel_decode_or_view':True,
         'sources':{n:sha(ROOT/n) for n in SOURCES}, 'metadata_sha256':sha(METADATA),
         'archive_sha256':sha(ARCHIVE), 'old_TEST_manifests':{n:sha(ROOT/n) for n in OLD},
         'target':TARGET, 'reason':'priority chosen using original filename-base label support, no model scores',
         'reserved_evaluation':'ALL original publisherVAL6417 across all5assets; target originalVAL31normal/33rust. Never FIT/internalVAL/CAL.',
         'audit':'all11525 photos decode directly ZIP in RAM, no reencode. rawSHA, RGBSHA, dimensions, EXIF None/1,8orientationpHash;4workers; max40millionpixels/32MiBmember; failure stops entire gate',
         'grouping':'union global same date+DJI frame across all assets/variants, exactRGB+size or any8orientationpHash<=2; similarity not physicalidentity proof',
         'exclude':'TRAIN in any source publisherVAL group or linked to retained8995TEST; no relaxation; originalVAL retained even if oldTEST match, reported',
         'candidates':'upper-shackle TRAIN original filename-base only; suffix strip only when base stem exists in samefolder; variants remain grouped but excluded from all development samples',
         'split':'stratify eligible connected groups by set of original target labels; within each stratum sortSHA256(insplad44:+groupminmember),firstceil15percent internalVAL,nextceil15percentCAL,restFIT; strata<3groups allFIT; once only',
         'minimums':{'fit':{'0':30,'1':20},'validation':{'0':20,'1':5},'cal':{'0':20},'test':{'0':20,'1':20},'groups_per_required_label':3},
         'export':'only target selected original TRAIN and all64publisherVAL originals, numeric filenames, unchanged memberbytes; max256MiBexports/64MiBmetadata and>=1GiBfree afterwards; no full877MBexpansion',
         'gate_failure':'preserve records; no new encoder/features/model/optimizer; do not change split or minimums',
         'scope':'cropped visible surface normal/rust, no pixel masks/whole-equipment mechanical or electrical health; no original natural-photo independence claim',
         'license':'InsPLAD Mendeley5n3fjgvfyzV1 DOI10.17632/5n3fjgvfyz.1 and pinned authorLICENSE CC BY-NC3.0; cite paper10.1080/01431161.2023.2283900/version/change; other licenses unchanged',
         'visual_review':'only after split, first3 sorted FIT perlabel; never VAL/TEST content viewing',
         'training_protocol_fixed':False,'new_encoder_features_scores_optimizer':0,'units_passed':3}
    OUT.mkdir(); RUN.mkdir(); new(POLICY,p)
    print({'policy_sha256':sha(POLICY),'new_pixel_decodes':0},flush=True)


def validate():
    p = read(POLICY)
    for n,h in p['sources'].items(): assert sha(ROOT/n)==h,n
    assert sha(METADATA)==p['metadata_sha256'] and sha(ARCHIVE)==p['archive_sha256']
    for n,h in p['old_TEST_manifests'].items(): assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def audit():
    p=validate(); assert not (OUT/'manifest.json').exists()
    assert shutil.disk_usage(ROOT).free > 1024**3+320*1024**2
    entries=sorted(read(METADATA)['rows'],key=lambda r:r['member'])
    assert len(entries)==11525 and sum(r['publisher_role']=='val' for r in entries)==6417
    stems=defaultdict(set)
    for r in entries: stems[str(PurePosixPath(r['member']).parent)].add(PurePosixPath(r['member']).stem)
    local=threading.local()
    def decode(row):
        if not hasattr(local,'archive'): local.archive=zipfile.ZipFile(ARCHIVE)
        assert parse_member(row['member'])=={k:row[k] for k in parse_member(row['member'])}
        raw=local.archive.read(row['member']); assert len(raw)<=32*1024**2
        with Image.open(io.BytesIO(raw)) as im:
            assert im.format in ('JPEG','PNG') and im.getexif().get(274) in (None,1),row['member']
            assert 0<im.width*im.height<=40000000
            im.load(); rgb=im.convert('RGB')
        path=PurePosixPath(row['member']); base=filename_base_proxy(row['member'],stems[str(path.parent)])
        return {**row,'image_sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
                'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'size':list(rgb.size),
                'orientation_phashes':fingerprints(rgb),'filename_base_proxy':base,
                'is_filename_base':path.stem==base,'category':'insplad_upper_shackle','mask':None}
    fingerprints_path=OUT/'decoded-fingerprints.json'
    if fingerprints_path.exists():
        saved=read(fingerprints_path); assert saved['policy_sha256']==sha(POLICY)
        rows=saved['rows']; assert [r['member'] for r in rows]==[r['member'] for r in entries]
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows=[]
            for i,r in enumerate(pool.map(decode,entries)):
                rows.append(r)
                if i%1000==0:print({'decoded':i+1,'total':len(entries)},flush=True)
        new(fingerprints_path,{'rows':rows,'policy_sha256':sha(POLICY)})
    heldout={}
    for name in OLD:
        for r in read(ROOT/name)['rows']:
            if r['phase'] in ('test','reserved_test'): heldout[r['image']]=r
    assert len(heldout)==8995
    def old_fingerprint(r):
        assert sha(ROOT/r['image'])==r['image_sha256']
        if r.get('rgb_sha256') and r.get('orientation_phashes') and r.get('size'):
            return {k:r[k] for k in ('image','rgb_sha256','orientation_phashes','size')}
        with Image.open(ROOT/r['image']) as im: rgb=im.convert('RGB')
        return {'image':r['image'],'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'size':list(rgb.size),'orientation_phashes':fingerprints(rgb)}
    old_path=OUT/'retained-TEST-fingerprints.json'
    if old_path.exists():
        saved=read(old_path);assert saved['policy_sha256']==sha(POLICY);old=saved['rows']
        for r in old:assert sha(ROOT/r['image'])==heldout[r['image']]['image_sha256']
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:old=list(pool.map(old_fingerprint,heldout.values()))
        new(old_path,{'rows':old,'policy_sha256':sha(POLICY)})
    index=Bands(); oldexact=defaultdict(set)
    for i,r in enumerate(old):index.add(r['orientation_phashes'],i);oldexact[(r['rgb_sha256'],tuple(r['size']))].add(i)
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i: parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        i,j=find(i),find(j);parents[max(i,j)]=min(i,j)
    seen=Bands(); exact={}; frames={}; links=[];matches={}
    for i,r in enumerate(rows):
        frame=r['frame_proxy'];key=(r['rgb_sha256'],tuple(r['size']))
        if frame in frames:union(i,frames[frame])
        else:frames[frame]=i
        if key in exact:union(i,exact[key])
        else:exact[key]=i
        for j in sorted(seen.candidates(r['orientation_phashes'])):
            if distance(r['orientation_phashes'],rows[j]['orientation_phashes'])<=2:union(i,j);links.append([i,j])
        seen.add(r['orientation_phashes'],i)
        hits=set(oldexact.get(key,()))
        for j in index.candidates(r['orientation_phashes']):
            if distance(r['orientation_phashes'],old[j]['orientation_phashes'])<=2:hits.add(j)
        if hits:matches[str(i)]=[old[j]['image'] for j in sorted(hits)]
    testgroups={find(i) for i,r in enumerate(rows) if r['publisher_role']=='val'}
    oldgroups={find(int(i)) for i in matches}
    eligible=[]
    for i,r in enumerate(rows):
        r['group_id']=rows[find(i)]['member']; r['id']='insplad/'+r['member']
        if r['publisher_role']=='val': r['phase']='reserved_publisher_VAL'
        elif find(i) in testgroups:r['phase']='excluded_TEST_group'
        elif find(i) in oldgroups:r['phase']='excluded_old_TEST_group'
        elif not r['is_filename_base']:r['phase']='excluded_filename_variant'
        elif r['asset']!=TARGET:r['phase']='unselected_asset_TRAIN'
        else:r['phase']='candidate';eligible.append(r)
    allocation=allocate(eligible)
    for r in eligible:r['phase']=allocation[r['group_id']]
    selected=[r.copy() for r in rows if (r['asset']==TARGET and r['publisher_role']=='val') or r['phase'] in ('fit','validation','cal')]
    for r in selected:
        if r['publisher_role']=='val':r['phase']='test'
    counts={phase:dict(Counter(str(r['label']) for r in selected if r['phase']==phase)) for phase in ('fit','validation','cal','test')}
    groupcounts={phase:{str(label):len({r['group_id'] for r in selected if r['phase']==phase and r['label']==label}) for label in (0,1)} for phase in counts}
    reasons=[]
    for phase,minimum in p['minimums'].items():
        if not isinstance(minimum,dict):continue
        for label,n in minimum.items():
            if counts[phase].get(label,0)<n:reasons.append(f'{phase} label{label} count below{n}')
            if groupcounts[phase][label]<3:reasons.append(f'{phase} label{label} groups below3')
    for a in counts:
        for b in counts:
            if a<b:assert {r['group_id'] for r in selected if r['phase']==a}.isdisjoint({r['group_id'] for r in selected if r['phase']==b})
    exportbytes=sum(r['bytes'] for r in selected)
    assert exportbytes<=256*1024**2 and shutil.disk_usage(ROOT).free-exportbytes>1024**3
    if not reasons:
        (OUT/'originals').mkdir()
        with zipfile.ZipFile(ARCHIVE) as archive:
            for i,r in enumerate(selected):
                path=OUT/'originals'/f'{i:05d}{PurePosixPath(r["member"]).suffix.lower()}'
                raw=archive.read(r['member']);assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
                with path.open('xb') as out:out.write(raw)
                r['image']=path.relative_to(ROOT).as_posix()
    result={'UTC':datetime.now(timezone.utc).isoformat(),'ready':not reasons,'gate_failures':reasons,
            'counts':counts,'groups_each_label':groupcounts,'all_source_entries':len(rows),
            'all_publisher_VAL_reserved':6417,'local_near_pairs':len(links),'local_connected_groups':len({find(i) for i in range(len(rows))}),
            'old_TEST_checked':len(old),'source_direct_old_TEST_matches':len(matches),
            'all_TRAIN_exclusions':dict(Counter(r['phase'] for r in rows if r['publisher_role']=='train')),
            'exported_originals':len(selected) if not reasons else 0,'exported_bytes':exportbytes if not reasons else 0,
            'source_sizes':dict(Counter(str(r['size']) for r in rows)),
            'policy_sha256':sha(POLICY),'decoded_fingerprints_sha256':sha(fingerprints_path),
            'new_encoder_features_scores_optimizer':0,'visual_photo_review_performed':False,
            'physical_part_and_flight_independence_unverified':True,'new_training_protocol_fixed':False,
            'source_label_scope':'visible normal/rust crops; no masks or whole equipment health',
            'independent_field_validation':False}
    new(OUT/'grouped-all-source.json',{'rows':rows,'near_pairs':links,'old_TEST_matches':matches})
    new(OUT/'manifest.json',{'rows':selected,'audit':result})
    result['manifest_sha256']=sha(OUT/'manifest.json')
    new(ROOT/'reports/industrial-insplad-v44-data-audit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','audit'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job: '+str(path))
        lock=RUN/'operation.lock'
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:audit()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'pid':os.getpid()});raise
        finally:lock.unlink()
