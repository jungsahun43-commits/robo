"""Full new-source pixel linkage, then one preregistered normal-only target split."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.belt_normal_roi import fingerprints,distance
from safelog_ai.insplad_pixel_groups import Groups,allocate
from safelog_ai.process_probe import process_exists
from scripts.prepare_magnetic_tiles import Bands
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

ARCHIVE=ROOT/'data/industrial-insplad-unsupervised-v55-source/publisher-unsupervised-anomaly-detection.zip'
META=ROOT/'data/industrial-insplad-unsupervised-v56-metadata/all-rows.json'
OLD=ROOT/'data/industrial-insplad-grouped-v44'
OUT=ROOT/'data/industrial-insplad-lower-shackle-v57'
RUN=ROOT/'runs/industrial-insplad-lower-shackle-data-v57'
POLICY=ROOT/'reports/industrial-insplad-lower-shackle-v57-data-policy.json'
SOURCES=('scripts/prepare_insplad_lower_shackle.py','safelog_ai/insplad_pixel_groups.py','tests/test_insplad_pixel_groups.py')
PRIOR=('data/industrial-insplad-grouped-v44/manifest.json','data/industrial-insplad-glass-v47/manifest.json',
       'data/industrial-insplad-lightning-v50/manifest.json','data/industrial-insplad-varigrip-v53/manifest.json')
TARGET='polymer-insulator-lower-shackle'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    assert sha(META)=='a98aa6eb783473f7f6af3f2a9f88d2afd14269448915542180516a9787b83889'
    assert sha(ARCHIVE)=='baab1d227b408ec6bbe5831cbcae5ea049a9b5132a74bf36f4e3d6051b0cb7e3'
    held=read(ROOT/'reports/industrial-insplad-v44-data-policy.json')['old_TEST_manifests']
    caches=('grouped-all-source.json','decoded-fingerprints.json','retained-TEST-fingerprints.json')
    assert shutil.disk_usage(ROOT).free>1024**3+512*1024**2
    p={'UTC':utc(),'fixed_before_new_pixel_decode_or_split':True,'sources':{n:sha(ROOT/n) for n in SOURCES},
       'metadata_sha256':sha(META),'archive_sha256':sha(ARCHIVE),'old_caches_sha256':{n:sha(OLD/n) for n in caches},
       'old_TEST_manifests':held,'prior_development_manifests':{n:sha(ROOT/n) for n in PRIOR},
       'target':TARGET,'target_initial_normal_candidates':143,'target_publisher_TEST':{'good':327,'rust':315},
       'reuse':'all27620 newrows,8990 rawbyteexact oldsupervised rows reuse verified size/RGB/8orientationpHash;remaining18630 rows ZIPtoRAM. Never rerun oldsource audit/download.',
       'decode':'JPEG/PNG <=32MiB encoded,positive<=40MP,EXIF None/1,4workers,500-row resumable shards;originalsha perdecoded member,decoder events counted. No pixelview or neuralfeatures.',
       'group':'seed v56 frame/raw/oldsourcegroups; union same size+RGB exact or any8orientationpHash Hamming<=2 new-new/new-oldsource/new-old8995TEST. Old-oldsource groups reused. Only joins across distinct currentcomponents logged; intra-component distances skipped without changing connectivity. Pixel similarity not verifiedphysicalidentity.',
       'held':'all6481newpublisherTEST plus all6417oldpublisherVAL and8995oldTEST nodes; anyconnected newTRAIN excluded. OriginalTEST kept evenif pastoldFITexposure, limits reported.',
       'candidates':'ONLY v56rawnovel_unassigned_TRAIN good TARGET143. No compensation by otherTRAIN afterpixel exclusions.',
       'split':'oldv45/v48/v51/rejectedv54 developmentrole one preserved, conflicts excluded. Unlocked targetgroups sortSHA256(insplad57:+smalleststablememberID),firstceil20percentVAL,nextceil20CAL,restFIT,onceonly.',
       'minimums':{'fit':{'0':30},'validation':{'0':20},'cal':{'0':20},'test':{'0':20,'1':20}},'minimum_groups_per_required_label':3,
       'failure':'ifminimum fails preserve andstop beforeimagesexport/features/optimizer, no threshold/split/minimum orcandidate adjustment',
       'export':'onlysuccessful targetselectedTRAIN plusall642originaltargetTEST unchanged bytes, numericfilename<=128MiB. No entireZIPextraction, no photoGT edits or classificationlabel masks.',
       'visual_review':'onlyafter successful finalsplit first3sortedFITnormal fixed. No choosingTESTcontent/VAL_CAL orclaiminghumanreview.',
       'cache':'bind rawSHA/member/order/metadata/archive/policySHA;500rowshards saved via unique partialfile then atomicrename, verifysha andids onresume. Incomplete partialfiles preserved; events record repeateddecodes. Interrupted graph scans can rerun only fromcached fingerprints into a new edgefile, no pixel/optimizer repetition claimed as newphotos.',
       'storage':'metadata<=256MiB andexport<=128MiB,plannedworkingbudget512MiB+1GiBreserve; original/failed models/log/cache/ZIP allpreserved',
       'source_scope':'original cropped lower-shackle visiblegood/rust, TRAINnormalonly. NoactualrustTRAIN/pixelGT/internalhealth labels; no oldbracket/tubeimprovement claim.',
       'license':'InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1 CC BY-NC3.0 noncommercialacademic author/source/version/change attribution paper10.1080/01431161.2023.2283900; oldlicensesunchanged',
       'model_protocol_fixed':False,'new_encoder_features_scores_gradient_optimizer':0,'new_units_passed':3,'independent_field_validation':False}
    OUT.mkdir();RUN.mkdir();new(POLICY,p);print({'policy_sha256':sha(POLICY),'new_neural_features_optimizer':0},flush=True)


def validate():
    p=read(POLICY)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    assert sha(META)==p['metadata_sha256'] and sha(ARCHIVE)==p['archive_sha256']
    for n,h in p['old_caches_sha256'].items():assert sha(OLD/n)==h,n
    for n,h in {**p['old_TEST_manifests'],**p['prior_development_manifests']}.items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def pixel_rows(p):
    rows=read(META)['rows'];old=read(OLD/'grouped-all-source.json')['rows'];decoded=read(OLD/'decoded-fingerprints.json')['rows']
    bymember={r['member']:r for r in decoded};cache={}
    for r in old:
        original=bymember[r['member']]
        for name in ('image_sha256','size','rgb_sha256','orientation_phashes'):assert r[name]==original[name]
        value={k:r[k] for k in ('size','rgb_sha256','orientation_phashes')}
        if r['image_sha256'] in cache:assert cache[r['image_sha256']]==value
        cache[r['image_sha256']]=value
    assert sum(r['image_sha256'] in cache for r in rows)==8990
    local=threading.local();eventlock=threading.Lock();eventpath=RUN/'decode-events.jsonl';complete=[]
    with eventpath.open('a',encoding='utf-8',buffering=1) as events:
        def decode(r):
            if r['image_sha256'] in cache:return {**r,**cache[r['image_sha256']],'pixel_origin':'old_raw_exact_cache'}
            try:
                if not hasattr(local,'archive'):local.archive=zipfile.ZipFile(ARCHIVE)
                raw=local.archive.read(r['member']);assert len(raw)==r['bytes'] and len(raw)<=32*1024**2
                assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
                with Image.open(io.BytesIO(raw)) as im:
                    assert im.format in ('JPEG','PNG') and 0<im.width*im.height<=40000000
                    assert im.getexif().get(274) in (None,1),r['member']
                    im.load();rgb=im.convert('RGB')
                result={**r,'size':list(rgb.size),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                        'orientation_phashes':fingerprints(rgb),'pixel_origin':'new_ZIP_RAM_decode'}
                with eventlock:events.write(json.dumps({'UTC':utc(),'member':r['member'],'status':'decoded'})+'\n')
                return result
            except BaseException as error:
                with eventlock:events.write(json.dumps({'UTC':utc(),'member':r['member'],'status':'failed','error':repr(error)})+'\n')
                raise
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0,len(rows),500):
                batch=rows[start:start+500];path=OUT/f'fingerprints-{start:05d}.json';receipt=OUT/f'fingerprints-{start:05d}-sha.json'
                if path.exists():
                    saved=read(path)
                    assert saved['policy_sha256']==sha(POLICY) and saved['metadata_sha256']==p['metadata_sha256']
                    values=saved['rows'];assert [r['member'] for r in values]==[r['member'] for r in batch]
                    assert [r['image_sha256'] for r in values]==[r['image_sha256'] for r in batch]
                    if receipt.exists():assert sha(path)==read(receipt)['sha256']
                    else:new(receipt,{'sha256':sha(path),'rows':len(values),'recovered_after_atomic_cache_write':True})
                else:
                    values=list(pool.map(decode,batch));partial=OUT/(path.stem+'-partial-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
                    new(partial,{'policy_sha256':sha(POLICY),'metadata_sha256':p['metadata_sha256'],'rows':values})
                    assert path.resolve().is_relative_to(OUT.resolve()) and not path.exists();partial.rename(path)
                    new(receipt,{'sha256':sha(path),'rows':len(values)})
                complete.extend(values);print({'fingerprinted_or_reused':len(complete),'total':len(rows)},flush=True)
    assert len(complete)==27620
    return complete,old


def audit():
    p=validate();assert not (OUT/'manifest.json').exists() and shutil.disk_usage(ROOT).free>1024**3+512*1024**2
    rows,old=pixel_rows(p);held=read(OLD/'retained-TEST-fingerprints.json')['rows'];assert len(old)==11525 and len(held)==8995
    oldheld={}
    for name in p['old_TEST_manifests']:
        for r in read(ROOT/name)['rows']:
            if r['phase'] in ('test','reserved_test'):oldheld[r['image']]=r['image_sha256']
    assert set(oldheld)=={r['image'] for r in held}
    n=len(rows);h=n+len(old);allrows=rows+old+held;g=Groups(len(allrows));oldgroup={};metagroup={}
    for j,r in enumerate(old):
        key=r['group_id']
        if key in oldgroup:g.union(n+j,oldgroup[key])
        else:oldgroup[key]=n+j
    for i,r in enumerate(rows):
        key=r['group_proxy']
        if key in metagroup:g.union(i,metagroup[key])
        else:metagroup[key]=i
        for key in r['old_source_group_links']:g.union(i,oldgroup[key])
    index=Bands();exact=defaultdict(list)
    for j,r in enumerate(old+held,n):index.add(r['orientation_phashes'],j);exact[(r['rgb_sha256'],tuple(r['size']))].append(j)
    counts=Counter()
    edgepath=OUT/'pixel-joining-edges.jsonl'
    if edgepath.exists():edgepath=OUT/('pixel-joining-edges-recovery-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.jsonl')
    with edgepath.open('x',encoding='utf-8',buffering=1) as edges:
        for i,r in enumerate(rows):
            key=(r['rgb_sha256'],tuple(r['size']))
            for j in exact.get(key,()):
                if g.find(i)!=g.find(j):g.union(i,j);counts['RGB_exact_joins']+=1;edges.write(json.dumps([i,j,'RGB_exact'])+'\n')
            for j in sorted(index.candidates(r['orientation_phashes'])):
                if g.find(i)==g.find(j):continue
                counts['cross_component_pHash_comparisons']+=1
                d=distance(r['orientation_phashes'],allrows[j]['orientation_phashes'])
                if d<=2:g.union(i,j);counts['pHash_joins']+=1;edges.write(json.dumps([i,j,'pHash',d])+'\n')
            index.add(r['orientation_phashes'],i);exact[key].append(i)
            if i%2500==0:print({'pixel_grouped':i+1,'total':n,'joins':dict(counts)},flush=True)
    nodekeys=['unsupervised:'+r['member'] for r in rows]+['supervised:'+r['member'] for r in old]+['held:'+r['image'] for r in held]
    canonical={}
    for i,key in enumerate(nodekeys):root=g.find(i);canonical[root]=min(canonical.get(root,key),key)
    heldroots={g.find(i) for i,r in enumerate(rows) if r['publisher_role']=='test'}
    heldroots.update(g.find(n+j) for j,r in enumerate(old) if r['publisher_role']=='val');heldroots.update(g.find(i) for i in range(h,len(allrows)))
    priorroles=defaultdict(set)
    for path in p['prior_development_manifests']:
        for r in read(ROOT/path)['rows']:
            if r['phase'] in ('fit','validation','cal'):priorroles[canonical[g.find(oldgroup[r['group_id']])]].add(r['phase'])
    candidates=[i for i,r in enumerate(rows) if r['asset']==TARGET and r['rawnovel_unassigned_TRAIN']]
    assert len(candidates)==143 and all(rows[i]['publisher_role']=='train' and rows[i]['condition']=='good' for i in candidates)
    phases=allocate([canonical[g.find(i)] for i in candidates],{canonical[root] for root in heldroots},priorroles)
    selected=[];excluded=[]
    for i in candidates:
        r={**rows[i],'v56_phase':rows[i]['phase'],'pixel_group_id':canonical[g.find(i)],'phase':phases[canonical[g.find(i)]],
           'label':0,'id':'insplad-unsupervised/'+rows[i]['member'],'category':'insplad_lower_shackle_rust','mask':None}
        (excluded if r['phase'].startswith('excluded_') else selected).append(r)
    test=[{**r,'v56_phase':r['phase'],'phase':'test','pixel_group_id':canonical[g.find(i)],'label':int(r['condition']=='rust'),
          'id':'insplad-unsupervised/'+r['member'],'category':'insplad_lower_shackle_rust','mask':None} for i,r in enumerate(rows) if r['asset']==TARGET and r['publisher_role']=='test']
    assert Counter(r['condition'] for r in test)=={'good':327,'rust':315}
    resultrows=sorted(selected+test,key=lambda r:r['member']);counter={phase:dict(Counter(str(r['label']) for r in resultrows if r['phase']==phase)) for phase in p['minimums']}
    groups={phase:{str(y):len({r['pixel_group_id'] for r in resultrows if r['phase']==phase and r['label']==y}) for y in (0,1)} for phase in counter}
    for a in counter:
        for b in counter:
            if a<b:assert {r['pixel_group_id'] for r in resultrows if r['phase']==a}.isdisjoint({r['pixel_group_id'] for r in resultrows if r['phase']==b})
    failures=[]
    for phase,minimums in p['minimums'].items():
        for label,minimum in minimums.items():
            if counter[phase].get(label,0)<minimum or groups[phase][label]<3:failures.append({'phase':phase,'label':label,'photos':counter[phase].get(label,0),'groups':groups[phase][label],'minimum_photos':minimum,'minimum_groups':3})
    size=sum(r['bytes'] for r in resultrows);assert size<=128*1024**2 and shutil.disk_usage(ROOT).free-size>1024**3
    if not failures:
        (OUT/'originals').mkdir()
        with zipfile.ZipFile(ARCHIVE) as archive:
            for i,r in enumerate(resultrows):
                raw=archive.read(r['member']);assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
                path=OUT/'originals'/f'{i:05d}{PurePosixPath(r["member"]).suffix.lower()}'
                with path.open('xb') as out:out.write(raw)
                r['image']=path.relative_to(ROOT).as_posix()
    events=[json.loads(line) for line in (RUN/'decode-events.jsonl').read_text(encoding='utf-8').splitlines()]
    result={'UTC':utc(),'ready':not failures,'target':TARGET,'counts':counter,'groups_each_label':groups,'gate_failures':failures,
        'initial_candidate_normal_photos':143,'excluded_candidates':dict(Counter(r['phase'] for r in excluded)),
        'original_target_TEST642_preserved':len(test)==642,'new_source_rows':len(rows),'old_raw_exact_cached_rows':sum(r['pixel_origin']=='old_raw_exact_cache' for r in rows),
        'new_decoded_unique_members':len({r['member'] for r in events if r['status']=='decoded'}),'actual_decode_events_including_repeats':len(events),
        'neural_features_scores_gradient_optimizer':0,'new_photo_content_viewed':0,'training_protocol_fixed':False,'independent_field_validation':False,
        'join_counts_current_completed_scan':dict(counts),'pixel_joining_edge_file':edgepath.relative_to(ROOT).as_posix(),'pixel_joining_edge_sha256':sha(edgepath),'proxy_grouping_not_physical_identity_proof':True,'old_original_images_SHA_all_recalculated':False,
        'oldcaches_reused_sha256':p['old_caches_sha256'],'new_units_passed':3,'policy_sha256':sha(POLICY),'exported_original_bytes':size if not failures else 0}
    new(OUT/'excluded-candidates.json',{'rows':excluded});new(OUT/'manifest.json',{'rows':resultrows,'audit':result})
    result['manifest_sha256']=sha(OUT/'manifest.json');new(ROOT/'reports/industrial-insplad-lower-shackle-v57-data-audit.json',result)
    assert sum(path.stat().st_size for path in OUT.glob('*.json*'))<=256*1024**2
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
