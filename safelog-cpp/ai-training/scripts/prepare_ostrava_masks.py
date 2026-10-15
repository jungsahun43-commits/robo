"""Audit real publisher masks and pixel groups without changing original assets."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import os
import re
import shutil
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.ostrava_split import assign
from safelog_ai.insplad_pixel_groups import Groups
from safelog_ai.belt_normal_roi import fingerprints,distance
from safelog_ai.process_probe import process_exists
from scripts.prepare_magnetic_tiles import Bands
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

SOURCE=ROOT/'data/industrial-new-rust-belt-v60-source'
ARCHIVE=SOURCE/'publisher-ostrava-v2.zip'
OUT=ROOT/'data/industrial-ostrava-masks-v61'
RUN=ROOT/'runs/industrial-ostrava-data-v61'
POLICY=ROOT/'reports/industrial-ostrava-v61-data-policy.json'
SOURCES=('scripts/prepare_ostrava_masks.py','safelog_ai/ostrava_split.py','tests/test_ostrava_split.py')
OLD=ROOT/'data/industrial-insplad-grouped-v44'
META=ROOT/'data/industrial-insplad-unsupervised-v56-metadata/all-rows.json'
V57=ROOT/'data/industrial-insplad-lower-shackle-v57'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    audit=read(SOURCE/'ostrava-archive-audit.json')
    assert audit['local_archive_sha256']=='874a60cd8e9e92476c2314bd6acfcac7dc0f64eb797ea220a091a96681de50c5'
    assert sha(ARCHIVE)==audit['local_archive_sha256']
    members=read(SOURCE/'ostrava-zip-members.json')['rows'];photos=[];masks={}
    for r in members:
        if r['directory']:continue
        path=PurePosixPath(r['member']);assert len(path.parts)==4
        prefix,role,kind,filename=path.parts
        assert prefix=='corrosion-in-industrial-complexes-in-ostrava' and role in ('train','validation','test')
        assert kind in ('images','masks') and path.suffix==('.jpg' if kind=='images' else '.png')
        value={**r,'publisher_role':role,'source_stem':path.stem}
        if kind=='images':photos.append(value)
        else:masks[(role,path.stem)]=value
    assert Counter(r['publisher_role'] for r in photos)=={'train':75,'validation':15,'test':15}
    assert len(masks)==105 and len({(r['publisher_role'],r['source_stem']) for r in photos})==105
    for r in photos:
        mask=masks[(r['publisher_role'],r['source_stem'])]
        r.update(mask_member=mask['member'],mask_encoded_sha256=mask['encoded_sha256'],mask_bytes=mask['bytes'],id='ostrava/'+r['publisher_role']+'/'+r['source_stem'])
    photos.sort(key=lambda r:r['member']);OUT.mkdir();RUN.mkdir()
    new(OUT/'reserved-original-TEST-before-pixels.json',{'rows':[r for r in photos if r['publisher_role']=='test'],'new_pixel_decode_contentview_features_optimizer':0})
    new(OUT/'publisher-pairs-before-pixels.json',{'rows':photos})
    caches={n:sha(OLD/n) for n in ('grouped-all-source.json','retained-TEST-fingerprints.json')}
    shards={v.relative_to(ROOT).as_posix():sha(v) for v in V57.glob('fingerprints-*.json')}
    p={'UTC':utc(),'fixed_before_ostrava_pixel_decode_or_new_split':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
       'archive_sha256':sha(ARCHIVE),'publisher_member_list_sha256':sha(SOURCE/'ostrava-zip-members.json'),
       'publisher_pairs_sha256':sha(OUT/'publisher-pairs-before-pixels.json'),'reserved_TEST_sha256':sha(OUT/'reserved-original-TEST-before-pixels.json'),
       'old_caches_sha256':caches,'v56_metadata_sha256':sha(META),'v57_shards_sha256':shards,
       'v57_edges_sha256':sha(V57/'pixel-joining-edges.jsonl'),'v57_manifest_sha256':sha(V57/'manifest.json'),
       'actual_publisher_photos':{'train':75,'validation':15,'test':15},'original_annotation_files':105,
       'decode':'ZIPtoRAM original encodedSHA eachimage/mask; <=32MiB each/<=40MP; imageJPEG RGB andmaskPNG L conversion, EXIForientationNone1 only, both2048square. Noautorotation/resize/correctedGT. Sequentialdecode events/errors preserved.',
       'mask':'author pinnedloader uses convertL >0 corrosionpositive; record originalmode/uniqueLvalues andpositive/background counts, exactnativecoordinate size match. No class/bbox to pixel conversion or autoGT edits.',
       'groups':'replay samev57 graph/meta/94edges andvalidateall783canonical IDs, no oldpixel/oldpairdistance reread. Addnew105 RGBexact+size or8orientationpHash<=2 againstallold48140 andnew; samevalid cameraModel+EXIFDateTimeOriginal exactsecond mergesnewcaptureproxy. MissingEXIF notindependenceproof. ExactnewRGB withdifferentLmasksha rejects study.',
       'split':'originalTEST15 immutable; externallyheldold8995TEST/oldpublisherVAL6417/newTEST6481 ornewTESTconnected nonTEST excluded; originalVAL15 retainsvalidation exceptheldconnections. TRAINnearVAL excluded. RemainingTRAINgroups sortSHA256(ostrava61:+canonicalgroupID) firstceil20percentCAL,restFIT once.',
       'minimum_positive_photos':{'fit':30,'validation':5,'cal':5,'test':5},'minimum_groups':{'fit':5,'validation':3,'cal':3,'test':3},
       'minimum_positive_and_background_pixels_each_phase':1024,
       'failure':'no minimumrelax/repartition/currentTESTasFIT; preservefailedmanifest/audit andstop before model features/gradient',
       'QA':'only afterpassed finalsplit first3FIT sortedoriginalmember fixed, originalRGBandoriginalLmaskcontactsheet; no selectedVAL_CAL_TESTcontentorGT edits',
       'export':'no originalfileexpansion; bind immutable archive/member/rawSHA/maskSHA inmanifest, futureloaderZIPtoRAM; no newencoder/cache/model training now',
       'source_limits':'15sourceTESTphotos<20 and normalwholephoto support notknown; no new app type20percent photoFP_FN pass claim, no fullmachinehealth or independent physicalpart/site proof',
       'license':'MatejFric Ostrava ZenodoV2 DOI10.5281/zenodo.11235637 dataCC BY4.0 author/source/version/changes attribution; otherlicensesunchanged',
       'storage':'metadata/contactsheet<64MiB plusRAMdecoding,>=1GiBreserve; no archive duplication/extraction/oldcache deletion',
       'new_units_passed':2,'new_model_features_scores_gradient_optimizer':0,'full_training_protocol_fixed':False,'independent_field_validation':False}
    assert shutil.disk_usage(ROOT).free>1024**3+64*1024**2
    new(POLICY,p);print({'policy_sha256':sha(POLICY),'reserved_TEST15_sha256':p['reserved_TEST_sha256']},flush=True)


def validate():
    p=read(POLICY)
    for n,h in {**p['source_sha256'],**p['v57_shards_sha256']}.items():assert sha(ROOT/n)==h,n
    for n,h in p['old_caches_sha256'].items():assert sha(OLD/n)==h,n
    for path,h in ((ARCHIVE,p['archive_sha256']),(META,p['v56_metadata_sha256']),(OUT/'publisher-pairs-before-pixels.json',p['publisher_pairs_sha256']),
            (OUT/'reserved-original-TEST-before-pixels.json',p['reserved_TEST_sha256']),(V57/'pixel-joining-edges.jsonl',p['v57_edges_sha256']),(V57/'manifest.json',p['v57_manifest_sha256'])):assert sha(path)==h,path
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def decode():
    pairs=read(OUT/'publisher-pairs-before-pixels.json')['rows'];rows=[]
    with zipfile.ZipFile(ARCHIVE) as archive,(RUN/'decode-events.jsonl').open('x',encoding='utf-8',buffering=1) as events:
        for r in pairs:
            raw=archive.read(r['member']);maskraw=archive.read(r['mask_member'])
            assert len(raw)==r['bytes'] and len(maskraw)==r['mask_bytes'] and max(len(raw),len(maskraw))<=32*1024**2
            assert hashlib.sha256(raw).hexdigest()==r['encoded_sha256'] and hashlib.sha256(maskraw).hexdigest()==r['mask_encoded_sha256']
            with Image.open(io.BytesIO(raw)) as im,Image.open(io.BytesIO(maskraw)) as mask:
                assert im.format=='JPEG' and mask.format=='PNG' and im.size==mask.size==(2048,2048)
                assert im.getexif().get(274) in (None,1) and mask.getexif().get(274) in (None,1)
                rgb=im.convert('RGB');m=np.asarray(mask.convert('L'));positive=int((m>0).sum())
                exif=im.getexif();sub=exif.get_ifd(34665) if exif.get(34665) else {}
                stamp=sub.get(36867);camera=exif.get(272)
                capture=str(camera)+'|'+str(stamp) if camera and isinstance(stamp,str) and re.fullmatch(r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}',stamp) else None
                row={**r,'size':list(rgb.size),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                     'orientation_phashes':fingerprints(rgb),'mask_L_sha256':hashlib.sha256(m.tobytes()).hexdigest(),
                     'original_mask_mode':mask.mode,'mask_L_values':np.unique(m).tolist(),'positive_pixels':positive,'background_pixels':m.size-positive,
                     'label':int(positive>0),'capture_second_proxy':capture,'archive':ARCHIVE.relative_to(ROOT).as_posix()}
            rows.append(row);events.write(json.dumps({'UTC':utc(),'id':r['id'],'decoded_original_RGB_and_mask':True})+'\n')
    new(OUT/'decoded-pairs.json',{'rows':rows});return rows


def audit():
    p=validate();assert not (OUT/'manifest.json').exists()
    rows=decode();old=read(OLD/'grouped-all-source.json')['rows'];held=read(OLD/'retained-TEST-fingerprints.json')['rows'];prior=[]
    for path in sorted(V57.glob('fingerprints-*.json')):
        if path.name.endswith('-sha.json'):continue
        prior.extend(read(path)['rows'])
    meta=read(META)['rows'];assert len(prior)==27620 and [r['member'] for r in prior]==[r['member'] for r in meta]
    a,b=len(prior),len(prior)+len(old);offset=b+len(held);allrows=prior+old+held+rows
    assert (len(prior),len(old),len(held))==(27620,11525,8995)
    g=Groups(len(allrows));oldgroups={};metagroups={}
    for j,r in enumerate(old):
        key=r['group_id']
        if key in oldgroups:g.union(a+j,oldgroups[key])
        else:oldgroups[key]=a+j
    for i,r in enumerate(prior):
        key=r['group_proxy']
        if key in metagroups:g.union(i,metagroups[key])
        else:metagroups[key]=i
        for key in r['old_source_group_links']:g.union(i,oldgroups[key])
    edges=[json.loads(v) for v in (V57/'pixel-joining-edges.jsonl').read_text(encoding='utf-8').splitlines()];assert len(edges)==94
    for i,j,*_ in edges:g.union(i,j)
    keys=['unsupervised:'+r['member'] for r in prior]+['supervised:'+r['member'] for r in old]+['held:'+r['image'] for r in held]
    canonical={}
    for i,key in enumerate(keys):root=g.find(i);canonical[root]=min(canonical.get(root,key),key)
    bymember={r['member']:i for i,r in enumerate(prior)}
    for r in read(V57/'manifest.json')['rows']:assert canonical[g.find(bymember[r['member']])]==r['pixel_group_id']
    index=Bands();exact=defaultdict(list);capture={};counts=Counter();newlinks=[]
    for i,r in enumerate(allrows[:offset]):index.add(r['orientation_phashes'],i);exact[(r['rgb_sha256'],tuple(r['size']))].append(i)
    for i,r in enumerate(rows,offset):
        key=(r['rgb_sha256'],tuple(r['size']))
        for j in exact.get(key,()):
            if g.find(i)!=g.find(j):g.union(i,j);newlinks.append([i,j,'RGB_exact']);counts['RGB_exact_joins']+=1
        for j in sorted(index.candidates(r['orientation_phashes'])):
            if g.find(i)==g.find(j):continue
            counts['new_cross_component_pHash_comparisons']+=1;d=distance(r['orientation_phashes'],allrows[j]['orientation_phashes'])
            if d<=2:g.union(i,j);newlinks.append([i,j,'pHash',d]);counts['pHash_joins']+=1
        stamp=r['capture_second_proxy']
        if stamp:
            if stamp in capture:g.union(i,capture[stamp]);counts['capture_second_joins']+=1
            else:capture[stamp]=i
        index.add(r['orientation_phashes'],i);exact[key].append(i)
    keys.extend('ostrava:'+r['member'] for r in rows);canonical={}
    for i,key in enumerate(keys):root=g.find(i);canonical[root]=min(canonical.get(root,key),key)
    external_held_indices=[i for i,r in enumerate(prior) if r['publisher_role']=='test']+[a+j for j,r in enumerate(old) if r['publisher_role']=='val']+list(range(b,offset))
    external={canonical[g.find(i)] for i in external_held_indices}
    values=[{**r,'group_id':canonical[g.find(offset+i)],'category':'ostrava_industrial_corrosion_mask',
             'storage':'ZIPmember','annotation_kind':'native_L_greater_than_zero_corrosion_mask'} for i,r in enumerate(rows)]
    values=assign(values,external);failures=[]
    for key,ids in exact.items():
        own=[j-offset for j in ids if j>=offset]
        if len({rows[j]['mask_L_sha256'] for j in own})>1:failures.append({'kind':'exactRGB_conflicting_mask','ids':[rows[j]['id'] for j in own]})
    counter={phase:dict(Counter(str(r['label']) for r in values if r['phase']==phase)) for phase in p['minimum_groups']}
    groups={phase:len({r['group_id'] for r in values if r['phase']==phase}) for phase in counter}
    pixelcounts={phase:{name:sum(r[name] for r in values if r['phase']==phase) for name in ('positive_pixels','background_pixels')} for phase in counter}
    for phase,minimum in p['minimum_positive_photos'].items():
        if counter[phase].get('1',0)<minimum:failures.append({'kind':'positive_photo_minimum','phase':phase,'actual':counter[phase].get('1',0),'minimum':minimum})
        if groups[phase]<p['minimum_groups'][phase]:failures.append({'kind':'group_minimum','phase':phase,'actual':groups[phase],'minimum':p['minimum_groups'][phase]})
        for name,n in pixelcounts[phase].items():
            if n<p['minimum_positive_and_background_pixels_each_phase']:failures.append({'kind':'pixel_minimum','phase':phase,'field':name,'actual':n})
    for x in counter:
        for y in counter:
            if x<y:assert {r['group_id'] for r in values if r['phase']==x}.isdisjoint({r['group_id'] for r in values if r['phase']==y})
    assert [r['id'] for r in values if r['publisher_role']=='test']==[r['id'] for r in read(OUT/'reserved-original-TEST-before-pixels.json')['rows']]
    new(OUT/'new-pixel-links.json',{'rows':newlinks});new(OUT/'excluded-rows.json',{'rows':[r for r in values if r['phase'].startswith('excluded_')]})
    result={'UTC':utc(),'ready':not failures,'counts':counter,'groups':groups,'native_pixel_counts':pixelcounts,'failures':failures,
       'excluded_photos':dict(Counter(r['phase'] for r in values if r['phase'].startswith('excluded_'))),'pixel_comparisons_and_joins':dict(counts),
       'new_image_and_mask_pairs_decoded':len(rows),'replayed_old_edges':94,'old783canonical_groups_exact':True,
       'external_TEST_connected_source_photos':sum(r['group_id'] in external for r in values),'capture_second_proxy_photos':sum(bool(r['capture_second_proxy']) for r in rows),
       'all_source_mask_L_values':sorted({v for r in rows for v in r['mask_L_values']}),'sourceTEST15_preserved':True,
       'new_neural_features_scores_gradient_optimizer':0,'new_photo_content_viewed':0,'training_protocol_fixed':False,
       'no_original_expansion_or_source_code_execution':True,'independent_field_validation':False,'physical_part_capture_groups_verified':False,
       'new_units_passed':2,'policy_sha256':sha(POLICY),'reserved_TEST_sha256':p['reserved_TEST_sha256'],'archive_sha256':p['archive_sha256']}
    new(OUT/'manifest.json',{'rows':values,'audit':result});result['manifest_sha256']=sha(OUT/'manifest.json')
    new(ROOT/'reports/industrial-ostrava-v61-data-audit.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','audit'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(path))
        lock=RUN/'operation.lock'
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:audit()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(error),'pid':os.getpid()});raise
        finally:lock.unlink()
