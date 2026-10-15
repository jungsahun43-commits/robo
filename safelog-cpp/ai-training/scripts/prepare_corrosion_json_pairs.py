"""Preregister and audit author embedded-image/polygon pairs, preserving v66 failure."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.corrosion_json_pairs import embedded_image,polygon_labels,locked_phase
from safelog_ai.insplad_pixel_groups import Groups
from safelog_ai.belt_normal_roi import fingerprints,distance
from safelog_ai.process_probe import process_exists
from scripts.prepare_magnetic_tiles import Bands
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts import prepare_corrosion_condition_pixels as prior

SOURCE=prior.SOURCE
OUT=ROOT/'data/industrial-corrosion-json-pairs-v68'
RUN=ROOT/'runs/industrial-corrosion-json-pair-data-v68'
POLICY=ROOT/'reports/industrial-corrosion-json-pair-v68-data-policy.json'
SOURCES=('scripts/prepare_corrosion_json_pairs.py','safelog_ai/corrosion_json_pairs.py','tests/test_corrosion_json_pairs.py')


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    assert sha(prior.OUT/'manifest.json')=='b4be1aa9405e4ab9982907557e7a20327ffbd37642fa5376dc3bbf3fec90b0c7'
    assert not read(prior.OUT/'manifest.json')['audit']['ready']
    paths=[ROOT/n for n in read(prior.RECOVERY2)['inputs_sha256']]
    paths += [prior.OUT/'manifest.json',prior.OUT/'new-pixel-links.json',prior.RECOVERY2,
              ROOT/'scripts/prepare_corrosion_condition_pixels.py',ROOT/'safelog_ai/corrosion_condition_data.py',
              ROOT/'safelog_ai/insplad_pixel_groups.py',ROOT/'safelog_ai/belt_normal_roi.py',ROOT/'scripts/prepare_magnetic_tiles.py']
    paths += [SOURCE/('labelme-v4.5.2-'+n+'-static.py.txt') for n in ('shape','label_file','image')]
    OUT.mkdir();RUN.mkdir()
    p={'UTC':utc(),'fixed_before_embedded_imageData_base64_or_pixel_decode':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'inputs_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in paths},
       'author_contract':'Pinned author labelme2voc uses LabelFile.imageData + original shapes. OfficialLabelMe4.5.2 static sources were read, not executed. Independent ordered polygon Lcanvas implementation; class ids0Good/1Fair/2Poor/3Severe/255ignore. Not a repair/relabel of provided native PNG, and v66 strict-native contract remainsFAILED.',
       'imageData':'Required string, originalJSON<=32MiB; strip onlyASCII whitespace then strictASCIIbase64; compact<=4ceil(32MiB/3), raw<=32MiB. ZIPmember/SHA verified. SingleframeJPEG/PNG, RGB/L or opaqueRGBA, <=40MP, EXIForientationNone1, actualsize must equal JSONwidth/height. No imagePath fallback, EXIF rotation, missingdata guess or copied nativeimage substitution.',
       'polygon':'Original exact labelstrings/order; shape_type absent/None/polygon only. >=3 finitepoints, <=100000perpolygon/1milliontotal, abscoordinate<=1million; allow outofcanvas vertices via official PIL raster clipping, do not move/clamp/scale vertices. Record them. Unknownshape/label/nonfinite/mismatchedsize/decode failure stops before learning and preserves logs.',
       'reuse':'Replay cachedv57 original48140nodes/94edges, Ostrava105 plus savedlinks/captureproxies, thenv66cached882views/explicitentrylinks/savedpixeljoins/captureproxies. Verify prior783/105/440 canonical IDs exact. No old RGBdecode or oldpairdistance rerun. Add only440newembedded views, explicitsameentry linkage, thenRGBexact+size or8orientationpHash<=2 againstoldandnew.',
       'held':'OriginalTEST44/220files immutable; reservedJSONSHA also reservesimageData. AddnewembeddedTEST to existing8995held+InsPLAD6417publisherVAL+6481unsupervisedTEST+Ostrava15TEST+allv66TESTcopies. Exclude every currentTEST-connected TRAIN transitively.',
       'roles':'Lock v66FIT276/VAL60/CAL60 plus priorv45/v48/v51/v54/v57lower/v61Ostrava one developmentrole. Conflictingnewgroups excluded; no reallocation/roletransfer/minimumrelaxation. OriginalTEST rows preserved regardless links.',
       'minimum_entries':{'fit':40,'validation':10,'cal':10,'test':20},
       'minimum_groups':{'fit':5,'validation':3,'cal':3,'test':3},
       'minimum_photos_each_class_0_1_2_3':{'fit':20,'validation':5,'cal':5,'test':5},
       'minimum_pixels_each_class_each_phase':1024,
       'conflict':'Exact embedded RGB+same size with different polygon class rasterSHA blocks study, no guessed GT merge or oneannotation selection. Native PNG versus new polygon raster is diagnostic only under this explicitly different authorJSON contract; never claim originalv66 exact-native gate passed.',
       'QA':'Only after allgates pass first3FIT sortedsourceJSONmember fixed; render embeddedRGB/4classoverlay contact sheet. No chosenVAL/CAL/TEST contentviewing, no manualGT corrections. AIvisualQA not expert verification.',
       'semantics':'Good includes no OR minimalcorrosion, notbinaryzero-rust. Sourcepolygonappearance4classes, no measured depth/engineeringgrade/wholemachinehealth. Wholehealthy-photoGT absent => no wholephotoFPR/newapp20percentpass claim.',
       'storage':'Sequential ZIPtoRAM, metadata+oneFITcontactsheet<=96MiB disk, <=1GiB transientnativebuffers, >=1GiBdiskreserve. No embeddedimage/mask wholeexpansion, denseencoder cache, fullparentcopy or originaldeletion.',
       'training':'No new encoderfeatures/scores/gradient/optimizer in thisstage. One later4classbranch protocol must be separatelyfixed/committed beforefeatures; v16/approvedthreeheads/researchv62 unchanged.',
       'license':'Bianchi/Hebdon VirginiaTechdatasetV2 dataCC0, cite10.7294/16624663.v2 andpaper10.1061/(ASCE)CP.1943-5487.0001045 plusauthors/source/version/changes. SourceMIT separate; olderdata terms unchanged.',
       'new_units_passed':3,'physical_capture_or_part_independence_verified':False,'independent_field_validation':False}
    assert shutil.disk_usage(ROOT).free>1024**3+96*1024**2
    new(POLICY,p);print({'policy_sha256':sha(POLICY)},flush=True)


def validate():
    p=read(POLICY)
    for n,h in {**p['source_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def decode():
    result=[];old=read(prior.OUT/'manifest.json')['rows']
    with zipfile.ZipFile(SOURCE/'publisher-ccs-v2.zip') as archive,(RUN/'decode-events.jsonl').open('x',encoding='utf-8',buffering=1) as events:
        for r in old:
            item=r['members']['original/json'];assert item['bytes']<=32*1024**2
            rawjson=archive.read(item['member']);assert hashlib.sha256(rawjson).hexdigest()==item['encoded_sha256']
            a=json.loads(rawjson);rgb,raw,meta=embedded_image(a);mask,geometry=polygon_labels(a,rgb.size)
            classes,counts=np.unique(mask,return_counts=True)
            stamp=meta['capture_time'];camera=meta['camera_model']
            capture=str(camera)+'|'+stamp if camera and isinstance(stamp,str) and re.fullmatch(r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}',stamp) else None
            value={'id':r['id'],'publisher_role':r['publisher_role'],'stem':r['stem'],'v66_group_id':r['group_id'],'v66_phase':r['phase'],
                   'source_json_member':item['member'],'source_json_encoded_sha256':item['encoded_sha256'],
                   'archive':(SOURCE/'publisher-ccs-v2.zip').relative_to(ROOT).as_posix(),'storage':'JSONimageData+originalpolygons',
                   'size':list(rgb.size),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'orientation_phashes':fingerprints(rgb),
                   'embedded_encoded_sha256':hashlib.sha256(raw).hexdigest(),'class_raster_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
                   'class_pixels':dict(zip(map(str,classes.tolist()),counts.tolist())),'geometry':geometry,'embedded_metadata':meta,
                   'capture_second_proxy':capture,'JSON_version':a.get('version'),'raw_labels':dict(Counter(s['label'] for s in a['shapes'])),
                   'same_size_RGB_exact_to_supplied_original':list(rgb.size)==r['views'][0]['size'] and hashlib.sha256(rgb.tobytes()).hexdigest()==r['views'][0]['rgb_sha256'],
                   'new_raster_matches_preserved_nativePNG_class_sha256':hashlib.sha256(mask.tobytes()).hexdigest()==r['native_mask']['class_sha256'],
                   'annotation_scope':'author supplied polygon classes; implicitGood/background includes minimalcorrosion; notwholehealth'}
            result.append(value);new(OUT/('decoded-'+r['publisher_role']+'-'+r['stem']+'.json'),value)
            events.write(json.dumps({'UTC':utc(),'id':r['id'],'embedded_RGB_and_polygon_raster':True})+'\n')
    new(OUT/'decoded-pairs.json',{'rows':result});return result


def join_graph(rows):
    unsup=[]
    for path in sorted(prior.V57.glob('fingerprints-*.json')):
        if not path.name.endswith('-sha.json'):unsup.extend(read(path)['rows'])
    old=read(prior.OLD/'grouped-all-source.json')['rows'];held=read(prior.OLD/'retained-TEST-fingerprints.json')['rows'];ost=read(prior.OST/'decoded-pairs.json')['rows'];ccs=read(prior.OUT/'manifest.json')['rows']
    assert (len(unsup),len(old),len(held),len(ost),len(ccs))==(27620,11525,8995,105,440)
    assert [r['member'] for r in unsup]==[r['member'] for r in read(prior.UNSUP)['rows']]
    views=[v for r in ccs for v in r['views']];assert len(views)==882
    a,b=len(unsup),len(unsup)+len(old);c=b+len(held);d=c+len(ost);e=d+len(views)
    allrows=unsup+old+held+ost+views+rows;g=Groups(len(allrows));oldgroups={};metagroups={}
    for j,r in enumerate(old):
        if r['group_id'] in oldgroups:g.union(a+j,oldgroups[r['group_id']])
        else:oldgroups[r['group_id']]=a+j
    for i,r in enumerate(unsup):
        if r['group_proxy'] in metagroups:g.union(i,metagroups[r['group_proxy']])
        else:metagroups[r['group_proxy']]=i
        for key in r['old_source_group_links']:g.union(i,oldgroups[key])
    edges=[json.loads(v) for v in (prior.V57/'pixel-joining-edges.jsonl').read_text(encoding='utf-8').splitlines()];assert len(edges)==94
    for i,j,*_ in edges:g.union(i,j)
    keys=['unsupervised:'+r['member'] for r in unsup]+['supervised:'+r['member'] for r in old]+['held:'+r['image'] for r in held]
    def canonical():
        result={}
        for i,key in enumerate(keys):root=g.find(i);result[root]=min(result.get(root,key),key)
        return result
    canon=canonical();um={r['member']:i for i,r in enumerate(unsup)}
    for r in read(prior.V57/'manifest.json')['rows']:assert canon[g.find(um[r['member']])]==r['pixel_group_id']
    keys+=['ostrava:'+r['member'] for r in ost];capture={}
    for i,j,*_ in read(prior.OST/'new-pixel-links.json')['rows']:g.union(i,j)
    for i,r in enumerate(ost,c):
        stamp=r['capture_second_proxy']
        if stamp:
            if stamp in capture:g.union(i,capture[stamp])
            else:capture[stamp]=i
    canon=canonical();om={r['member']:c+i for i,r in enumerate(ost)}
    for r in read(prior.OST/'manifest.json')['rows']:assert canon[g.find(om[r['member']])]==r['group_id']
    keys+=['ccs66:'+v['id'] for v in views];entry={}
    for i,v in enumerate(views,d):
        if v['entry_id'] in entry:g.union(i,entry[v['entry_id']])
        else:entry[v['entry_id']]=i
    for i,j,*_ in read(prior.OUT/'new-pixel-links.json')['rows']:g.union(i,j)
    for i,r in enumerate(views,d):
        stamp=r['capture_second_proxy'] if r['variant']=='original/images' else None
        if stamp:
            if stamp in capture:g.union(i,capture[stamp])
            else:capture[stamp]=i
    canon=canonical()
    for r in ccs:assert canon[g.find(entry[r['id']])]==r['group_id']
    keys+=['ccs68:'+r['id']+'/embedded' for r in rows]
    for i,r in enumerate(rows,e):g.union(i,entry[r['id']])
    index=Bands();exact=defaultdict(list);counts=Counter();links=[]
    for i,r in enumerate(allrows[:e]):index.add(r['orientation_phashes'],i);exact[(r['rgb_sha256'],tuple(r['size']))].append(i)
    for i,r in enumerate(rows,e):
        key=(r['rgb_sha256'],tuple(r['size']))
        for j in exact.get(key,()):
            if g.find(i)!=g.find(j):g.union(i,j);links.append([i,j,'RGB_exact']);counts['RGB_exact_joins']+=1
        for j in sorted(index.candidates(r['orientation_phashes'])):
            if g.find(i)==g.find(j):continue
            counts['new_cross_component_pHash_comparisons']+=1;delta=distance(r['orientation_phashes'],allrows[j]['orientation_phashes'])
            if delta<=2:g.union(i,j);links.append([i,j,'pHash',delta]);counts['pHash_joins']+=1
        stamp=r['capture_second_proxy']
        if stamp:
            if stamp in capture:g.union(i,capture[stamp]);counts['capture_second_joins']+=1
            else:capture[stamp]=i
        index.add(r['orientation_phashes'],i);exact[key].append(i)
    canon=canonical();heldindices=[i for i,r in enumerate(unsup) if r['publisher_role']=='test']+[a+j for j,r in enumerate(old) if r['publisher_role']=='val']+list(range(b,c))
    heldindices += [om[r['member']] for r in read(prior.OST/'manifest.json')['rows'] if r['phase']=='test']
    heldindices += [entry[r['id']] for r in ccs if r['publisher_role']=='Test']+[e+i for i,r in enumerate(rows) if r['publisher_role']=='Test']
    heldroots={g.find(i) for i in heldindices};roles=defaultdict(set)
    for name in prior.PRIOR:
        for r in read(ROOT/name)['rows']:
            if r['phase'] in ('fit','validation','cal'):roles[g.find(oldgroups[r['group_id']])].add(r['phase'])
    for path,mapping,key in [(prior.V57/'manifest.json',um,'member'),(prior.OST/'manifest.json',om,'member')]:
        for r in read(path)['rows']:
            if r['phase'] in ('fit','validation','cal'):roles[g.find(mapping[r[key]])].add(r['phase'])
    for r in ccs:
        if r['phase'] in ('fit','validation','cal'):roles[g.find(entry[r['id']])].add(r['phase'])
    result=[]
    for i,r in enumerate(rows,e):
        root=g.find(i);phase=locked_phase(r['publisher_role'],root in heldroots,roles[root])
        result.append({**r,'group_id':canon[root],'phase':phase,'prior_development_roles':sorted(roles[root])})
    conflicts=[]
    for ids in exact.values():
        own=[j-e for j in ids if j>=e]
        if len({rows[j]['class_raster_sha256'] for j in own})>1:conflicts.append([rows[j]['id'] for j in own])
    new(OUT/'new-pixel-links.json',{'rows':links,'counts':dict(counts),'explicit_sameentry_links':440})
    return result,counts,conflicts


def audit():
    p=validate();assert not (OUT/'decoded-pairs.json').exists() and not (OUT/'manifest.json').exists()
    decoded=decode();rows,counts,conflicts=join_graph(decoded);failures=[];summary={}
    if conflicts:failures.append({'kind':'exact_embedded_RGB_conflicting_polygons','groups':conflicts})
    for phase,minimum in p['minimum_entries'].items():
        chosen=[r for r in rows if r['phase']==phase];groups=len({r['group_id'] for r in chosen});pixel=Counter()
        for r in chosen:pixel.update(r['class_pixels'])
        photos={str(k):sum(r['class_pixels'].get(str(k),0)>0 for r in chosen) for k in range(4)}
        summary[phase]={'entries':len(chosen),'groups':groups,'photos_per_class':photos,'class_pixels':dict(pixel)}
        if len(chosen)<minimum:failures.append({'kind':'entry_minimum','phase':phase,'actual':len(chosen),'minimum':minimum})
        if groups<p['minimum_groups'][phase]:failures.append({'kind':'group_minimum','phase':phase,'actual':groups})
        for k in range(4):
            if photos[str(k)]<p['minimum_photos_each_class_0_1_2_3'][phase]:failures.append({'kind':'class_photo_minimum','phase':phase,'class':k,'actual':photos[str(k)]})
            if pixel[str(k)]<p['minimum_pixels_each_class_each_phase']:failures.append({'kind':'class_pixel_minimum','phase':phase,'class':k,'actual':pixel[str(k)]})
    for x in summary:
        for y in summary:
            if x<y:assert {r['group_id'] for r in rows if r['phase']==x}.isdisjoint({r['group_id'] for r in rows if r['phase']==y})
    expected=[r['id'] for r in read(prior.META/'reserved-original-and-resized-TEST-before-pixels.json')['rows']]
    assert [r['id'] for r in rows if r['phase']=='test']==expected
    audit={'UTC':utc(),'ready':not failures,'policy_sha256':sha(POLICY),'counts':summary,'failures':failures,
           'exclusions':dict(Counter(r['phase'] for r in rows if r['phase'].startswith('excluded'))),'new_pixel_joins':dict(counts),
           'canonical_v57_783_Ostrava105_v66_440_exact':True,'new_embedded_RGB_views':440,'new_polygon_rasters':440,
           'same_size_RGB_exact_to_supplied_original':sum(r['same_size_RGB_exact_to_supplied_original'] for r in rows),
           'new_raster_exact_to_preserved_nativePNG':sum(r['new_raster_matches_preserved_nativePNG_class_sha256'] for r in rows),
           'outside_canvas_vertex_photos':sum(r['geometry']['outside_canvas_vertices_clipped_by_rasterizer_not_moved']>0 for r in rows),
           'original_TEST44_and220files_preserved':True,'v66_native_contract_still_failed':True,
           'new_source_visual_content_views':0,'new_encoder_features_scores_gradient_optimizer':0,'training_protocol_fixed':False,
           'physical_part_capture_independence_verified':False,'independent_field_validation':False,
           'wholehealthy_photoGT':0,'wholephoto_app20percent_gate_passed':False,'new_units_passed':3,'disk_free_bytes':shutil.disk_usage(ROOT).free}
    new(OUT/'manifest.json',{'rows':rows,'audit':audit});audit['manifest_sha256']=sha(OUT/'manifest.json')
    new(ROOT/'reports/industrial-corrosion-json-pair-v68-data-audit.json',audit);print(audit,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','audit'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(path))
        with (RUN/'operation.lock').open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
        try:audit()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(error),'pid':os.getpid()});raise
        finally:(RUN/'operation.lock').unlink()
