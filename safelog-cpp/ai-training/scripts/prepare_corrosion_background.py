"""Preregister sparse ROI coordinates, reserved TEST, and cached-group exclusions."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.corrosion_background_roi import rectangle,sparse_target,target512
from safelog_ai.corrosion_cached_graph import replay
from safelog_ai.belt_normal_roi import fingerprints,distance
from safelog_ai.process_probe import process_exists
from scripts.prepare_magnetic_tiles import Bands
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

SOURCE=ROOT/'data/industrial-corrosion-background-v75-source'
ARCHIVE=SOURCE/'publisher-labeled-images-v1.zip'
OUT=ROOT/'data/industrial-corrosion-background-v76'
RUN=ROOT/'runs/industrial-corrosion-background-data-v76'
POLICY=ROOT/'reports/industrial-corrosion-background-v76-data-policy.json'
SOURCES=('scripts/prepare_corrosion_background.py','safelog_ai/corrosion_background_roi.py','safelog_ai/corrosion_cached_graph.py','tests/test_corrosion_background_roi.py')


def utc():return datetime.now(timezone.utc).isoformat()


def mapping():
    members=read(SOURCE/'publisher-labeled-members.json')['rows'];rows=[]
    for n in sorted({re.search(r'Imag(\d+)\.tif',r['member']).group(1) for r in members if '/Labels/Corrosion/' in r['member'] and r['member'].endswith('.zip')}):
        linked=[r for r in members if not r['directory'] and re.search(r'Imag'+n+r'(?:-1)?\.tif(?:\.zip)?$',r['member'])]
        choose=lambda token:[r for r in linked if token in r['member']]
        original=choose('/Original images/');pos=choose('/Labels/Corrosion/');neg=choose('/Labels/Background/')
        assert len(original)<=1 and len(pos)==len(neg)==1
        rows.append({'id':'rios/'+n,'stem':n,'original':original[0] if original else None,'positive_zip':pos[0],'negative_zip':neg[0],
                     'all_same_ID_copies':linked,'unavailable_reason':'missing_original83_preregistered_exclusion' if not original else None})
    assert len(rows)==35 and [r['stem'] for r in rows if r['original'] is None]==['83']
    return rows


def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    assert sha(ARCHIVE)=='6fbd29524a81f4d8250a3cf978da09ad09d48f65522677792156d0bdf454fcce'
    rows=mapping();order=sorted(rows,key=lambda r:(hashlib.sha256(('rios76TEST:'+r['id']).encode()).hexdigest(),r['id']));held={r['id'] for r in order[:math.ceil(.2*len(order))]}
    for r in rows:r['reserved_role']='test' if r['id'] in held else 'development_unassigned'
    OUT.mkdir();RUN.mkdir();new(OUT/'reserved-TEST-and-copies-before-pixels.json',{'rows':rows,'reserved_ids':sorted(held),'rule':'firstceil20percentof35authorlabelIDs by SHA256(rios76TEST:+id), tieid. Missing83 excludedbeforepixels and allcopies bound evenifreserved.', 'original_author_split':'No TRAIN_VAL_TEST components in publisher file paths; our reservation is not an author split.'})
    paths=[ARCHIVE,SOURCE/'publisher-labeled-members.json',SOURCE/'archive-audit.json',SOURCE/'imagej-static-receipt.json']+[SOURCE/n for n in ('imagej-RoiDecoder.java.txt','imagej-Roi.java.txt','imagej-ImageProcessor.java.txt')]
    paths += [ROOT/'data/industrial-corrosion-json-pairs-v68/manifest.json',ROOT/'data/industrial-corrosion-json-pairs-v68/new-pixel-links.json',ROOT/'reports/industrial-corrosion-json-pair-v68-data-policy.json']
    for n in read(ROOT/'reports/industrial-corrosion-json-pair-v68-data-policy.json')['inputs_sha256']:
        if n.endswith(('.json','.jsonl')):paths.append(ROOT/n)
    paths=sorted(set(paths))
    p={'UTC':utc(),'fixed_before_new_source_pixels_or_inner_ROI_content':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'inputs_sha256':{n.relative_to(ROOT).as_posix():sha(n) for n in paths},'reserved_sha256':sha(OUT/'reserved-TEST-and-copies-before-pixels.json'),
       'missing83':'Exclude83unavailablebeforepixels; preservebothannotationZIPs/visualization. Do notdownloadfull2.2GB, substitutevisualization, inpaintlabels, infercleanGT orfixoriginalbytes. Extra203-1visualization isbound toID203andnotnewphoto.',
       'nested':'70nestedZIPstotal CRC/encodedSHA/safepaths notpreviouslychecked. Thisstageinspectsavailable34x2 annotationZIPs only; missing83twoZIPs remainunparsed. EachZIP<=1MiB/<=16members/<=8MiBexpanded/eachROI<=1MiB, namesrelative/no traversal-colon-backslash/symlink/encryption/duplicatecasefoldpaths. Require5ROI files perclass, no recursiveZIP/codeexecution/extraction.',
       'coordinate':'Independent parserbasedonImageJofficialrevision4c4975d6df7cf89334e7bc7bb56c48f7b204f244 staticRoiDecoder/Roi. MagicIout, version217..229, type1integerrectangle only, noncomposite/no subtype/subpixel/rounded/coordinatearray. Nativeleft/top inclusive,right/bottom exclusive (width=right-left), no1basedshift/clipping/relocation. Onlystackposition/CZT0or1 forsingleTIFF. Exactlyfive5x5boxes/class, allinbounds, no same/crossclassoverlap. Unknownformat/coordinate/label/mapping stopsandpreservesfailure; neverconvertfailuretonormal.',
       'decode':'Only34originalTIFFs, <=128MiBencoded/40MP/singleframe/modeRGB,L,P oropaqueRGBA/EXIForientationNone1. VerifyencodedSHA, originalpixelRGBhash/phashonlynew34; no suppliedvisualizationpixels. SequentialRAM<512MiBtransient; no wholeexpansion. Header/ROIdecodeeventsdistinctfromRGBdecode andfromvisualcontentview.',
       'GT':'ProvidedCorrosionROI->1/BackgroundROI->0, outsidebothunknown255. Raw5x5patchsupport only, no fullmask/wholehealthyGT. Preservebytesandnativecoords; collect512nearestletterboxcounts/padding255withoutGTdilation. ExactRGBsamecoordinates withconflictingROI rasterblocks.',
       'graph':'Replayonlycompletedv57edges94/Ostrava105/v66views882/v68embedded440 plusalloldroles/held. Verifycanonicalv57rows783/Ostrava105/v66rows440/v68rows440exact. No oldRGBdecode/oldpairdistance. Addnew34RGB nodes, exactRGB+size/8orientationpHashdistance<=2 andknownEXIFcamera+secondproxy; allreservedTEST/copiesandold8995/InsPLAD6417/new6481/Ostrava15/VT44 held transitivelyexcludeTRAIN. SameIDcopiesarebound; numbersarenotphysicalIDs.',
       'allocation':'ReservedTESTIDsimmutable. Afternewjoins, oldsingleFIT/VAL/CALrolelocked; conflictingrolesexclude, heldconnectexclude. Unassignednewgroupcanonicalkeys SHA256(rios76split:+key), firstceil20percentVAL,nextceil20percentCAL,restFIT once. No relocation/reallocation/minimumrelaxation. MissingreservedTEST staysunavailable, notdeleted/relabelled.',
       'minimum_entries':{'fit':10,'validation':3,'cal':3,'test':3},'minimum_groups':{'fit':5,'validation':2,'cal':2,'test':2},
       'minimum_photos_with_both_classes_native':{'fit':10,'validation':3,'cal':3,'test':3},'minimum_native_pixels_each_class_each_phase':375,
       'minimum_512_pixels_each_class_each_phase':16,'minimum_512_photos_with_both_classes_each_phase':3,
       'QA':'Onlyifallgatespass first3FITbyoriginalmember. NativeRGBwithprovided5x5marks andfixedcontextpatches contact sheet; FITonly, notinferencetrainingcrops. NoVAL_CAL_TESTvisualviews/GTedits/expertvalidationclaim.',
       'storage':'MetadataandoneFITcontactsheet<96MiB disk, no newTIFF/mask expansion orencoderfeaturesfiles/parentcopy/deletion; >=1GiBreserve. Sequentialnativebuffers<512MiB, oldcachedfingerprintindexboundedRAM.',
       'training':'No newfeatures/scores/gradient/optimizer; separatelypreregisteroneinput/sampling/model/loss/VAL/CAL/guards/delta/devicecontractonlyafterqualifieddata. Tiny5x5patches mayvanishat512; do notweaken minimaifso. No newappwholephoto20percentpass orfieldindependence.',
       'license':'Rios/Netto/Roehl,MendeleyV1DOI10.17632/kcyn4nhv2c.1,dataCC BY4.0 plusauthor/source/version/changes andidentifiedthirdpartyterms. ImageJofficialcodeonlyreadastext/notexecuted. AllparentVTCC0/OstravaCC BY4.0/InsPLADCC BY-NC3.0 andothertermsunchanged.',
       'new_units_passed':3,'independent_field_validation':False}
    assert shutil.disk_usage(ROOT).free>1024**3+96*1024**2
    new(POLICY,p);print({'policy_sha256':sha(POLICY),'reserved_TEST_ids':sorted(held)},flush=True)


def validate():
    p=read(POLICY)
    for n,h in {**p['source_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/n)==h,n
    assert sha(OUT/'reserved-TEST-and-copies-before-pixels.json')==p['reserved_sha256']
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def boxes(archive,item):
    raw=archive.read(item['member']);assert hashlib.sha256(raw).hexdigest()==item['encoded_sha256'] and len(raw)<=1024**2
    result=[];seen=set()
    with zipfile.ZipFile(BytesIO(raw)) as inner:
        assert len(inner.infolist())<=16 and sum(v.file_size for v in inner.infolist())<=8*1024**2
        for v in inner.infolist():
            path=PurePosixPath(v.filename);assert not path.is_absolute() and '..' not in path.parts and ':' not in v.filename and '\\' not in v.filename and not stat.S_ISLNK(v.external_attr>>16) and not v.flag_bits&1
            assert v.filename.casefold() not in seen;seen.add(v.filename.casefold())
            if v.is_dir():continue
            assert v.filename.lower().endswith('.roi') and v.file_size<=1024**2
            encoded=inner.read(v);result.append({**rectangle(encoded),'member':v.filename,'encoded_sha256':hashlib.sha256(encoded).hexdigest(),'bytes':len(encoded),'CRC32':f'{v.CRC:08x}'})
    assert len(result)==5
    return result


def decode():
    rows=[]
    with zipfile.ZipFile(ARCHIVE) as archive,(RUN/'decode-events.jsonl').open('x',encoding='utf-8',buffering=1) as events:
        for r in read(OUT/'reserved-TEST-and-copies-before-pixels.json')['rows']:
            if not r['original']:continue
            positives=boxes(archive,r['positive_zip']);negatives=boxes(archive,r['negative_zip'])
            events.write(json.dumps({'UTC':utc(),'id':r['id'],'nested_ZIP_and_ROI_files_checked':10,'RGB_decoded':False})+'\n')
            encoded=archive.read(r['original']['member']);assert hashlib.sha256(encoded).hexdigest()==r['original']['encoded_sha256'] and len(encoded)<=128*1024**2
            with Image.open(BytesIO(encoded)) as im:
                assert im.format=='TIFF' and im.n_frames==1 and im.width*im.height<=40_000_000 and im.mode in ('RGB','L','P','RGBA')
                exif=im.getexif();assert exif.get(274) in (None,1)
                if im.mode=='RGBA':assert im.getchannel('A').getextrema()==(255,255)
                rgb=im.convert('RGB');camera=exif.get(272);stamp=exif.get(36867) or exif.get(306)
            native=sparse_target(rgb.size,positives,negatives);small,geometry=target512(native)
            capture=str(camera)+'|'+stamp if camera and isinstance(stamp,str) and re.fullmatch(r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}',stamp) else None
            value={**r,'size':list(rgb.size),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'orientation_phashes':fingerprints(rgb),'capture_second_proxy':capture,'positive_ROIs':positives,'negative_ROIs':negatives,'native_sparse_SHA256':hashlib.sha256(native.tobytes()).hexdigest(),'target512_SHA256':hashlib.sha256(small.tobytes()).hexdigest(),'native_pixels':{str(k):int((native==k).sum()) for k in (0,1,255)},'pixels512':{str(k):int((small==k).sum()) for k in (0,1,255)},'geometry':geometry,'archive':ARCHIVE.relative_to(ROOT).as_posix()}
            new(OUT/('decoded-'+r['stem']+'.json'),value);rows.append(value)
            events.write(json.dumps({'UTC':utc(),'id':r['id'],'new_RGB_view_decoded':True,'original_size':list(rgb.size),'visual_content_viewed':False})+'\n')
    new(OUT/'decoded-pairs.json',{'rows':rows});return rows


def join(rows):
    old,keys,g,held,role_entries,capture=replay(ROOT);offset=len(old);assert offset==49567
    g.parents.extend(range(offset,offset+len(rows)));keys+=['rios76:'+r['id'] for r in rows]
    index=Bands();exact=defaultdict(list);links=[];counts=Counter()
    for i,r in enumerate(old):index.add(r['orientation_phashes'],i);exact[(r['rgb_sha256'],tuple(r['size']))].append(i)
    allrows=old+rows
    for i,r in enumerate(rows,offset):
        key=(r['rgb_sha256'],tuple(r['size']))
        for j in exact.get(key,()):
            if g.find(i)!=g.find(j):g.union(i,j);links.append([i,j,'RGB_exact']);counts['exact_joins']+=1
        for j in sorted(index.candidates(r['orientation_phashes'])):
            if g.find(i)==g.find(j):continue
            counts['new_cross_component_pHash_comparisons']+=1;delta=distance(r['orientation_phashes'],allrows[j]['orientation_phashes'])
            if delta<=2:g.union(i,j);links.append([i,j,'pHash',delta]);counts['pHash_joins']+=1
        stamp=r['capture_second_proxy']
        if stamp:
            if stamp in capture:g.union(i,capture[stamp]);links.append([i,capture[stamp],'capture_second_proxy'])
            else:capture[stamp]=i
        index.add(r['orientation_phashes'],i);exact[key].append(i)
    held += [offset+i for i,r in enumerate(rows) if r['reserved_role']=='test'];heldroots={g.find(i) for i in held};roles=defaultdict(set);canon={}
    for i,role in role_entries:roles[g.find(i)].add(role)
    for i,key in enumerate(keys):root=g.find(i);canon[root]=min(canon.get(root,key),key)
    phases={};unlocked=[]
    for i,r in enumerate(rows,offset):
        root=g.find(i)
        if r['reserved_role']=='test':continue
        if root in heldroots:phases[root]='excluded_TEST_connected'
        elif len(roles[root])>1:phases[root]='excluded_prior_role_conflict'
        elif roles[root]:phases[root]=next(iter(roles[root]))
        else:unlocked.append(root)
    ordered=sorted(set(unlocked),key=lambda k:(hashlib.sha256(('rios76split:'+canon[k]).encode()).hexdigest(),canon[k]));n=math.ceil(.2*len(ordered))
    for j,k in enumerate(ordered):phases[k]='validation' if j<n else 'cal' if j<2*n else 'fit'
    result=[{**r,'group_id':canon[g.find(offset+i)],'phase':'test' if r['reserved_role']=='test' else phases[g.find(offset+i)],'prior_roles':sorted(roles[g.find(offset+i)])} for i,r in enumerate(rows)]
    conflicts=[]
    for indices in exact.values():
        own=[j-offset for j in indices if j>=offset]
        if len({rows[j]['native_sparse_SHA256'] for j in own})>1:conflicts.append([rows[j]['id'] for j in own])
    new(OUT/'new-pixel-links.json',{'rows':links,'counts':dict(counts),'old_canonical_groups_replayed_exact':True,'old_pair_distance_recalculated':0})
    return result,conflicts,counts


def audit():
    p=validate();decoded=decode();rows,conflicts,distances=join(decoded);failures=[];summary={}
    if conflicts:failures.append({'kind':'same_RGB_conflicting_sparse_GT','groups':conflicts})
    for phase,minimum in p['minimum_entries'].items():
        selected=[r for r in rows if r['phase']==phase];groups=len({r['group_id'] for r in selected});bothnative=sum(all(r['native_pixels'][str(k)]>0 for k in (0,1)) for r in selected);both512=sum(all(r['pixels512'][str(k)]>0 for k in (0,1)) for r in selected)
        native={str(k):sum(r['native_pixels'][str(k)] for r in selected) for k in (0,1,255)};small={str(k):sum(r['pixels512'][str(k)] for r in selected) for k in (0,1,255)}
        summary[phase]={'photos':len(selected),'groups':groups,'photos_both_native_classes':bothnative,'photos_both_512_classes':both512,'native_pixels':native,'pixels512':small}
        if len(selected)<minimum or groups<p['minimum_groups'][phase] or bothnative<p['minimum_photos_with_both_classes_native'][phase]:failures.append({'kind':'photo_group_minimum','phase':phase})
        if min(native['0'],native['1'])<375:failures.append({'kind':'native_valid_support_minimum','phase':phase})
        if min(small['0'],small['1'])<16 or both512<3:failures.append({'kind':'512_valid_support_minimum','phase':phase})
    for a in summary:
        for b in summary:
            if a<b:assert {r['group_id'] for r in rows if r['phase']==a}.isdisjoint({r['group_id'] for r in rows if r['phase']==b})
    reserved=read(OUT/'reserved-TEST-and-copies-before-pixels.json');missing=[r for r in reserved['rows'] if not r['original']]
    assert sorted([r['id'] for r in rows if r['phase']=='test']+[r['id'] for r in missing if r['reserved_role']=='test'])==reserved['reserved_ids']
    result={'UTC':utc(),'ready':not failures,'policy_sha256':sha(POLICY),'counts':summary,'failures':failures,'exclusions':dict(Counter(r['phase'] for r in rows if r['phase'].startswith('excluded'))),'missing_originals':[r['id'] for r in missing],'decoded_original_RGB':len(decoded),'nested_ZIPs_checked':2*len(decoded),'ROI_files_checked':10*len(decoded),'new_distance_audit':dict(distances),'new_visual_content_views':0,'new_features_scores_gradient_optimizer':0,'training_protocol_fixed':False,'wholephoto_app_gate_passed':False,'physical_part_independence_verified':False,'independent_field_validation':False,'new_units_passed':3,'disk_free_bytes':shutil.disk_usage(ROOT).free}
    new(OUT/'manifest.json',{'rows':rows,'missing_originals':missing,'audit':result});new(ROOT/'reports/industrial-corrosion-background-v76-data-audit.json',{**result,'manifest_sha256':sha(OUT/'manifest.json')});print(result,flush=True)


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
