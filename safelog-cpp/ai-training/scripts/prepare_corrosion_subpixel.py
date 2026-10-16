"""Separate direct-file coordinate contract, preserving the failed v76 description contract."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import json
import os
import re
from pathlib import Path,PurePosixPath
import shutil
import stat
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.corrosion_subpixel_roi import subpixel_rectangle,provided_target
from safelog_ai.corrosion_background_roi import target512
from safelog_ai.belt_normal_roi import fingerprints
from safelog_ai.process_probe import process_exists
from scripts import prepare_corrosion_background as prior
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

OUT=ROOT/'data/industrial-corrosion-subpixel-v78'
RUN=ROOT/'runs/industrial-corrosion-subpixel-data-v78'
POLICY=ROOT/'reports/industrial-corrosion-subpixel-v78-data-policy.json'
SOURCES=('scripts/prepare_corrosion_subpixel.py','safelog_ai/corrosion_subpixel_roi.py','tests/test_corrosion_subpixel_roi.py')
OLD_OUT=prior.OUT
HEADERS=ROOT/'runs/industrial-corrosion-roi-format-v77/ROI-headers.json'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    d=read(ROOT/'runs/industrial-corrosion-roi-format-v77/diagnosis.json');assert d['v76_integer_contract_still_failed'] and d['ROI_headers']==703 and d['RGB_pixels_loaded']==0
    OUT.mkdir();RUN.mkdir();original=OLD_OUT/'reserved-TEST-and-copies-before-pixels.json';(OUT/original.name).write_bytes(original.read_bytes())
    p=read(prior.POLICY)
    files=[ROOT/n for n in p['inputs_sha256']]+[prior.POLICY,prior.RECOVERY,HEADERS,ROOT/'runs/industrial-corrosion-roi-format-v77/TIFF-headers.json',ROOT/'runs/industrial-corrosion-roi-format-v77/diagnosis.json',ROOT/'safelog_ai/corrosion_cached_graph.py',ROOT/'scripts/prepare_corrosion_background.py']
    value={'UTC':utc(),'fixed_before_new_RGB_pixel_decode_or_direct_ROI_raster':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'inputs_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in sorted(set(files))},'reservation_sha256':sha(original),
       'different_contract':'Use all actualprovidedsubpixelROI geometry, not publisherdescriptionfive5x5patch conversion. v76strictintegerfive5x5contract remainsFAILED, notretrospectivelyqualified. Actualheaders703version227/type1/subpix128,usually20x20/count9..12 perclassZIP. Explanationofdifferencefromwebdescription unresolved; no guessedscale4/resize/coordinatefix.',
       'coordinates':'PinnedImageJ4c4975d6df7cf89334e7bc7bb56c48f7b204f244 RoiDecoder readsfloat32xywh forversion>=223+SUB_PIXEL. Roi(double...corner) castsx/ytoint andceilwidth/height; getBounds/contains integerrectangle halfopen. Independentlyimplementexactcast/ceil, preservefloatmetadata; do notsnap20.0000038to20, shift1basedcoords, rescale/dilate orclipinvalidbounds. Unsupportedshape/frame/nonfinite/negative/outofbounds/oppositeclassconflict stops, keepsfailure, no GTcorrection.',
       'GT':'Corrosionfolderprovidedrectspositive1,Backgroundfoldernegative0, allotherpixelsunknown255. Useallprovidedrectangles>=5/<=16 eachclass; sameclassoverlapunioncountedonce, crossclassoverlapblocks. Filefolderlabelsnotwholehealthyphoto/entirebinarymask. Broad/weaksourceannotationsarenotexpertfieldverified.',
       'reuse':'Usev77nestedROIheaderSHA/CRC inventory andavailable34TIFFIFDheaders. Re-read encodedROIbytes toverifyandderiveactualprovidedmask; no oldRGBdecode/pHashaudit. Firstactual34originalRGB pixeldecodes now. Sourcevisualizationneverinput/GT. Missing83excludedunchanged; no2.2GBdownload; identicalv76reserved7TESTIDs/allcopies, roles/splitrule/entry-group-native-512minimumsunchanged.',
       'graph_allocation':p['graph']+' '+p['allocation'],'minimum_entries':p['minimum_entries'],'minimum_groups':p['minimum_groups'],'minimum_photos_with_both_classes_native':p['minimum_photos_with_both_classes_native'],'minimum_native_pixels_each_class_each_phase':375,'minimum_512_pixels_each_class_each_phase':16,'minimum_512_photos_with_both_classes_each_phase':3,
       'decode':p['decode'],'QA':p['QA'],'storage':p['storage'],'training':p['training'],'license':p['license'],'new_units_passed':3,'independent_field_validation':False}
    assert shutil.disk_usage(ROOT).free>=1024**3+96*1024**2
    new(POLICY,value);print({'policy_sha256':sha(POLICY),'reservation_unchanged':True},flush=True)


def validate():
    p=read(POLICY)
    for n,h in {**p['source_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/n)==h,n
    assert sha(OUT/'reserved-TEST-and-copies-before-pixels.json')==p['reservation_sha256']
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def provided_boxes(archive,item,expected):
    raw=archive.read(item['member']);assert hashlib.sha256(raw).hexdigest()==item['encoded_SHA256'];result=[]
    with zipfile.ZipFile(BytesIO(raw)) as inner:
        for v in inner.infolist():
            if v.is_dir():continue
            encoded=inner.read(v);key=(item['member'],v.filename);assert hashlib.sha256(encoded).hexdigest()==expected[key]['SHA256']
            result.append({**subpixel_rectangle(encoded),'member':v.filename,'encoded_sha256':expected[key]['SHA256']})
    return result


def decode():
    expected={(r['outer_member'],r['roi_member']):r for r in read(HEADERS)['rows']};rows=[]
    headers={r['member']:r for r in read(ROOT/'runs/industrial-corrosion-roi-format-v77/TIFF-headers.json')['rows']}
    with zipfile.ZipFile(prior.ARCHIVE) as archive,(RUN/'decode-events.jsonl').open('x',encoding='utf-8',buffering=1) as events:
        for r in read(OUT/'reserved-TEST-and-copies-before-pixels.json')['rows']:
            if not r['original']:continue
            pos=provided_boxes(archive,r['positive_zip'],expected);neg=provided_boxes(archive,r['negative_zip'],expected)
            raw=archive.read(r['original']['member']);assert hashlib.sha256(raw).hexdigest()==r['original']['encoded_SHA256'] and len(raw)<=128*1024**2
            with Image.open(BytesIO(raw)) as im:
                assert im.format=='TIFF' and im.n_frames==1 and list(im.size)==headers[r['original']['member']]['size'] and im.width*im.height<=40_000_000 and im.mode=='RGB' and im.getexif().get(274) in (None,1)
                exif=im.getexif();camera=exif.get(272);stamp=exif.get(36867) or exif.get(306);rgb=im.convert('RGB')
            events.write(json.dumps({'UTC':utc(),'id':r['id'],'first_actual_RGB_decode':True,'not_new_natural_photo_from_ROIcount':True})+'\n')
            mask,overlap=provided_target(rgb.size,pos,neg);small,g=target512(mask)
            capture=str(camera)+'|'+stamp if camera and isinstance(stamp,str) and re.fullmatch(r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}',stamp) else None
            value={**r,'size':list(rgb.size),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'orientation_phashes':fingerprints(rgb),'capture_second_proxy':capture,
                   'positive_ROIs':pos,'negative_ROIs':neg,'native_sparse_SHA256':hashlib.sha256(mask.tobytes()).hexdigest(),'target512_SHA256':hashlib.sha256(small.tobytes()).hexdigest(),'sameclass_overlap_pixels':overlap,
                   'native_pixels':{str(k):int((mask==k).sum()) for k in (0,1,255)},'pixels512':{str(k):int((small==k).sum()) for k in (0,1,255)},'geometry':g,'archive':prior.ARCHIVE.relative_to(ROOT).as_posix()}
            new(OUT/('decoded-'+r['stem']+'.json'),value);rows.append(value)
    new(OUT/'decoded-pairs.json',{'rows':rows});return rows


def audit():
    p=validate();decoded=decode();prior.OUT=OUT;prior.RUN=RUN;rows,conflicts,counts=prior.join(decoded);failures=[];summary={}
    if conflicts:failures.append({'kind':'same_RGB_conflicting_sparse_GT','groups':conflicts})
    for phase,minimum in p['minimum_entries'].items():
        selected=[r for r in rows if r['phase']==phase];groups=len({r['group_id'] for r in selected});both=sum(all(r['native_pixels'][str(k)]>0 for k in (0,1)) for r in selected);both512=sum(all(r['pixels512'][str(k)]>0 for k in (0,1)) for r in selected)
        native={str(k):sum(r['native_pixels'][str(k)] for r in selected) for k in (0,1,255)};small={str(k):sum(r['pixels512'][str(k)] for r in selected) for k in (0,1,255)}
        summary[phase]={'photos':len(selected),'groups':groups,'photos_both_native_classes':both,'photos_both_512_classes':both512,'native_pixels':native,'pixels512':small}
        if len(selected)<minimum or groups<p['minimum_groups'][phase] or both<p['minimum_photos_with_both_classes_native'][phase]:failures.append({'kind':'photo_group_minimum','phase':phase})
        if min(native['0'],native['1'])<375:failures.append({'kind':'native_valid_support_minimum','phase':phase})
        if min(small['0'],small['1'])<16 or both512<3:failures.append({'kind':'512_valid_support_minimum','phase':phase})
    for a in summary:
        for b in summary:
            if a<b:assert {r['group_id'] for r in rows if r['phase']==a}.isdisjoint({r['group_id'] for r in rows if r['phase']==b})
    reserved=read(OUT/'reserved-TEST-and-copies-before-pixels.json');assert sorted(r['id'] for r in rows if r['phase']=='test')==reserved['reserved_ids']
    result={'UTC':utc(),'ready':not failures,'policy_sha256':sha(POLICY),'counts':summary,'failures':failures,'exclusions':dict(Counter(r['phase'] for r in rows if r['phase'].startswith('excluded'))),'missing_originals':['rios/83'],'decoded_original_RGB':len(decoded),'provided_ROI_geometry_reads':sum(len(r['positive_ROIs'])+len(r['negative_ROIs']) for r in rows),'new_distance_audit':dict(counts),'old_graph_canonical_exact':True,'v76_integer_contract_still_failed':True,'publisher5x5_description_discrepancy_unresolved':True,'new_visual_content_views':0,'new_features_scores_gradient_optimizer':0,'training_protocol_fixed':False,'wholephoto_app_gate_passed':False,'physical_part_independence_verified':False,'independent_field_validation':False,'new_units_passed':3,'disk_free_bytes':shutil.disk_usage(ROOT).free}
    new(OUT/'manifest.json',{'rows':rows,'audit':result});new(ROOT/'reports/industrial-corrosion-subpixel-v78-data-audit.json',{**result,'manifest_sha256':sha(OUT/'manifest.json')});print(result,flush=True)


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
