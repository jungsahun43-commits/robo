"""v91 existing fixed FIT files only: diagnose representation, never repair GT."""
import argparse,hashlib,json,os,shutil,struct,subprocess,sys,traceback
from collections import defaultdict
from datetime import datetime,timezone
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.liaci_coordinate_contract import checked_polygons
from safelog_ai.belt_normal_roi import fingerprints
from safelog_ai.process_probe import process_exists
SOURCE=ROOT/'data/industrial-liaci-v90-coordinates'
OUT=ROOT/'data/industrial-liaci-v91-bitmap-diagnosis'
RUN=ROOT/'runs/industrial-liaci-bitmap-v91'
POLICY=ROOT/'reports/industrial-liaci-v91-bitmap-policy.json'
CODE=['scripts/diagnose_liaci_bitmap.py','safelog_ai/liaci_coordinate_contract.py','safelog_ai/belt_normal_roi.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    p=read(ROOT/'reports/industrial-liaci-v90-coordinate-policy.json');r=read(SOURCE/'member-receipts.json')
    assert len(r['rows'])==42 and all(x['CRC_verified'] for x in r['rows'])
    files={x['path']:x['encoded_SHA256'] for x in r['rows']}
    for name,h in files.items():assert sha(ROOT/name)==h
    failures=list((ROOT/'runs/industrial-liaci-coordinate-v90').glob('failure-*.json'));assert len(failures)==1
    assert 'category presence differs' in read(failures[0])['error']
    OUT.mkdir();RUN.mkdir()
    save(POLICY,{'UTC':utc(),'scope':'Existing six presetFIT v90 files only; representation diagnosis, not corrected GT or ready data',
        'code_SHA256':{n:sha(ROOT/n) for n in CODE},'files_SHA256':files,
        'inputs_SHA256':{n:sha(ROOT/n) for n in ['reports/industrial-liaci-v90-coordinate-policy.json','data/industrial-liaci-v90-coordinates/member-receipts.json','data/industrial-liaci-v89-metadata/publisher-coco-labels.json']},
        'probe_rows':p['probe_rows'],'failure_SHA256':sha(failures[0]),'new_downloads':0,
        'decode':'6RGB once each, 30BMP and6PNG numeric arrays only, <=40MP/32MiB/singleframe/EXIFNone1/exactnative COCO size; formats JPEG_RGB,BMP_1_L_P_RGB,PNG_P_L_RGB_RGBA only, no resize/flip/rewrite. Unknown format/size/coordinates stops.',
        'diagnosis':'Record BMP header/depth/palette, native value counts, COCO/PIL geometry. Compare explicit white/black support only when exactly binary0/255; independentPIL not proven author exporter. Neither best polarity nor matching subset becomes GT. Segmentation colors are recorded without guessing a class map.',
        'GT_and_roles':'Keep original COCO floats/category order, supplied BMP/PNG, v90 role lock and TEST191/allcopies/not_used332. v90 failed pairing remains failed. Same six files only, no replacements, no VAL_CAL_TEST content.',
        'graph':'Cache six RGB/pHashes for future entire-source graph; oldheld join/omitted bridges remain pending, no old global decode/distance audit.',
        'readiness':False,'new_model_features_scores_gradients_optimizer':0,'new_visual_views':0,
        'storage_bound_bytes':8*1024**2,'reserve_bytes':1024**3,'data_license':'LIACiCC BY-NC-SA4.0, paperCC BY4.0 separate; all parent terms retained.'})
    print(json.dumps({'policy_SHA256':sha(POLICY),'new_network':0,'images':6}),flush=True)
def run():
    p=read(POLICY);git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in [*CODE,POLICY.relative_to(ROOT).as_posix()]:assert subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])==(ROOT/n).read_bytes()
    for n,h in {**p['code_SHA256'],**p['files_SHA256'],**p['inputs_SHA256']}.items():assert sha(ROOT/n)==h
    assert shutil.disk_usage(ROOT).free>=p['reserve_bytes']+p['storage_bound_bytes']
    for lock in (ROOT/'runs').glob('**/*.lock'):
        if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(lock))
    with (RUN/'operation.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        receipts=read(SOURCE/'member-receipts.json')['rows'];files={(x['image_id'],x['role']):ROOT/x['path'] for x in receipts}
        annotations=defaultdict(list)
        for a in read(ROOT/'data/industrial-liaci-v89-metadata/publisher-coco-labels.json')['annotations']:annotations[(a['image_id'],a['category_id'])].append(a)
        results=[]
        for row in p['probe_rows']:
            identifier=row['image_id'];size=tuple(row['size']);assert size[0]*size[1]<=40_000_000
            with Image.open(files[(identifier,'RGB')]) as im:
                assert im.format=='JPEG' and im.mode=='RGB' and im.size==size and getattr(im,'n_frames',1)==1 and im.getexif().get(274) in (None,1)
                rgb=im.copy();exif=im.getexif();capture={str(k):exif.get(k) for k in (272,306,36867)}
            layers={}
            for role,category in [('corrosion',5),('ship_hull',10),('paint_peel',2),('marine_growth',9),('defect',4)]:
                raw=files[(identifier,role)].read_bytes();assert len(raw)<=32*1024**2
                with Image.open(BytesIO(raw)) as im:
                    assert im.format=='BMP' and im.size==size and getattr(im,'n_frames',1)==1 and im.mode in ('1','L','P','RGB')
                    mode=im.mode;palette=im.getpalette();pixels=np.asarray(im.convert('RGB'));luminance=np.asarray(im.convert('L'))
                colors,counts=np.unique(pixels.reshape(-1,3),axis=0,return_counts=True)
                assert len(colors)<=256
                polys=checked_polygons(annotations[(identifier,category)],*size)
                ref=Image.new('1',size);d=ImageDraw.Draw(ref)
                for poly in polys:d.polygon(poly,fill=1)
                ref=np.asarray(ref,dtype=bool)
                assert struct.unpack_from('<I',raw,14)[0]>=40
                value={'mode':mode,'palette':palette,'BMP_signature':raw[:2].decode('ascii'),'DIB_bytes':struct.unpack_from('<I',raw,14)[0],
                    'bits_per_pixel':struct.unpack_from('<H',raw,28)[0],'pixel_offset':struct.unpack_from('<I',raw,10)[0],
                    'native_RGB_colors_and_counts':[[c.tolist(),int(n)] for c,n in zip(colors,counts)],
                    'COCO_annotation_count':len(annotations[(identifier,category)]),'PIL_reference_pixels':int(ref.sum()),
                    'no_new_GT_or_original_exporter_claim':True}
                if set(map(tuple,colors.tolist())).issubset({(0,0,0),(255,255,255)}):
                    white=luminance==255;black=luminance==0
                    value.update({'white_pixels':int(white.sum()),'black_pixels':int(black.sum()),
                        'PIL_reference_vs_white_disagreement':int((ref!=white).sum()),'PIL_reference_vs_black_disagreement':int((ref!=black).sum()),
                        'PIL_reference_white_intersection':int((ref&white).sum()),'PIL_reference_black_intersection':int((ref&black).sum())})
                layers[role]=value
            with Image.open(files[(identifier,'segmentation')]) as im:
                assert im.format=='PNG' and im.size==size and getattr(im,'n_frames',1)==1 and im.mode in ('P','L','RGB','RGBA')
                mode=im.mode;palette=im.getpalette();arr=np.asarray(im)
            colors,counts=np.unique(arr.reshape(-1,arr.shape[-1]),axis=0,return_counts=True) if arr.ndim==3 else np.unique(arr,return_counts=True)
            png={'mode':mode,'palette':palette,'unique_count':len(colors),'value_counts':[[c.tolist(),int(n)] for c,n in zip(colors,counts)] if len(colors)<=256 else None,'no_guessed_class_map':True}
            result={'image_id':identifier,'file_name':row['file_name'],'phase':'fit','size':list(size),'RGB_SHA256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                'orientation_phashes':fingerprints(rgb),'capture_metadata':capture,'layers':layers,'segmentation_PNG':png}
            save(OUT/f'decoded-{identifier:04d}.json',result);results.append(result)
            print(json.dumps({'image_id':identifier,'corrosion':layers['corrosion'],'segmentation_unique_count':png['unique_count']}),flush=True)
        first_failure=next(({'image_index':i,'image_id':r['image_id'],'layer_index':j,'role':role,'prior_RGB_and_BMP_decode_invocations':5*i+j+1,
            'prior_unique_RGB':i+1} for i,r in enumerate(results) for j,role in enumerate(('corrosion','ship_hull','paint_peel','marine_growth','defect'))
            if bool(r['layers'][role]['COCO_annotation_count'])!=bool(r['layers'][role].get('white_pixels',0))),None)
        assert first_failure is not None
        result={'UTC':utc(),'policy_SHA256':sha(POLICY),'status':'diagnosis complete; GT contract and complete graph still pending',
            'rows':results,'RGB_decode_invocations':6,'BMP_decode_invocations':30,'PNG_decode_invocations':6,'new_unique_source_RGB':6-first_failure['prior_unique_RGB'],
            'prior_v90_first_presence_failure_reconstructed_from_numeric_counts':first_failure,
            'new_download_bytes':0,'new_visual_views':0,'new_features_scores_gradients_optimizer':0,'ready_for_training':False,
            'source_pairing_passed':False,'v90_failure_preserved':True,'oldheld_and_fullsource_closure_pending':True,'disk_free_bytes':shutil.disk_usage(ROOT).free}
        save(OUT/'bitmap-diagnosis.json',result);save(ROOT/'reports/industrial-liaci-v91-bitmap-diagnosis.json',result)
    except BaseException as exc:
        save(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(exc),'traceback':traceback.format_exc(),'new_optimizer':0});raise
    finally:(RUN/'operation.lock').unlink()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));globals()[parser.parse_args().stage]()
