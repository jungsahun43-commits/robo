"""v93 fixed remaining FIT diagnostics with observed PNG signature, no GT repair."""
import argparse,hashlib,json,os,shutil,subprocess,sys,traceback
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.liaci_coordinate_contract import checked_polygons
from safelog_ai.belt_normal_roi import fingerprints
from safelog_ai.process_probe import process_exists
SOURCE=ROOT/'data/industrial-liaci-v90-coordinates';PRIOR=ROOT/'data/industrial-liaci-v91-bitmap-diagnosis'
OUT=ROOT/'data/industrial-liaci-v93-local-complete';RUN=ROOT/'runs/industrial-liaci-local-v93'
POLICY=ROOT/'reports/industrial-liaci-v93-local-policy.json';CODE='scripts/complete_liaci_local_probe.py'
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    inputs=[SOURCE/'member-receipts.json',ROOT/'reports/industrial-liaci-v90-coordinate-policy.json',ROOT/'reports/industrial-liaci-v92-remaining-headers.json',
        ROOT/'data/industrial-liaci-v89-metadata/publisher-coco-labels.json',ROOT/'data/industrial-liaci-v87-source/public-zip-directory.json',*PRIOR.glob('decoded-*.json')]
    files={x['path']:x['encoded_SHA256'] for x in read(SOURCE/'member-receipts.json')['rows']}
    OUT.mkdir();RUN.mkdir()
    save(POLICY,{'UTC':utc(),'scope':'Existing remaining3presetFIT numeric arrays only; reuse completed3, not qualified data or corrected GT',
        'code_SHA256':{n:sha(ROOT/n) for n in [CODE,'safelog_ai/liaci_coordinate_contract.py','safelog_ai/belt_normal_roi.py']},
        'input_SHA256':{x.relative_to(ROOT).as_posix():sha(x) for x in inputs},'files_SHA256':files,'remaining_ids':[1,3,8],
        'decode':'Observed v92 PNG bytes in .jpg names1/3 accepted as PNG_RGB singleframe;8JPEG_RGB. ExactCOCO size/EXIFNone1/40MP/32MiB. Each once, native BMP1 and PNG exactsize; no resize/flip/rewrite or class inference.',
        'comparison':'Preserve COCO float/category and independentPIL raster; record white-mask support/disagreement only, no pairing pass or exporter/healthy-negative claim. Reuse v91 three complete rows; no repeats/downloads/VAL_CAL_TEST/visual views.',
        'storage_bound_bytes':8*1024**2,'reserve_bytes':1024**3,'new_features_scores_gradients_optimizer':0,'ready_for_training':False,
        'future_budget':'Metadata-only conservative estimate: all remaining compressedRGB+128headerallowance each,16MiBgraph,32MiBtemporary,64MiBactivity,1GiBreserve; then256MiBrun. No acquisition/feature/model authorization from estimate; separate preregistration required.',
        'data_license':'LIACiCC BY-NC-SA4.0; paperCC BY4.0 separate, all parent rights retained.'})
    print(json.dumps({'policy_SHA256':sha(POLICY),'remaining_RGB':3}),flush=True)
def run():
    p=read(POLICY);git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in [CODE,POLICY.relative_to(ROOT).as_posix()]:assert subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])==(ROOT/n).read_bytes()
    for n,h in {**p['code_SHA256'],**p['input_SHA256'],**p['files_SHA256']}.items():assert sha(ROOT/n)==h
    assert shutil.disk_usage(ROOT).free>=p['reserve_bytes']+p['storage_bound_bytes']
    for lock in (ROOT/'runs').glob('**/*.lock'):
        if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(lock))
    with (RUN/'operation.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        receipts=read(SOURCE/'member-receipts.json')['rows'];files={(x['image_id'],x['role']):ROOT/x['path'] for x in receipts}
        annotations=defaultdict(list)
        for a in read(ROOT/'data/industrial-liaci-v89-metadata/publisher-coco-labels.json')['annotations']:annotations[(a['image_id'],a['category_id'])].append(a)
        results=[read(f) for f in sorted(PRIOR.glob('decoded-*.json'))];assert len(results)==3
        for row in read(ROOT/'reports/industrial-liaci-v90-coordinate-policy.json')['probe_rows']:
            identifier=row['image_id']
            if identifier not in p['remaining_ids']:continue
            size=tuple(row['size']);assert size[0]*size[1]<=40_000_000 and files[(identifier,'RGB')].stat().st_size<=32*1024**2
            with Image.open(files[(identifier,'RGB')]) as im:
                assert im.format==('PNG' if identifier in (1,3) else 'JPEG') and im.mode=='RGB' and im.size==size and getattr(im,'n_frames',1)==1 and im.getexif().get(274) in (None,1)
                rgb=im.copy();fmt=im.format
            layers={}
            for role,category in [('corrosion',5),('ship_hull',10),('paint_peel',2),('marine_growth',9),('defect',4)]:
                with Image.open(files[(identifier,role)]) as im:
                    assert im.format=='BMP' and im.mode=='1' and im.size==size and getattr(im,'n_frames',1)==1
                    white=np.asarray(im,dtype=bool)
                polys=checked_polygons(annotations[(identifier,category)],*size);reference=Image.new('1',size);draw=ImageDraw.Draw(reference)
                for poly in polys:draw.polygon(poly,fill=1)
                ref=np.asarray(reference,dtype=bool)
                layers[role]={'mode':'1','white_pixels':int(white.sum()),'COCO_annotation_count':len(annotations[(identifier,category)]),
                    'PIL_reference_pixels':int(ref.sum()),'PIL_reference_vs_white_disagreement':int((ref!=white).sum()),'no_new_GT_or_original_exporter_claim':True}
            with Image.open(files[(identifier,'segmentation')]) as im:
                assert im.format=='PNG' and im.size==size and getattr(im,'n_frames',1)==1 and im.mode in ('P','L','RGB','RGBA')
                arr=np.asarray(im);mode=im.mode;palette=im.getpalette()
            colors,counts=np.unique(arr.reshape(-1,arr.shape[-1]),axis=0,return_counts=True) if arr.ndim==3 else np.unique(arr,return_counts=True)
            result={'image_id':identifier,'file_name':row['file_name'],'phase':'fit','size':list(size),'actual_RGB_format':fmt,
                'RGB_SHA256':hashlib.sha256(rgb.tobytes()).hexdigest(),'orientation_phashes':fingerprints(rgb),'layers':layers,
                'segmentation_PNG':{'mode':mode,'palette':palette,'unique_count':len(colors),'value_counts':[[c.tolist(),int(n)] for c,n in zip(colors,counts)] if len(colors)<=256 else None,'no_guessed_class_map':True}}
            save(OUT/f'decoded-{identifier:04d}.json',result);results.append(result)
        assert len(results)==6
        existing={r['member']['member'] for r in receipts if r['role']=='RGB'}
        remaining=[r for r in read(ROOT/'data/industrial-liaci-v87-source/public-zip-directory.json')['rows'] if '/images/' in r['member'] and not r['directory'] and r['member'] not in existing]
        assert len(remaining)==1887
        compressed=sum(r['packed']+128 for r in remaining);free=shutil.disk_usage(ROOT).free
        prep=p['reserve_bytes']+compressed+(16+32+64)*1024**2
        budget={'free_bytes':free,'remaining_RGB_members':1887,'preserve_compressed_and_header_bound_bytes':compressed,'reserve_bytes':p['reserve_bytes'],
            'graph_temp_activity_bound_bytes':112*1024**2,'preparation_required_free_bytes':prep,'preparation_deficit_bytes':max(0,prep-free),
            'preparation_plus_256MiB_run_required_free_bytes':prep+256*1024**2,'with_run_deficit_bytes':max(0,prep+256*1024**2-free),
            'extra_mask_or_new_source_contract_not_included':True,'all_fullsource_decode_and_acquisition_still_unapproved_until_separate_policy':True}
        summary={'UTC':utc(),'policy_SHA256':sha(POLICY),'rows':results,'reused_numeric_RGB':3,'new_RGB_decodes':3,'new_BMP_decodes':15,'new_PNG_decodes':3,
            'new_visual_views':0,'new_download_bytes':0,'new_features_scores_gradients_optimizer':0,'ready_for_training':False,
            'v90_pair_and_v91_format_failure_preserved':True,'fullsource_graph_and_GT_contract_pending':True,'future_budget':budget}
        save(OUT/'local-probe-complete.json',summary);save(ROOT/'reports/industrial-liaci-v93-local-complete.json',summary)
        print(json.dumps({'corrosion':[{k:r[k] for k in ('image_id','file_name')}|r['layers']['corrosion'] for r in results],'future_budget':budget}),flush=True)
    except BaseException as exc:
        save(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(exc),'traceback':traceback.format_exc()});raise
    finally:(RUN/'operation.lock').unlink()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));globals()[parser.parse_args().stage]()
