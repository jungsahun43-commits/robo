"""v90 fixed TRAIN-only numeric pairing probe; never qualifies a dataset/model."""
import argparse,csv,hashlib,json,math,os,shutil,subprocess,sys,traceback
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import threading
import requests
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.liaci_coordinate_contract import checked_polygons,numeric_pair,read_binary_member
from safelog_ai.public_zip_metadata import check_range
from safelog_ai.belt_normal_roi import fingerprints
from safelog_ai.process_probe import process_exists

OUT=ROOT/'data/industrial-liaci-v90-coordinates';RUN=ROOT/'runs/industrial-liaci-coordinate-v90'
POLICY=ROOT/'reports/industrial-liaci-v90-coordinate-policy.json'
INDEX=ROOT/'data/industrial-liaci-v87-source/public-zip-directory.json'
ANNOTATION=ROOT/'data/industrial-liaci-v89-metadata/publisher-coco-labels.json'
RESERVATION=ROOT/'data/industrial-liaci-v89-metadata/reserved-publisher-evaluation-before-annotation-content.json'
CODE=['scripts/probe_liaci_coordinates.py','safelog_ai/liaci_coordinate_contract.py','tests/test_liaci_coordinate_contract.py','safelog_ai/public_zip_metadata.py','safelog_ai/belt_normal_roi.py']
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    assert sha(ANNOTATION)=='4acd1fd3d0498c091803d4d5f2617bffcef5427cab0bd4d209c32d72e917de94'
    assert sha(RESERVATION)=='2b7bab9a88db75e1a465eac0f82ebbbf40d92813c73510e17488f191976e5706'
    obj=read(ANNOTATION);reservation=read(RESERVATION);corrosion={a['image_id'] for a in obj['annotations'] if a['category_id']==5}
    hull={a['image_id'] for a in obj['annotations'] if a['category_id']==10};images={r['file_name']:r for r in obj['images']}
    rows=[]
    for r in reservation['rows']:
        im=images[r['file_name']];condition='corrosion_annotated' if im['id'] in corrosion else 'no_corrosion_annotation'
        phase='test' if r['publisher_split']=='test' else 'excluded_publisher_not_used' if r['publisher_split']=='not_used' else None
        rows.append({**r,'image_id':im['id'],'size':[im['width'],im['height']],'condition':condition,'hull_annotated':im['id'] in hull,'phase':phase})
    for condition in sorted({r['condition'] for r in rows}):
        selected=[r for r in rows if r['phase'] is None and r['condition']==condition]
        selected.sort(key=lambda r:hashlib.sha256(('liaci90:'+r['file_name']).encode()).hexdigest())
        n=math.ceil(.15*len(selected))
        for i,r in enumerate(selected):r['phase']='validation' if i<n else 'cal' if i<2*n else 'fit'
    probe=[]
    for condition in ('corrosion_annotated','no_corrosion_annotation'):
        candidates=sorted((r for r in rows if r['phase']=='fit' and r['condition']==condition and r['hull_annotated']),key=lambda r:r['file_name'])
        assert len(candidates)>=3;probe.extend(candidates[:3])
    members={r['member']:r for r in read(INDEX)['rows']};tasks=[]
    for r in probe:
        stem=r['file_name'].removesuffix('.jpg')
        names={'RGB':'images/'+stem+'.jpg','corrosion':'masks/corrosion/'+stem+'.bmp','ship_hull':'masks/ship_hull/'+stem+'.bmp','paint_peel':'masks/paint_peel/'+stem+'.bmp','marine_growth':'masks/marine_growth/'+stem+'.bmp','defect':'masks/defect/'+stem+'.bmp','segmentation':'masks/segmentation/'+stem+'.png'}
        for role,name in names.items():tasks.append({'image_id':r['image_id'],'role':role,'member':members['LIACi_dataset_pretty/'+name]})
    budget=sum(t['member']['bytes']+t['member']['packed']+131072 for t in tasks)
    assert budget<=32*1024**2 and shutil.disk_usage(ROOT).free>=1024**3+budget+16*1024**2
    OUT.mkdir();RUN.mkdir();save(OUT/'preset-roles-before-pixels.json',{'rows':rows,'probe_ids':[r['image_id'] for r in probe],'original_TEST':191,'not_used':332,'final_graph_audit_pending':True})
    prior=read(ROOT/'reports/industrial-liaci-v89-annotation-metadata-policy.json')
    save(POLICY,{'UTC':utc(),'scope':'Six fixed presetFIT numeric coordinate probes only; not full graph audit, readyFIT or training',
        'URL':prior['URL'],'ETag':prior['ETag'],'archive_bytes':prior['archive_bytes'],'source_sha256':{n:sha(ROOT/n) for n in CODE},
        'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [INDEX,ANNOTATION,RESERVATION,ROOT/'data/industrial-liaci-v90-paper/publisher-article-text-only.json']},
        'preset_roles_SHA256':sha(OUT/'preset-roles-before-pixels.json'),'probe_rows':probe,'tasks':tasks,'storage_bound_bytes':budget,'reserve_bytes':1024**3,'max_network_bytes':32*1024**2,
        'HTTP':'Exact public206/ContentRange/length/ETag/identity, pinned local filename/method/size/CRC; keep all received ranges/original encoded members, fail closed without full fallback.2workers.',
        'GT':'Native provided corrosionBMP is a separate author layer; preserve other labels, overlap and unannotated background. No wholephoto normal label, binary background substitution or GT edits. Only numeric size/mode/presence and named PIL-raster diagnostic, not proven original exporter.',
        'decode':'JPEG RGB, BMP singleframe1bit0/255; PNG original mode/palette only. Native dimensions must match COCO/JPEG; no resize, <=40MP/32MiB perfile, EXIFNone1; finite polygon>=3vertices within[0,width]x[0,height], iscrowd0. Unknown shape/label/palette/coordinate stops.',
        'allocation':'Freeze single label-presence-stratifiedSHA256(liaci90:+filename)ceil15percentVAL/next15CAL/restFIT now. OriginalTEST191/not_used332 and14copies locked. Future complete graph: exclude every TEST/not_used-connected TRAIN, cross-role component, or prior role conflict; preserve one matching prior role, never reassign/replenish/relax. These six probes stayFIT or becomeexcluded.',
        'future_full_graph_minimums':{'fit_each_annotation_condition_photos':30,'validation_and_cal_each_corrosion':10,'validation_and_cal_each_other':20,'test_each_condition_photos':20,'each_phase_each_condition_connected_groups':3,'each_phase_each_valid_native_class_pixels':4096},
        'readiness':'Alwaysfalse in this probe. All1893 source RGB fingerprint closure/oldheld exclusion/coordinate and semantic contract still required before any final manifest. Reuse all old graphs/fingerprints/roles inclRios7/Ostrava15/VT44; no old global RGB/distance rerun. Partial probes cannot prove independence.',
        'QA':'No source visual display at this stage, no VAL_CAL_TEST pixels or models. Only six presetFIT RGB/masks/fingerprints; cache these for later complete graph.',
        'paper':'16commercial inspections plusresearchvessel; layered annotations overlap and waterbackground unlabelled; not_used comes from author similarity filtering, not extra TRAIN. Missing physical mapping retained.',
        'data_license':'LIACiCC BY-NC-SA4.0 attribution/noncommercial/SA/third-party rights; paperCC BY4.0 separate. All parent terms unchanged.',
        'new_encoder_scores_gradients_optimizer':0,'model_protocol_fixed':False,'new_units_passed':3})
    print(json.dumps({'policy_SHA256':sha(POLICY),'probe_ids':[r['image_id'] for r in probe],'storage_bound_bytes':budget,'preset_roles':dict(Counter(r['phase'] for r in rows))}),flush=True)

def run():
    p=read(POLICY);git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for name in [*CODE,POLICY.relative_to(ROOT).as_posix()]:assert subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])==(ROOT/name).read_bytes()
    for name,h in {**p['source_sha256'],**p['input_sha256']}.items():assert sha(ROOT/name)==h
    assert sha(OUT/'preset-roles-before-pixels.json')==p['preset_roles_SHA256']
    assert shutil.disk_usage(ROOT).free>=p['reserve_bytes']+p['storage_bound_bytes']+16*1024**2
    for lock in (ROOT/'runs').glob('**/*.lock'):
        if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(lock))
    with (RUN/'operation.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        guard=threading.Lock();local=threading.local();received=0
        def worker(task):
            nonlocal received
            row=task['member'];folder=OUT/f'{task["image_id"]:04d}-{task["role"]}';folder.mkdir()
            ranges=[]
            if not hasattr(local,'session'):local.session=requests.Session()
            def getter(start,count):
                nonlocal received
                assert count>0 and 0<=start<start+count<=p['archive_bytes']
                with guard:
                    assert received+count<=p['max_network_bytes'];received+=count
                    assert shutil.disk_usage(ROOT).free>=p['reserve_bytes']+count+65536
                dest=folder/f'range-{start}-{count}.bin';size=0;h=hashlib.sha256()
                with local.session.get(p['URL'],headers={'Range':f'bytes={start}-{start+count-1}','If-Range':p['ETag'],'Accept-Encoding':'identity'},stream=True,timeout=(15,40)) as response:
                    check_range(response.status_code,response.headers,start,count,p['archive_bytes'],p['ETag']);assert response.url==p['URL']
                    with dest.open('xb') as f:
                        for b in response.iter_content(65536):size+=len(b);assert size<=count;f.write(b);h.update(b)
                    assert size==count
                ranges.append({'start':start,'bytes':count,'SHA256':h.hexdigest()});return dest.read_bytes()
            raw=read_binary_member(row,getter,32*1024**2);path=folder/('publisher'+Path(row['member']).suffix)
            with path.open('xb') as f:f.write(raw)
            result={**task,'path':path.relative_to(ROOT).as_posix(),'encoded_SHA256':sha(path),'ranges':ranges,'CRC_verified':True}
            save(folder/'receipt.json',result);print(json.dumps({'downloaded_image_id':task['image_id'],'role':task['role'],'bytes':len(raw)}),flush=True);return result
        with ThreadPoolExecutor(max_workers=2) as pool:objects=list(pool.map(worker,p['tasks']))
        save(OUT/'member-receipts.json',{'rows':objects,'range_bytes':received})
        by_key={(r['image_id'],r['role']):r for r in objects};coco=read(ANNOTATION);by_annotation=defaultdict(list)
        for a in coco['annotations']:by_annotation[(a['image_id'],a['category_id'])].append(a)
        results=[]
        for row in p['probe_rows']:
            identifier=row['image_id'];rgb_raw=(ROOT/by_key[(identifier,'RGB')]['path']).read_bytes();layers={};first_rgb=None
            for role,category in [('corrosion',5),('ship_hull',10),('paint_peel',2),('marine_growth',9),('defect',4)]:
                raw=(ROOT/by_key[(identifier,role)]['path']).read_bytes();rgb,mask,value=numeric_pair(rgb_raw,raw,row['size'],by_annotation[(identifier,category)])
                if first_rgb is None:first_rgb=rgb
                layers[role]=(mask,value)
            png_path=ROOT/by_key[(identifier,'segmentation')]['path']
            with Image.open(png_path) as im:
                if im.format!='PNG' or im.n_frames!=1 or list(im.size)!=row['size'] or im.mode not in ('P','RGB','RGBA','L'):raise ValueError('Segmentation PNG contract')
                arr=np.asarray(im);colors=np.unique(arr.reshape(-1,arr.shape[-1]),axis=0) if arr.ndim==3 else np.unique(arr)
                png={'mode':im.mode,'size':list(im.size),'unique_color_or_index_count':int(len(colors)),'palette':im.getpalette() if im.mode=='P' else None}
            corrosion=layers['corrosion'][0];others=layers['paint_peel'][0]|layers['marine_growth'][0]|layers['defect'][0]
            result={'image_id':identifier,'file_name':row['file_name'],'preset_phase':'fit','size':row['size'],
                'orientation_phashes':fingerprints(first_rgb),'layers':{k:v[1] for k,v in layers.items()},'segmentation_PNG':png,
                'corrosion_overlaps_other_inspection_pixels':int((corrosion&others).sum()),'ship_hull_outside_all_inspection_pixels':int((layers['ship_hull'][0]&~corrosion&~others).sum()),
                'these_counts_are_not_measured_rust_absence_or_machine_health':True}
            save(OUT/f'decoded-{identifier:04d}.json',result);results.append(result)
        result={'UTC':utc(),'status':'numeric probe complete, full source preparation pending','policy_SHA256':sha(POLICY),'probe_rows':results,
            'source_original_RGB':6,'native_binary_mask_reads':30,'segmentation_PNG_reads':6,'RGB_decode_invocations':30,'new_visual_views':0,
            'PIL_reference_not_proven_exporter':True,'old_graph_RGB_or_distance_reaudits':0,'oldheld_join_and_fullsource_closure_pending':True,
            'new_encoder_features_scores_gradients_optimizer':0,'wholephoto_app_gate_passed':False,'ready_for_training':False,
            'received_range_bytes':received,'disk_free_bytes':shutil.disk_usage(ROOT).free}
        save(OUT/'coordinate-probe.json',result);save(ROOT/'reports/industrial-liaci-v90-coordinate-probe.json',result);print(json.dumps({k:v for k,v in result.items() if k!='probe_rows'}),flush=True)
    except BaseException as exc:
        save(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(exc),'traceback':traceback.format_exc(),'new_optimizer':0,'originals_and_partial_ranges_preserved':True});raise
    finally:(RUN/'operation.lock').unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));globals()[parser.parse_args().stage]()
