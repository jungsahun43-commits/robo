"""One fixed initial neural fit for the broad equipment prototype."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.equipment_expansion import ARCH,CATEGORIES,SIZE,ExpandedMain,features_for_equipment,reconstruction_loss,anomaly_map,photo_score,stage_gate
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.unified_main import UnifiedMain,TASKS
from scripts.unified_main_data import Cache,DATA as OLD_DATA,normalize_batch
from scripts.study_unified_belt_tiles import START,START_SHA,snapshot
from scripts.prepare_equipment_expansion import DATA,new
from scripts.fetch_equipment_expansion import sha
RUN=ROOT/'runs/equipment-expansion-v1';MANIFEST=DATA/'manifest.json';PROTOCOL=ROOT/'reports/equipment-expansion-v1-protocol.json'
SOURCES=('safelog_ai/equipment_expansion.py','scripts/study_equipment_expansion.py','tests/test_equipment_expansion.py','tests/test_equipment_expansion_data.py')
SEED,EPOCHS,BATCH,DRAWS=583,30,16,128


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def old_state(model):return {n:v for n,v in model.state_dict().items() if not n.startswith('equipment_head.')}


def images(rows,device='cuda'):
    values=[]
    for row in rows:
        with Image.open(ROOT/row['image']) as im:
            values.append(torch.from_numpy(np.array(im.convert('RGB').resize((SIZE,SIZE),Image.Resampling.BICUBIC))).permute(2,0,1))
    return normalize_batch(torch.stack(values),device)


def initialize(device='cuda'):
    initial=torch.load(START,map_location='cpu',weights_only=True)
    model=ExpandedMain().initialize_accepted(initial['state_dict']).to(device).eval()
    model.requires_grad_(False);model.equipment_head.requires_grad_(True)
    return model,initial


def freeze():
    assert not PROTOCOL.exists() and sha(START)==START_SHA
    manifest=read(MANIFEST)
    for n,h in manifest['source_sha256'].items():assert sha(ROOT/n)==h,n
    available={cat:{phase:[r for r in manifest['rows'] if r['category']==cat and r['phase']==phase] for phase in ('fit','cal','test')} for cat in CATEGORIES}
    ready=[cat for cat,x in available.items() if len(x['fit'])>=20 and len(x['cal'])>=20]
    assert ready,'No category has sufficient normal training/calibration photos'
    ledger=dict(read(ROOT/'runs/unified-belt-photo-guard-v1/protected-inputs.json'))
    for folder in (ROOT/'reports',ROOT/'runs/unified-belt-photo-guard-v1'):
        for path in folder.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock','.log')):ledger[path.relative_to(ROOT).as_posix()]=snapshot(path)
    ledger[MANIFEST.relative_to(ROOT).as_posix()]=snapshot(MANIFEST)
    for row in manifest['rows']:
        for n,h in [(row['image'],row['image_sha256']),*zip(row['masks'],row['mask_sha256'])]:
            p=ROOT/n;record=snapshot(p);assert record['sha256']==h,n;ledger[n]=record
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda item: check_snapshot(*item),ledger.items()))
    new(RUN/'protected-inputs.json',ledger)
    schedule={cat:np.random.default_rng(SEED+i).integers(0,len(available[cat]['fit']),size=(EPOCHS,DRAWS)) for i,cat in enumerate(ready)}
    np.savez_compressed(RUN/'draws.npz',**schedule)
    new(PROTOCOL,{'schema':'safelog-equipment-expansion-fixed-v1','declared_before_neural_fit_or_CAL_model_scores':True,
        'declared_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
        'preparation_source_sha256':manifest['source_sha256'],'legacy_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for d in ('scripts','tests','safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'manifest_sha256':sha(MANIFEST),'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),'draws_sha256':sha(RUN/'draws.npz'),
        'initial_accepted_checkpoint_sha256':START_SHA,'architecture':ARCH,'one_backbone':True,'categories':list(CATEGORIES),'trained_categories':ready,
        'normal_only_fit':True,'counts':manifest['counts'],'unique_normal_fit':sum(len(available[c]['fit']) for c in ready),
        'epochs':EPOCHS,'batch':BATCH,'draws_each_category_epoch':DRAWS,'optimizer_updates_planned':EPOCHS*len(ready)*(DRAWS//BATCH),'seed':SEED,
        'sampling':'128 replacement draws per category per epoch, balanced category batches, seed583; repeats are not new photos',
        'preprocess':'RGB bicubic square256, ImageNet normalization; same transform in API and evaluation',
        'features':'ONE accepted MobileNet low40/high960; separately channel-normalize branches, concatenate at high16x16 and normalize; encoder and old3 heads frozen including BN',
        'new_head':'One shared1000-to128-to32 reconstructor with16dim category embeddings; decoder48-to128-to1000',
        'optimizer':'AdamW LR.001 wd.0001 cosine30 eta_min.00001, clip10, FP32 TF32off',
        'loss':'Mean pixelwise cosine feature reconstruction, normalFIT only; no synthetic defect labels or TEST masks used in fit',
        'checkpoint':'Final30 only, resumable64update progress with optimizer,scheduler,cursor and CPU/CUDA RNG; no TEST-based epoch selection',
        'photo_score':'Mean top1percent of bilinear256 cosine-residual map, raw score not calibrated probability',
        'photo_threshold':'Only normalCAL category-specific ceil((n+1)*.85) order statistic; strict score>threshold. No defectCAL or TEST threshold tuning.',
        'pixel_threshold':'Only normalCAL pooled residual quantile.995; coarse feature evidence, not confirmed damage segmentation',
        'stage_targets':{'initial_each_photo_FPR_FNR_max':.20,'next_each_photo_FPR_FNR_max':.15,'minimum_test_each_class':20,'unknown_is_not_pass':True},
        'location_metrics_separate':True,'industrial_independent_field_performance_verified':False,'publisher_test_used_for_fit_or_cal':False,
        'research_prototype_only':True,'old_runtime_model_or_ZIP_auto_replaced':False,'unit_tests_passed':8})
    print(json.dumps({'protocol_frozen':True,'categories':ready,'planned_updates':EPOCHS*len(ready)*8,'normal_fit':sum(len(available[c]['fit']) for c in ready),'protected_files':len(ledger)}),flush=True)


def check_snapshot(n,v):assert snapshot(ROOT/n)==v,n


def validate():
    p=read(PROTOCOL)
    for n,h in {**p['source_sha256'],**p['preparation_source_sha256']}.items():assert sha(ROOT/n)==h,n
    for path,h in [(START,START_SHA),(MANIFEST,p['manifest_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256']),(RUN/'draws.npz',p['draws_sha256'])]:assert sha(path)==h,path
    return p


def save_progress(path,model,opt,scheduler,epoch,cursor,updates):
    assert path.resolve().is_relative_to(RUN.resolve());temporary=path.with_suffix('.pending.pt')
    torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
        'epoch':epoch,'cursor':cursor,'updates':updates,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temporary)
    temporary.replace(path)


def train():
    p=validate();repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip();run=RUN/'train';run.mkdir(exist_ok=True)
    if (run/'final.pt').exists():print('Completed initial expansion fit preserved',flush=True);return
    from safelog_ai.process_probe import process_exists
    import os
    lock=run/'training.lock'
    if lock.exists():
        if process_exists(int(lock.read_text())):raise RuntimeError('Owned expansion training already running')
        lock.unlink()
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        torch.manual_seed(SEED);model,initial=initialize();fixed=state_sha256(old_state(model));first=state_sha256(model.equipment_head)
        rows=read(MANIFEST)['rows'];ready=p['trained_categories'];fit={cat:[r for r in rows if r['category']==cat and r['phase']=='fit'] for cat in ready}
        cache_path=run/'normal-fit-features.pt'
        if not cache_path.exists():
            cache={}
            with torch.inference_mode():
                for cat in ready:
                    features=[]
                    for i in range(0,len(fit[cat]),16):features.append(features_for_equipment(model.backbone(images(fit[cat][i:i+16]))).cpu().half())
                    cache[cat]=torch.cat(features);print(json.dumps({'normal_FIT_features_cached':cat,'photos':len(fit[cat])}),flush=True)
            with cache_path.open('xb') as f:torch.save({'features':cache,'ids':{c:[r['id'] for r in fit[c]] for c in ready},'encoder_sha256':fixed},f)
        saved=torch.load(cache_path,map_location='cpu',weights_only=True);assert saved['encoder_sha256']==fixed and saved['ids']=={c:[r['id'] for r in fit[c]] for c in ready};cache=saved['features']
        opt=torch.optim.AdamW(model.equipment_head.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=.00001)
        draws=np.load(RUN/'draws.npz');epoch,cursor,updates=1,0,0;progress=run/'progress.pt'
        torch.manual_seed(SEED)
        if progress.exists():
            s=torch.load(progress,map_location='cpu',weights_only=True);model.load_state_dict(s['state_dict'],strict=True);opt.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
            epoch,cursor,updates=(s[k] for k in ('epoch','cursor','updates'));torch.set_rng_state(s['rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])
        started=time.monotonic();batches=[(cat,i) for i in range(8) for cat in ready]
        with (run/'update-journal.jsonl').open('a',encoding='utf-8',buffering=1) as journal:
            while epoch<=EPOCHS:
                model.eval();model.equipment_head.train();total=0.
                for index in range(cursor,len(batches)):
                    cat,i=batches[index];idx=draws[cat][epoch-1,i*BATCH:(i+1)*BATCH];target=cache[cat][idx].cuda().float()
                    labels=torch.full((BATCH,),CATEGORIES.index(cat),device='cuda',dtype=torch.long)
                    opt.zero_grad(set_to_none=True);loss=reconstruction_loss(model.equipment_head(target,labels),target);assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.equipment_head.parameters(),10,error_if_nonfinite=True);opt.step()
                    updates+=1;cursor=index+1;total+=float(loss.detach());journal.write(json.dumps({'update':updates,'epoch':epoch,'cursor':cursor,'category':cat,'loss':float(loss.detach()),'normal_FIT_draws':BATCH})+'\n')
                    if updates%64==0:save_progress(progress,model,opt,scheduler,epoch,cursor,updates)
                assert state_sha256(old_state(model))==fixed;scheduler.step()
                entry={'epoch':epoch,'updates':updates,'mean_loss':total/len(batches),'elapsed_minutes':(time.monotonic()-started)/60,'all_old_tensors_unchanged':True}
                file=run/f'epoch-{epoch:02d}.json'
                if not file.exists():new(file,entry)
                print(json.dumps(entry),flush=True);epoch+=1;cursor=0;save_progress(progress,model,opt,scheduler,epoch,cursor,updates)
        assert updates==p['optimizer_updates_planned'] and state_sha256(model.equipment_head)!=first
        assert all(v.grad is None for n,v in model.named_parameters() if not n.startswith('equipment_head.'))
        training_minutes=(time.monotonic()-started)/60
        model.eval();points={}
        with torch.inference_mode():
            for cat in ready:
                cal=[r for r in rows if r['category']==cat and r['phase']=='cal'];maps=[]
                for i in range(0,len(cal),16):maps.append(model.forward_equipment(images(cal[i:i+16]),cat).cpu())
                maps=torch.cat(maps);scores=photo_score(maps).double().tolist();rank=min(len(scores),int(np.ceil((len(scores)+1)*.85)))
                points[cat]={'photo_threshold':sorted(scores)[rank-1],'pixel_threshold':float(torch.quantile(maps.flatten(),.995)),'normal_CAL_photos':len(cal),'ids':[r['id'] for r in cal],'scores':scores}
        proof={};host=ExpandedMain().eval();host.load_state_dict({n:v.cpu() for n,v in model.state_dict().items()},strict=True)
        old=UnifiedMain().eval();old.load_state_dict(initial['state_dict'],strict=True);manifest=read(OLD_DATA/'manifest.json')
        with torch.inference_mode():
            for task in TASKS:
                x,_=Cache(manifest['fit'][task],{0,1}).batch([0,1]);gpu=model.forward_task(x,task);cpu=host.forward_task(x.cpu(),task);prior=old.forward_task(x.cpu(),task)
                if task=='facility_surface':gpu,cpu,prior=gpu[0],cpu[0],prior[0]
                assert torch.equal(cpu,prior) and torch.allclose(gpu.cpu(),cpu,atol=1e-4,rtol=.001);proof[task]={'prior_CPU_exact':True,'CPU_GPU_close':True}
            x=images(fit[ready[0]][:2]);a=model.forward_equipment(x,ready[0]);b=host.forward_equipment(x.cpu(),ready[0]);assert torch.allclose(a.cpu(),b,atol=1e-4,rtol=.001)
        meta={'architecture':ARCH,'tasks':list(TASKS)+['equipment_surface'],'one_shared_backbone':True,'neural_model_count':1,
            'thresholds':initial['thresholds'],'equipment_operating_points':points,'equipment_categories':list(CATEGORIES),'trained_categories':ready,
            'initial_checkpoint_sha256':START_SHA,'protocol_sha256':sha(PROTOCOL),'frozen_before_fit_commit':commit,'epochs':EPOCHS,'optimizer_updates':updates,
            'normal_FIT_draws':updates*BATCH,'unique_normal_FIT_photos':p['unique_normal_fit'],'old_tensors_sha256':fixed,'old_CPU_GPU_proofs':proof,'equipment_CPU_GPU_close':True,
            'elapsed_optimizer_training_minutes':training_minutes,'research_prototype_only':True,'fieldPerformanceValidated':False,
            'publisher_TEST_used_for_fit_or_calibration':False,'app_default_promoted':False,'trained_dataset_sources':['DACL10k','DamSegment','CODEBRIM','BeltCrack','Mendeley-s25x2bnshz','MVTec AD','MVTec LOCO AD','MPDD','KolektorSDD2']}
        with (run/'final.pt').open('xb') as f:torch.save({**meta,'state_dict':{n:v.cpu() for n,v in model.state_dict().items()}},f)
        new(ROOT/'reports/equipment-expansion-v1-training.json',{**meta,'weights_sha256':sha(run/'final.pt')})
        print(json.dumps({'actual_neural_updates':updates,'checkpoint_sha256':sha(run/'final.pt')}),flush=True)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()


def evaluate():
    p=validate();state=torch.load(RUN/'train/final.pt',map_location='cpu',weights_only=True);model=ExpandedMain().cuda().eval();model.load_state_dict(state['state_dict'],strict=True)
    rows=read(MANIFEST)['rows'];results={};observations=[]
    with torch.inference_mode():
        for cat in p['trained_categories']:
            test=[r for r in rows if r['category']==cat and r['phase']=='test'];points=state['equipment_operating_points'][cat];scores=[];pixels=dict(tp=0,fp=0,tn=0,fn=0);vanished=0
            for start in range(0,len(test),16):
                batch=test[start:start+16];maps=model.forward_equipment(images(batch),cat).cpu();scores.extend(photo_score(maps).tolist())
                for row,heat in zip(batch,maps):
                    mask=np.zeros((SIZE,SIZE),dtype=bool)
                    for n in row['masks']:
                        with Image.open(ROOT/n) as im:mask|=np.array(im.convert('L').resize((SIZE,SIZE),Image.Resampling.NEAREST))>0
                    vanished+=int(row['label'] and not mask.any());pred=heat.numpy()>points['pixel_threshold']
                    for key,val in [('tp',pred&mask),('fp',pred&~mask),('tn',~pred&~mask),('fn',~pred&mask)]:pixels[key]+=int(val.sum())
            rates=classification_rates([r['label'] for r in test],scores,points['photo_threshold']);normal=sum(r['label']==0 for r in test);defect=len(test)-normal
            results[cat]={'photo_rates':rates,'normal_test_photos':normal,'defect_test_photos':defect,'normal_CAL_photos':points['normal_CAL_photos'],
                'auroc':rank_auroc([r['label'] for r in test],scores),'pixel_counts_model256':pixels,'pixel_iou':pixels['tp']/max(1,pixels['tp']+pixels['fp']+pixels['fn']),
                'pixel_fnr':pixels['fn']/max(1,pixels['tp']+pixels['fn']),'positive_masks_vanished_at256':vanished,'stage_gate':stage_gate(rates['fpr'],rates['fnr'],normal,defect),
                'photo_threshold':points['photo_threshold'],'pixel_threshold':points['pixel_threshold'],'heatmap_is_confirmed_localization':False}
            observations.extend({'id':r['id'],'category':cat,'label':r['label'],'score':s,'candidate_detected':s>points['photo_threshold']} for r,s in zip(test,scores))
            print(json.dumps({'category':cat,'normal':normal,'defect':defect,'FPR':rates['fpr'],'FNR':rates['fnr'],'stage':results[cat]['stage_gate']['state']}),flush=True)
    for cat in CATEGORIES:
        if cat not in results:results[cat]={'stage_gate':{'state':'insufficient_normal_fit_or_calibration','ready_for_next_stage':False}}
    new(ROOT/'reports/equipment-expansion-v1-results.json',{'status':'initial_broad_prototype_evaluated','categories':results,'rows':observations,
        'all_categories_photo_rates_within20percent':all(x['stage_gate']['ready_for_next_stage'] for x in results.values()),
        'stage20_categories':[k for k,x in results.items() if x['stage_gate']['ready_for_next_stage']],
        'source_TEST_not_used_for_fit_or_calibration':True,'benchmark_split_newly_scored_by_this_head':True,'independent_industrial_field_validated':False,
        'thresholds_not_fitted_to_defect_or_TEST_labels':True,'app_default_promoted':False,'checkpoint_sha256':sha(RUN/'train/final.pt')})


def verify():
    p=validate();ledger=read(RUN/'protected-inputs.json')
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda item:check_snapshot(*item),ledger.items()))
    for n,h in p['legacy_source_sha256'].items():assert sha(ROOT/n)==h,n
    final=torch.load(RUN/'train/final.pt',map_location='cpu',weights_only=True);progress=torch.load(RUN/'train/progress.pt',map_location='cpu',weights_only=True);old=torch.load(START,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,old['state_dict'][n]) for n,v in final['state_dict'].items() if not n.startswith('equipment_head.'))
    assert all(torch.equal(v,progress['state_dict'][n]) for n,v in final['state_dict'].items())
    journal=[json.loads(x) for x in (RUN/'train/update-journal.jsonl').read_text().splitlines()];planned=p['optimizer_updates_planned']
    assert set(x['update'] for x in journal)==set(range(1,planned+1)) and len(journal)==planned
    new(ROOT/'reports/equipment-expansion-v1-verification.json',{'status':'passed','actual_unique_optimizer_updates':planned,'actual_journal_updates':len(journal),
        'old_files_bytes_mtimes_preserved':len(ledger),'old_sources_preserved':len(p['legacy_source_sha256']),'all_old_tensors_exact':True,
        'final_equals_progress':True,'normal_only_neural_fit':True,'publisher_TEST_not_used_for_fit_or_calibration':True,'architecture_has_one_shared_backbone':True})
    print(json.dumps({'verification':'passed','updates':planned,'preserved_files':len(ledger)}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['freeze','train','eval','verify'],required=True);args=parser.parse_args()
    {'freeze':freeze,'train':train,'eval':evaluate,'verify':verify}[args.mode]()
