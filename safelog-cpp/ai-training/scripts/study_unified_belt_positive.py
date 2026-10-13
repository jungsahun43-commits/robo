"""One fixed six-epoch positive-view + score-retention head continuation."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from safelog_ai.belt_positive_retention import pixel_retention
from safelog_ai.belt_detail import configure_belt_only,non_belt_state,paired_flip
from safelog_ai.belt_tiles import predict,policy
from safelog_ai.belt_fine_tiles import guards,BASELINE
from safelog_ai.belt_evaluation import summarize
from safelog_ai.belt_segmentation import pixel_counts
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.unified_main import ARCH,TASKS,UnifiedMain,freeze_bn_statistics,loss_for_task
from safelog_ai.process_probe import process_exists
from scripts.unified_main_data import DATA,Cache,read,sha,normalize_batch
from scripts.study_unified_belt_tiles import native,snapshot,START,START_SHA
from scripts.study_unified_belt_fine import slices

BASE=ROOT/'runs/unified-belt-positive-retention-v1'
PROTOCOL=ROOT/'reports/unified-belt-positive-protocol.json'
NATIVE=ROOT/'data/unified-belt-native-detail-v1/manifest.json'
UNIFORM=ROOT/'data/unified-belt-uniform-tiles-v1/manifest.json'
SOURCES=('safelog_ai/belt_positive_retention.py','scripts/study_unified_belt_positive.py','tests/test_belt_positive_retention.py')
SEED,EPOCHS=439,6


def new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def freeze():
    assert not BASE.exists() and sha(START)==START_SHA; BASE.mkdir()
    source=read(DATA/'manifest.json')['fit']['belt_damage']; crops=read(NATIVE)['rows']; uniform=read(UNIFORM)['rows']
    assert len(source)==len(crops)==1397
    assert all(a['id']==b['id'] and a['split']==b['split']=='train' and b['positive_pixels']>0 for a,b in zip(source,crops))
    choices=defaultdict(list)
    for i,r in enumerate(uniform):
        assert r['split']=='train' and r['source_id']==source[r['source_index']]['id']
        if r['positive_pixels']>0: choices[r['source_index']].append(i)
    draws=np.load(ROOT/'runs/unified-belt-native-detail-v1/draws.npz')['belt_damage'][:6].copy()
    assert draws.shape==(6,1024)
    selected=np.full(draws.shape,-1,np.int64); rng=np.random.default_rng(SEED); fallback=0
    for e in range(6):
        for j in range(2,1024,4):
            options=choices[int(draws[e,j])]
            if options: selected[e,j]=options[int(rng.integers(len(options)))]
            else: fallback+=1
    np.savez_compressed(BASE/'draws.npz',belt_damage=draws,positive_uniform=selected)
    old=ROOT/'runs/unified-belt-fine-tiles-v1/protected-inputs.json'; files=dict(read(old))
    for folder in [ROOT/'reports',ROOT/'runs/unified-belt-fine-tiles-v1']:
        for path in folder.rglob('*') if folder.name=='unified-belt-fine-tiles-v1' else folder.iterdir():
            if path.is_file() and not path.name.endswith(('.log','.lock')): files[path.relative_to(ROOT).as_posix()]=snapshot(path)
    for path in [NATIVE,UNIFORM]: files[path.relative_to(ROOT).as_posix()]=snapshot(path)
    def check(item):
        name,value=item; assert snapshot(ROOT/name)==value,name
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check,files.items()))
    new(BASE/'protected-inputs.json',files)
    prior=read(ROOT/'reports/unified-belt-fine-global_tile_mean-validation.json')
    new(PROTOCOL,{'schema':'unified-belt-positive-retention-v1','declared_before_fit_and_VAL':True,'declared_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for d in ('scripts','safelog_ai','tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'protected_ledger_sha256':sha(BASE/'protected-inputs.json'),'manifest_sha256':sha(DATA/'manifest.json'),
        'native_manifest_sha256':sha(NATIVE),'uniform_manifest_sha256':sha(UNIFORM),'draws_sha256':sha(BASE/'draws.npz'),
        'start_sha256':START_SHA,'epochs':6,'updates_planned':768,'batch':8,'seed':SEED,
        'available_source_TRAIN_photos':1397,'scheduled_unique_TRAIN_photos':len(set(draws.flatten().tolist())),
        'scheduled_draws':6144,'positive_uniform_draws':int((selected>=0).sum()),'positive_uniform_unique_views':len(set(selected[selected>=0].tolist())),
        'uniform_positive_fallback_to_existing_native_crop_draws':fallback,'new_independent_photos':0,'new_labels':0,
        'sampling':'4 original full positive frames +2 existing first-shape contextual crops +2 author-positive uniform crops (fallback native); same TRAIN IDs/order as accepted native study, first6epochs. Selection uses TRAIN masks only, never VAL/TEST.',
        'optimizer':'Belt-head only fresh AdamW LR5e-5 wd.0002 cosine6 eta_min5e-6; clip10; FP32 TF32 off; all BN and nonbelt tensors frozen',
        'loss':'Unchanged publisher BCE posweight20+Dice +10*mean Bernoulli KL(teacher||student) over valid pixels at T1; frozen accepted checkpoint teacher is a regularizer, never new truth or deployment ensemble',
        'checkpoint_selection':'Final epoch6 only; no intermediate VAL or LR search','inference_policy':policy(BASELINE),
        'baseline':prior['summary'],'baseline_slices':prior['slices'],
        'selection':'Same fixed8guards as fine protocol, evaluated against current accepted global_tile_mean baseline; otherwise retain current model and ZIP',
        'evaluation':'Known reused VAL353/4 sequences; threshold.5; reserved author TEST untouched; normal whole-photo FPR null',
        'tests_passed':2,'checkpoint_every_updates':64,'app_default_promoted':False})
    print(json.dumps({'frozen':True,'source_TRAIN':len(set(draws.flatten().tolist())),'uniform_positive_draws':int((selected>=0).sum()),'fallback':fallback,'old_files':len(files)}),flush=True)


def validate():
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items(): assert sha(ROOT/n)==h,n
    for path,h in [(START,START_SHA),(DATA/'manifest.json',p['manifest_sha256']),(NATIVE,p['native_manifest_sha256']),(UNIFORM,p['uniform_manifest_sha256']),
        (BASE/'draws.npz',p['draws_sha256']),(BASE/'protected-inputs.json',p['protected_ledger_sha256'])]: assert sha(path)==h,path
    return p


def progress(path,model,opt,scheduler,epoch,cursor,updates,losses,elapsed):
    temp=path.with_suffix('.pending.pt'); assert path.resolve().is_relative_to(BASE.resolve())
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
        'epoch':epoch,'cursor':cursor,'updates':updates,'losses':losses,'elapsed_seconds':elapsed,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp)
    temp.replace(path)


def train():
    p=validate(); repo=ROOT.parents[1]; git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip(); run=BASE/'train'; run.mkdir(exist_ok=True)
    if (run/'final.pt').exists():
        final=torch.load(run/'final.pt',map_location='cpu',weights_only=True)
        report=ROOT/'reports/unified-belt-positive-training.json'
        if not report.exists(): new(report,{**{k:v for k,v in final.items() if k!='state_dict'},'weights_sha256':sha(run/'final.pt')})
        assert sha(run/'final.pt')==read(report)['weights_sha256']
        print('Complete; preserving trained weights',flush=True); return
    lock=run/'training.lock'
    if lock.exists():
        if process_exists(int(lock.read_text())): raise RuntimeError('Owned training already running')
        lock.unlink()
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        torch.manual_seed(SEED); start=torch.load(START,map_location='cpu',weights_only=True)
        model=UnifiedMain().cuda(); model.load_state_dict(start['state_dict'],strict=True)
        teacher=UnifiedMain().cuda().eval(); teacher.load_state_dict(start['state_dict'],strict=True); teacher.requires_grad_(False)
        fixed=state_sha256(non_belt_state(model)); initial=state_sha256(model.belt_head); teacher_sha=state_sha256(teacher)
        opt=torch.optim.AdamW(configure_belt_only(model),lr=5e-5,weight_decay=.0002); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=6,eta_min=5e-6)
        manifest=read(DATA/'manifest.json'); draws=np.load(BASE/'draws.npz'); ids,views=draws['belt_damage'],draws['positive_uniform']
        allowed=set(ids.flatten().tolist()); full=Cache(manifest['fit']['belt_damage'],allowed); contextual=Cache(read(NATIVE)['rows'],allowed)
        uniform=Cache(read(UNIFORM)['rows'],set(views[views>=0].tolist())); epoch,cursor,updates,losses,elapsed=1,0,0,0.,0.
        checkpoint=run/'progress.pt'; torch.manual_seed(SEED)
        if checkpoint.exists():
            s=torch.load(checkpoint,map_location='cpu',weights_only=True); model.load_state_dict(s['state_dict'],strict=True)
            opt.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
            epoch,cursor,updates,losses,elapsed=(s[k] for k in ('epoch','cursor','updates','losses','elapsed_seconds'))
            torch.set_rng_state(s['torch_rng']); torch.cuda.set_rng_state_all(s['cuda_rng'])
        started=time.monotonic()
        with (run/'update-journal.jsonl').open('a',encoding='utf-8',buffering=1) as journal:
            while epoch<=6:
                model.eval(); model.belt_head.train(); freeze_bn_statistics(model)
                for i in range(cursor,128):
                    offset=i*8; samples=[]
                    for j in range(8):
                        index=int(ids[epoch-1,offset+j]); view=int(views[epoch-1,offset+j])
                        samples.append(full.row(index) if j%2 else uniform.row(view) if j%4==2 and view>=0 else contextual.row(index))
                    images=torch.stack([v[0] for v in samples]); masks=torch.stack([v[1]['mask'] for v in samples]); valid=torch.stack([v[1]['valid'] for v in samples])
                    assert (masks.flatten(1).sum(1)>0).all()
                    images,masks,valid=paired_flip(images,masks,valid,torch.rand(8)<.5); images=normalize_batch(images)
                    truth={'mask':masks.cuda().float(),'valid':valid.cuda().float()}
                    with torch.no_grad(): targets=teacher.forward_task(images,'belt_damage')
                    opt.zero_grad(set_to_none=True); output=model.forward_task(images,'belt_damage')
                    retention=pixel_retention(output,targets,truth['valid']); value=loss_for_task('belt_damage',output,truth)+10*retention
                    assert torch.isfinite(value); value.backward(); torch.nn.utils.clip_grad_norm_(model.belt_head.parameters(),10,error_if_nonfinite=True); opt.step()
                    updates+=1; cursor=i+1; losses+=float(value.detach())
                    journal.write(json.dumps({'epoch':epoch,'cursor':cursor,'update':updates,'all_batch_masks_publisher_positive':True,'retention':float(retention.detach())})+'\n')
                    if updates%64==0: progress(checkpoint,model,opt,scheduler,epoch,cursor,updates,losses,elapsed+time.monotonic()-started)
                assert state_sha256(non_belt_state(model))==fixed; scheduler.step()
                entry={'epoch':epoch,'updates':updates,'mean_loss':losses/128,'elapsed_minutes':(elapsed+time.monotonic()-started)/60,'non_belt_unchanged':True}
                dest=run/f'epoch-{epoch:02d}.json'
                if not dest.exists(): new(dest,entry)
                else: assert read(dest)['updates']==updates
                print(json.dumps(entry),flush=True); epoch+=1; cursor=0; losses=0.
                progress(checkpoint,model,opt,scheduler,epoch,cursor,updates,losses,elapsed+time.monotonic()-started)
        assert updates==768 and state_sha256(model.belt_head)!=initial and state_sha256(teacher)==teacher_sha
        assert all(v.grad is None for n,v in model.named_parameters() if not n.startswith('belt_head.'))
        model.eval(); cpu=UnifiedMain().eval(); cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)
        old=UnifiedMain().eval(); old.load_state_dict(start['state_dict'],strict=True); proofs={}
        with torch.inference_mode():
            for task in TASKS:
                images,_=Cache(manifest['fit'][task],{0,1}).batch([0,1]); a,b=model.forward_task(images,task),cpu.forward_task(images.cpu(),task)
                if task=='facility_surface': a,b=a[0],b[0]
                assert torch.allclose(a.cpu(),b,atol=1e-4,rtol=.001)
                if task!='belt_damage':
                    expected=old.forward_task(images.cpu(),task); expected=expected[0] if task=='facility_surface' else expected; assert torch.equal(expected,b)
                proofs[task]={'CPU_GPU_close':True,'prior_CPU_exact':task!='belt_damage'}
            rgb,_,_=native(manifest['fit']['belt_damage'][0]); a,box,count=predict(model,rgb,BASELINE,'cuda'); b,box_cpu,count_cpu=predict(cpu,rgb,BASELINE,'cpu')
            assert box==box_cpu and count==count_cpu==6 and torch.allclose(a.cpu(),b,atol=1e-4,rtol=.001)
        metadata={'status':'trained_fixed6','architecture':ARCH,'tasks':list(TASKS),'one_shared_backbone':True,'input_size':640,'thresholds':start['thresholds'],
            'epochs':6,'optimizer_updates':768,'initial_checkpoint_sha256':START_SHA,'protocol_sha256':sha(PROTOCOL),'fit_protocol_commit':commit,
            'inference_policy_sha256':policy(BASELINE)['sha256'],'non_belt_state_sha256':fixed,'strict_CPU_GPU_three_tasks':proofs,'native_tile_CPU_GPU_close':True,
            'teacher_unchanged_and_not_for_deployment':True,'elapsed_training_minutes':(elapsed+time.monotonic()-started)/60,
            'actual_journal_updates_including_replay':sum(1 for _ in (run/'update-journal.jsonl').open()),'new_independent_photos':0,'new_labels':0}
        with (run/'final.pt').open('xb') as f: torch.save({**metadata,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}},f)
        new(ROOT/'reports/unified-belt-positive-training.json',{**metadata,'weights_sha256':sha(run/'final.pt')})
        print(json.dumps({'trained_updates':768,'weights_sha256':sha(run/'final.pt')}),flush=True)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()): lock.unlink()


def evaluate():
    validate(); model=UnifiedMain().cuda().eval(); model.load_state_dict(torch.load(BASE/'train/final.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    rows=read(DATA/'manifest.json')['belt_known_validation']; assert len(rows)==353; truth=Cache(rows,range(len(rows))); observations=[]
    for i,row in enumerate(rows):
        rgb,_,_=native(row); probability,box,count=predict(model,rgb,BASELINE,'cuda'); _,t=truth.batch([i])
        observations.append({'id':row['id'],'sequence':row['sequence'],'photo_positive':row['photo_positive'],'detail_views':count,
            'score':float(probability[t['valid'].bool()].max()),'pixels':pixel_counts(probability,t['mask'],t['valid'])})
        if (i+1)%100==0: print(json.dumps({'evaluated_photos':i+1,'total':353}),flush=True)
    summary=summarize(observations); summary['photo_score_definition']='Maximum fused global/tile probability over known nonpadding pixels; not calibrated'
    new(ROOT/'reports/unified-belt-positive-validation.json',{'status':'known_reused_VAL353','rows':observations,'summary':summary,'slices':slices(observations),
        'checkpoint_sha256':sha(BASE/'train/final.pt'),'inference_policy':policy(BASELINE),'new_neural_updates':768,'normal_photo_FPR_available':False,'reserved_TEST_used':False})
    print(json.dumps({'summary':summary}),flush=True)


def select_verify():
    p=validate(); results=read(ROOT/'reports/unified-belt-positive-validation.json'); checks=guards(results['summary'],p['baseline'],results['slices'],p['baseline_slices'])
    selected='positive_retention' if all(checks.values()) else 'prior_accepted_tiles'
    chosen=BASE/'train/final.pt' if selected=='positive_retention' else START
    new(ROOT/'reports/unified-belt-positive-selection.json',{'selected':selected,'guards':checks,'checkpoint_path':chosen.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(chosen),
        'inference_policy':policy(BASELINE),'new_neural_updates_performed':768,'selected_new_neural_updates':768 if selected=='positive_retention' else 0,
        'normal_photo_FPR_available':False,'reserved_TEST_used':False,'app_default_promoted':False})
    ledger=read(BASE/'protected-inputs.json')
    def check(item):
        name,value=item; assert snapshot(ROOT/name)==value,name
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check,ledger.items()))
    for n,h in p['legacy_source_sha256'].items(): assert sha(ROOT/n)==h,n
    final=torch.load(BASE/'train/final.pt',map_location='cpu',weights_only=True); saved=torch.load(BASE/'train/progress.pt',map_location='cpu',weights_only=True)
    initial=torch.load(START,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,saved['state_dict'][n]) for n,v in final['state_dict'].items())
    assert all(torch.equal(v,initial['state_dict'][n]) for n,v in final['state_dict'].items() if not n.startswith('belt_head.'))
    journal=[json.loads(line) for line in (BASE/'train/update-journal.jsonl').read_text().splitlines()]
    assert set(r['update'] for r in journal)==set(range(1,769))
    new(ROOT/'reports/unified-belt-positive-verification.json',{'status':'passed','protected_files':len(ledger),'legacy_sources':len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved':True,'non_belt_tensors_exact_prior':True,'final_equals_progress':True,'actual_journal_updates':len(journal),
        'committed_unique_updates':768,'native_tile_CPU_GPU_close':True,'reserved_TEST_used':False})
    print(json.dumps({'selected':selected,'guards':checks,'verified_files':len(ledger),'actual_updates':len(journal)}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['freeze','train','eval','select-verify'],required=True)
    args=parser.parse_args(); {'freeze':freeze,'train':train,'eval':evaluate,'select-verify':select_verify}[args.mode]()
