"""Resumable, predeclared common-backbone preservation comparison after main v1."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from safelog_ai.unified_main import ARCH,TASKS,FACILITY_CLASSES,UnifiedMain,freeze_bn_statistics,loss_for_task
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.foreign_calibration import fit_threshold
from safelog_ai.equipment_anomaly import classification_rates
from safelog_ai.belt_evaluation import summarize as belt_summary
from safelog_ai.belt_segmentation import pixel_counts
from scripts.unified_main_data import DATA,read,sha,Cache,photo,normalize_batch

BASE=ROOT/'runs/unified-preservation-v2'
PROTOCOL=ROOT/'reports/unified-preservation-protocol.json'
LEDGER=BASE/'protected-inputs.json'
START=ROOT/'runs/unified-main-v1/final.pt'
TEACHER=ROOT/'runs/facility-presence-target-roi-control/best.pt'
START_SHA='362f9cab4a137419e63118f73b6b622dccfacc37c94a8f52af62f217b1491598'
VARIANTS=('joint','frozen')
EPOCHS,BATCH,SEED=10,8,427
NEW_SOURCES=('scripts/study_unified_preservation.py','tests/test_unified_preservation.py')


def new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2)


def snapshot(path):
    stat=path.stat();return {'sha256':sha(path),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def schedule(old):
    assert old['facility_surface'].shape==(20,512)
    result={'facility_surface':old['facility_surface'].reshape(EPOCHS,1024).copy(),
            'belt_damage':old['belt_damage'][:EPOCHS].copy(),
            'conveyor_foreign':old['conveyor_foreign'][:EPOCHS].copy()}
    rng=np.random.default_rng(SEED)
    for task in TASKS:
        for row in result[task]:rng.shuffle(row)
        assert result[task].shape==(EPOCHS,1024)
    return result


def configure(model,variant):
    if variant not in VARIANTS:raise ValueError('Explicit joint or frozen comparison required')
    model.zero_grad(set_to_none=True)
    model.train();freeze_bn_statistics(model)
    model.backbone.requires_grad_(variant=='joint')
    if variant=='frozen':model.backbone.eval()
    groups=[]
    for prefix,lr in [('backbone.',1e-5),('belt_head.',3e-4),('foreign_head.',5e-4)]:
        params=[p for name,p in model.named_parameters() if name.startswith(prefix) and p.requires_grad]
        if params:groups.append({'params':params,'lr':lr,'name':prefix})
    groups.append({'params':[p for name,p in model.named_parameters() if not name.startswith(('backbone.','belt_head.','foreign_head.')) and p.requires_grad],
                   'lr':1e-4,'name':'facility_heads'})
    return groups


def research_selection(rows,baseline):
    """Use reused facility/belt VAL only. Foreign TEST cannot participate."""
    accepted=[];gates={}
    for variant in VARIANTS:
        result=rows[variant]
        gate={'facility_primary_max_not_worse':result['facility_primary_max']<=baseline['facility_primary_max'],
              'belt_FNR_not_worse':result['belt']['photo_rates']['fnr']<=baseline['belt']['photo_rates']['fnr'],
              'belt_IoU_not_worse':result['belt']['pixel_iou']>=baseline['belt']['pixel_iou']}
        gates[variant]=gate
        if all(gate.values()):accepted.append(variant)
    selected=min(accepted,key=lambda v:(rows[v]['facility_primary_max'],-rows[v]['belt']['pixel_iou'],v!='frozen')) if accepted else 'prior_v1'
    return {'selected':selected,'validation_guards':gates,'normal_belt_photo_FPR_available':False,
            'foreign_TEST_used_for_selection':False,'independent_factory_proof':False,'app_default_promoted':False}


def preflight():
    assert not BASE.exists() and sha(START)==START_SHA
    BASE.mkdir();torch.manual_seed(SEED)
    state=torch.load(START,map_location='cpu',weights_only=True)
    model=UnifiedMain().cuda();model.load_state_dict(state['state_dict'],strict=True)
    manifest=read(DATA/'manifest.json')
    proof={}
    for variant in VARIANTS:
        model.load_state_dict(state['state_dict'],strict=True)
        groups=configure(model,variant);original=state_sha256(model.backbone)
        opt=torch.optim.AdamW(groups,weight_decay=.0002);rows=manifest['fit']['conveyor_foreign']
        x,truth=Cache(rows,range(BATCH)).batch(range(BATCH));opt.zero_grad(set_to_none=True)
        value=loss_for_task('conveyor_foreign',model.forward_task(x,'conveyor_foreign'),truth)
        value.backward();grad=sum(float(p.grad.abs().sum()) for p in model.backbone.parameters() if p.grad is not None)
        opt.step();changed=state_sha256(model.backbone)!=original
        assert changed==(variant=='joint') and (grad>0)==(variant=='joint')
        proof[variant]={'backbone_grad_abs_sum':grad,'backbone_changed':changed,'disposable_updates':1}
    new(BASE/'preflight.json',{'status':'passed','variants':proof,'not_part_of_declared_epochs':True,
                            'source_sha256':{n:sha(ROOT/n) for n in NEW_SOURCES},
                            'peak_GPU_bytes':torch.cuda.max_memory_allocated()})
    print(json.dumps({'preflight':'passed','comparison':proof}),flush=True)


def freeze():
    flight=read(BASE/'preflight.json')
    assert flight['status']=='passed' and flight['source_sha256']=={n:sha(ROOT/n) for n in NEW_SOURCES}
    previous=read(ROOT/'reports/unified-main-protocol.json')
    for name,digest in {**previous['source_sha256'],**previous['legacy_source_sha256']}.items():assert sha(ROOT/name)==digest,name
    old=np.load(DATA/'draws.npz',allow_pickle=False);draws=schedule(old)
    np.savez_compressed(BASE/'draws.npz',**draws)
    old_ledger=read(ROOT/'runs/unified-main-input-ledger.json')
    protected=dict(old_ledger['previous_models_and_reports'])
    for folder in [ROOT/'reports',ROOT/'runs/unified-main-v1']:
        for path in folder.iterdir():
            if path.is_file() and (folder==ROOT/'runs/unified-main-v1' or path.name.startswith(('unified-main-','UNIFIED_MAIN'))):
                protected[path.relative_to(ROOT).as_posix()]=snapshot(path)
    def check(pair):
        name,entry=pair;assert snapshot(ROOT/name)==entry,name
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check,old_ledger['current_inputs'].items()))
        list(pool.map(check,protected.items()))
    new(LEDGER,{'current_inputs':old_ledger['current_inputs'],'previous_models_and_reports':protected})
    legacy={p.relative_to(ROOT).as_posix():sha(p) for folder in ['safelog_ai','scripts','tests'] for p in (ROOT/folder).glob('*.py') if p.relative_to(ROOT).as_posix() not in NEW_SOURCES}
    baseline=read(ROOT/'reports/unified-main-results.json')
    primary=max(v['unified'][k] for rows in baseline['facility_fixed_old_thresholds'].values() for c,v in rows.items() if c in FACILITY_CLASSES[:2] for k in ('fpr','fnr'))
    manifest=read(DATA/'manifest.json')
    new(PROTOCOL,{'schema':'shared-main-preservation-v2','declared_before_fit':True,'declared_utc':datetime.now(timezone.utc).isoformat(),
                 'source_sha256':{n:sha(ROOT/n) for n in NEW_SOURCES},'legacy_source_sha256':legacy,
                 'input_ledger_sha256':sha(LEDGER),'manifest_sha256':sha(DATA/'manifest.json'),'draws_sha256':sha(BASE/'draws.npz'),
                 'start_checkpoint_sha256':START_SHA,'original_main_teacher_sha256':sha(TEACHER),
                 'variants':list(VARIANTS),'epochs_per_variant':EPOCHS,'updates_per_variant':3840,'combined_updates':7680,
                 'batch':BATCH,'seed':SEED,'draws_per_task_per_epoch':1024,
                 'facility_sampling':'Two successive old512 TRAIN draws combined; identical total20-epoch facility exposure in10 epochs',
                 'equipment_sampling':'First10 old1024 TRAIN draws for each equipment task',
                 'unique_scheduled_records':{t:len(set(draws[t].flatten().tolist())) for t in TASKS},
                 'dataset_sources':manifest['sources'],'new_data_or_labels':0,'unknowns_preserved':True,
                 'initialization':'Both variants exact same completed unified v1 state; fresh AdamW, no optimizer state inherited',
                 'changed_training_all_variants':'Equal1024 draws per task per epoch, increased facility fraction; seeded original horizontal flips only',
                 'only_difference_between_variants':'Backbone parameters including BN affine trainable(joint) versus requires_grad false(frozen); all BN stats frozen in both',
                 'head_lrs':{'facility':1e-4,'belt':3e-4,'foreign':5e-4},'joint_backbone_lr':1e-5,'weight_decay':.0002,
                 'cosine_eta_min':5e-6,'loss':'Same frozen v1 loss, teacher, weights, belt posweight20 and known/padding rules',
                 'precision':'FP32; TF32 disabled','gradient_clip':10,'teacher_used_for_new_labels':False,
                 'thresholds_facility_belt':read(ROOT/'reports/unified-main-training.json')['thresholds'],
                 'foreign_threshold':'Refit using same CAL320-only algorithm after fixed10 epochs, before TEST',
                 'checkpoint_selection':'Final10 only; no early stopping or per-epoch VAL tuning',
                 'research_variant_selection':'VAL-only guards: primary facility max<=v1, belt FNR<=v1 and IoU>=v1; accepted ordered by facility max then belt IoU then frozen; else preserve prior_v1',
                 'baseline_for_guards':{'facility_primary_max':primary,'belt':baseline['belt_val353']},
                 'foreign_TEST_scoring':'Only after research-selection.json is written; score both fixed candidates but never alter selection',
                 'all_evaluation_is_known_reused':True,'normal_belt_photo_labels':0,'field_validation':False,'app_default_promoted':False,
                 'progress':'Checkpoint every64 optimizer calls; mutable progress only; every attempted completed update appended to journal',
                 'tests_passed':3,'preflight_updates_excluded':2})
    print(json.dumps({'protocol':'frozen','protected_files':len(protected),'legacy_sources':len(legacy),'updates_each':3840}),flush=True)


def validate():
    protocol=read(PROTOCOL)
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(START)==START_SHA and sha(TEACHER)==protocol['original_main_teacher_sha256']
    assert sha(DATA/'manifest.json')==protocol['manifest_sha256'] and sha(BASE/'draws.npz')==protocol['draws_sha256']
    assert sha(LEDGER)==protocol['input_ledger_sha256']
    return protocol


def supervision(manifest):
    rows=manifest['fit']['facility_surface'];labels=torch.tensor([r['targets'] for r in rows]);aux=torch.tensor([r['auxiliary_targets'] for r in rows])
    counts=read(ROOT/'data/facility-spatial-training/train.json')['audit']['per_label_pixel_cells']
    return {'photo':((labels==0).sum(0)/(labels==1).sum(0).clamp_min(1)).clamp(.2,6).float().cuda(),
            'pixel':torch.tensor([min(20,max(1,counts[c]['negative']/max(1,counts[c]['positive']))) for c in FACILITY_CLASSES]).float().cuda(),
            'aux':((aux==0).sum(0)/(aux==1).sum(0).clamp_min(1)).clamp(.2,6).float().cuda()}


def save_progress(path,model,opt,scheduler,epoch,cursor,updates,counts,losses,elapsed):
    assert path.resolve().is_relative_to(BASE.resolve())
    temp=path.with_suffix('.pending.pt')
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),
                'scheduler':scheduler.state_dict(),'epoch':epoch,'cursor':cursor,'updates':updates,
                'counts':dict(counts),'losses':dict(losses),'elapsed_seconds':elapsed,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp)
    temp.replace(path)


def train_variant(variant):
    p=validate();repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for name in (*NEW_SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])).hexdigest()==sha(ROOT/name)
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
    run=BASE/variant;run.mkdir(exist_ok=True)
    if (run/'final.pt').exists():
        report=ROOT/f'reports/unified-preservation-{variant}-training.json'
        if not report.exists():
            stored=torch.load(run/'final.pt',map_location='cpu',weights_only=True)
            assert stored['variant']==variant and stored['epochs']==EPOCHS and stored['optimizer_updates']==3840 and stored['strict_CPU_GPU_all_three']
            new(report,{'status':'trained','weights_sha256':sha(run/'final.pt'),'checkpoint_count':1,'backbone_count':1,
                        **{k:v for k,v in stored.items() if k!='state_dict'}})
        trained=read(report)
        assert sha(run/'final.pt')==trained['weights_sha256'];print(variant+' already completed, preserving final checkpoint',flush=True);return
    lock=run/'training.lock'
    if lock.exists():
        old_pid=int(lock.read_text())
        try:os.kill(old_pid,0)
        except OSError:
            assert lock.resolve().is_relative_to(BASE.resolve());lock.unlink()
        else:raise RuntimeError(f'{variant} training already running with PID {old_pid}')
    with lock.open('x') as handle:handle.write(str(os.getpid()))
    try:
        manifest=read(DATA/'manifest.json');draws=np.load(BASE/'draws.npz',allow_pickle=False)
        caches={t:Cache(manifest['fit'][t],set(draws[t].flatten().tolist())) for t in TASKS}
        torch.manual_seed(SEED);start=torch.load(START,map_location='cpu',weights_only=True)
        model=UnifiedMain().cuda();model.load_state_dict(start['state_dict'],strict=True)
        initial=state_sha256(model);initial_backbone=state_sha256(model.backbone)
        teacher=AuxiliaryClassifier(7,False).cuda().eval();teacher.load_state_dict(torch.load(TEACHER,map_location='cpu',weights_only=True)['state_dict'],strict=True);teacher.requires_grad_(False)
        teacher_initial=state_sha256(teacher);truth_weights=supervision(manifest)
        opt=torch.optim.AdamW(configure(model,variant),weight_decay=.0002)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=5e-6)
        progress=run/'progress.pt';epoch=1;cursor=0;updates=0;counts=Counter();losses=Counter();elapsed=0.
        torch.manual_seed(SEED)
        if progress.exists():
            saved=torch.load(progress,map_location='cpu',weights_only=True)
            model.load_state_dict(saved['state_dict'],strict=True);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            epoch,cursor,updates=saved['epoch'],saved['cursor'],saved['updates'];counts=Counter(saved['counts']);losses=Counter(saved['losses']);elapsed=saved['elapsed_seconds']
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            print(json.dumps({'resumed':variant,'epoch':epoch,'cursor':cursor,'committed_updates':updates}),flush=True)
        started=time.monotonic()
        with (run/'update-journal.jsonl').open('a',encoding='utf-8',buffering=1) as journal:
            while epoch<=EPOCHS:
                model.train();freeze_bn_statistics(model)
                if variant=='frozen':model.backbone.eval()
                chunks={t:np.split(draws[t][epoch-1],128) for t in TASKS}
                order=[t for t in TASKS for _ in range(128)];np.random.default_rng(SEED+epoch).shuffle(order)
                assert cursor==sum(counts.values())
                for index in range(cursor,len(order)):
                    task=order[index];selected=chunks[task][counts[task]];images,truth=caches[task].batch(selected,training=True)
                    with torch.no_grad():reference=teacher(images) if task=='facility_surface' else None
                    opt.zero_grad(set_to_none=True);value=loss_for_task(task,model.forward_task(images,task),truth,truth_weights,reference)
                    assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
                    opt.step();updates+=1;counts[task]+=1;losses[task]+=float(value.detach());cursor=index+1
                    journal.write(json.dumps({'epoch':epoch,'cursor':cursor,'task':task,'committed_model_update_index':updates})+'\n')
                    if updates%64==0:save_progress(progress,model,opt,scheduler,epoch,cursor,updates,counts,losses,elapsed+time.monotonic()-started)
                scheduler.step()
                entry={'variant':variant,'epoch':epoch,'optimizer_updates':updates,'task_updates':dict(counts),
                       'task_mean_loss':{t:losses[t]/counts[t] for t in TASKS},'elapsed_minutes':(elapsed+time.monotonic()-started)/60}
                record=run/f'epoch-{epoch:02d}.json'
                if record.exists():assert read(record)['optimizer_updates']==updates
                else:new(record,entry)
                print(json.dumps(entry),flush=True)
                epoch+=1;cursor=0;counts=Counter();losses=Counter()
                save_progress(progress,model,opt,scheduler,epoch,cursor,updates,counts,losses,elapsed+time.monotonic()-started)
        assert updates==3840 and state_sha256(model)!=initial and state_sha256(teacher)==teacher_initial
        backbone_changed=state_sha256(model.backbone)!=initial_backbone
        assert backbone_changed==(variant=='joint')
        assert all(param.grad is None for param in teacher.parameters())
        if variant=='frozen':assert all(param.grad is None for param in model.backbone.parameters())
        model.eval();cpu=UnifiedMain().eval();cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)
        with torch.inference_mode():
            for task in TASKS:
                images,_=caches[task].batch(draws[task][0][:2]);a=model.forward_task(images,task);b=cpu.forward_task(images.cpu(),task)
                if task=='facility_surface':a,b=a[0],b[0]
                assert torch.allclose(a.cpu(),b,atol=1e-4,rtol=.001)
            rows=manifest['foreign_calibration'];cal=[]
            for offset in range(0,len(rows),BATCH):
                batch=rows[offset:offset+BATCH];images=normalize_batch(torch.stack([photo(r) for r in batch]))
                scores=model.forward_task(images,'conveyor_foreign').sigmoid().cpu().tolist()
                cal.extend({'id':r['id'],'label':r['label'],'score':score} for r,score in zip(batch,scores))
        calibration_path=run/'foreign-calibration.json'
        if calibration_path.exists():
            previous=read(calibration_path)
            assert previous['model_state_sha256']==state_sha256(model)
            point=previous
        else:
            point=fit_threshold(cal)
            new(calibration_path,{'rows':cal,**point,'neural_epochs_in_threshold_fit':0,'model_state_sha256':state_sha256(model)})
        thresholds={**start['thresholds'],'conveyor_foreign':point['threshold']}
        calls=sum(1 for _ in (run/'update-journal.jsonl').open())
        metadata={'architecture':ARCH,'input_size':640,'facility_classes':list(FACILITY_CLASSES),'tasks':list(TASKS),
                  'one_shared_backbone':True,'thresholds':thresholds,'epochs':EPOCHS,'optimizer_updates':updates,
                  'actual_optimizer_calls_including_discarded_replay':calls,'variant':variant,'dataset_sources':manifest['sources'],
                  'initial_main_sha256':START_SHA,'initial_state_sha256':initial,'final_state_sha256':state_sha256(model),
                  'initial_backbone_sha256':initial_backbone,'final_backbone_sha256':state_sha256(model.backbone),
                  'backbone_changed':backbone_changed,'teacher_unchanged':True,'protocol_sha256':sha(PROTOCOL),
                  'fit_protocol_commit':commit,'elapsed_training_and_CAL_minutes':(elapsed+time.monotonic()-started)/60,
                  'TEST_used_for_fit_or_CAL':False,'app_default_promoted':False,'strict_CPU_GPU_all_three':True}
        torch.save({**metadata,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}},run/'final.pt')
        new(ROOT/f'reports/unified-preservation-{variant}-training.json',{'status':'trained','strict_CPU_GPU_all_three':True,
            'weights_sha256':sha(run/'final.pt'),'checkpoint_count':1,'backbone_count':1,**metadata})
        print(json.dumps({'completed':variant,'epochs':EPOCHS,'optimizer_updates':updates,'actual_calls':calls,'sha256':sha(run/'final.pt')}),flush=True)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()


def model_for(variant):
    report=read(ROOT/f'reports/unified-preservation-{variant}-training.json');run=BASE/variant
    assert sha(run/'final.pt')==report['weights_sha256']
    model=UnifiedMain().cuda().eval();model.load_state_dict(torch.load(run/'final.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    return model,report,run


def evaluate_validation():
    validate();manifest=read(DATA/'manifest.json');baseline=read(ROOT/'reports/unified-main-results.json')
    for variant in VARIANTS:
        record=ROOT/f'reports/unified-preservation-{variant}-validation.json'
        if record.exists():continue
        model,trained,run=model_for(variant);facility={}
        for domain,rows in manifest['facility_known_validation'].items():
            scores=[]
            with torch.inference_mode():
                for offset in range(0,len(rows),BATCH):
                    batch=rows[offset:offset+BATCH];images=normalize_batch(torch.stack([photo(dict(r,task='facility_surface')) for r in batch]))
                    scores.extend(model(images).sigmoid().cpu().tolist())
            previous=read(ROOT/f'runs/unified-main-v1/{domain}-validation.json')
            assert previous['image_ids']==[r['image'] for r in rows]
            result={}
            for i,c in enumerate(FACILITY_CLASSES):
                indices=[j for j,r in enumerate(rows) if r['targets'][i]>=0];labels=[rows[j]['targets'][i] for j in indices]
                threshold=trained['thresholds']['facility_surface'][c]
                result[c]={'previous_v1':baseline['facility_fixed_old_thresholds'][domain][c]['unified'],
                           'candidate':classification_rates(labels,[scores[j][i] for j in indices],threshold),'threshold':threshold}
            facility[domain]=result
            new(run/f'{domain}-validation.json',{'image_ids':[r['image'] for r in rows],'scores':scores,'per_class':result})
        rows=manifest['belt_known_validation'];cache=Cache(rows,range(len(rows)));belt=[]
        with torch.inference_mode():
            for offset in range(0,len(rows),BATCH):
                images,truth=cache.batch(range(offset,min(offset+BATCH,len(rows))));scores=model.forward_task(images,'belt_damage').sigmoid()
                for i,r in enumerate(rows[offset:offset+BATCH]):
                    score=scores[i:i+1];valid=truth['valid'][i:i+1];mask=truth['mask'][i:i+1]
                    belt.append({'id':r['id'],'sequence':r['sequence'],'photo_positive':True,
                                 'score':float(score[valid.bool()].max()),'pixels':pixel_counts(score,mask,valid)})
        summary=belt_summary(belt);new(run/'belt-validation.json',{'rows':belt,'summary':summary})
        maximum=max(v['candidate'][k] for rows in facility.values() for c,v in rows.items() if c in FACILITY_CLASSES[:2] for k in ('fpr','fnr'))
        new(record,{'status':'evaluated_known_VAL','variant':variant,'model_sha256':trained['weights_sha256'],
                    'facility':facility,'facility_primary_max':maximum,'belt':summary,'foreign_TEST_scored':False,
                    'normal_belt_FPR_available':False,'known_benchmarks_reused':True})
        print(json.dumps({'validation':variant,'facility_primary_max':maximum,'belt_IoU':summary['pixel_iou'],'belt_FNR':summary['photo_rates']['fnr']}),flush=True)


def select():
    p=validate();rows={v:read(ROOT/f'reports/unified-preservation-{v}-validation.json') for v in VARIANTS}
    choice=research_selection(rows,p['baseline_for_guards'])
    path=START if choice['selected']=='prior_v1' else BASE/choice['selected']/'final.pt'
    choice.update(checkpoint_path=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),
                  decided_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PROTOCOL))
    new(BASE/'research-selection.json',choice);new(ROOT/'reports/unified-preservation-selection.json',choice)
    print(json.dumps(choice),flush=True)


def evaluate_foreign():
    validate();selection=read(BASE/'research-selection.json');assert not selection['foreign_TEST_used_for_selection']
    before=sha(BASE/'research-selection.json');manifest=read(DATA/'manifest.json')
    for variant in VARIANTS:
        record=ROOT/f'reports/unified-preservation-{variant}-foreign.json'
        if record.exists():continue
        model,trained,run=model_for(variant);foreign={}
        for name,key in [('common448','foreign_known_test'),('known783','foreign_known_extra_test')]:
            rows=manifest[key];observations=[]
            with torch.inference_mode():
                for offset in range(0,len(rows),BATCH):
                    batch=rows[offset:offset+BATCH];images=normalize_batch(torch.stack([photo(r) for r in batch]))
                    scores=model.forward_task(images,'conveyor_foreign').sigmoid().cpu().tolist()
                    observations.extend({'id':r['id'],'label':r['label'],'score':s} for r,s in zip(batch,scores))
            rate=classification_rates([r['label'] for r in observations],[r['score'] for r in observations],trained['thresholds']['conveyor_foreign'])
            foreign[name]=rate;new(run/f'foreign-{name}.json',{'rows':observations,'rates':rate})
        new(record,{'status':'known_TEST_scored_after_VAL_selection','variant':variant,'foreign':foreign,
                    'selection_sha256_before_TEST':before,'selection_sha256_after_TEST':sha(BASE/'research-selection.json'),
                    'model_sha256':trained['weights_sha256'],'known_recordings_reused':True,'independent_factory_verified':False})
        assert sha(BASE/'research-selection.json')==before
        print(json.dumps({'foreign':variant,'rates':foreign}),flush=True)


def verify():
    p=validate();ledger=read(LEDGER)
    for name,digest in p['legacy_source_sha256'].items():assert sha(ROOT/name)==digest,name
    def check(pair):
        name,entry=pair;assert snapshot(ROOT/name)==entry,name
    with ThreadPoolExecutor(max_workers=4) as pool:
        for group in ledger.values():list(pool.map(check,group.items()))
    actual=0;models={}
    for variant in VARIANTS:
        trained=read(ROOT/f'reports/unified-preservation-{variant}-training.json');run=BASE/variant
        history=[read(run/f'epoch-{e:02d}.json') for e in range(1,EPOCHS+1)]
        assert history[-1]['optimizer_updates']==trained['optimizer_updates']==3840
        assert all(row['task_updates']=={t:128 for t in TASKS} for row in history)
        assert trained['strict_CPU_GPU_all_three'] and trained['teacher_unchanged'] and sha(run/'final.pt')==trained['weights_sha256']
        if variant=='frozen':assert trained['initial_backbone_sha256']==trained['final_backbone_sha256']
        actual+=trained['actual_optimizer_calls_including_discarded_replay'];models[variant]=trained['weights_sha256']
    choice=read(BASE/'research-selection.json');assert choice==read(ROOT/'reports/unified-preservation-selection.json')
    assert sha(ROOT/choice['checkpoint_path'])==choice['checkpoint_sha256']
    new(ROOT/'reports/unified-preservation-verification.json',{'status':'passed','epochs_per_candidate':EPOCHS,
        'additional_candidate_epochs_total':20,'committed_updates_total':7680,'actual_optimizer_calls_total':actual,
        'two_research_candidates_not_two_models_loaded_in_app':True,'candidate_weights':models,
        'frozen_candidate_backbone_exactly_unchanged':True,'previous_sources_and_inputs_preserved':True,
        'input_files_protected':len(ledger['current_inputs']),'previous_artifacts_protected':len(ledger['previous_models_and_reports']),
        'selected':choice,'preflight_updates_excluded':2,'new_data_or_labels':0,
        'all_industrial_tasks_5percent_verified':False,'app_default_promoted':False})
    print('Both real fits, resume journals, unchanged backbone, inputs and selection verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','freeze','train','validation','select','foreign','verify'])
    parser.add_argument('--variant',choices=VARIANTS);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='train':
        for v in ([args.variant] if args.variant else VARIANTS):train_variant(v)
    else:{'preflight':preflight,'freeze':freeze,'validation':evaluate_validation,'select':select,'foreign':evaluate_foreign,'verify':verify}[args.mode]()
