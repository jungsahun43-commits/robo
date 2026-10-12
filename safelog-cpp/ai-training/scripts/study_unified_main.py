"""Actual existing-main extension, joint five-source fit, and fixed benchmark comparison."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from safelog_ai.unified_main import ARCH,TASKS,FACILITY_CLASSES,initialize_from_main,UnifiedMain,freeze_bn_statistics,loss_for_task
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.foreign_calibration import fit_threshold
from safelog_ai.equipment_anomaly import classification_rates
from safelog_ai.belt_evaluation import summarize as belt_summary
from safelog_ai.belt_segmentation import pixel_counts
from scripts.unified_main_data import ROOT,DATA,read,sha,Cache,photo,normalize_batch

RUN = ROOT/'runs/unified-main-v1'
PROTOCOL = ROOT/'reports/unified-main-protocol.json'
LEDGER = ROOT/'runs/unified-main-input-ledger.json'
MAIN = ROOT/'runs/facility-presence-target-roi-control/best.pt'
BELT = ROOT/'runs/beltcrack-cached-expansion-v2/final.pt'
SOURCES = ('safelog_ai/unified_main.py','scripts/prepare_unified_main.py','scripts/unified_main_data.py',
           'scripts/study_unified_main.py','tests/test_unified_main.py')
EPOCHS,BATCH,SEED = 20,8,271


def new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as output: json.dump(value,output,indent=2,ensure_ascii=False)


def sources(): return {name:sha(ROOT/name) for name in SOURCES}
def snapshot(path):
    stat=path.stat();return {'sha256':sha(path),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def initialize():
    torch.manual_seed(SEED)
    main=torch.load(MAIN,map_location='cpu',weights_only=True); belt=torch.load(BELT,map_location='cpu',weights_only=True)
    assert main['architecture']=='lraspp_mobilenet_facility_auxiliary_v1' and main['classes']==list(FACILITY_CLASSES)
    model=initialize_from_main(main['state_dict'],belt['state_dict'])
    assert all(torch.equal(model.state_dict()[k],v) for k,v in main['state_dict'].items())
    teacher=AuxiliaryClassifier(7,False).eval();teacher.load_state_dict(main['state_dict'],strict=True);teacher.requires_grad_(False)
    return model.cuda(),teacher.cuda()


def weights(manifest):
    rows=manifest['fit']['facility_surface']; labels=torch.tensor([r['targets'] for r in rows]);aux=torch.tensor([r['auxiliary_targets'] for r in rows])
    counts=read(ROOT/'data/facility-spatial-training/train.json')['audit']['per_label_pixel_cells']
    return {'photo':((labels==0).sum(0)/(labels==1).sum(0).clamp_min(1)).clamp(.2,6).float().cuda(),
        'pixel':torch.tensor([min(20,max(1,counts[c]['negative']/max(1,counts[c]['positive']))) for c in FACILITY_CLASSES]).float().cuda(),
        'aux':((aux==0).sum(0)/(aux==1).sum(0).clamp_min(1)).clamp(.2,6).float().cuda()}


def preflight():
    manifest=read(DATA/'manifest.json'); model,teacher=initialize(); initial=state_sha256(model);frozen=state_sha256(teacher)
    model.eval();x=normalize_batch(torch.stack([photo(manifest['fit']['facility_surface'][0])]))
    with torch.inference_mode(): assert torch.equal(model(x),teacher(x)), 'Original main outputs must initially be identical'
    opt=torch.optim.AdamW(model.parameters(),lr=.0001); gradients={};supervision=weights(manifest)
    cases=[('facility_surface',d) for d in ['dacl','damsegment','codebrim']]+[('belt_damage','BeltCrack'),('conveyor_foreign','Mendeley')]
    for task,source in cases:
        rows=manifest['fit'][task]; selected=[i for i,r in enumerate(rows) if task!='facility_surface' or r['domain']==source][:BATCH]
        images,truth=Cache(rows,selected).batch(selected)
        model.train();freeze_bn_statistics(model);opt.zero_grad(set_to_none=True)
        with torch.no_grad(): t=teacher(images) if task=='facility_surface' else None
        value=loss_for_task(task,model.forward_task(images,task),truth,supervision,t);assert torch.isfinite(value)
        value.backward();gradients[source]=sum(float(p.grad.abs().sum()) for p in model.backbone.parameters() if p.grad is not None)
        assert gradients[source]>0
        if task!='facility_surface': assert all(p.grad is None for p in model.photo_head.parameters())
        if task!='belt_damage': assert all(p.grad is None for p in model.belt_head.parameters())
        if task!='conveyor_foreign': assert all(p.grad is None for p in model.foreign_head.parameters())
        opt.step()
    assert state_sha256(model)!=initial and state_sha256(teacher)==frozen
    assert all(p.grad is None for p in teacher.parameters())
    model.eval();cpu=UnifiedMain().eval();cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)
    with torch.inference_mode():
        for task in TASKS:
            a=model.forward_task(x,task);b=cpu.forward_task(x.cpu(),task)
            if task=='facility_surface':a,b=a[0],b[0]
            assert torch.allclose(a.cpu(),b,atol=.0001,rtol=.001)
    new(ROOT/'runs/unified-main-preflight.json',{'status':'passed','five_actual_disposable_updates':5,'actual_epochs':0,
        'each_dataset_gradient_on_same_backbone':gradients,'original_main_initialized_exactly':True,
        'main_initial_outputs_identical':True,'strict_CPU_GPU_all_three_tasks':True,'teacher_frozen':True,
        'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'source_sha256':sources()})
    print(json.dumps({'preflight':'passed','same_backbone_gradients':gradients}),flush=True)


def freeze():
    flight=read(ROOT/'runs/unified-main-preflight.json');assert flight['status']=='passed' and flight['source_sha256']==sources()
    manifest=read(DATA/'manifest.json');draws=np.load(DATA/'draws.npz',allow_pickle=False)
    paths={DATA/'manifest.json',DATA/'draws.npz',ROOT/'data/facility-spatial-training/train.json',ROOT/'data/facility-auxiliary-training/train.json'}
    for task in TASKS:
        for i in set(draws[task].flatten().tolist()):
            r=manifest['fit'][task][i];paths.add(ROOT/r['image'])
            for key in ['pixel_target','mask','valid']:
                if key in r: paths.add(ROOT/r[key])
    for rows in manifest['facility_known_validation'].values(): paths.update(ROOT/r['image'] for r in rows)
    for r in manifest['belt_known_validation']:
        paths.update(ROOT/r[k] for k in ['image','mask','valid'])
    for field in ['foreign_calibration','foreign_known_test','foreign_known_extra_test']: paths.update(ROOT/r['image'] for r in manifest[field])
    paths=sorted(paths);records={}
    def record(path): return path.relative_to(ROOT).as_posix(),snapshot(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(name,entry) in enumerate(pool.map(record,paths)):
            records[name]=entry
            if i%5000==0:print(json.dumps({'input_files_frozen':i,'total':len(paths)}),flush=True)
    previous=read(ROOT/'reports/pcb-multisource-protocol.json')
    legacy={p.relative_to(ROOT).as_posix():sha(p) for folder in ['safelog_ai','scripts','tests'] for p in (ROOT/folder).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES}
    for name,digest in {**previous['source_sha256'],**previous['legacy_source_sha256']}.items():assert legacy[name]==digest,name
    artifacts=dict(read(ROOT/'runs/pcb-multisource-input-ledger.json')['prior_model_and_report_artifacts'])
    for p in [MAIN,BELT,ROOT/'runs/pcb-multisource-v3/final.pt',ROOT/'runs/conveyor-foreign-calibrated-v3/final.pt']:
        artifacts[p.relative_to(ROOT).as_posix()]=snapshot(p)
    for pattern in ['pcb-multisource-*','PCB_MULTISOURCE*','foreign-calibration*','beltcrack-expansion*']:
        for p in (ROOT/'reports').glob(pattern):
            if p.is_file():artifacts[p.relative_to(ROOT).as_posix()]=snapshot(p)
    for name,entry in artifacts.items():assert snapshot(ROOT/name)==entry,name
    new(LEDGER,{'current_inputs':records,'previous_models_and_reports':artifacts})
    thresholds={c:(read(MAIN.parent/'TARGET-SELECTION.json')['selected']['per_class'][c]['threshold'] if c in FACILITY_CLASSES[:2] else .5) for c in FACILITY_CLASSES}
    new(PROTOCOL,{'schema':'existing-main-shared-three-task-v1','declared_before_fit':True,'declared_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sources(),'legacy_source_sha256':legacy,'manifest_sha256':sha(DATA/'manifest.json'),'draws_sha256':sha(DATA/'draws.npz'),
        'input_ledger_sha256':sha(LEDGER),'architecture':ARCH,'initial_main_path':MAIN.relative_to(ROOT).as_posix(),'initial_main_sha256':sha(MAIN),
        'initial_belt_head_path':BELT.relative_to(ROOT).as_posix(),'initial_belt_head_sha256':sha(BELT),
        'original_main_state_copied_exactly':True,'backbone_count':1,'checkpoint_count':1,'tasks':list(TASKS),'dataset_sources':manifest['sources'],
        'available_fit_records':manifest['available_fit_records'],'actually_scheduled_unique':manifest['unique_fit_records_scheduled'],
        'epochs':EPOCHS,'batch':BATCH,'seed':SEED,'draws_per_epoch':manifest['draws_per_epoch'],'actual_updates_planned':6400,
        'optimizer':'AdamW decay0.0002; backbone1e-5, old facility heads1e-4, belt head3e-4, foreign head5e-4; cosine20 eta5e-6',
        'BN_running_statistics':'All eval during training; affine remains trainable','gradient_clip':10,'teacher':'Original main frozen; facility known labels only',
        'facility_loss':'Masked photo focal + spatial0.5 + aux19*0.25 + original-main known Bernoulli KL4/T2',
        'belt_loss':'Same known-valid weighted BCE20 + Dice; other task labels unknown',
        'foreign_loss':'Binary BCE, balanced512/512 label draws; no location labels, other tasks unknown',
        'facility_input':'Same640 square bicubic as old main','belt_input':'Same640 letterbox, original author masks and valid padding',
        'foreign_input':'Native publisher RGB resized640 letterbox, different from old384 model',
        'augmentation':'Seeded TRAIN-only horizontal flip; masks flipped together; no new labels',
        'checkpoint_selection':'Final20 fixed; no per-epoch evaluation or best-epoch search',
        'facility_thresholds_fixed_from_old_main':thresholds,'belt_threshold_fixed':.5,
        'foreign_threshold':'Same declared CAL-only minmax FPR/FNR midpoint algorithm on320; computed before known TEST',
        'evaluation':'Known facility VAL1745, belt VAL353, reused foreign TEST448+783; no fresh field/sequence claim',
        'building_reserved_TEST_used':False,'belt_reserved_TEST_used':False,'foreign_TEST_used_for_fit_or_CAL':False,
        'normal_belt_photo_count':0,'belt_photo_FPR_is_null':True,'no_new_AI_or_human_labels':True,
        'PCB_or_IDEA_ND_data_used':False,'research_noncommercial_use':True,'app_default_promoted':False,
        'acceptance':'Report all per-source regressions; do not replace app automatically; all-photo5% cannot pass without normal belt gold',
        'tests_passed':3,'disposable_preflight_updates':5})
    print(json.dumps({'protocol':'frozen','inputs':len(records),'protected_sources':len(legacy),'original_main_sha256':sha(MAIN)}),flush=True)


def validate():
    p=read(PROTOCOL);assert p['source_sha256']==sources() and p['manifest_sha256']==sha(DATA/'manifest.json')
    assert p['draws_sha256']==sha(DATA/'draws.npz') and p['input_ledger_sha256']==sha(LEDGER)
    return p


def train():
    protocol=validate();repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for name in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])).hexdigest()==sha(ROOT/name)
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip();assert not RUN.exists();RUN.mkdir()
    manifest=read(DATA/'manifest.json');draws=np.load(DATA/'draws.npz',allow_pickle=False)
    caches={task:Cache(manifest['fit'][task],set(draws[task].flatten().tolist())) for task in TASKS}
    model,teacher=initialize();frozen=state_sha256(teacher);initial=state_sha256(model);supervision=weights(manifest)
    groups=[{'params':[p for name,p in model.named_parameters() if name.startswith(prefix)],'lr':lr,'name':prefix}
        for prefix,lr in [('backbone.',.00001),('belt_head.',.0003),('foreign_head.',.0005)]]
    groups.append({'params':[p for name,p in model.named_parameters() if not name.startswith(('backbone.','belt_head.','foreign_head.'))],'lr':.0001,'name':'original_facility_heads'})
    opt=torch.optim.AdamW(groups,weight_decay=.0002);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=.000005)
    history=[];updates=0;started=time.monotonic();torch.manual_seed(SEED)
    for epoch in range(1,EPOCHS+1):
        model.train();freeze_bn_statistics(model); rng=np.random.default_rng(SEED+epoch)
        chunks={task:[v for v in np.array_split(draws[task][epoch-1],len(draws[task][epoch-1])//BATCH)] for task in TASKS}
        order=[task for task in TASKS for _ in chunks[task]];rng.shuffle(order);at=Counter();losses=Counter()
        for task in order:
            selected=chunks[task][at[task]];at[task]+=1;images,truth=caches[task].batch(selected,training=True)
            with torch.no_grad():teacher_logits=teacher(images) if task=='facility_surface' else None
            opt.zero_grad(set_to_none=True);value=loss_for_task(task,model.forward_task(images,task),truth,supervision,teacher_logits)
            assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
            opt.step();updates+=1;losses[task]+=float(value.detach())
        scheduler.step();entry={'epoch':epoch,'optimizer_updates':updates,'task_updates':dict(at),
            'task_mean_loss':{task:losses[task]/at[task] for task in TASKS},'elapsed_minutes':(time.monotonic()-started)/60}
        new(RUN/f'epoch-{epoch:02d}.json',entry);history.append(entry);print(json.dumps(entry),flush=True)
    assert updates==6400 and state_sha256(model)!=initial and state_sha256(teacher)==frozen
    model.eval(); cpu=UnifiedMain().eval();cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)
    for task in TASKS:
        indices=draws[task][0][:2];images,_=caches[task].batch(indices)
        with torch.inference_mode():a=model.forward_task(images,task);b=cpu.forward_task(images.cpu(),task)
        if task=='facility_surface':a,b=a[0],b[0]
        assert torch.allclose(a.cpu(),b,atol=.0001,rtol=.001)
    # CAL changes the decision point only, after the fixed neural fit. TEST is still unscored.
    cal=[]
    with torch.inference_mode():
        rows=manifest['foreign_calibration']
        for offset in range(0,len(rows),BATCH):
            batch=rows[offset:offset+BATCH];images=normalize_batch(torch.stack([photo(r) for r in batch]))
            scores=model.forward_task(images,'conveyor_foreign').sigmoid().cpu().tolist()
            cal.extend({'id':r['id'],'split':'calibration','label':r['label'],'score':s} for r,s in zip(batch,scores))
    point=fit_threshold(cal);new(RUN/'foreign-calibration.json',{'rows':cal,**point,'neural_epochs_in_threshold_step':0})
    thresholds={'facility_surface':protocol['facility_thresholds_fixed_from_old_main'],'belt_damage':.5,'conveyor_foreign':point['threshold']}
    metadata={'architecture':ARCH,'input_size':640,'facility_classes':list(FACILITY_CLASSES),'tasks':list(TASKS),
        'one_shared_backbone':True,'thresholds':thresholds,'epochs':EPOCHS,'optimizer_updates':updates,
        'dataset_sources':manifest['sources'],'actually_scheduled_unique':manifest['unique_fit_records_scheduled'],
        'initial_main_sha256':sha(MAIN),'initial_state_sha256':initial,'final_state_sha256':state_sha256(model),
        'teacher_unchanged':True,'protocol_sha256':sha(PROTOCOL),'fit_protocol_commit':commit,
        'elapsed_training_and_CAL_minutes':(time.monotonic()-started)/60,'TEST_used_for_fit_or_CAL':False}
    torch.save({**metadata,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}},RUN/'final.pt')
    new(RUN/'history.json',history);new(ROOT/'reports/unified-main-training.json',{'status':'joint_main_trained_and_calibrated_before_TEST',
        **metadata,'weights_sha256':sha(RUN/'final.pt'),'strict_CPU_GPU_all_three_tasks':True,'checkpoint_count':1,'backbone_count':1})
    print('Existing main extended: one backbone, three heads, five sources, actual20epoch complete',flush=True)


def load_model():
    trained=read(ROOT/'reports/unified-main-training.json');assert sha(RUN/'final.pt')==trained['weights_sha256']
    checkpoint=torch.load(RUN/'final.pt',map_location='cpu',weights_only=True);model=UnifiedMain().cuda().eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True);return model,trained


def evaluate():
    validate();assert not (ROOT/'reports/unified-main-results.json').exists();model,trained=load_model();manifest=read(DATA/'manifest.json')
    facilities={}
    for domain,rows in manifest['facility_known_validation'].items():
        probabilities=[]
        with torch.inference_mode():
            for offset in range(0,len(rows),BATCH):
                batch=rows[offset:offset+BATCH];images=normalize_batch(torch.stack([photo(dict(r,task='facility_surface')) for r in batch]))
                probabilities.extend(model(images).sigmoid().cpu().tolist())
        old=read(MAIN.parent/f'target-validation-{domain}-grid1.json')['probabilities']
        assert len(old)==len(rows)
        result={}
        for i,c in enumerate(FACILITY_CLASSES):
            chosen=[j for j,r in enumerate(rows) if r['targets'][i]>=0];labels=[rows[j]['targets'][i] for j in chosen]
            threshold=trained['thresholds']['facility_surface'][c]
            result[c]={'baseline':classification_rates(labels,[old[j][i] for j in chosen],threshold),
                       'unified':classification_rates(labels,[probabilities[j][i] for j in chosen],threshold),'threshold':threshold}
        facilities[domain]=result;new(RUN/f'{domain}-validation.json',{'image_ids':[r['image'] for r in rows],'scores':probabilities,'per_class':result})
    belt=[];rows=manifest['belt_known_validation'];cache=Cache(rows,range(len(rows)))
    with torch.inference_mode():
        for offset in range(0,len(rows),BATCH):
            images,truth=cache.batch(range(offset,min(offset+BATCH,len(rows))));scores=model.forward_task(images,'belt_damage').sigmoid()
            for i,r in enumerate(rows[offset:offset+BATCH]):
                p=scores[i:i+1];valid=truth['valid'][i:i+1];mask=truth['mask'][i:i+1]
                belt.append({'id':r['id'],'sequence':r['sequence'],'photo_positive':True,
                    'score':float(p[valid.bool()].max()),'pixels':pixel_counts(p,mask,valid)})
    belt_result=belt_summary(belt);new(RUN/'belt-validation.json',{'rows':belt,'summary':belt_result})
    foreign={}
    for name,key in [('common448','foreign_known_test'),('known783','foreign_known_extra_test')]:
        rows=manifest[key];observations=[]
        with torch.inference_mode():
            for offset in range(0,len(rows),BATCH):
                batch=rows[offset:offset+BATCH];images=normalize_batch(torch.stack([photo(r) for r in batch]))
                scores=model.forward_task(images,'conveyor_foreign').sigmoid().cpu().tolist()
                observations.extend({'id':r['id'],'label':r['label'],'score':s} for r,s in zip(batch,scores))
        rates=classification_rates([r['label'] for r in observations],[r['score'] for r in observations],trained['thresholds']['conveyor_foreign'])
        foreign[name]=rates;new(RUN/f'foreign-{name}.json',{'rows':observations,'rates':rates,'threshold':trained['thresholds']['conveyor_foreign']})
    new(ROOT/'reports/unified-main-results.json',{'status':'evaluated','facility_fixed_old_thresholds':facilities,'belt_val353':belt_result,
        'belt_baseline_val353':read(ROOT/'reports/beltcrack-expansion-results.json')['expanded_val'],
        'foreign':foreign,'foreign_baseline':'reports/foreign-calibration-results.json','one_checkpoint_sha256':trained['weights_sha256'],
        'known_benchmarks_reused':True,'all_tasks_industrial_5percent_validated':False,'normal_belt_photo_labels':0,
        'checkpoint_selected_before_evaluation':True,'foreign_CAL_resubstitution_is_not_validation':True,
        'independent_factory_verified':False,'app_default_promoted':False})
    print(json.dumps({'belt':belt_result['photo_rates'],'foreign':foreign}),flush=True)


def verify():
    p=validate();ledger=read(LEDGER)
    for name,digest in p['legacy_source_sha256'].items():assert sha(ROOT/name)==digest,name
    for group in ['current_inputs','previous_models_and_reports']:
        def check(pair):
            name,record=pair;assert snapshot(ROOT/name)==record,name
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check,ledger[group].items()))
    history=read(RUN/'history.json');trained=read(ROOT/'reports/unified-main-training.json')
    assert [r['epoch'] for r in history]==list(range(1,21)) and history[-1]['optimizer_updates']==trained['optimizer_updates']==6400
    assert all(r['task_updates']=={'facility_surface':64,'belt_damage':128,'conveyor_foreign':128} for r in history)
    assert trained['strict_CPU_GPU_all_three_tasks'] and sha(RUN/'final.pt')==trained['weights_sha256']
    new(ROOT/'reports/unified-main-verification.json',{'status':'passed','actual_epochs':20,'actual_optimizer_updates':6400,
        'original_main_extended':True,'one_shared_backbone':True,'one_checkpoint':True,'three_task_gradients_verified':True,
        'five_source_preflight_gradients_verified':True,'strict_CPU_GPU_all_three':True,'tests_passed':3,'preflight_updates_excluded':5,
        'current_inputs_preserved':len(ledger['current_inputs']),'prior_artifacts_preserved':len(ledger['previous_models_and_reports']),
        'legacy_sources_preserved':len(p['legacy_source_sha256']),'new_labels':0,'app_default_promoted':False})
    print('Actual main extension, one checkpoint and original data/models preserved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','freeze','train','evaluate','verify'])
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'preflight':preflight,'freeze':freeze,'train':train,'evaluate':evaluate,'verify':verify}[args.mode]()
