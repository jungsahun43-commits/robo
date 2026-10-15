"""One preregistered cross-part contrastive trial on the same shared encoder."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.gear_cross_group import GearCrossGroupMain,VERSION,sample_cross_group_batch
from safelog_ai.gear_cross_group_loss import cross_group_loss
from safelog_ai.research_delta import pack_delta,restore_delta
from safelog_ai.process_probe import process_exists
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from scripts import study_gear_spatial_mil as spatial
from scripts import study_gear_only as global_study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_unified_belt_tiles import snapshot

RUN=ROOT/'runs/industrial-gear-cross-group-v33'
PROTOCOL=ROOT/'reports/industrial-gear-cross-group-v33-protocol.json'
MANIFEST=spatial.MANIFEST;START=spatial.START;EXPECTED=spatial.EXPECTED
WARM=global_study.RUN/'trained/final.pt';WARM_SHA='0669d9b988aa5ab3e4bfe414a078e140f61757d9bf06828dc56538cf474bf96f'
CHECKPOINT=RUN/'trained/final-delta.pt'
SOURCES=('safelog_ai/gear_cross_group.py','safelog_ai/gear_cross_group_loss.py','scripts/study_gear_cross_group.py','tests/test_gear_cross_group.py')
EPOCHS,STEPS,BATCH,SEED=30,64,16,3333
TEMPERATURE,AUXILIARY_WEIGHT=.2,.1


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(START)==EXPECTED and sha(WARM)==WARM_SHA
    probe=ROOT/'runs/industrial-gear-adapter-v31-probe'
    assert read(probe/'delta-test.json')['status']=='passed' and read(probe/'adapter-probe.json')['status']=='passed'
    assert shutil.disk_usage(ROOT).free>1024**3
    RUN.mkdir();ledger=read(ROOT/'runs/industrial-gear-task-adapter-v31/protected-inputs.json')
    for folder in (ROOT/'runs/industrial-gear-task-adapter-v31',ROOT/'runs/industrial-gear-selection-v32-diagnostic'):
        for path in folder.rglob('*'):
            if path.is_file() and not path.name.endswith(('.log','.lock')):
                n=path.relative_to(ROOT).as_posix()
                if n not in ledger:ledger[n]=snapshot(path)
    new(RUN/'protected-inputs.json',ledger)
    p={'UTC':datetime.now(timezone.utc).isoformat(),'schema':VERSION,'fixed_before_new_training_gradients_model_scores_optimizer':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'parent_sha256':EXPECTED,'warm_global_research_checkpoint_sha256':WARM_SHA,
       'manifest_sha256':sha(MANIFEST),'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),
       'legacy_source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for d in ('safelog_ai','scripts','tests') for f in (ROOT/d).glob('*.py') if f.relative_to(ROOT).as_posix() not in SOURCES},
       'fine_cache_sha256':{phase:sha(spatial.RUN/(phase+'-fine.pt')) for phase in ('fit','validation','cal','test')},
       'probe_receipts_sha256':{name:sha(probe/name) for name in ('delta-test.json','adapter-probe.json')},
       'v32_diagnostic_sha256':sha(ROOT/'reports/industrial-gear-selection-v32-diagnostic.json'),
       'data':'unchangedv28 gear-only FIT5466/VAL1440/CAL1116/TEST2179,groups8/3/3/4; original labels, fullsource failures/TEST and all exclusions preserved',
       'architecture':'same accepted encoder stage3 FP16fine -> zero-output24to4/GELU/4to24 residual -> original shared later modules ONCE -> FP16 maps/3072pooled descriptor -> frozen v28 FIT standardization clamp8 -> existing v28 initialized linearhead. The same normalized3072 embedding serves cross-group loss; no new encoder/projection, hooks or category/group inference input.',
       'initialization':'acceptedv16 preserved; v28 researchhead/FITmean/std, fresh zero-output220parameter adapter down seed3333. No failed epoch0 promoted as learned gain.',
       'initial_guard':'VAL BCE drift<=.005 from v28 before first optimizer, quantized tail as preregistered v31 probe atol.02/rtol.01. No claim of bitwise original fullRGB latent.',
       'trainable':'220adapter +3073globalhead only; accepted backbone and all old branches frozen/eval; v28 FIT mean/std frozen',
       'loss':'patch logitBCE +1.0*sum(globalhead.weight**2) +.1*cross_group_loss(normalized3072embedding, actual labels, groups,T=.2). Helper L2 normalizes each embedding, same-label different-group positives, same-group pairs excluded from numerator AND denominator. No pixelGT.',
       'sampling':'batch16: independently perlabel uniformly select4 distinct FITgroups with>=2 patches, then2 distinct patches pergroup. No duplicate photo within batch, replacement across steps only; no augmentation. Sampling+auxloss is one joint hypothesis, not isolated causal attribution.',
       'temperature':TEMPERATURE,'auxiliary_weight':AUXILIARY_WEIGHT,
       'optimizer':'AdamW adapterlr1e-4/headlr1e-5 wd1e-4 clip5, cosine factor .01+.99*.5*(1+cos(pi*epoch/30)); FP32 TF32off; seed3333',
       'epochs':EPOCHS,'steps':STEPS,'batch':BATCH,'seed':SEED,'planned_optimizer_updates':EPOCHS*STEPS,
       'VAL':'epoch0 and each completedepoch, actual-label macro(label,group) BCE only; earliest strictminimum, no auxiliary selection or last-weight promotion; epoch0 cannot release',
       'CAL':'Only selectedepoch>0 proceeds. NORMAL CAL sorted order statistic ceil(.85*n)-1, strict >. No labeled minimax/TEST cut search. Save threshold/checkpointSHA before TEST. epoch0 is rejected with no new CAL/TEST scoring. Normal85percent differs from older labeled-minimax studies, so do not attribute threshold-policy effects to learning.',
       'adoption':['selectedepoch>0','gearTEST eachlabel>=20 groups>=3','gearpatchFPR<=.20 andFNR<=.20','accepted oldstate exact','delta restores exact combinedstate','new packagedCPU/GPU/C++/aliases/old maps before release'],
       'storage':'existing2GBfine cache reused, selected/progress delta<5MB plus small logs/ledger, reserve>=1GiB; no original fullweight/cache deletion. Delta bound to accepted parentfile/state/added/combined SHA and exact reread restore, not standalone app delivery.',
       'cache':'v29 IDs/RGB SHA/manifest/geometry256/fine24@64 FP16/order/encoder SHA verified; no new RGB/features. Original prefix once and same halfboundary/adapter/tail once for later inference.',
       'resume':'atomic progress head+adapter/optimizer/scheduler/NumPy/TorchRNG; actual journal counts each real update including partialepoch replay; preserve failures',
       'new_unit_tests_passed':3,'existing_loss_unit_tests_passed':3,'independent_field_validated':False,
       'scope':'Re-GBC128gear surface-patch visible-wear acceptance, not machine function/internalfault/fullsite photo/verified damage location. Existing other categories untouched.',
       'license':'Re-GBCv1 CC BY4.0 cite author dataset/DOI10.1109/ETFA65518.2025.11205777; inherited sources keep their conditions; paper motivation, not reproduction'}
    new(PROTOCOL,p);print({'frozen':True,'planned_optimizer':1920,'new_RGB_features':0,'trainable_parameters':3293},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in ((START,EXPECTED),(WARM,WARM_SHA),(MANIFEST,p['manifest_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256'])):assert sha(path)==h,path
    if committed:
        git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def checkpoint_state(path):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value.get('schema')!='safelog.research-delta.v1':return value
    parent_path=(ROOT/value['parent_relative_path']).resolve();assert parent_path.is_relative_to(ROOT.resolve()) and sha(parent_path)==EXPECTED==value['parent_sha256']
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    return restore_delta(parent,value,sha(parent_path))


def load(path=WARM,initialize=False,device='cuda'):
    state=checkpoint_state(path);model=GearCrossGroupMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and (not missing or initialize and all(n.startswith('gear_task_adapter.') for n in missing))
    return model.eval().requires_grad_(False).to(device),state


def cache(phase,model):
    if phase=='test':assert (RUN/'calibration.json').exists()
    if phase=='cal':assert (RUN/'trained/training.json').exists()
    assert sha(spatial.RUN/(phase+'-fine.pt'))==read(PROTOCOL)['fine_cache_sha256'][phase]
    return spatial.cache(phase,model)


def logits_for(model,values):
    result=[]
    with torch.inference_mode():
        for i in range(0,len(values),BATCH):result.append(model.forward_gear_cross_group_features(values[i:i+BATCH].cuda())[0].cpu())
    return torch.cat(result)


def train():
    validate(committed=True);assert not (RUN/'trained/training.json').exists()
    model,_=load(initialize=True);fit,x=cache('fit',model);val,vx=cache('validation',model);x=x.cuda();vx=vx.cuda()
    y=torch.tensor([r['label'] for r in fit],dtype=torch.float32,device='cuda');vy=torch.tensor([r['label'] for r in val],dtype=torch.float32)
    initial=float(global_study.prior.macro_loss(F.binary_cross_entropy_with_logits(logits_for(model,vx),vy,reduction='none'),val));reference=read(global_study.RUN/'trained/training.json')['VAL_macro_group_BCE']
    assert abs(initial-reference)<=.005,'Initial quantized-tail drift; preserve and stop before optimizer'
    for module in (model.gear_task_adapter,model.gearbox_head):module.requires_grad_(True)
    parameters=[*model.gear_task_adapter.parameters(),*model.gearbox_head.parameters()];assert sum(p.numel() for p in parameters)==3293
    opt=torch.optim.AdamW([{'params':model.gear_task_adapter.parameters(),'lr':1e-4},{'params':model.gearbox_head.parameters(),'lr':1e-5}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:.01+.99*.5*(1+math.cos(math.pi*e/30)));rng=np.random.default_rng(SEED)
    progress=RUN/'progress.pt';journal=RUN/'optimizer.jsonl';start=0;best=initial;bestepoch=0
    def head_state():return {n:v for n,v in model.state_dict().items() if n.startswith(('gear_task_adapter.','gearbox_head.'))}
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);missing,extra=model.load_state_dict(saved['head'],strict=False);assert not extra
        opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);start,best,bestepoch=saved['epoch'],saved['best'],saved['bestepoch'];rng.bit_generator.state=saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        assert not journal.exists();journal.touch(exist_ok=False)
        new(RUN/'initial-VAL.json',{'VAL_macro_group_BCE':initial,'v28_reference_VAL':reference,'absolute_precision_drift':abs(initial-reference),'epoch':0})
        torch.save({'head':head_state(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'best-head.pt')
    actual=len(journal.read_text(encoding='utf-8').splitlines())
    with journal.open('a',encoding='utf-8',buffering=1) as out:
        for epoch in range(start+1,EPOCHS+1):
            model.gear_task_adapter.train();model.gearbox_head.train();bces=[];auxiliaries=[]
            for step in range(STEPS):
                ids=sample_cross_group_batch(fit,rng);groups=[str(fit[i]['group_id']) for i in ids]
                logits,embedding=model.forward_gear_cross_group_features(x[ids])
                bce=F.binary_cross_entropy_with_logits(logits,y[ids]);auxiliary=cross_group_loss(embedding,y[ids],groups,TEMPERATURE)
                loss=bce+model.gearbox_head.weight.square().sum()+AUXILIARY_WEIGHT*auxiliary;assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward()
                if actual==0:
                    assert all(v.grad is None for v in model.backbone.parameters())
                    assert float(model.gear_task_adapter[-1].weight.grad.abs().sum())>0
                gradient_norm=float(torch.nn.utils.clip_grad_norm_(parameters,5));assert math.isfinite(gradient_norm)
                opt.step();actual+=1
                out.write(json.dumps({'actual_update':actual,'epoch':epoch,'step':step+1,'BCE':float(bce.detach()),'cross_group_loss':float(auxiliary.detach()),'objective':float(loss.detach()),'gradient_norm_before_clip':gradient_norm,'distinct_photos':len(set(ids.tolist()))})+'\n')
                bces.append(float(bce.detach()));auxiliaries.append(float(auxiliary.detach()))
            scheduler.step();model.gear_task_adapter.eval();model.gearbox_head.eval();score=float(global_study.prior.macro_loss(F.binary_cross_entropy_with_logits(logits_for(model,vx),vy,reduction='none'),val))
            if score<best:
                best,bestepoch=score,epoch;torch.save({'head':head_state(),'epoch':epoch,'VAL':score,'actual_updates':actual},RUN/'best-head.pt')
            torch.save({'head':head_state(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'bestepoch':bestepoch,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},RUN/'progress.partial.pt');(RUN/'progress.partial.pt').replace(progress)
            new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'train_BCE':float(np.mean(bces)),'train_cross_group_loss':float(np.mean(auxiliaries)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual})
            print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'cross_group_loss':float(np.mean(auxiliaries)),'actual_optimizer':actual},flush=True)
    chosen=torch.load(RUN/'best-head.pt',map_location='cpu',weights_only=True);model.load_state_dict(chosen['head'],strict=False);model.eval()
    summary={'completed_epochs':EPOCHS,'actual_optimizer_updates':actual,'selected_epoch':chosen['epoch'],'selected_actual_updates':chosen['actual_updates'],'VAL_macro_group_BCE':chosen['VAL'],'initial_VAL_macro_group_BCE':initial,'FIT_patches':len(fit),'VAL_patches':len(val),'replayed_partial_epoch_updates':actual-EPOCHS*STEPS,'trainable_parameters':3293,'temperature':TEMPERATURE,'auxiliary_weight':AUXILIARY_WEIGHT}
    parent=torch.load(START,map_location='cpu',weights_only=True);result=deepcopy(parent);result['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()}
    result.update(gear_cross_group_training=summary,gear_cross_group_protocol_sha256=sha(PROTOCOL),gear_cross_group_initial_global_sha256=WARM_SHA,gear_cross_group_scope='gear128surface_acceptance_not_verified_location')
    payload=pack_delta(parent,result,EXPECTED,START.relative_to(ROOT).as_posix(),('gearbox_','gear_task_adapter.'))
    (RUN/'trained').mkdir()
    with CHECKPOINT.open('xb') as out:torch.save(payload,out)
    restored=checkpoint_state(CHECKPOINT)
    for n,v in result['state_dict'].items():assert torch.equal(v,restored['state_dict'][n]),n
    for n,v in result.items():
        if n!='state_dict':assert v==restored[n],n
    new(RUN/'trained/training.json',summary)
    new(RUN/'trained/delta-verification.json',{'all_state_and_metadata_exact':True,'delta_sha256':sha(CHECKPOINT),'delta_bytes':CHECKPOINT.stat().st_size,'parent_sha256':EXPECTED,'combined_state_sha256':payload['combined_state_sha256'],'old_files_deleted':0})


def calibrate():
    validate();assert not (RUN/'calibration.json').exists()
    if read(RUN/'trained/training.json')['selected_epoch']==0:
        new(RUN/'calibration.json',{'skipped':True,'reason':'selectedepoch0_ineligible_before_CAL_TEST','new_CAL_scores':0,'new_TEST_scores':0,'research_delta_sha256':sha(CHECKPOINT)})
        return
    validate();assert not (RUN/'calibration.json').exists();model,_=load(CHECKPOINT);rows,features=cache('cal',model);scores=logits_for(model,features).sigmoid().tolist();labels=[r['label'] for r in rows];normal=sorted(s for s,label in zip(scores,labels) if label==0);assert len(normal)>=20;cut=normal[math.ceil(.85*len(normal))-1]
    new(RUN/'calibration.json',{'research_delta_sha256':sha(CHECKPOINT),'threshold':cut,'policy':'normal85percent_order_statistic_strict_greater','normal_CAL_patches':len(normal),'rates':classification_rates(labels,scores,cut),'newTEST_scores_before_this':False})
    new(RUN/'calibration-scores.json',{'ids':[r['id'] for r in rows],'scores':scores})


def evaluate():
    validate();assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['research_delta_sha256']==sha(CHECKPOINT)
    if cal.get('skipped'):
        result={'UTC':datetime.now(timezone.utc).isoformat(),'research_delta_sha256':sha(CHECKPOINT),'training':read(RUN/'trained/training.json'),'eligible_pending_package_checks':False,'CAL_TEST_skipped':True,'reason':cal['reason'],'new_CAL_scores':0,'new_TEST_scores':0,'TEST_patches':0,'new_independent_field_validation':False,'existing_bracket_tube_transferred_or_replaced':False}
        new(RUN/'selection.json',result);new(ROOT/'reports/industrial-gear-cross-group-v33-selection.json',result);print(result,flush=True)
        return
    model,_=load(CHECKPOINT);rows,features=cache('test',model);scores=logits_for(model,features).sigmoid().tolist();labels=[r['label'] for r in rows];rates=classification_rates(labels,scores,cal['threshold']);by={}
    for field in ('physical_part_id','group_id'):
        by[field]={}
        for g in sorted({r[field] for r in rows},key=str):
            ids=[i for i,r in enumerate(rows) if r[field]==g];by[field][str(g)]=classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['threshold'])
    train=read(RUN/'trained/training.json');groups=sorted({r['group_id'] for r in rows});eligible=train['selected_epoch']>0 and len(groups)>=3 and min(rates['positive_photos'],rates['negative_photos'])>=20 and max(rates['fpr'],rates['fnr'])<=.20
    rng=np.random.default_rng(SEED+99);intervals=[]
    for _ in range(1000):
        sample=rng.choice(groups,len(groups),replace=True);ids=[i for g in sample for i,r in enumerate(rows) if r['group_id']==g];value=classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['threshold'])
        if value['fpr'] is not None and value['fnr'] is not None:intervals.append([value['fpr'],value['fnr']])
    result={'UTC':datetime.now(timezone.utc).isoformat(),'research_delta_sha256':sha(CHECKPOINT),'training':train,'threshold':cal['threshold'],'rates':rates,'AUROC':rank_auroc(labels,scores),'by':by,'eligible_pending_package_checks':bool(eligible),'TEST_patches':len(rows),'TEST_connected_groups':len(groups),'group_bootstrap_95percent':np.quantile(intervals,[.025,.975],axis=0).tolist(),'bootstrap_valid_draws':len(intervals),'new_independent_field_validation':False,'whole_equipment_health_or_verified_location':False,'existing_bracket_tube_transferred_or_replaced':False}
    new(RUN/'test-scores.json',{'ids':[r['id'] for r in rows],'scores':scores});new(RUN/'selection.json',result);new(ROOT/'reports/industrial-gear-cross-group-v33-selection.json',result)
    print({k:v for k,v in result.items() if k not in ('by','group_bootstrap_95percent')},flush=True)


def verify():
    p=validate();s=read(RUN/'selection.json');parent=torch.load(START,map_location='cpu',weights_only=True);candidate=checkpoint_state(CHECKPOINT);warm=torch.load(WARM,map_location='cpu',weights_only=True)
    for n,v in parent['state_dict'].items():assert torch.equal(v,candidate['state_dict'][n]),n
    for n,v in parent.items():
        if n!='state_dict':assert v==candidate[n],n
    for n in ('gearbox_mean','gearbox_std'):assert torch.equal(warm['state_dict'][n],candidate['state_dict'][n]),n
    for n,h in p['legacy_source_sha256'].items():assert sha(ROOT/n)==h,n
    unchanged=rehashed=0
    for n,before in read(RUN/'protected-inputs.json').items():
        path=ROOT/n;st=path.stat()
        if st.st_size==before['size_bytes'] and st.st_mtime_ns==before['mtime_ns']:unchanged+=1
        else:assert sha(path)==before['sha256'],n;rehashed+=1
    journal=[json.loads(v) for v in (RUN/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()];assert all(v['distinct_photos']==16 and math.isfinite(v['cross_group_loss']) for v in journal);assert len(journal)==s['training']['actual_optimizer_updates'] and len(journal)>=1920 and [v['actual_update'] for v in journal]==list(range(1,len(journal)+1))
    for phase,h in p['fine_cache_sha256'].items():assert sha(spatial.RUN/(phase+'-fine.pt'))==h
    delta=read(RUN/'trained/delta-verification.json');assert delta['delta_sha256']==sha(CHECKPOINT) and delta['all_state_and_metadata_exact']
    result={'status':'passed','accepted_v16_all_tensor_buffer_metadata_exact':True,'v28_FIT_mean_std_exact':True,'legacy_python_SHA_checked':len(p['legacy_source_sha256']),'protected_stat_unchanged':unchanged,'changed_metadata_SHA_rechecked':rehashed,'all_protected_SHA_reread':False,'actual_optimizer_updates':len(journal),'new_unit_tests_passed':3,'existing_loss_unit_tests_passed':3,'research_delta_bytes':CHECKPOINT.stat().st_size,'research_delta_restore_exact':True,'new_RGB_encoderfeatures':0,'eligible_pending_package_checks':s['eligible_pending_package_checks'],'new_API_CPP_ZIP_checks':0,'active_v16_retained_pending_release':True}
    new(RUN/'verification.json',result);new(ROOT/'reports/industrial-gear-cross-group-v33-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run','train','calibrate','evaluate','verify'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        validate(committed=True)
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work '+str(path))
        lock=RUN/'operation.lock'
        if lock.exists():raise RuntimeError('Preserve and review stale lock before resuming')
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:
            if args.stage=='run':train();calibrate();evaluate();verify()
            else:globals()[args.stage]()
        except BaseException as error:
            new(RUN/(args.stage+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'pid':os.getpid()});raise
        finally:lock.unlink()
