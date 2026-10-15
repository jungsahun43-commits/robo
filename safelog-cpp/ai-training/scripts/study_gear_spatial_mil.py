"""One preregistered fine24 spatial MIL study; retain accepted v16 and old outputs."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.gear_spatial_mil import GearSpatialMain,VERSION
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.gearbox_supervised import gearbox_descriptor
from safelog_ai.gear_only_cache import select_descriptors
from safelog_ai.process_probe import process_exists
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from scripts import study_gear_only as prior
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_bsclass_supervised import calibrated_threshold

RUN=ROOT/'runs/industrial-gear-spatial-v29'
PROTOCOL=ROOT/'reports/industrial-gear-spatial-v29-protocol.json'
MANIFEST=prior.MANIFEST;START=prior.prior.START;EXPECTED=prior.prior.EXPECTED
SOURCES=('safelog_ai/gear_spatial_mil.py','scripts/study_gear_spatial_mil.py','tests/test_gear_spatial_mil.py')
EPOCHS,STEPS,SEED,BATCH=30,64,2929,32


def rows_for(phase):return [r for r in read(MANIFEST)['rows'] if r['phase']==phase]


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(START)==EXPECTED
    assert read(prior.RUN/'verification.json')['status']=='passed'
    assert sha(MANIFEST)=='fe7fce446e34002d47a24c0ad6b106468e8c128209739164d2e511705ae8779d'
    probe=ROOT/'runs/industrial-gear-spatial-probe-v29/probe.json'
    assert read(probe)['maximum_fine_descriptor_abs_delta']==0
    counts={p:len(rows_for(p)) for p in ('fit','validation','cal','test')}
    feature_bytes=sum(counts.values())*24*64*64*2
    assert counts=={'fit':5466,'validation':1440,'cal':1116,'test':2179}
    assert shutil.disk_usage(ROOT).free>feature_bytes+3*1024**3
    RUN.mkdir();ledger=read(prior.RUN/'protected-inputs.json')
    for folder in (prior.RUN,probe.parent):
        for path in folder.rglob('*'):
            if path.is_file() and not path.name.endswith(('.log','.lock')):
                n=path.relative_to(ROOT).as_posix()
                if n not in ledger:ledger[n]=snapshot(path)
    new(RUN/'protected-inputs.json',ledger)
    p={'UTC':datetime.now(timezone.utc).isoformat(),'schema':VERSION,
       'fixed_before_full_new_cache_gradients_optimizer_VAL_CAL_TEST_scores':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'parent_sha256':EXPECTED,'manifest_sha256':sha(MANIFEST),
       'probe_sha256':sha(probe),'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),
       'legacy_source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for d in ('safelog_ai','scripts','tests') for f in (ROOT/d).glob('*.py') if f.relative_to(ROOT).as_posix() not in SOURCES},
       'original_descriptor_cache_SHA256':{phase:sha(prior.prior.RUN/(phase+'-features.pt')) for phase in ('fit','validation','cal','test')},
       'data':read(MANIFEST)['audit'],'full_fine_cache_tensor_bytes':feature_bytes,
       'disk_gate':'fine tensor bytes +3GiB reserve before extraction; preserve all old caches/weights/raws',
       'input':'original128x128 RGB patch bicubic256square/ImageNet normalization, identical source-independent geometry, no TEST/GT crop',
       'cache':'same frozen encoder fine24@64x64 FP16, IDs/imageSHA/native128/input256/encoderSHA/protocolSHA fixed; raw feature maps not pooled descriptors',
       'cache_compatibility':'new fine mean/popstd/max72 FP16 vs existing descriptor prefix, atol.002 rtol.002 every input; first FITbatch ordinarylow/high bitwiseexact; no tolerance relaxation',
       'normalization':'FIT only, FP64 sum/sumsquared over every FIT patch/pixel per24channels, population variance max0, std floor1e-3, FP32buffers, clamp[-8,8]',
       'head':'Conv24to32 kernel3 pad1,GroupNorm4,GELU,Conv32to1 kernel1 zero weight/bias; one new head only; mean of topceil10percent maplogits becomes patchlogit; no location GT',
       'trainable':'only gear_spatial_head; all original encoder/heads/tensors/buffers/metadata frozen, new mean/std set once from FIT',
       'loss':'scalar patch logit BCE against actual publisher OK/nOK. No pixel targets or mask loss.',
       'sampling':'batch32 16OK/16nOK; within eachlabel inverse group patch count, uniformgroup then patch effect, replacement not new photos',
       'augmentation':'independent H/V flips probability.5 of cachedfine feature maps, not new flippedRGB encoder passes; patchlabel preserved, no mask created',
       'optimizer':'AdamW lr1e-4 wd1e-4 cosine30 min1e-6 clip5 FP32 TF32off',
       'epochs':30,'steps':64,'batch':32,'seed':2929,'planned_optimizer_updates':1920,
       'VAL':'epoch0 and each completeepoch; same perlabel pergroup meanBCE then macro two labels; earliest strictminimum, epoch0 cannotrelease',
       'CAL':'selectedhead only; labeledCAL minimax FPR/FNR same ties sum/distance.5/lowercut; strict>; save before TEST newfeatures/scores',
       'adoption':['selectedepoch>0','TEST eachlabel>=20 connectedgroups>=3','gearpatch FPR<=.20 and FNR<=.20','all old state exactly unchanged','new packagedCPU/GPU/C++/aliases/oldmaps before apprelease'],
       'interpretation':'gear128patch acceptance only, same previously seen publicTEST, no fullgearbox health/internalfailure/verifiedlocation, no existingbracket/tube transfer or replacement',
       'resume':'per-epoch head/optimizer/scheduler/RNG, actual stepjournal including repeated partialepoch updates; preserve failed/partial state; completed studies never rerun',
       'new_unit_tests_passed':3,'license':'Re-GBCv1 CC BY4.0 cite originaldataset and DOI10.1109/ETFA65518.2025.11205777; inherited sources retain own conditions',
       'independent_field_validated':False}
    new(PROTOCOL,p);print({'frozen':True,'counts':counts,'fine_tensor_bytes':feature_bytes,'planned_optimizer':1920},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in ((START,EXPECTED),(MANIFEST,p['manifest_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256'])):assert sha(path)==h,path
    if committed:
        git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def load(path=START,device='cuda'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=GearSpatialMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and all(n.startswith('gear_spatial_') for n in missing)
    return model.eval().requires_grad_(False).to(device),state


def reference_fine(phase,rows):
    p=read(PROTOCOL);ids=[];features=[]
    for oldphase in (('test',) if phase=='test' else ('fit','validation','cal')):
        path=prior.prior.RUN/(oldphase+'-features.pt');assert sha(path)==p['original_descriptor_cache_SHA256'][oldphase]
        value=torch.load(path,map_location='cpu',weights_only=True)
        assert value['ids']==[r['id'] for r in prior.prior.rows_for(oldphase)]
        ids.extend(value['ids']);features.append(value['features'][:,:72])
    return select_descriptors([r['id'] for r in rows],ids,torch.cat(features)).float()


def cache(phase,model):
    p=read(PROTOCOL);rows=rows_for(phase);path=RUN/(phase+'-fine.pt');receipt=RUN/(phase+'-cache.json')
    if phase=='test':assert (RUN/'calibration.json').exists()
    if phase=='cal':assert (RUN/'trained/training.json').exists()
    if path.exists():
        assert sha(path)==read(receipt)['sha256']
        value=torch.load(path,map_location='cpu',weights_only=True)
        assert value['ids']==[r['id'] for r in rows] and value['image_SHA256']==[r['image_sha256'] for r in rows]
        assert value['manifest_sha256']==sha(MANIFEST) and value['protocol_sha256']==sha(PROTOCOL) and value['encoder_checkpoint_sha256']==EXPECTED
        return rows,value['fine']
    expected=reference_fine(phase,rows);values=[];maximum_delta=0.;exact=None
    with torch.inference_mode():
        for i in range(0,len(rows),32):
            images=prior.prior.images(rows[i:i+32]).cuda();features=forward_with_fine(model.backbone,images)
            fine=features['fine'].half().cpu();assert tuple(fine.shape[1:])==(24,64,64)
            if phase=='fit' and i==0:
                ordinary=model.backbone(images);exact=all(torch.equal(ordinary[k],features[k]) for k in ('low','high'));assert exact
            descriptor=gearbox_descriptor(features).cpu()[:,:72];target=expected[i:i+len(fine)]
            delta=float((descriptor-target).abs().max());maximum_delta=max(maximum_delta,delta)
            if not torch.allclose(descriptor,target,atol=.002,rtol=.002):
                new(RUN/(phase+f'-cache-mismatch-{i}.json'),{'start':i,'maximum_abs_delta':delta,'fixed_atol':.002,'fixed_rtol':.002});raise RuntimeError('Fine descriptor incompatible; do not relax fixed tolerance')
            values.append(fine)
            if i%1024==0:print({'cache':phase,'patches':min(i+32,len(rows)),'total':len(rows)},flush=True)
    saved={'ids':[r['id'] for r in rows],'image_SHA256':[r['image_sha256'] for r in rows],'fine':torch.cat(values),
           'manifest_sha256':sha(MANIFEST),'protocol_sha256':sha(PROTOCOL),'encoder_checkpoint_sha256':EXPECTED,'original_size':[128,128],'input_size':[256,256]}
    partial=path.with_suffix('.partial.pt')
    with partial.open('xb') as out:torch.save(saved,out)
    partial.rename(path)
    new(receipt,{'sha256':sha(path),'bytes':path.stat().st_size,'patches':len(rows),'shape':list(saved['fine'].shape),
        'maximum_fine_descriptor_abs_delta':maximum_delta,'canonical_descriptor_tolerance_passed':True,'firstFIT_low_high_bitwiseexact':exact})
    return rows,saved['fine']


def fit_stats(values):
    total=torch.zeros(24,dtype=torch.float64);squared=total.clone();count=0
    for i in range(0,len(values),64):
        chunk=values[i:i+64].double();total+=chunk.sum((0,2,3));squared+=chunk.square().sum((0,2,3));count+=len(chunk)*64*64
    mean=total/count;std=(squared/count-mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
    return mean.float().reshape(1,24,1,1),std.float().reshape(1,24,1,1)


def logits_for(model,values):
    result=[]
    with torch.inference_mode():
        for i in range(0,len(values),BATCH):result.append(model.forward_gear_spatial_features(values[i:i+BATCH].cuda()).cpu())
    return torch.cat(result)


def train():
    validate(committed=True);assert not (RUN/'trained/training.json').exists()
    model,parent=load();fit,x=cache('fit',model);val,vx=cache('validation',model)
    mean,std=fit_stats(x);model.gear_spatial_mean.copy_(mean.cuda());model.gear_spatial_std.copy_(std.cuda())
    new_stats=RUN/'FIT-statistics.pt'
    if not new_stats.exists():torch.save({'mean':mean,'std':std,'patches':len(fit),'manifest_sha256':sha(MANIFEST)},new_stats)
    else:
        old=torch.load(new_stats,map_location='cpu',weights_only=True);assert torch.equal(old['mean'],mean) and torch.equal(old['std'],std)
    x=x.cuda();vx=vx.cuda();y=torch.tensor([r['label'] for r in fit],dtype=torch.float32,device='cuda');vy=torch.tensor([r['label'] for r in val],dtype=torch.float32)
    model.gear_spatial_head.requires_grad_(True);opt=torch.optim.AdamW(model.gear_spatial_head.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6);rng=np.random.default_rng(SEED)
    progress=RUN/'progress.pt';journal=RUN/'optimizer.jsonl';start=0;best=float('inf');bestepoch=0
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);model.gear_spatial_head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start,best,bestepoch=saved['epoch'],saved['best'],saved['bestepoch'];rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        assert not journal.exists();journal.touch(exist_ok=False)
        best=float(prior.prior.macro_loss(F.binary_cross_entropy_with_logits(logits_for(model,vx),vy,reduction='none'),val))
        new(RUN/'initial-VAL.json',{'VAL_macro_group_BCE':best,'epoch':0})
        torch.save({'head':model.gear_spatial_head.state_dict(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'best-head.pt')
    actual=len(journal.read_text(encoding='utf-8').splitlines());pools={};probs={}
    for label in (0,1):
        ids=[i for i,r in enumerate(fit) if r['label']==label];counts=Counter(fit[i]['group_id'] for i in ids);w=np.array([1/counts[fit[i]['group_id']] for i in ids],dtype=float)
        pools[label]=ids;probs[label]=w/w.sum()
    with journal.open('a',encoding='utf-8',buffering=1) as out:
        for epoch in range(start+1,EPOCHS+1):
            model.gear_spatial_head.train();losses=[]
            for step in range(STEPS):
                ids=np.concatenate([rng.choice(pools[label],16,replace=True,p=probs[label]) for label in (0,1)])
                batch=x[ids]
                horizontal=torch.rand(len(batch),device='cuda')<.5;vertical=torch.rand(len(batch),device='cuda')<.5
                batch=torch.where(horizontal[:,None,None,None],batch.flip(-1),batch);batch=torch.where(vertical[:,None,None,None],batch.flip(-2),batch)
                loss=F.binary_cross_entropy_with_logits(model.forward_gear_spatial_features(batch),y[ids]);assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.gear_spatial_head.parameters(),5);opt.step()
                actual+=1;out.write(json.dumps({'actual_update':actual,'epoch':epoch,'step':step+1,'BCE':float(loss.detach())})+'\n');losses.append(float(loss.detach()))
            scheduler.step();model.gear_spatial_head.eval()
            score=float(prior.prior.macro_loss(F.binary_cross_entropy_with_logits(logits_for(model,vx),vy,reduction='none'),val))
            if score<best:
                best,bestepoch=score,epoch;torch.save({'head':model.gear_spatial_head.state_dict(),'epoch':epoch,'VAL':score,'actual_updates':actual},RUN/'best-head.pt')
            saved={'head':model.gear_spatial_head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'bestepoch':bestepoch,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
            torch.save(saved,RUN/'progress.partial.pt');(RUN/'progress.partial.pt').replace(progress)
            new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'train_BCE':float(np.mean(losses)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual})
            print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
    chosen=torch.load(RUN/'best-head.pt',map_location='cpu',weights_only=True);model.gear_spatial_head.load_state_dict(chosen['head']);model.eval()
    summary={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':chosen['epoch'],'selected_actual_updates':chosen['actual_updates'],'VAL_macro_group_BCE':chosen['VAL'],'FIT_patches':len(fit),'VAL_patches':len(val),'replayed_partial_epoch_updates':actual-1920}
    result=deepcopy(parent);result['state_dict']={n:t.detach().cpu() for n,t in model.state_dict().items()};result.update(gear_spatial_protocol_sha256=sha(PROTOCOL),gear_spatial_training=summary,gear_spatial_scope='gear128patch_acceptance_not_verified_location')
    (RUN/'trained').mkdir()
    with (RUN/'trained/final.pt').open('xb') as out:torch.save(result,out)
    new(RUN/'trained/training.json',summary)


def calibrate():
    validate();assert not (RUN/'calibration.json').exists()
    model,_=load(RUN/'trained/final.pt');rows,features=cache('cal',model);scores=logits_for(model,features).sigmoid().tolist();labels=[r['label'] for r in rows]
    cut=calibrated_threshold(labels,scores)
    new(RUN/'calibration.json',{'checkpoint_sha256':sha(RUN/'trained/final.pt'),'threshold':cut,'rates':classification_rates(labels,scores,cut),'TEST_new_features_or_scores_before_this':False})
    new(RUN/'calibration-scores.json',{'ids':[r['id'] for r in rows],'scores':scores})


def evaluate():
    validate();assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['checkpoint_sha256']==sha(RUN/'trained/final.pt')
    model,_=load(RUN/'trained/final.pt');rows,features=cache('test',model);scores=logits_for(model,features).sigmoid().tolist();labels=[r['label'] for r in rows]
    rates=classification_rates(labels,scores,cal['threshold']);by={}
    for field in ('physical_part_id','group_id'):
        by[field]={}
        for g in sorted({r[field] for r in rows},key=str):
            ids=[i for i,r in enumerate(rows) if r[field]==g];by[field][str(g)]=classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['threshold'])
    train=read(RUN/'trained/training.json');groups=sorted({r['group_id'] for r in rows})
    eligible=train['selected_epoch']>0 and len(groups)>=3 and min(rates['positive_photos'],rates['negative_photos'])>=20 and max(rates['fpr'],rates['fnr'])<=.20
    rng=np.random.default_rng(SEED+99);intervals=[]
    for _ in range(1000):
        sample=rng.choice(groups,len(groups),replace=True);ids=[i for g in sample for i,r in enumerate(rows) if r['group_id']==g]
        value=classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['threshold'])
        if value['fpr'] is not None and value['fnr'] is not None:intervals.append([value['fpr'],value['fnr']])
    result={'UTC':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':cal['checkpoint_sha256'],'training':train,'threshold':cal['threshold'],'rates':rates,'AUROC':rank_auroc(labels,scores),'by':by,'eligible_pending_package_checks':bool(eligible),'TEST_patches':len(rows),'TEST_connected_groups':len(groups),'group_bootstrap_95percent':np.quantile(intervals,[.025,.975],axis=0).tolist(),'bootstrap_valid_draws':len(intervals),'new_independent_field_validation':False,'whole_equipment_health_or_verified_location':False,'existing_bracket_tube_transferred_or_replaced':False}
    new(RUN/'test-scores.json',{'ids':[r['id'] for r in rows],'scores':scores});new(RUN/'selection.json',result);new(ROOT/'reports/industrial-gear-spatial-v29-selection.json',result)
    print({k:v for k,v in result.items() if k not in ('by','group_bootstrap_95percent')},flush=True)


def verify():
    p=validate();selection=read(RUN/'selection.json');parent=torch.load(START,map_location='cpu',weights_only=True);candidate=torch.load(RUN/'trained/final.pt',map_location='cpu',weights_only=True)
    for name,tensor in parent['state_dict'].items():assert torch.equal(tensor,candidate['state_dict'][name]),name
    for name,value in parent.items():
        if name!='state_dict':assert value==candidate[name],name
    for name,digest in p['legacy_source_sha256'].items():assert sha(ROOT/name)==digest,name
    unchanged=rehashed=0
    for name,before in read(RUN/'protected-inputs.json').items():
        path=ROOT/name;st=path.stat()
        if st.st_size==before['size_bytes'] and st.st_mtime_ns==before['mtime_ns']:unchanged+=1
        else:assert sha(path)==before['sha256'],name;rehashed+=1
    journal=[json.loads(v) for v in (RUN/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(journal)==selection['training']['actual_optimizer_updates'] and len(journal)>=1920 and [v['actual_update'] for v in journal]==list(range(1,len(journal)+1))
    for phase in ('fit','validation','cal','test'):assert sha(RUN/(phase+'-fine.pt'))==read(RUN/(phase+'-cache.json'))['sha256']
    result={'status':'passed','parent_all_tensor_buffer_metadata_exact':True,'legacy_python_SHA_checked':len(p['legacy_source_sha256']),'protected_stat_unchanged':unchanged,'changed_metadata_SHA_rechecked':rehashed,'all_protected_SHA_reread':False,'actual_optimizer_updates':len(journal),'new_unit_tests_passed':3,'eligible_pending_package_checks':selection['eligible_pending_package_checks'],'new_API_CPP_ZIP_checks':0,'active_v16_retained_pending_release':True,'new_spatial_maps_are_not_verified_defect_location':True}
    new(RUN/'verification.json',result);new(ROOT/'reports/industrial-gear-spatial-v29-verification.json',result);print(result,flush=True)


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
