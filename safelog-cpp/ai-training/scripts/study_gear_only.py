"""Predeclared gear-only task; same linear optimizer and immutable source descriptors."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from safelog_ai.gearbox_linear import LinearGearboxMain
from safelog_ai.process_probe import process_exists
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from scripts import study_gearbox_supervised as prior
from scripts.prepare_gear_only import MANIFEST
from safelog_ai.gear_only_cache import select_descriptors
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_bsclass_supervised import calibrated_threshold

RUN = ROOT/'runs/industrial-gear-only-v28'
PROTOCOL = ROOT/'reports/industrial-gear-only-v28-protocol.json'
SOURCES = ('safelog_ai/gear_only_cache.py','scripts/study_gear_only.py','tests/test_gear_only_cache.py')
EPOCHS,STEPS,SEED = 30,64,2828


def freeze():
    assert RUN.exists() and not PROTOCOL.exists()
    ready=read(RUN/'data-ready.json');assert ready['ready'] and ready['manifest_sha256']==sha(MANIFEST)
    old=ROOT/'runs/industrial-gearbox-linear-v27'
    assert read(old/'verification.json')['status']=='passed'
    ledger=read(old/'protected-inputs.json')
    for path in old.rglob('*'):
        if path.is_file() and not path.name.endswith(('.log','.lock')):
            n=path.relative_to(ROOT).as_posix()
            if n not in ledger:ledger[n]=snapshot(path)
    new(RUN/'protected-inputs.json',ledger)
    p={'UTC':datetime.now(timezone.utc).isoformat(),'schema':'safelog-gear-only-v28',
        'fixed_before_new_gradients_or_new_task_model_scores':True,
        'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'parent_sha256':prior.EXPECTED,
        'linear_architecture_source_sha256':sha(ROOT/'safelog_ai/gearbox_linear.py'),
        'manifest_sha256':sha(MANIFEST),'prior_protocol_sha256':sha(prior.PROTOCOL),
        'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),
        'legacy_source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for d in ('safelog_ai','scripts','tests') for f in (ROOT/d).glob('*.py') if f.relative_to(ROOT).as_posix() not in SOURCES},
        'cached_features_SHA256':{p:sha(prior.RUN/(p+'-features.pt')) for p in ('fit','validation','cal','test')},
        'source_runner_template_sha256':sha(ROOT/'scripts/study_gearbox_linear.py'),
        'task_scope':'ONLY gear128surface patch acceptance, distinct from full-source v26/v27 failed task. Not overall gearbox improvement or other components.',
        'data':read(MANIFEST)['audit'],
        'source_reuse':'regroup only original eligible publisherTRAIN gear groups, previously viewed FIT examples forcedFIT; same original reservedTEST gear subset2179; no TEST/excludedTRAIN moved',
        'initialization':'unchanged acceptedv16 parent plus fresh zero-initialized Linear3072to1, no old fitted new-head weights reused',
        'descriptor':'same frozen encoder fine24/low40/high960 mean/popstd/max3072 FP16roundtrip; sourceID-join from old fit/val/cal caches for development ONLY, oldTESTcache read only after newCALsaved',
        'standardization':'new FIT-only mean/popstd floor1e-3 clamp[-8,8], no oldCAL mean used',
        'trainable':'new linear head weight/bias only; preserve all original encoder/heads/buffers/metadata',
        'loss':'balancedbatch meanBCE+1.0*sum(weight**2), bias unpenalized; same v27 settings',
        'sampling':'32OK/32nOK, inverse label-specific group patch count, seed2828; no augmentation',
        'optimizer':'Adam lr1e-4 weightdecay0 cosine30 min1e-6 clip5 FP32 TF32off',
        'epochs':30,'steps':64,'batch':64,'seed':2828,'planned_optimizer_updates':1920,
        'VAL':'same macro label/groupBCE, epoch0 and every completedepoch, earliest strictminimum; initialepoch0 cannotrelease',
        'CAL':'same labeledCAL minimaxFPR/FNR threshold and tie rule; save before newTEST scoring, no threshold variants',
        'adoption':['selected_epoch>0','TEST both labels>=20,connected groups>=3','gear patch FPR<=.20,FNR<=.20','old tensor/buffer/metadata exact','fresh packagedCPU/GPU/C++ and original aliases/maps before release'],
        'old_full_source_failures_preserved':True,'existing_bracket_tube_not_evaluated_or_replaced':True,
        'new_unit_tests_passed':2,'new_RGB_encoder_features':0,
        'independent_field_validation':False,'development_and_TEST_previously_scored_in_other_failed_tasks':True,
        'label_scope':'publisher gear128surface acceptance, not whole machine functioning or verified pixel localization',
        'license':'Re-GBCv1 CC BY4.0 cite originalauthor dataset and DOI10.1109/ETFA65518.2025.11205777; inherited licenses unchanged'}
    new(PROTOCOL,p);print({'frozen':True,'counts':ready['audit']['counts'],'planned_optimizer':1920},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items(): assert sha(ROOT/n)==h,n
    assert sha(MANIFEST)==p['manifest_sha256'] and sha(prior.START)==p['parent_sha256']
    assert sha(prior.PROTOCOL)==p['prior_protocol_sha256'] and sha(RUN/'protected-inputs.json')==p['protected_inputs_sha256']
    if committed:
        git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):
            assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def load(path=prior.START):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=LinearGearboxMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and all(n.startswith('gearbox_') for n in missing)
    return model.eval().requires_grad_(False).cuda(),state


def cache(phase):
    p=read(PROTOCOL)
    rows=[r for r in read(MANIFEST)['rows'] if r['phase']==phase]
    source_ids=[];values=[]
    phases=('test',) if phase=='test' else ('fit','validation','cal')
    if phase=='test':assert (RUN/'calibration.json').exists()
    for oldphase in phases:
        path=prior.RUN/(oldphase+'-features.pt');assert sha(path)==p['cached_features_SHA256'][oldphase]
        value=torch.load(path,map_location='cpu',weights_only=True)
        oldrows=prior.rows_for(oldphase)
        assert value['ids']==[r['id'] for r in oldrows] and value['protocol_sha256']==p['prior_protocol_sha256']
        source_ids.extend(value['ids']);values.append(value['features'])
    selected=select_descriptors([r['id'] for r in rows],source_ids,torch.cat(values))
    return rows,selected.float().cuda()


def train():
    validate(committed=True);assert not (RUN/'trained/training.json').exists()
    model,parent=load();fit,x=cache('fit');val,vx=cache('validation')
    model.gearbox_mean.copy_(x.mean(0));model.gearbox_std.copy_(x.std(0,correction=0).clamp_min(1e-3))
    y=torch.tensor([r['label'] for r in fit],dtype=torch.float32,device='cuda');vy=torch.tensor([r['label'] for r in val],dtype=torch.float32,device='cuda')
    model.gearbox_head.requires_grad_(True)
    opt=torch.optim.Adam(model.gearbox_head.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6);rng=np.random.default_rng(SEED)
    progress=RUN/'progress.pt';journal=RUN/'optimizer.jsonl';start=0;best=float('inf');bestepoch=0
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);model.gearbox_head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start,best,bestepoch=saved['epoch'],saved['best'],saved['bestepoch'];rng.bit_generator.state=saved['numpy_rng']
    else:
        assert not journal.exists();journal.touch(exist_ok=False)
        with torch.no_grad():best=float(prior.macro_loss(F.binary_cross_entropy_with_logits(model.forward_gearbox_descriptor(vx),vy,reduction='none'),val))
        new(RUN/'initial-VAL.json',{'VAL_macro_group_BCE':best,'epoch':0})
        torch.save({'head':model.gearbox_head.state_dict(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'best-head.pt')
    actual=len(journal.read_text(encoding='utf-8').splitlines());pools={};probs={}
    for label in (0,1):
        ids=[i for i,r in enumerate(fit) if r['label']==label];counts=Counter(fit[i]['group_id'] for i in ids);w=np.array([1/counts[fit[i]['group_id']] for i in ids],dtype=float)
        pools[label]=ids;probs[label]=w/w.sum()
    with journal.open('a',encoding='utf-8',buffering=1) as out:
        for epoch in range(start+1,EPOCHS+1):
            losses=[]
            for step in range(STEPS):
                ids=np.concatenate([rng.choice(pools[label],32,replace=True,p=probs[label]) for label in (0,1)])
                bce=F.binary_cross_entropy_with_logits(model.forward_gearbox_descriptor(x[ids]),y[ids]);loss=bce+model.gearbox_head.weight.square().sum()
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.gearbox_head.parameters(),5);opt.step()
                actual+=1;out.write(json.dumps({'actual_update':actual,'epoch':epoch,'step':step+1,'loss':float(loss.detach()),'BCE':float(bce.detach())})+'\n');losses.append(float(bce.detach()))
            scheduler.step()
            with torch.no_grad():score=float(prior.macro_loss(F.binary_cross_entropy_with_logits(model.forward_gearbox_descriptor(vx),vy,reduction='none'),val))
            if score<best:
                best,bestepoch=score,epoch;torch.save({'head':model.gearbox_head.state_dict(),'epoch':epoch,'VAL':score,'actual_updates':actual},RUN/'best-head.pt')
            torch.save({'head':model.gearbox_head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'bestepoch':bestepoch,'numpy_rng':rng.bit_generator.state},RUN/'progress.partial.pt');(RUN/'progress.partial.pt').replace(progress)
            new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'BCE':float(np.mean(losses)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual})
            print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
    chosen=torch.load(RUN/'best-head.pt',map_location='cpu',weights_only=True);model.gearbox_head.load_state_dict(chosen['head'])
    summary={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':chosen['epoch'],'selected_actual_updates':chosen['actual_updates'],'VAL_macro_group_BCE':chosen['VAL'],'FIT_patches':len(fit),'VAL_patches':len(val)}
    result=deepcopy(parent);result['state_dict']={n:t.detach().cpu() for n,t in model.state_dict().items()};result.update(gearbox_protocol_sha256=sha(PROTOCOL),gearbox_training=summary,gearbox_variant='gearOnly_linearL2_v28')
    (RUN/'trained').mkdir()
    with (RUN/'trained/final.pt').open('xb') as out:torch.save(result,out)
    new(RUN/'trained/training.json',summary)


def calibrate():
    validate();assert not (RUN/'calibration.json').exists()
    model,_=load(RUN/'trained/final.pt');rows,features=cache('cal');scores=prior.score_rows(model,features);labels=[r['label'] for r in rows]
    cut=calibrated_threshold(labels,scores)
    new(RUN/'calibration.json',{'checkpoint_sha256':sha(RUN/'trained/final.pt'),'threshold':cut,'rates':classification_rates(labels,scores,cut),'TEST_scored_before_this':False})
    new(RUN/'calibration-scores.json',{'ids':[r['id'] for r in rows],'scores':scores})


def evaluate():
    validate();assert not (RUN/'selection.json').exists()
    cal=read(RUN/'calibration.json');assert cal['checkpoint_sha256']==sha(RUN/'trained/final.pt')
    model,_=load(RUN/'trained/final.pt');rows,features=cache('test');scores=prior.score_rows(model,features);labels=[r['label'] for r in rows]
    rates=classification_rates(labels,scores,cal['threshold']);by={}
    for field in ('functional_component_category','physical_part_id','group_id'):
        by[field]={}
        for g in sorted({r[field] for r in rows},key=str):
            ids=[i for i,r in enumerate(rows) if r[field]==g];by[field][str(g)]=classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['threshold'])
    train=read(RUN/'trained/training.json');adequate=[v for v in by['functional_component_category'].values() if min(v['positive_photos'],v['negative_photos'])>=20]
    eligible=len({r['group_id'] for r in rows})>=3 and train['selected_epoch']>0 and max(rates['fpr'],rates['fnr'])<=.20 and min(rates['positive_photos'],rates['negative_photos'])>=20 and all(max(v['fpr'],v['fnr'])<=.25 for v in adequate)
    groups=sorted({r['group_id'] for r in rows});rng=np.random.default_rng(SEED+99);intervals=[]
    for _ in range(1000):
        sample=rng.choice(groups,len(groups),replace=True);ids=[i for g in sample for i,r in enumerate(rows) if r['group_id']==g]
        value=classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['threshold'])
        if value['fpr'] is not None and value['fnr'] is not None:intervals.append([value['fpr'],value['fnr']])
    result={'UTC':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':cal['checkpoint_sha256'],'training':train,'threshold':cal['threshold'],'rates':rates,'AUROC':rank_auroc(labels,scores),'by':by,'eligible_pending_package_checks':bool(eligible),'TEST_patches':len(rows),'TEST_connected_groups':len(groups),'group_bootstrap_95percent':np.quantile(intervals,[.025,.975],axis=0).tolist(),'bootstrap_valid_draws':len(intervals),'new_independent_field_validation':False,'whole_equipment_health_or_verified_location':False,'existing_bracket_tube_transferred_or_replaced':False}
    new(RUN/'test-scores.json',{'ids':[r['id'] for r in rows],'scores':scores});new(RUN/'selection.json',result);new(ROOT/'reports/industrial-gear-only-v28-selection.json',result)
    print({k:v for k,v in result.items() if k not in ('by','group_bootstrap_95percent')},flush=True)


def verify():
    p=validate();selection=read(RUN/'selection.json');parent=torch.load(prior.START,map_location='cpu',weights_only=True);candidate=torch.load(RUN/'trained/final.pt',map_location='cpu',weights_only=True)
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
    for phase,h in p['cached_features_SHA256'].items():assert sha(prior.RUN/(phase+'-features.pt'))==h
    result={'status':'passed','parent_all_tensor_buffer_metadata_exact':True,'legacy_python_SHA_checked':len(p['legacy_source_sha256']),'protected_stat_unchanged':unchanged,'changed_metadata_SHA_rechecked':rehashed,'all_protected_SHA_reread':False,'actual_optimizer_updates':len(journal),'new_unit_tests_passed':2,'new_RGB_encoder_features':0,'eligible_pending_package_checks':selection['eligible_pending_package_checks'],'new_API_CPP_ZIP_checks':0,'active_v16_retained_pending_release':True}
    new(RUN/'verification.json',result);new(ROOT/'reports/industrial-gear-only-v28-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run','train','calibrate','evaluate','verify'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        validate(committed=True)
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work: '+str(path))
        lock=RUN/'operation.lock'
        if lock.exists():raise RuntimeError('Review and preserve stale lock first')
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:
            if args.stage=='run':train();calibrate();evaluate();verify()
            else:globals()[args.stage]()
        except BaseException as error:
            new(RUN/(args.stage+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'pid':os.getpid()});raise
        finally:lock.unlink()
