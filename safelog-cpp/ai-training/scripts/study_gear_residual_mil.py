"""Single bounded learned residual; no TEST-driven mixture or threshold variants."""
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
from safelog_ai.gear_residual_mil import GearResidualMain,VERSION
from safelog_ai.process_probe import process_exists
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from scripts import study_gear_spatial_mil as spatial
from scripts import study_gear_only as global_study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_bsclass_supervised import calibrated_threshold

RUN=ROOT/'runs/industrial-gear-residual-v30'
PROTOCOL=ROOT/'reports/industrial-gear-residual-v30-protocol.json'
MANIFEST=spatial.MANIFEST;START=spatial.START;EXPECTED=spatial.EXPECTED
WARM=global_study.RUN/'trained/final.pt'
WARM_SHA='0669d9b988aa5ab3e4bfe414a078e140f61757d9bf06828dc56538cf474bf96f'
LOCAL=spatial.RUN/'trained/final.pt'
LOCAL_SHA='208240ca2bc64c398010f3cbdbc3609ed077e3875081bcfbddedadf3770195f6'
SOURCES=('safelog_ai/gear_residual_mil.py','scripts/study_gear_residual_mil.py','tests/test_gear_residual_mil.py')
EPOCHS,STEPS,BATCH,SEED=30,64,32,3030


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(START)==EXPECTED
    assert sha(WARM)==WARM_SHA and sha(LOCAL)==LOCAL_SHA
    assert read(spatial.RUN/'verification.json')['status']=='passed'
    assert not read(spatial.RUN/'selection.json')['eligible_pending_package_checks']
    assert shutil.disk_usage(ROOT).free>2*1024**3
    RUN.mkdir();ledger=read(spatial.RUN/'protected-inputs.json')
    for path in spatial.RUN.rglob('*'):
        if path.is_file() and not path.name.endswith(('.log','.lock')):
            n=path.relative_to(ROOT).as_posix()
            if n not in ledger:ledger[n]=snapshot(path)
    new(RUN/'protected-inputs.json',ledger)
    p={'UTC':datetime.now(timezone.utc).isoformat(),'schema':VERSION,'fixed_before_new_gradients_scores_optimizer':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'parent_sha256':EXPECTED,'global_research_checkpoint_sha256':WARM_SHA,'spatial_research_checkpoint_sha256':LOCAL_SHA,
       'manifest_sha256':sha(MANIFEST),'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),
       'legacy_source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for d in ('safelog_ai','scripts','tests') for f in (ROOT/d).glob('*.py') if f.relative_to(ROOT).as_posix() not in SOURCES},
       'fine_cache_sha256':{p:sha(spatial.RUN/(p+'-fine.pt')) for p in ('fit','validation','cal','test')},
       'descriptor_cache_sha256':read(global_study.PROTOCOL)['cached_features_SHA256'],
       'v29_protocol_sha256':sha(spatial.PROTOCOL),'v28_protocol_sha256':sha(global_study.PROTOCOL),
       'motivation':'VAL globalv28.587897 vs standalone localv29.653942; keep global evidence and learn bounded spatial correction. No new TEST grid or manually selected mixture.',
       'data':'unchangedv28 gear-only FIT5466/VAL1440/CAL1116/TEST2179, groups8/3/3/4; no new images/split or RGB encoderfeatures',
       'initialization':'acceptedv16 oldstate retained; frozen v28selectedglobalhead+globalFITmean/std; copy v29selected Conv/GN and fineFITmean/std, reset final local1x1weight/bias0. Both are prior rejected research initializers, not app adoption.',
       'initial_check':'newVAL all logits bitwiseequal frozen global when correction0; macroVAL within1e-6 of storedv28; otherwise preservefailure andstop',
       'architecture':'one sharedencoder and onecheckpoint; global logit + .5*tanh(localMILlogit), fixed .5 beforetraining; localConv24to32/GN4/GELU/1x1to1/topceil10percent logitmean',
       'trainable':'only new gear_spatial_head; globalhead/globalnorm and all acceptedv16tensor/buffer/metadata frozen; fineFITnorm copied exactly fromv29',
       'loss':'patch acceptance logitBCE, actualOK/nOK only; no pixelmask or loss, no distillationlabel creation',
       'sampling':'batch32 16OK/16nOK, eachlabel inversegroupcount; seed3030, cachedfine H/Vflip.5, unchanged pooleddescriptor is permutation invariant; no freshRGB augmentation',
       'optimizer':'AdamW lr1e-4 wd1e-4 cosine30 min1e-6 clip5 FP32 TF32off',
       'epochs':30,'steps':64,'batch':32,'seed':3030,'planned_optimizer_updates':1920,
       'VAL':'epoch0 and each completeepoch, macro label/groupBCE, earliest strictminimum; epoch0 means frozenv28initialization and cannotrelease',
       'CAL':'selectedhead only, labeledCAL minimaxFPR/FNR same ties sum/distance.5/lowercut strict>; save before newTEST scoring',
       'adoption':['selectedepoch>0','TEST eachlabel>=20 groups>=3','gearpatch FPR<=.20 and FNR<=.20','oldacceptedstate and globalresearchstate frozenexact','packagedCPU/GPU/C++/aliases/oldmaps before release'],
       'resume':'epoch optimizer/scheduler/RNG and actualjournal preserve partialepochreplay; never reruncompletedstudy',
       'new_unit_tests_passed':2,'source_pixel_GT_available':False,'new_RGB_encoderfeatures':0,
       'scope':'gear128surface acceptance only, priorpublicTESTdevelopment, no fullgearbox/wholemachinehealth/verifiedlocation or oldtargettransfer',
       'independent_field_validated':False,'source_license':'Re-GBCv1 CC BY4.0 cite originaldataset and DOI10.1109/ETFA65518.2025.11205777, other source conditions unchanged'}
    new(PROTOCOL,p);print({'frozen':True,'new_RGB_features':0,'planned_optimizer':1920},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in ((START,EXPECTED),(WARM,WARM_SHA),(LOCAL,LOCAL_SHA),(MANIFEST,p['manifest_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256'])):assert sha(path)==h,path
    assert sha(spatial.PROTOCOL)==p['v29_protocol_sha256'] and sha(global_study.PROTOCOL)==p['v28_protocol_sha256']
    if committed:
        git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def load(path=WARM,initialize=False,device='cuda'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=GearResidualMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and (not missing or initialize and all(n.startswith('gear_spatial_') for n in missing))
    if initialize:
        local=torch.load(LOCAL,map_location='cpu',weights_only=True)['state_dict'];prefix='gear_spatial_head.'
        model.gear_spatial_head.load_state_dict({n[len(prefix):]:t for n,t in local.items() if n.startswith(prefix)},strict=True)
        with torch.no_grad():
            model.gear_spatial_head[-1].weight.zero_();model.gear_spatial_head[-1].bias.zero_()
            model.gear_spatial_mean.copy_(local['gear_spatial_mean']);model.gear_spatial_std.copy_(local['gear_spatial_std'])
    return model.eval().requires_grad_(False).to(device),state


def cache(phase,model):
    p=read(PROTOCOL)
    if phase=='test':assert (RUN/'calibration.json').exists()
    if phase=='cal':assert (RUN/'trained/training.json').exists()
    assert sha(spatial.RUN/(phase+'-fine.pt'))==p['fine_cache_sha256'][phase]
    rows,fine=spatial.cache(phase,model)
    drows,descriptor=global_study.cache(phase)
    assert [r['id'] for r in rows]==[r['id'] for r in drows]
    return rows,fine.cuda(),descriptor


def logits_for(model,fine,descriptor):
    values=[];globalvalues=[]
    with torch.inference_mode():
        for i in range(0,len(fine),BATCH):
            values.append(model.forward_gear_residual_features(fine[i:i+BATCH],descriptor[i:i+BATCH]).cpu())
            globalvalues.append(model.forward_gearbox_descriptor(descriptor[i:i+BATCH]).cpu())
    values,globalvalues=torch.cat(values),torch.cat(globalvalues)
    assert float((values-globalvalues).abs().max())<=.500001
    return values,globalvalues


def train():
    validate(committed=True);assert not (RUN/'trained/training.json').exists()
    model,parent=load(initialize=True);fit,x,dx=cache('fit',model);val,vx,vd=cache('validation',model)
    y=torch.tensor([r['label'] for r in fit],dtype=torch.float32,device='cuda');vy=torch.tensor([r['label'] for r in val],dtype=torch.float32)
    with torch.no_grad():
        initial,base=logits_for(model,vx,vd);assert torch.equal(initial,base)
        initial_loss=float(global_study.prior.macro_loss(F.binary_cross_entropy_with_logits(initial,vy,reduction='none'),val))
    assert abs(initial_loss-read(global_study.RUN/'trained/training.json')['VAL_macro_group_BCE'])<=1e-6
    model.gear_spatial_head.requires_grad_(True);opt=torch.optim.AdamW(model.gear_spatial_head.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6);rng=np.random.default_rng(SEED)
    progress=RUN/'progress.pt';journal=RUN/'optimizer.jsonl';start=0;best=initial_loss;bestepoch=0
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);model.gear_spatial_head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start,best,bestepoch=saved['epoch'],saved['best'],saved['bestepoch'];rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        assert not journal.exists();journal.touch(exist_ok=False)
        new(RUN/'initial-VAL.json',{'VAL_macro_group_BCE':best,'all_logits_bitwiseequal_frozen_global':True,'epoch':0})
        torch.save({'head':model.gear_spatial_head.state_dict(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'best-head.pt')
    actual=len(journal.read_text(encoding='utf-8').splitlines());pools={};probs={}
    for label in (0,1):
        ids=[i for i,r in enumerate(fit) if r['label']==label];counts=Counter(fit[i]['group_id'] for i in ids);w=np.array([1/counts[fit[i]['group_id']] for i in ids],dtype=float)
        pools[label]=ids;probs[label]=w/w.sum()
    with journal.open('a',encoding='utf-8',buffering=1) as out:
        for epoch in range(start+1,EPOCHS+1):
            model.gear_spatial_head.train();losses=[]
            for step in range(STEPS):
                ids=np.concatenate([rng.choice(pools[label],16,replace=True,p=probs[label]) for label in (0,1)]);batch=x[ids]
                horizontal=torch.rand(len(batch),device='cuda')<.5;vertical=torch.rand(len(batch),device='cuda')<.5
                batch=torch.where(horizontal[:,None,None,None],batch.flip(-1),batch);batch=torch.where(vertical[:,None,None,None],batch.flip(-2),batch)
                loss=F.binary_cross_entropy_with_logits(model.forward_gear_residual_features(batch,dx[ids]),y[ids]);assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.gear_spatial_head.parameters(),5);opt.step()
                actual+=1;out.write(json.dumps({'actual_update':actual,'epoch':epoch,'step':step+1,'BCE':float(loss.detach())})+'\n');losses.append(float(loss.detach()))
            scheduler.step();model.gear_spatial_head.eval();values,_=logits_for(model,vx,vd)
            score=float(global_study.prior.macro_loss(F.binary_cross_entropy_with_logits(values,vy,reduction='none'),val))
            if score<best:
                best,bestepoch=score,epoch;torch.save({'head':model.gear_spatial_head.state_dict(),'epoch':epoch,'VAL':score,'actual_updates':actual},RUN/'best-head.pt')
            saved={'head':model.gear_spatial_head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'bestepoch':bestepoch,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
            torch.save(saved,RUN/'progress.partial.pt');(RUN/'progress.partial.pt').replace(progress)
            new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'train_BCE':float(np.mean(losses)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual})
            print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
    chosen=torch.load(RUN/'best-head.pt',map_location='cpu',weights_only=True);model.gear_spatial_head.load_state_dict(chosen['head']);model.eval()
    summary={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':chosen['epoch'],'selected_actual_updates':chosen['actual_updates'],'VAL_macro_group_BCE':chosen['VAL'],'initial_VAL_macro_group_BCE':initial_loss,'FIT_patches':len(fit),'VAL_patches':len(val),'replayed_partial_epoch_updates':actual-1920}
    result=deepcopy(parent);result['state_dict']={n:t.detach().cpu() for n,t in model.state_dict().items()};result.update(gear_residual_protocol_sha256=sha(PROTOCOL),gear_residual_training=summary,gear_residual_scope='gear128patch_acceptance_not_verified_location')
    (RUN/'trained').mkdir()
    with (RUN/'trained/final.pt').open('xb') as out:torch.save(result,out)
    new(RUN/'trained/training.json',summary)


def calibrate():
    validate();assert not (RUN/'calibration.json').exists()
    model,_=load(RUN/'trained/final.pt');rows,fine,descriptor=cache('cal',model);values,_=logits_for(model,fine,descriptor);scores=values.sigmoid().tolist();labels=[r['label'] for r in rows]
    cut=calibrated_threshold(labels,scores)
    new(RUN/'calibration.json',{'checkpoint_sha256':sha(RUN/'trained/final.pt'),'threshold':cut,'rates':classification_rates(labels,scores,cut),'newTEST_scores_before_this':False})
    new(RUN/'calibration-scores.json',{'ids':[r['id'] for r in rows],'scores':scores})


def evaluate():
    validate();assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['checkpoint_sha256']==sha(RUN/'trained/final.pt')
    model,_=load(RUN/'trained/final.pt');rows,fine,descriptor=cache('test',model);values,base=logits_for(model,fine,descriptor);scores=values.sigmoid().tolist();labels=[r['label'] for r in rows]
    rates=classification_rates(labels,scores,cal['threshold']);by={}
    for field in ('physical_part_id','group_id'):
        by[field]={}
        for g in sorted({r[field] for r in rows},key=str):
            ids=[i for i,r in enumerate(rows) if r[field]==g];by[field][str(g)]=classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['threshold'])
    train=read(RUN/'trained/training.json');groups=sorted({r['group_id'] for r in rows});eligible=train['selected_epoch']>0 and len(groups)>=3 and min(rates['positive_photos'],rates['negative_photos'])>=20 and max(rates['fpr'],rates['fnr'])<=.20
    rng=np.random.default_rng(SEED+99);intervals=[]
    for _ in range(1000):
        sample=rng.choice(groups,len(groups),replace=True);ids=[i for g in sample for i,r in enumerate(rows) if r['group_id']==g];value=classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['threshold'])
        if value['fpr'] is not None and value['fnr'] is not None:intervals.append([value['fpr'],value['fnr']])
    result={'UTC':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':cal['checkpoint_sha256'],'training':train,'threshold':cal['threshold'],'rates':rates,'AUROC':rank_auroc(labels,scores),'by':by,'eligible_pending_package_checks':bool(eligible),'TEST_patches':len(rows),'TEST_connected_groups':len(groups),'group_bootstrap_95percent':np.quantile(intervals,[.025,.975],axis=0).tolist(),'bootstrap_valid_draws':len(intervals),'max_abs_spatial_logit_correction':float((values-base).abs().max()),'new_independent_field_validation':False,'whole_equipment_health_or_verified_location':False,'existing_bracket_tube_transferred_or_replaced':False}
    new(RUN/'test-scores.json',{'ids':[r['id'] for r in rows],'scores':scores,'global_scores':base.sigmoid().tolist(),'spatial_logit_correction':(values-base).tolist()});new(RUN/'selection.json',result);new(ROOT/'reports/industrial-gear-residual-v30-selection.json',result)
    print({k:v for k,v in result.items() if k not in ('by','group_bootstrap_95percent')},flush=True)


def verify():
    p=validate();s=read(RUN/'selection.json');parent=torch.load(START,map_location='cpu',weights_only=True);warm=torch.load(WARM,map_location='cpu',weights_only=True);local=torch.load(LOCAL,map_location='cpu',weights_only=True);candidate=torch.load(RUN/'trained/final.pt',map_location='cpu',weights_only=True)
    for name,tensor in warm['state_dict'].items():assert torch.equal(tensor,candidate['state_dict'][name]),name
    for name,value in warm.items():
        if name!='state_dict':assert value==candidate[name],name
    for name,tensor in parent['state_dict'].items():assert torch.equal(tensor,candidate['state_dict'][name]),name
    for name in ('gear_spatial_mean','gear_spatial_std'):assert torch.equal(local['state_dict'][name],candidate['state_dict'][name]),name
    for name,digest in p['legacy_source_sha256'].items():assert sha(ROOT/name)==digest,name
    unchanged=rehashed=0
    for name,before in read(RUN/'protected-inputs.json').items():
        path=ROOT/name;st=path.stat()
        if st.st_size==before['size_bytes'] and st.st_mtime_ns==before['mtime_ns']:unchanged+=1
        else:assert sha(path)==before['sha256'],name;rehashed+=1
    journal=[json.loads(v) for v in (RUN/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()];assert len(journal)==s['training']['actual_optimizer_updates'] and len(journal)>=1920 and [v['actual_update'] for v in journal]==list(range(1,len(journal)+1))
    for phase,h in p['fine_cache_sha256'].items():assert sha(spatial.RUN/(phase+'-fine.pt'))==h
    for phase,h in p['descriptor_cache_sha256'].items():assert sha(global_study.prior.RUN/(phase+'-features.pt'))==h
    r={'status':'passed','accepted_v16_all_tensor_buffer_metadata_exact':True,'v28_global_research_tensor_buffer_metadata_exact':True,'v29_FIT_mean_std_exact':True,'legacy_python_SHA_checked':len(p['legacy_source_sha256']),'protected_stat_unchanged':unchanged,'changed_metadata_SHA_rechecked':rehashed,'all_protected_SHA_reread':False,'actual_optimizer_updates':len(journal),'new_unit_tests_passed':2,'new_RGB_encoderfeatures':0,'eligible_pending_package_checks':s['eligible_pending_package_checks'],'new_API_CPP_ZIP_checks':0,'active_v16_retained_pending_release':True}
    new(RUN/'verification.json',r);new(ROOT/'reports/industrial-gear-residual-v30-verification.json',r);print(r,flush=True)


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
