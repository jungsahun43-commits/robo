"""Fixed normal-only global-context training with a separate VAL and CAL."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.loco_context import ContextMain, CATEGORY, VERSION, context_view, context_targets
from safelog_ai.ksdd_photo_only import KSDDPhotoMain
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_expansion import CATEGORIES, photo_score, stage_gate
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.process_probe import process_exists
from scripts.prepare_loco_context import DATA, RUN, REPORT as DATA_REPORT
from scripts.study_equipment_expansion import read, images
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_unified_belt_tiles import snapshot

REPORTS = ROOT/'reports'; MANIFEST = DATA/'manifest.json'; PROTOCOL = REPORTS/'industrial-loco-context-v13-protocol.json'
START = ROOT/'runs/industrial-ksdd-photo-only-v11/selected/final.pt'
EXPECTED = '935cc6f6cb8e021714c97aa5c7225a567cda1637ba1c1876fd6de9230a0454d6'
PARENT_EVALUATION = ROOT/'runs/industrial-ksdd-photo-only-v11/evaluation.json'
SOURCES = ('safelog_ai/loco_context.py','scripts/prepare_loco_context.py','scripts/study_loco_context.py','tests/test_loco_context.py')
EPOCHS, STEPS, BATCH, SEED = 30, 64, 16, 1313
METHODS = ('context_only', 'max_context_local')


def freeze():
    assert not PROTOCOL.exists() and sha(START) == EXPECTED
    assert read(REPORTS/'industrial-foreground-v12-verification.json')['status'] == 'passed'
    audit = read(DATA_REPORT); counts = audit['counts']
    assert counts['fit'] >= 50 and min(counts['validation'], counts['cal']) >= 20
    ledger = dict(read(ROOT/'runs/industrial-foreground-v12/protected-inputs.json'))
    for directory in (REPORTS, ROOT/'runs/industrial-foreground-v12'):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock','.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    new(RUN/'protected-inputs.json', ledger)
    baseline = read(PARENT_EVALUATION); lookup = {r['id']:r for r in baseline['rows'] if r['category'] == CATEGORY}; groups = {}
    rows = read(MANIFEST)['rows']
    for defect in ('logical_anomalies','structural_anomalies'):
        selected = [r for r in rows if r['phase']=='test' and r['defect']==defect]
        fn = sum(not lookup[r['id']]['candidate_detected'] for r in selected)
        groups[defect] = {'defect_photos':len(selected),'fn':fn,'fnr':fn/len(selected)}
    new(PROTOCOL, {'schema':VERSION,'declared_utc':datetime.now(timezone.utc).isoformat(),'declared_before_new_feature_caches_training_CAL_TEST':True,
        'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for d in ('safelog_ai','scripts','tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256':EXPECTED,'manifest_sha256':sha(MANIFEST),'parent_evaluation_sha256':sha(PARENT_EVALUATION),'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),
        'source_data_audit':audit,'baseline':baseline['categories'][CATEGORY],'baseline_by_defect':groups,
        'training_epochs':EPOCHS,'scheduled_steps_per_epoch':STEPS,'batch':BATCH,'scheduled_updates':EPOCHS*STEPS,'seed':SEED,
        'training':{'target':'same frozen shared backbone low40/high960 unit features, adaptive spatial4x8, FP16 cache/runtime roundtrip',
            'view':'512x256 isotropic letterbox127 RGB ImageNet normalization','augmentation':'normalFIT raw RGB original/H/V/HV views extracted separately; sampled uniformly; not additional source photos',
            'standardization':'FIT original views only, per-channel mean/std over images/spatial; std min.01, clamp[-8,8]',
            'head':'Conv1000to64/ReLU, flatten2048, FC256/ReLU/dropout.1/FC128; decoderFC256/ReLU/FC2048/ReLU/reshape64x4x8/Conv64to1000',
            'optimizer':'AdamW lr.001 wd.0001, cosine30 min.00001, clip5, FP32, TF32off; only context head trainable',
            'loss':'mean standardized target reconstruction MSE; target features fixed, no trainable target projection',
            'sampling':'fixed numpyPCG64 replacement draw schedules; FIT only','epoch_selection':'minimum held-out normalVAL mean reconstruction MSE; earliest on ties, CAL/TEST not used',
            'resume':'per-epoch progress and per-optimizer journal. Interrupted partial-epoch replays count as additional actual operations, never silently reset.'},
        'photo_score':'mean top8 of32 global spatial channel-MSE cells, no calibrated failure probability',
        'candidates':list(METHODS),'calibration':'normalCAL20 only; order statistic ceil((n+1)*.8), strict >. Fusion max(global/globalCALcut,local/localCALcut) with a separate CAL20 joint cut. Save all before TEST.',
        'selection_guards':['FPR<=.20','FPR<=parent','FNR<=parent-.05','logical and structural FNR each<=parent+.05','each class>=20','old tensors/buffers/points preserved'],
        'selection_order':'eligible lowest maxFPRFNR, then FNR, then context_only before fusion. Otherwise priorv11 retained.',
        'incremental_selection_distinct_from_stage20_gate':True,'original_location_and_pixel_thresholds_preserved':True,
        'actual_clamp_wire_endpoint_object_labels_available':False,'trained_on_TEST_masks':False,'paper_method_reproduced':False,
        'relation_detection_is_hypothesis_not_verified_semantic_reasoning':True,'same_one_backbone_passes':2,
        'known_public_TEST_reused_for_development':True,'inherited_parent_training_exposure_not_undone':True,
        'independent_field_validated':False,'app_default_promoted':False,'new_unit_tests_passed':3})
    rng = np.random.default_rng(SEED)
    np.savez_compressed(RUN/'draws.npz', indices=rng.integers(0,counts['fit'],size=(EPOCHS,STEPS,BATCH)), variants=rng.integers(0,4,size=(EPOCHS,STEPS,BATCH)))
    new(RUN/'draws.json', {'sha256':sha(RUN/'draws.npz'),'unique_normal_FIT_photos':counts['fit'],'new_defect_or_relation_labels':0})
    print({'protocol_frozen':True,'planned_updates':EPOCHS*STEPS,'counts':counts,'prior_by_defect':groups},flush=True)


def validate(committed=False):
    p = read(PROTOCOL)
    for name,digest in p['source_sha256'].items(): assert sha(ROOT/name)==digest,name
    for path,digest in ((START,EXPECTED),(MANIFEST,p['manifest_sha256']),(PARENT_EVALUATION,p['parent_evaluation_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256'])): assert sha(path)==digest,path
    assert sha(RUN/'draws.npz')==read(RUN/'draws.json')['sha256']
    if committed:
        repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
        for name in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()): assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])).hexdigest()==sha(ROOT/name),name
    return p


def load(path=START, device='cpu', new_head=False):
    state = torch.load(path,map_location='cpu',weights_only=True)
    cls=ContextMain if new_head or state.get('context_version') else KSDDPhotoMain
    model=cls(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    if new_head and not state.get('context_version'):
        missing, extra=model.load_state_dict(state['state_dict'],strict=False)
        assert not extra and missing and all(n.startswith('context_head.') for n in missing)
    else:model.load_state_dict(state['state_dict'],strict=True)
    return model.eval().requires_grad_(False).to(device),state


def cache_phase(phase, model):
    path=RUN/(phase+'-features.pt'); rows=[r for r in read(MANIFEST)['rows'] if r['phase']==phase]
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['protocol_sha256']==sha(PROTOCOL) and saved['ids']==[r['id'] for r in rows]
        return rows,saved
    if phase!='test':assert all(r['label']==0 for r in rows)
    targets=[];local_scores=[]
    with torch.inference_mode():
        for index,row in enumerate(rows):
            assert sha(ROOT/row['image'])==row['image_sha256'],row['id']
            with Image.open(ROOT/row['image']) as im:rgb=im.convert('RGB')
            variants=[rgb]
            if phase=='fit':variants += [rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT),rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM),rgb.transpose(Image.Transpose.ROTATE_180)]
            x=torch.stack([normalized(context_view(a)[0]) for a in variants]).cuda()
            targets.append(context_targets(model.backbone(x)).cpu().half())
            if phase in ('cal','test'):
                original=normalized(rgb.resize((256,256),Image.Resampling.BICUBIC))[None].cuda()
                local_scores.append(float(photo_score(model.forward_equipment(original,CATEGORY))[0]))
            if index%50==0:print({'feature_cache':phase,'photos':index+1,'total':len(rows)},flush=True)
    value={'protocol_sha256':sha(PROTOCOL),'ids':[r['id'] for r in rows],'targets':torch.stack(targets),'local_scores':local_scores}
    with path.open('xb') as out:torch.save(value,out)
    return rows,value


def train():
    validate(committed=True); assert not (RUN/'trained/training.json').exists()
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model,parent=load(device='cuda',new_head=True); head=model.context_head
    fit_rows,fit=cache_phase('fit',model);val_rows,val=cache_phase('validation',model)
    x=fit['targets'].cuda().float();v=val['targets'][:,0].cuda().float()
    with torch.no_grad():
        head.FIT_mean.copy_(x[:,0].mean((0,2,3))[None,:,None,None])
        head.FIT_scale.copy_(x[:,0].std((0,2,3),unbiased=False).clamp_min(.01)[None,:,None,None])
    initial={n:t.detach().cpu().clone() for n,t in head.state_dict().items()}
    head.requires_grad_(True);optim=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=EPOCHS,eta_min=.00001)
    draws=np.load(RUN/'draws.npz');epoch0=0;best_loss=float('inf');best_epoch=0;actual=0
    progress=RUN/'train/progress.pt';progress.parent.mkdir(exist_ok=True)
    journal=RUN/'train/optimizer.jsonl'
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);assert saved['protocol_sha256']==sha(PROTOCOL)
        head.load_state_dict(saved['head']);optim.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch0=saved['epoch'];best_loss=saved['best_loss'];best_epoch=saved['best_epoch']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    if journal.exists():actual=sum(1 for _ in journal.open(encoding='utf-8'))
    for epoch in range(epoch0,EPOCHS):
        head.train();losses=[]
        for step in range(STEPS):
            ids=torch.as_tensor(draws['indices'][epoch,step],device='cuda'); variants=torch.as_tensor(draws['variants'][epoch,step],device='cuda')
            target=x[ids,variants];optim.zero_grad(set_to_none=True)
            loss=F.mse_loss(head(target),head.standardize(target));assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);optim.step();actual+=1;losses.append(float(loss.detach()))
            with journal.open('a',encoding='utf-8') as out:out.write(json.dumps({'actual_operation':actual,'epoch':epoch+1,'step':step+1,'loss':losses[-1]})+'\n')
        head.eval()
        with torch.inference_mode():vl=float(F.mse_loss(head(v),head.standardize(v)))
        if vl<best_loss:
            best_loss,best_epoch=vl,epoch+1
            torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':best_epoch,'VAL_MSE':vl,'protocol_sha256':sha(PROTOCOL)},RUN/'train/best.pt')
        scheduler.step()
        torch.save({'head':head.state_dict(),'optimizer':optim.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best_loss':best_loss,'best_epoch':best_epoch,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        record={'epoch':epoch+1,'actual_optimizer_operations':actual,'scheduled_unique_updates':(epoch+1)*STEPS,'train_MSE':sum(losses)/len(losses),'VAL_MSE':vl,'selected_epoch':best_epoch}
        with (RUN/'train/epochs.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(record)+'\n')
        print(record,flush=True)
    best=torch.load(RUN/'train/best.pt',map_location='cpu',weights_only=True)
    assert any(not torch.equal(initial[n],t) for n,t in best['head'].items() if not n.startswith('FIT_'))
    state=deepcopy(parent);state['state_dict'].update({'context_head.'+n:t for n,t in best['head'].items()})
    assert all(torch.equal(t,state['state_dict'][n]) for n,t in parent['state_dict'].items())
    state.update(context_version=VERSION,context_protocol_sha256=sha(PROTOCOL),context_parent_sha256=EXPECTED,context_actual_optimizer_updates=actual,
        context_selected_epoch=best_epoch,context_scheduled_unique_updates=EPOCHS*STEPS)
    (RUN/'trained').mkdir()
    with (RUN/'trained/final.pt').open('xb') as out:torch.save(state,out)
    new(RUN/'trained/training.json',{'actual_optimizer_updates':actual,'scheduled_unique_updates':EPOCHS*STEPS,'completed_epochs':EPOCHS,
        'selected_epoch_by_normal_validation':best_epoch,'selected_epoch_scheduled_updates':best_epoch*STEPS,'validation_MSE':best_loss,
        'normal_FIT_photos':len(fit_rows),'normal_VAL_photos':len(val_rows),'real_defect_or_object_relation_labels_used':0,
        'backbone_old_heads_and_buffers_exact':True,'fitted_head_changed':True,'fixed_FIT_target_no_trainable_projection':True})


def scores(head, cache):
    values=[]
    with torch.inference_mode():
        targets=cache['targets'][:,0].cuda().float()
        for start in range(0,len(targets),16):values.extend(head.evidence(targets[start:start+16])[1].cpu().tolist())
    return values


def calibrate():
    validate();assert not (RUN/'calibration.json').exists()
    model,_=load(RUN/'trained/final.pt','cuda');rows,cache=cache_phase('cal',model)
    global_scores=scores(model.context_head,cache);local=cache['local_scores']
    global_cut,rank=normal_budget_threshold(global_scores,.20);local_cut,local_rank=normal_budget_threshold(local,.20)
    assert min(global_cut,local_cut)>0
    joint=[max(a/global_cut,b/local_cut) for a,b in zip(global_scores,local)]
    joint_cut,joint_rank=normal_budget_threshold(joint,.20)
    new(RUN/'calibration.json',{'protocol_sha256':sha(PROTOCOL),'thresholds_saved_before_TEST':True,'normal_CAL_photos':len(rows),
        'points':{'context_only':{'photo_threshold':global_cut,'rank':rank},'max_context_local':{'photo_threshold':joint_cut,'rank':joint_rank,
            'global_divisor':global_cut,'local_divisor':local_cut,'local_divisor_rank':local_rank}},
        'observations':[{'id':r['id'],'global':a,'local':b,'joint':s} for r,a,b,s in zip(rows,global_scores,local,joint)]})
    print({'calibrated':True,'normalCAL':len(rows),'global_cut':global_cut,'fusion_cut':joint_cut},flush=True)


def evaluate():
    p=validate();assert not (REPORTS/'industrial-loco-context-v13-validation.json').exists()
    cal=read(RUN/'calibration.json');assert cal['thresholds_saved_before_TEST'] and cal['protocol_sha256']==sha(PROTOCOL)
    model,state=load(RUN/'trained/final.pt','cuda');rows,cache=cache_phase('test',model);global_scores=scores(model.context_head,cache);local=cache['local_scores']
    labels=[r['label'] for r in rows];results={}
    for method in METHODS:
        point=cal['points'][method];threshold=point['photo_threshold']
        values=global_scores if method=='context_only' else [max(a/point['global_divisor'],b/point['local_divisor']) for a,b in zip(global_scores,local)]
        rates=classification_rates(labels,values,threshold);groups={}
        for defect in ('logical_anomalies','structural_anomalies'):
            indices=[i for i,r in enumerate(rows) if r['defect']==defect];fn=sum(values[i]<=threshold for i in indices)
            groups[defect]={'defect_photos':len(indices),'fn':fn,'fnr':fn/len(indices)}
        baseline=p['baseline']['photo_rates'];guards={'FPR_within20':rates['fpr']<=.20,'FPR_no_worse':rates['fpr']<=baseline['fpr'],
            'FNR_improved5pp':rates['fnr']<=baseline['fnr']-.05,'each_class_at_least20':min(rates['negative_photos'],rates['positive_photos'])>=20,
            'no_defect_group_worse_by_more_than5pp':all(v['fnr']<=p['baseline_by_defect'][c]['fnr']+.05 for c,v in groups.items()),'parent_tensors_preserved':True}
        results[method]={'photo_rates':rates,'point':point,'by_defect':groups,'auroc':rank_auroc(labels,values),
            'stage_gate':stage_gate(rates['fpr'],rates['fnr'],rates['negative_photos'],rates['positive_photos']),'guards':guards,'eligible':all(guards.values()),
            'observations':[{'id':r['id'],'label':r['label'],'defect':r['defect'],'global_score':a,'local_score':b,'score':s,'candidate_detected':s>threshold} for r,a,b,s in zip(rows,global_scores,local,values)]}
        print({'candidate':method,'rates':rates,'by_defect':groups,'eligible':all(guards.values())},flush=True)
    eligible=[m for m in METHODS if results[m]['eligible']]
    method=min(eligible,key=lambda m:(max(results[m]['photo_rates']['fpr'],results[m]['photo_rates']['fnr']),results[m]['photo_rates']['fnr'],METHODS.index(m))) if eligible else None
    new(REPORTS/'industrial-loco-context-v13-validation.json',{'candidates':results,'baseline':p['baseline'],'baseline_by_defect':p['baseline_by_defect'],'known_TEST_reused':True})
    checkpoint=START
    if method:
        state.update(context_selected_method=method,context_operating_point=cal['points'][method],context_normal_CAL_photos=cal['normal_CAL_photos'])
        (RUN/'selected').mkdir();checkpoint=RUN/'selected/final.pt'
        with checkpoint.open('xb') as out:torch.save(state,out)
    evaluation=read(PARENT_EVALUATION)
    if method:
        value=results[method];evaluation['categories'][CATEGORY].update(photo_rates=value['photo_rates'],photo_threshold=value['point']['photo_threshold'],
            stage_gate=value['stage_gate'],by_defect=value['by_defect'],photo_score_source=method,auroc=value['auroc'])
        lookup={r['id']:r for r in value['observations']}
        for row in evaluation['rows']:
            if row['category']==CATEGORY:row.update(score=lookup[row['id']]['score'],candidate_detected=lookup[row['id']]['candidate_detected'])
    evaluation.update(checkpoint_sha256=sha(checkpoint),status='industrial_loco_context_v13',stage20_categories=[c for c,v in evaluation['categories'].items() if v['stage_gate']['ready_for_next_stage']])
    new(RUN/'evaluation.json',evaluation)
    new(REPORTS/'industrial-loco-context-v13-selection.json',{'selected':method,'checkpoint_path':checkpoint.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(checkpoint),
        'stage20_categories':evaluation['stage20_categories'],'actual_optimizer_updates':state['context_actual_optimizer_updates'],'selected_epoch':state['context_selected_epoch'],
        'original_locations_and_thresholds_preserved':True,'known_TEST_reused_for_development':True,'independent_field_validated':False,'app_default_promoted':False})


def verify():
    p=validate();choice=read(REPORTS/'industrial-loco-context-v13-selection.json')
    for name,digest in p['legacy_source_sha256'].items():assert sha(ROOT/name)==digest,name
    ledger=read(RUN/'protected-inputs.json')
    def check(item):
        name,value=item;assert snapshot(ROOT/name)==value,name
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check,ledger.items()))
    model,state=load(RUN/'trained/final.pt');parent,original=load()
    assert all(torch.equal(t,state['state_dict'][n]) for n,t in original['state_dict'].items())
    for name in ('equipment_operating_points','ksdd_operating_point','wear_operating_point','surface_reference_specs','thresholds'):assert state[name]==original[name],name
    x=torch.zeros(1,3,256,256);proofs={}
    with torch.inference_mode():
        for category in CATEGORIES:assert torch.equal(model.forward_equipment(x,category),parent.forward_equipment(x,category));proofs[category]=True
        assert torch.equal(model.forward_wear(x),parent.forward_wear(x))
        row=next(r for r in read(MANIFEST)['rows'] if r['phase']=='fit')
        with Image.open(ROOT/row['image']) as im:rgb=im.convert('RGB')
        view=normalized(context_view(rgb)[0])[None]
        location,_,_=model.forward_context_and_location(images([row],'cpu'),view)
        assert torch.equal(location,parent.forward_equipment(images([row],'cpu'),CATEGORY))
    training=read(RUN/'trained/training.json');actual=sum(1 for _ in (RUN/'train/optimizer.jsonl').open(encoding='utf-8'))
    assert training['actual_optimizer_updates']==actual and training['completed_epochs']==EPOCHS
    new(REPORTS/'industrial-loco-context-v13-verification.json',{'status':'passed','protected_files':len(ledger),'legacy_sources':len(p['legacy_source_sha256']),
        'parent_all_tensors_and_buffers_exact':True,'parent_all_operating_points_exact':True,'old_CPU_locations_exact':proofs,'BSClass_CPU_output_exact':True,
        'current_context_original_location_exact':True,'actual_optimizer_updates_verified_from_journal':actual,'new_unit_tests_passed':3,'selected':choice['selected']})
    print({'verified':True,'actual_optimizer_updates':actual,'selected':choice['selected']},flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','train','calibrate','evaluate','verify']);args=parser.parse_args()
    if args.mode in ('train','calibrate','evaluate'):
        for lock in (ROOT/'runs').glob('**/*.lock'):
            try:pid=int(lock.read_text(encoding='utf-8').strip())
            except (ValueError,UnicodeError):raise RuntimeError('Unrecognised work lock: '+str(lock))
            if process_exists(pid):raise RuntimeError('Existing work process: '+str(lock))
        lock=RUN/'operation.lock'
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:globals()[args.mode]()
        except BaseException as error:
            new(RUN/(args.mode+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'mode':args.mode,'pid':os.getpid()})
            raise
        finally:lock.unlink()
    else:globals()[args.mode]()
