"""One preregistered actual fine-mask native-view study, with preserved v16 parent."""
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
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.commutator_surface import VERSION,CATEGORY,WIDTH,HEIGHT,CommutatorSurfaceMain,commutator_view,fine_mask_loss,restore_map,mixed_logits,freeze_projection
from safelog_ai.ksdd_supervised import ksdd_photo_score
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_mixed_real_mask import get_cache as target_cache,parent_target_scores,draw_group_balanced
from scripts.study_real_mask_transfer import group_rates

RUN=ROOT/'runs/industrial-ksdd1-v18';DATA=ROOT/'data/industrial-ksdd1-v18'
MANIFEST=DATA/'manifest-r1.json';REPORTS=ROOT/'reports';PROTOCOL=REPORTS/'industrial-commutator-v18-protocol.json'
START=ROOT/'runs/industrial-magnetic-tiles-v16/released/final.pt'
EXPECTED='5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e'
PARENT_EVAL=ROOT/'runs/industrial-magnetic-tiles-v16/evaluation.json'
AUDIT=REPORTS/'industrial-ksdd1-v18-data-audit-r1.json'
SOURCES=('safelog_ai/commutator_surface.py','scripts/study_commutator_surface.py','tests/test_commutator_surface.py')
TARGETS=('mpdd_bracket_black','mpdd_tubes');METHODS=('commutator_only','max_commutator_parent')
EPOCHS,STEPS,BATCH,SEED=30,64,4,1818


def rows_for(phase):return [r for r in read(MANIFEST)['rows'] if r['phase']==phase]
def rates(labels,predictions):return classification_rates(labels,predictions,.5)


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(START)==EXPECTED
    assert sha(MANIFEST)=='aa506252c5842aef5869a4982e4b10cc746ee3ddc713cc29d33e0a9e76126736'
    assert read(ROOT/'runs/industrial-mixed-real-mask-v17/verification.json')['status']=='passed'
    assert not read(ROOT/'runs/industrial-mixed-real-mask-v17/selection.json')['selected']
    assert read(AUDIT)['status']=='ready_for_small_sample_research_protocol'
    assert shutil.disk_usage(ROOT).free>8*1024**3
    RUN.mkdir()
    ledger=dict(read(ROOT/'runs/industrial-mixed-real-mask-v17/protected-inputs.json'))
    for folder in (ROOT/'runs/industrial-mixed-real-mask-v17',DATA,REPORTS):
        for path in folder.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock','.log')) and path.relative_to(ROOT).as_posix() not in ledger:ledger[path.relative_to(ROOT).as_posix()]=snapshot(path)
    new(RUN/'protected-inputs.json',ledger)
    target_hashes={}
    for c in TARGETS:
        for phase in ('cal','test'):
            path=ROOT/'runs/industrial-magnetic-tiles-v16'/f'{c}-{phase}-features.pt';target_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    pixels={'positive':0,'valid':0}
    for r in rows_for('fit'):
        w,h=r['size'];scale=min(WIDTH/w,HEIGHT/h);pixels['valid']+=round(w*scale)*round(h*scale);pixels['positive']+=r['positive_pixels_at_native512x1408']
    positive_weight=min(100.,max(10.,math.sqrt((pixels['valid']-pixels['positive'])/pixels['positive'])))
    counts={p:dict(Counter('defect' if r['label'] else 'normal' for r in rows_for(p))) for p in ('fit','validation','cal','test')}
    p={'schema':VERSION,'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_new_encoder_features_training_VAL_CAL_TEST_scores':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'legacy_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for d in ('safelog_ai','scripts','tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
       'manifest_sha256':sha(MANIFEST),'source_audit_sha256':sha(AUDIT),'parent_sha256':EXPECTED,'parent_evaluation_sha256':sha(PARENT_EVAL),'target_cache_sha256':target_hashes,
       'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),'source_counts':counts,'one_candidate_only':True,
       'initialization':'clone accepted ksdd_head into new commutator_surface_head; high_projection frozen; only new fusion parameters updated; all accepted tensors/metadata exact',
       'view':'width512 height1408 isotropic RGB bicubic letterbox127; real fine mask>0 nearest; valid is only source rectangle; target MPDD retains old512square view',
       'features':'one existing shared backbone low40/high960; cache float16 then head float32, identical runtime half roundtrip',
       'training':'30 loop epochs x64steps x4photos,2normal/2actual fine-mask defects, inverse group-size within each label, replacement; repeated draws not new photos',
       'epochs':EPOCHS,'steps':STEPS,'batch':BATCH,'seed':SEED,'planned_optimizer_updates':EPOCHS*STEPS,
       'augmentation':'paired per-photo cached low/high/mask/valid H/V flips.5; then whole batch quarter-rotation k drawn uniformly0..3, including changed output H/W; not newly encoded RGB augmentation',
       'loss':'valid BCEWithLogits fixedFITpositiveweight + softDice +.5 top32-valid photoBCE; valid masks exclude unknown padding even in photo labels',
       'FIT_pixel_counts':pixels,'positive_weight_formula':'clamp(sqrt(FIT_valid_negative/FIT_positive),10,100)','positive_weight':positive_weight,
       'optimizer':'AdamW new fusion only lr1e-4 wd1e-4 cosine30 minimum1e-6 clip5 FP32 TF32off',
       'VAL_selection':'initial epoch0 and each complete epoch; mean of normal and defect per-photo-average loss, earliest minimum;5positive validation views only',
       'resume':'per-epoch optimizer/scheduler/RNG checkpoint and per-step actual operation journal; partial replay operations count separately',
       'source_CAL':'candidate and unadapted accepted KSDD head each own labelledCAL40 minimax maxFPRFNR then sum abs(cut-.5) cut; candidates0/1/distinctscores, strict>; frozen before TEST',
       'source_pixel':'both candidate and unadapted head evaluated original fine-mask source coordinates, fixed probability>.5, no pixel-threshold search',
       'source_registration':'never eligible this study: TEST18defects<20minimum even0FN; do not register app category or claim20percentPASS',
       'target_methods':list(METHODS),'target_CAL':'same existing normalCAL rows, rankceil((n+1)*.85); max(new/newCALcut,currentParent/currentParentCut) has own joint normalCAL cut',
       'target_guards':['each photo label>=20','FPR<=.20','FPR<=current_v16+.05','FNR<=current_v16-.05','every defect subtype>=10 FNR<=current_v16+.05','selected_epoch>0'],
       'target_selection':'eligible minimum maxFPRFNR, then FNR, then method order; no TEST cut retuning',
       'pixel_target_policy':'existing original target location map and pixel threshold exact unchanged; new map not validated target localization',
       'source_negative_scope':'no visible annotated surface defect, not whole commutator or mechanical/electrical functional health',
       'license':'CC BY-NC-SA4.0 academic noncommercial attribution/share-alike, DOI10.1007/s10845-019-01476-x',
       'known_MPDD_TEST_reused':True,'source_reserved_TEST_first_model_scoring':True,'physical_groups':'author kos object folders kept together plus conservative near links; cross-source acquisition independence unverified',
       'independent_field_validated':False,'unreviewed_v14_relations_used':False,'app_default_promoted':False,'new_unit_tests_passed':3}
    new(PROTOCOL,p)
    fit=rows_for('fit');fit=[{**r,'group_id':r['appearance_object_group']} for r in fit]
    normal=[i for i,r in enumerate(fit) if not r['label']];defect=[i for i,r in enumerate(fit) if r['label']]
    rng=np.random.default_rng(SEED);draws=[]
    for _ in range(EPOCHS*STEPS):draws.append([draw_group_balanced(fit,subset,rng) for subset in (normal,normal,defect,defect)])
    np.savez_compressed(RUN/'draws.npz',indices=np.array(draws).reshape(EPOCHS,STEPS,BATCH),flips=rng.random((EPOCHS,STEPS,BATCH,2))<.5,rotations=rng.integers(0,4,size=(EPOCHS,STEPS)))
    new(RUN/'draws.json',{'sha256':sha(RUN/'draws.npz')})
    print({'frozen':True,'planned_optimizer_updates':EPOCHS*STEPS,'positive_weight':positive_weight,'counts':counts},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in [(START,EXPECTED),(PARENT_EVAL,p['parent_evaluation_sha256']),(MANIFEST,p['manifest_sha256']),(AUDIT,p['source_audit_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256'])]:assert sha(path)==h,path
    assert sha(RUN/'draws.npz')==read(RUN/'draws.json')['sha256']
    if committed:
        repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def load(path=START,device='cpu',initialize=False):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=CommutatorSurfaceMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and (not missing or initialize and all(n.startswith('commutator_surface_head.') for n in missing))
    if initialize:model.commutator_surface_head.load_state_dict(model.ksdd_head.state_dict())
    return model.eval().requires_grad_(False).to(device),state


def cache(phase,model):
    path=RUN/(phase+'-features.pt');rows=rows_for(phase)
    if phase=='test':assert (RUN/'calibration.json').exists(),'No reserved TEST inference before frozen model/CAL'
    if path.exists():
        assert sha(path)==read(RUN/(phase+'-cache-receipt.json'))['sha256']
        a=torch.load(path,map_location='cpu',weights_only=True);assert a['protocol_sha256']==sha(PROTOCOL) and a['ids']==[r['id'] for r in rows];return rows,a
    def prepare(r):
        assert sha(ROOT/r['image'])==r['image_sha256'] and sha(ROOT/r['mask'])==r['mask_sha256']
        with Image.open(ROOT/r['image']) as im:rgb=im.convert('RGB')
        with Image.open(ROOT/r['mask']) as im:truth=im.convert('L')
        view,mask,valid,g=commutator_view(rgb,truth);assert bool(mask.any())==bool(r['label'])
        return normalized(view),mask,valid,g
    low=[];high=[];masks=[];valid=[];geometry=[]
    with ThreadPoolExecutor(max_workers=4) as pool,torch.inference_mode():
        for start in range(0,len(rows),4):
            batch=list(pool.map(prepare,rows[start:start+4]));features=model.backbone(torch.stack([r[0] for r in batch]).cuda())
            low.append(features['low'].cpu().half());high.append(features['high'].cpu().half());masks.extend(r[1] for r in batch);valid.extend(r[2] for r in batch);geometry.extend(r[3] for r in batch)
            if start%64==0:print({'cache_phase':phase,'photos':min(start+4,len(rows)),'total':len(rows)},flush=True)
    a={'protocol_sha256':sha(PROTOCOL),'ids':[r['id'] for r in rows],'low':torch.cat(low),'high':torch.cat(high),'masks':torch.stack(masks),'valid':torch.stack(valid),'geometry':geometry}
    partial=path.with_suffix('.partial.pt')
    with partial.open('xb') as out:torch.save(a,out)
    partial.rename(path)
    new(RUN/(phase+'-cache-receipt.json'),{'sha256':sha(path),'bytes':path.stat().st_size,'protocol_sha256':sha(PROTOCOL),'photos':len(rows),'source_ids_exact':True,'feature_dtypes':'float16','masks_and_valid_same_native_coordinates':True})
    return rows,a


def validation(head,values,rows,positive_weight):
    result={};head.eval()
    with torch.inference_mode():
        for label in (0,1):
            ids=[i for i,r in enumerate(rows) if r['label']==label];total=0
            for start in range(0,len(ids),BATCH):
                at=ids[start:start+BATCH];batch={k:values[k][at].cuda() for k in ('low','high','masks','valid')}
                out=mixed_logits(head,batch['low'],batch['high'],batch['masks'].shape[-2:]);total+=float(fine_mask_loss(out,batch['masks'][:,None],batch['valid'][:,None],positive_weight))*len(at)
            result[str(label)]=total/len(ids)
    result['balanced_loss']=(result['0']+result['1'])/2
    return result


def train():
    p=validate(committed=True);assert not (RUN/'trained/training.json').exists()
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);model,parent=load(device='cuda',initialize=True);head=model.commutator_surface_head
    fr,fit=cache('fit',model);vr,val=cache('validation',model);params=freeze_projection(head)
    initial={n:t.detach().cpu().clone() for n,t in head.state_dict().items()}
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)
    folder=RUN/'train';folder.mkdir(exist_ok=True);progress=folder/'progress.pt';journal=folder/'optimizer.jsonl';draws=np.load(RUN/'draws.npz')
    epoch0=0;actual=0;best=float('inf');best_epoch=0
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);assert saved['protocol_sha256']==sha(PROTOCOL)
        head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);epoch0=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        baseline=validation(head,val,vr,p['positive_weight']);best=baseline['balanced_loss'];new(folder/'initial-VAL.json',baseline)
        torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':0,'VAL_loss':best,'protocol_sha256':sha(PROTOCOL)},folder/'best.pt')
        torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':0,'best':best,'best_epoch':0,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        print({'initial_VAL':baseline},flush=True)
    if journal.exists():actual=sum(1 for _ in journal.open(encoding='utf-8'))
    for epoch in range(epoch0,EPOCHS):
        head.train();losses=[]
        for step in range(STEPS):
            at=draws['indices'][epoch,step];batch={k:fit[k][at].clone().cuda() for k in ('low','high','masks','valid')}
            for i in range(BATCH):
                axes=tuple(a for enabled,a in zip(draws['flips'][epoch,step,i],(-1,-2)) if enabled)
                if axes:
                    for k in batch:batch[k][i]=batch[k][i].flip(axes)
            rotation=int(draws['rotations'][epoch,step])
            if rotation:batch={k:torch.rot90(t,rotation,(-2,-1)) for k,t in batch.items()}
            optimizer.zero_grad(set_to_none=True);out=mixed_logits(head,batch['low'],batch['high'],batch['masks'].shape[-2:]);loss=fine_mask_loss(out,batch['masks'][:,None],batch['valid'][:,None],p['positive_weight']);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,5);optimizer.step();actual+=1;losses.append(float(loss.detach()))
            with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'actual_operation':actual,'epoch':epoch+1,'step':step+1,'loss':losses[-1]})+'\n')
        vl=validation(head,val,vr,p['positive_weight'])
        if vl['balanced_loss']<best:
            best=vl['balanced_loss'];best_epoch=epoch+1;torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':best_epoch,'VAL_loss':best,'protocol_sha256':sha(PROTOCOL)},folder/'best.pt')
        scheduler.step();torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        item={'epoch':epoch+1,'actual_optimizer_updates':actual,'scheduled_unique_updates':(epoch+1)*STEPS,'training_loss':sum(losses)/len(losses),'VAL':vl,'selected_epoch':best_epoch}
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
        print(item,flush=True)
    chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(t,chosen['head'][n]) for n,t in initial.items() if n.startswith('high_projection.'))
    state=deepcopy(parent);state['state_dict'].update({'commutator_surface_head.'+n:t for n,t in chosen['head'].items()})
    assert all(torch.equal(t,state['state_dict'][n]) for n,t in parent['state_dict'].items())
    state.update(commutator_surface_version=VERSION,commutator_parent_sha256=EXPECTED,commutator_protocol_sha256=sha(PROTOCOL),commutator_actual_optimizer_updates=actual,commutator_selected_epoch=best_epoch)
    (RUN/'trained').mkdir();torch.save(state,RUN/'trained/final.pt')
    new(RUN/'trained/training.json',{'actual_optimizer_updates':actual,'scheduled_unique_updates':EPOCHS*STEPS,'selected_epoch':best_epoch,'selected_epoch_updates':best_epoch*STEPS,'completed_epochs':EPOCHS,'VAL_balanced_loss':best,'source_counts':p['source_counts'],'positive_weight':p['positive_weight'],'parent_all_tensors_exact':True,'new_projection_frozen':True})


def scores_cached(model,values,baseline=False):
    scores=[]
    head=model.ksdd_head if baseline else model.commutator_surface_head
    with torch.inference_mode():
        for start in range(0,len(values['low']),BATCH):
            at=slice(start,start+BATCH);prob=mixed_logits(head,values['low'][at].cuda(),values['high'][at].cuda(),values['valid'].shape[-2:])[:,0].sigmoid().cpu()
            scores.extend(ksdd_photo_score(prob,values['valid'][at]).tolist())
    return scores


def labelled_cal(labels,scores):
    candidates=[]
    for cut in sorted(set([0.,1.,*scores])):
        r=classification_rates(labels,scores,cut);candidates.append((max(r['fpr'],r['fnr']),r['fpr']+r['fnr'],abs(cut-.5),cut,r))
    choice=min(candidates,key=lambda r:r[:4]);return {'photo_threshold':choice[3],'CAL_rates':choice[4]}


def get_target(category,phase,p):
    path=ROOT/'runs/industrial-magnetic-tiles-v16'/f'{category}-{phase}-features.pt'
    assert sha(path)==p['target_cache_sha256'][path.relative_to(ROOT).as_posix()]
    return target_cache(category,phase)


def calibrate():
    p=validate();assert not (RUN/'calibration.json').exists();model,state=load(RUN/'trained/final.pt','cuda')
    rows,v=cache('cal',model);labels=[r['label'] for r in rows];scores=scores_cached(model,v);baseline=scores_cached(model,v,True)
    result={CATEGORY:{'candidate':labelled_cal(labels,scores),'unadapted_ksdd_head':labelled_cal(labels,baseline),'observations':[{'id':r['id'],'label':r['label'],'score':s,'unadapted_score':b} for r,s,b in zip(rows,scores,baseline)]}}
    for c in TARGETS:
        rows,v=get_target(c,'cal',p);assert not any(r['label'] for r in rows)
        scores=scores_cached(model,v);cut,rank=normal_budget_threshold(scores,.15);old,oldcut=parent_target_scores(model,state,c,v);assert min(cut,oldcut)>0
        fusion=[max(n/cut,o/oldcut) for n,o in zip(scores,old)];joint,jrank=normal_budget_threshold(fusion,.15)
        result[c]={'points':{'commutator_only':{'photo_threshold':cut,'rank':rank},'max_commutator_parent':{'photo_threshold':joint,'rank':jrank,'commutator_divisor':cut,'parent_divisor':oldcut}},'normal_CAL':len(rows),'observations':[{'id':r['id'],'score':s,'parent_score':o} for r,s,o in zip(rows,scores,old)]}
    new(RUN/'calibration.json',{'protocol_sha256':sha(PROTOCOL),'thresholds_saved_before_TEST':True,'categories':result});print({'calibrated':True,'source_points':{k:v for k,v in result[CATEGORY].items() if k!='observations'}},flush=True)


def wilson(error,n):
    z=1.959963984540054;p=error/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-radius),min(1,mid+radius)]


def evaluate():
    p=validate();assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['thresholds_saved_before_TEST'] and cal['protocol_sha256']==sha(PROTOCOL)
    model,state=load(RUN/'trained/final.pt','cuda');rows,v=cache('test',model);labels=[r['label'] for r in rows];result={CATEGORY:{}};selected={};all_rows=[]
    for variant in ('candidate','unadapted_ksdd_head'):
        baseline=variant!='candidate';scores=scores_cached(model,v,baseline);cut=cal['categories'][CATEGORY][variant]['photo_threshold'];pred=[s>cut for s in scores];r=rates(labels,pred);pixels=dict(tp=0,fp=0,tn=0,fn=0)
        with torch.inference_mode():
            head=model.ksdd_head if baseline else model.commutator_surface_head
            for i,row in enumerate(rows):
                prob=mixed_logits(head,v['low'][i:i+1].cuda(),v['high'][i:i+1].cuda(),v['valid'].shape[-2:])[:,0].sigmoid().cpu()
                guess=restore_map(prob,v['geometry'][i])[0]>.5
                with Image.open(ROOT/row['mask']) as im:truth=torch.from_numpy(np.array(im.convert('L'))>0)
                for k,m in [('tp',guess&truth),('fn',~guess&truth),('fp',guess&~truth),('tn',~guess&~truth)]:pixels[k]+=int(m.sum())
        result[CATEGORY][variant]={'rates':r,'photo_FPR_Wilson95':wilson(r['fp'],r['negative_photos']),'photo_FNR_Wilson95':wilson(r['fn'],r['positive_photos']),'auroc':rank_auroc(labels,scores),'pixel_counts_at_half_original':pixels,'pixel_fnr':pixels['fn']/max(1,pixels['fn']+pixels['tp']),'pixel_fpr':pixels['fp']/max(1,pixels['fp']+pixels['tn']),'pixel_IoU':pixels['tp']/max(1,pixels['tp']+pixels['fp']+pixels['fn']),'photo_threshold':cut,'eligible_app_category':False,'reason':'TEST18defects<20minimum, surface view only; limited-sample research'}
        all_rows.extend({'id':row['id'],'category':CATEGORY,'variant':variant,'label':row['label'],'score':s,'candidate_detected':q} for row,s,q in zip(rows,scores,pred))
    prior={r['id']:r for r in read(PARENT_EVAL)['rows']};trained=read(RUN/'trained/training.json')['selected_epoch']>0
    for c in TARGETS:
        rows,v=get_target(c,'test',p);labels=[r['label'] for r in rows];scores=scores_cached(model,v);old,oldcut=parent_target_scores(model,state,c,v);oldpred=[s>oldcut for s in old]
        assert oldpred==[prior[r['id']]['candidate_detected'] for r in rows],'Current-parent cached baseline drift'
        base=rates(labels,oldpred);bg=group_rates(rows,oldpred);result[c]={'parent':base,'parent_by_defect':bg,'methods':{}}
        for method in METHODS:
            point=cal['categories'][c]['points'][method];score=scores if method=='commutator_only' else [max(n/point['commutator_divisor'],o/point['parent_divisor']) for n,o in zip(scores,old)]
            pred=[s>point['photo_threshold'] for s in score];r=rates(labels,pred);groups=group_rates(rows,pred)
            eligible=trained and min(r['positive_photos'],r['negative_photos'])>=20 and r['fpr']<=.20 and r['fpr']<=base['fpr']+.05 and r['fnr']<=base['fnr']-.05 and all(g['photos']<10 or g['fnr']<=bg[d]['fnr']+.05 for d,g in groups.items())
            result[c]['methods'][method]={'rates':r,'by_defect':groups,'eligible':eligible,'auroc':rank_auroc(labels,score)}
            all_rows.extend({'id':row['id'],'category':c,'method':method,'label':row['label'],'defect':row['defect'],'score':s,'candidate_detected':q} for row,s,q in zip(rows,score,pred))
        eligible=[m for m in METHODS if result[c]['methods'][m]['eligible']]
        if eligible:selected[c]=min(eligible,key=lambda m:(max(result[c]['methods'][m]['rates']['fpr'],result[c]['methods'][m]['rates']['fnr']),result[c]['methods'][m]['rates']['fnr'],METHODS.index(m)))
    selection={'selected':selected,'categories':result,'training':read(RUN/'trained/training.json'),'CAL_sha256':sha(RUN/'calibration.json'),'source_reserved_TEST_first_model_scoring':True,'known_MPDD_TEST_reused':True,'independent_field_validated':False}
    new(RUN/'candidate-evaluation.json',{'rows':all_rows,'categories':result});new(RUN/'selection.json',selection);new(REPORTS/'industrial-commutator-v18-selection.json',selection)
    if selected:
        state['commutator_surface_methods']=selected;state['commutator_surface_points']={c:cal['categories'][c]['points'][m] for c,m in selected.items()};(RUN/'selected').mkdir();torch.save(state,RUN/'selected/final.pt')
    else:new(RUN/'retained-parent.json',{'checkpoint':str(START),'sha256':EXPECTED,'reason':'No preregistered target candidate eligible; source evidence insufficient for category'})
    print({'selection':selection},flush=True)


def verify():
    p=validate();selection=read(RUN/'selection.json');training=read(RUN/'trained/training.json')
    for n,h in p['legacy_source_sha256'].items():assert sha(ROOT/n)==h,n
    unchanged=0;rehashed=0
    for n,before in read(RUN/'protected-inputs.json').items():
        path=ROOT/n;st=path.stat()
        if st.st_size==before['size_bytes'] and st.st_mtime_ns==before['mtime_ns']:unchanged+=1
        else:assert sha(path)==before['sha256'],n;rehashed+=1
    parent=torch.load(START,map_location='cpu',weights_only=True);candidate=torch.load(RUN/'trained/final.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(t,candidate['state_dict'][n]) for n,t in parent['state_dict'].items())
    assert all(candidate[k]==v for k,v in parent.items() if k!='state_dict')
    for n,t in parent['state_dict'].items():
        if n.startswith('ksdd_head.high_projection.'):assert torch.equal(t,candidate['state_dict'][n.replace('ksdd_head.','commutator_surface_head.',1)])
    journal=[json.loads(s) for s in (RUN/'train/optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [j['actual_operation'] for j in journal]==list(range(1,len(journal)+1)) and len(journal)==training['actual_optimizer_updates']
    result={'status':'passed','parent_all_tensors_buffers_metadata_exact':True,'new_projection_frozen':True,'legacy_python_SHA_checked':len(p['legacy_source_sha256']),'protected_stat_unchanged':unchanged,'changed_metadata_SHA_rechecked':rehashed,'all_protected_SHA_reread':False,'actual_optimizer_updates':len(journal),'new_unit_tests_passed':3,'selected':selection['selected'],'new_source_not_app_registered':True,'new_CPU_CPP_release_required_if_selected':bool(selection['selected']),'active_parent_still_retained':True,'independent_field_validated':False}
    new(RUN/'verification.json',result);new(REPORTS/'industrial-commutator-v18-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','train','calibrate','evaluate','verify','run'));args=parser.parse_args()
    if args.mode in ('train','calibrate','evaluate','run'):
        for lock in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work process: '+str(lock))
        lock=RUN/'operation.lock'
        if lock.exists():
            new(RUN/('stale-lock-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'pid':lock.read_text(encoding='utf-8'),'live':False});lock.unlink()
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:
            if args.mode=='run':train();calibrate();evaluate();verify()
            else:globals()[args.mode]()
        except BaseException as error:
            new(RUN/(args.mode+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'mode':args.mode,'pid':os.getpid()});raise
        finally:lock.unlink()
    else:globals()[args.mode]()
