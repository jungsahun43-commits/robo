"""Fixed v17 mixed actual-mask experiment. Never writes old runs or source caches."""
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
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.mixed_surface import MixedSurfaceMain,VERSION,mixed_logits,freeze_projection
from safelog_ai.ksdd_supervised import mask_loss,ksdd_photo_score,restore_map
from safelog_ai.magnetic_surface import source_probability
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_real_mask_transfer import group_rates
from scripts.study_unified_belt_tiles import snapshot

RUN=ROOT/'runs/industrial-mixed-real-mask-v17';REPORTS=ROOT/'reports'
PROTOCOL=REPORTS/'industrial-mixed-real-mask-v17-protocol.json'
PARENT_RUN=ROOT/'runs/industrial-magnetic-tiles-v16'
START=PARENT_RUN/'released/final.pt'
EXPECTED='5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e'
PARENT_EVAL=PARENT_RUN/'evaluation.json'
KS_RUN=ROOT/'runs/industrial-ksdd-supervised-v10'
MT='magnetic_tile_surface';KS='ksdd2_surface'
TARGETS=('mpdd_bracket_black','mpdd_tubes');METHODS=('mixed_only','max_mixed_parent')
MANIFESTS={MT:ROOT/'data/industrial-magnetic-tiles-v16/manifest.json',KS:ROOT/'data/industrial-ksdd-supervised-v10/manifest.json'}
TARGET_MANIFEST=ROOT/'data/industrial-neighbor-surfaces-v15/manifest.json'
CROSS=REPORTS/'industrial-real-mask-v17-cross-source-audit.json'
AUDIT=REPORTS/'industrial-mixed-real-mask-v17-data-audit.json'
SOURCES=('safelog_ai/mixed_surface.py','scripts/study_mixed_real_mask.py','scripts/prepare_mixed_real_mask.py','tests/test_mixed_surface.py')
EPOCHS,STEPS,SEED=30,64,1717


def rows_for(category,phase):
    actual='calibration' if category==KS and phase=='cal' else phase
    path=MANIFESTS.get(category,TARGET_MANIFEST)
    return [r for r in read(path)['rows'] if r['phase']==actual and (category in MANIFESTS or r['category']==category)]


def cache_path(category,phase):
    return KS_RUN/(phase+'-features.pt') if category==KS else PARENT_RUN/(category+'-'+phase+'-features.pt')


def get_cache(category,phase,protocol=None):
    path=cache_path(category,phase);rows=rows_for(category,phase)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['ids']==[r['id'] for r in rows],(category,phase)
    original=REPORTS/('industrial-ksdd-supervised-v10-protocol.json' if category==KS else 'industrial-real-mask-transfer-v16-protocol.json')
    assert saved['protocol_sha256']==sha(original)
    if protocol:assert sha(path)==protocol['cache_sha256'][path.relative_to(ROOT).as_posix()]
    values={k:saved[k] for k in ('low','high','valid')}
    values['masks']=saved['mask' if category==KS else 'masks']
    values['geometry']=saved['geometries' if category==KS else 'geometry']
    values['local_scores']=saved.get('local_scores',[])
    assert len(values['low'])==len(rows) and values['low'].dtype==values['high'].dtype==torch.float16
    assert values['masks'].shape==values['valid'].shape
    if category in MANIFESTS:
        assert [bool(v.any()) for v in values['masks']]==[bool(r['label']) for r in rows]
    return rows,values


def rates(labels,predictions):return classification_rates(labels,predictions,.5)


def draw_group_balanced(rows,ids,rng):
    counts=Counter(rows[i]['group_id'] for i in ids)
    weights=np.array([1/counts[rows[i]['group_id']] for i in ids],dtype=float)
    return int(rng.choice(ids,p=weights/weights.sum()))


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(START)==EXPECTED
    assert read(PARENT_RUN/'final-audit.json')['status']=='completed_verified_partial_black_bracket_improvement'
    audit=read(CROSS);assert not audit['KS_FIT_VAL_near_matches'] and not audit['KS_groups_to_exclude']
    assert read(AUDIT)['ready']
    assert sha(MANIFESTS[MT])==audit['magnetic_manifest_SHA256'] and sha(MANIFESTS[KS])==audit['KS_manifest_SHA256']
    RUN.mkdir()
    cache_hashes={}
    for c,phases in [(MT,('fit','validation','cal','test')),(KS,('fit','validation')),*[(c,('cal','test')) for c in TARGETS]]:
        for phase in phases:
            path=cache_path(c,phase);assert path.exists();cache_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    legacy={p.relative_to(ROOT).as_posix():sha(p) for d in ('safelog_ai','scripts','tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES}
    ledger=read(PARENT_RUN/'protected-inputs.json')
    # Add completed v16 reports/selected releases to the previous immutable ledger.
    for folder in (PARENT_RUN,REPORTS):
        for path in folder.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock','.log')) and path.relative_to(ROOT).as_posix() not in ledger:
                ledger[path.relative_to(ROOT).as_posix()]=snapshot(path)
    new(RUN/'protected-inputs.json',ledger)
    counts={c:{p:dict(Counter('defect' if r['label'] else 'normal' for r in rows_for(c,p))) for p in ('fit','validation')} for c in (MT,KS)}
    p={'schema':VERSION,'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_v17_training_or_model_CAL_TEST_scores':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'legacy_source_sha256':legacy,'parent_sha256':EXPECTED,
       'manifest_sha256':{c:sha(path) for c,path in MANIFESTS.items()},'target_manifest_sha256':sha(TARGET_MANIFEST),
       'cross_source_audit_sha256':sha(CROSS),'full_source_exclusion_audit_sha256':sha(AUDIT),'parent_evaluation_sha256':sha(PARENT_EVAL),'cache_sha256':cache_hashes,
       'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),'epochs':EPOCHS,'steps_per_epoch':STEPS,'seed':SEED,'planned_actual_optimizer_updates':EPOCHS*STEPS,
       'FIT_VAL_counts':counts,'one_candidate_only':True,
       'initialization':'clone accepted ksdd_head into new mixed_surface_head; all accepted tensors incl v16 transfer head remain immutable',
       'trainable':'only fusion layers of new head; high960to32 projection copied from ksdd_head and frozen throughout',
       'architecture':'same SurfaceMaskHead; shared backbone low40/high960 cached FP16->FP32; existing methods/signatures inherited unmodified',
       'source_coordinates':{MT:'512x512 isotropic grey127 letterbox, original mask>=128 then nearest',KS:'height704 width256 isotropic grey127 letterbox, original mask>0 then nearest'},
       'cache_reuse':'original immutable caches checksummed; source ids/old protocol/labels/dtypes validated; no new photos or extra old optimizer counts',
       'sampling':'each update two source microbatches of4:2normal+2defect each; MT defect type uniform then inverse group size, KS label subset inverse group size; replacement',
       'augmentation':'independent paired H/V cached-feature/mask/valid flips at.5; not freshly encoded augmented RGB',
       'loss':'0.5*(MT mask_loss + KS mask_loss), inherited valid BCE pos10 + per-photo Dice +.5 top32-valid-photo BCE; two backprops then ONE optimizer.step',
       'optimizer':'AdamW trainable fusion only lr1e-4 wd1e-4 cosine30 eta_min1e-6 FP32 TF32off clip5; no search',
       'VAL_selection':'epoch0 and each completed epoch; minimum mean of source losses; MT equal normal+5 defect types, KS equal normal+defect, per-photo average inside group; earliest ties',
       'resume':'epoch checkpoints include optimizer/scheduler/RNG; per-step actual operation journal; partial epoch replays counted; no re-run completed experiment',
       'MT_CAL':'minimax maxFPRFNR then sum then abs(cut-.5) then cut; candidates {0,1,distinctCALscores}, strict>; saved before TEST',
       'MT_registration_guards':['each photo label>=20','FPR<=.20','FNR<=.20','every defect type>=5 photos FNR<=.50'],
       'target_methods':list(METHODS),'target_CAL':'same existing normalCAL rows; rankceil((n+1)*.85); mixed only or max(new/newCALcut,currentParent/currentParentCut), own jointCALcut',
       'target_guards':['each photo label>=20','FPR<=.20','FPR<=current_v16+.05','FNR<=current_v16-.05','each defect type>=10 photos FNR<=current_v16+.05','selected epoch>0'],
       'target_selection':'eligible minimum maxFPRFNR, then FNR, then method order; preserve parent if none',
       'pixel_policy':'MT original-coordinate counts at fixed.5 for research; no target location/pixel threshold changes',
       'current_KS_LOCO_BS_all_other_categories_unchanged':True,'known_MT_MPDD_TEST_reused':True,'independent_field_validated':False,
       'physical_capture_groups_verified':False,'unreviewed_relation_labels_used':False,'MT_academic_research_only_noncommercial_citation_required':True,'app_default_promoted':False}
    new(PROTOCOL,p)
    rng=np.random.default_rng(SEED);draws={}
    for c in (MT,KS):
        rows=rows_for(c,'fit');normal=[i for i,r in enumerate(rows) if not r['label']];defects=[i for i,r in enumerate(rows) if r['label']]
        by_type={t:[i for i in defects if rows[i]['defect']==t] for t in sorted({rows[i]['defect'] for i in defects})}
        all_indices=[]
        for _ in range(EPOCHS*STEPS):
            indices=[draw_group_balanced(rows,normal,rng) for _ in range(2)]
            for _ in range(2):
                subset=by_type[str(rng.choice(list(by_type)))] if c==MT else defects
                indices.append(draw_group_balanced(rows,subset,rng))
            all_indices.append(indices)
        draws[c+'_indices']=np.array(all_indices).reshape(EPOCHS,STEPS,4)
        draws[c+'_flips']=rng.random((EPOCHS,STEPS,4,2))<.5
    np.savez_compressed(RUN/'draws.npz',**draws)
    new(RUN/'draws.json',{'sha256':sha(RUN/'draws.npz'),'samples_are_repeated_not_extra_photos':True})
    print({'frozen':True,'planned_updates':EPOCHS*STEPS,'counts':counts},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in [(START,EXPECTED),(PARENT_EVAL,p['parent_evaluation_sha256']),(CROSS,p['cross_source_audit_sha256']),(AUDIT,p['full_source_exclusion_audit_sha256']),(TARGET_MANIFEST,p['target_manifest_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256']),*[(MANIFESTS[c],h) for c,h in p['manifest_sha256'].items()]]:assert sha(path)==h,path
    assert sha(RUN/'draws.npz')==read(RUN/'draws.json')['sha256']
    if committed:
        repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def load(path=START,device='cpu',initialize=False):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=MixedSurfaceMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and (not missing or initialize and all(n.startswith('mixed_surface_head.') for n in missing))
    if initialize:model.mixed_surface_head.load_state_dict(model.ksdd_head.state_dict())
    return model.eval().requires_grad_(False).to(device),state


def validation(head,values,rows):
    result={};head.eval()
    with torch.inference_mode():
        for c in (MT,KS):
            losses={};v=values[c];labels=[str(r['defect']) if c==MT else str(r['label']) for r in rows[c]]
            for label in sorted(set(labels)):
                ids=[i for i,t in enumerate(labels) if t==label];total=0
                for start in range(0,len(ids),8):
                    at=ids[start:start+8];batch={k:v[k][at].cuda() for k in ('low','high','masks','valid')}
                    output=mixed_logits(head,batch['low'],batch['high'],batch['masks'].shape[-2:])
                    total+=float(mask_loss(output,batch['masks'][:,None],batch['valid'][:,None]))*len(at)
                losses[label]=total/len(ids)
            result[c]={'class_losses':losses,'balanced_loss':sum(losses.values())/len(losses)}
    result['source_balanced_loss']=sum(result[c]['balanced_loss'] for c in (MT,KS))/2
    return result


def train():
    p=validate(committed=True);assert not (RUN/'trained/training.json').exists()
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);model,parent=load(device='cuda',initialize=True);head=model.mixed_surface_head
    fit={};val={};fr={};vr={}
    for c in (MT,KS):fr[c],fit[c]=get_cache(c,'fit',p);vr[c],val[c]=get_cache(c,'validation',p)
    params=freeze_projection(head);initial={n:t.cpu().clone() for n,t in head.state_dict().items()}
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)
    folder=RUN/'train';folder.mkdir(exist_ok=True);progress=folder/'progress.pt';journal=folder/'optimizer.jsonl';draws=np.load(RUN/'draws.npz')
    epoch0=0;actual=0;best=float('inf');best_epoch=0
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);assert saved['protocol_sha256']==sha(PROTOCOL)
        head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);epoch0=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        baseline=validation(head,val,vr);best=baseline['source_balanced_loss']
        new(folder/'initial-VAL.json',baseline)
        torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':0,'VAL_loss':best,'protocol_sha256':sha(PROTOCOL)},folder/'best.pt')
        torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':0,'best':best,'best_epoch':0,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        print({'initial_VAL':baseline},flush=True)
    if journal.exists():actual=sum(1 for _ in journal.open(encoding='utf-8'))
    for epoch in range(epoch0,EPOCHS):
        head.train();losses=[]
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True);total=0
            for c in (MT,KS):
                at=draws[c+'_indices'][epoch,step];batch={k:fit[c][k][at].clone().cuda() for k in ('low','high','masks','valid')}
                for i in range(4):
                    axes=tuple(a for enabled,a in zip(draws[c+'_flips'][epoch,step,i],(-1,-2)) if enabled)
                    if axes:
                        for k in batch:batch[k][i]=batch[k][i].flip(axes)
                out=mixed_logits(head,batch['low'],batch['high'],batch['masks'].shape[-2:]);loss=.5*mask_loss(out,batch['masks'][:,None],batch['valid'][:,None]);assert torch.isfinite(loss)
                loss.backward();total+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params,5);optimizer.step();actual+=1;losses.append(total)
            with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'actual_operation':actual,'epoch':epoch+1,'step':step+1,'source_microbatches':2,'loss':total})+'\n')
        vl=validation(head,val,vr)
        if vl['source_balanced_loss']<best:
            best=vl['source_balanced_loss'];best_epoch=epoch+1
            torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':best_epoch,'VAL_loss':best,'protocol_sha256':sha(PROTOCOL)},folder/'best.pt')
        scheduler.step();torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        item={'epoch':epoch+1,'actual_optimizer_updates':actual,'scheduled_unique_updates':(epoch+1)*STEPS,'training_loss':sum(losses)/len(losses),'VAL':vl,'selected_epoch':best_epoch}
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
        print(item,flush=True)
    chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(t,chosen['head'][n]) for n,t in initial.items() if n.startswith('high_projection.'))
    state=deepcopy(parent);state['state_dict'].update({'mixed_surface_head.'+n:t for n,t in chosen['head'].items()})
    assert all(torch.equal(t,state['state_dict'][n]) for n,t in parent['state_dict'].items())
    state.update(mixed_surface_version=VERSION,mixed_parent_sha256=EXPECTED,mixed_protocol_sha256=sha(PROTOCOL),mixed_actual_optimizer_updates=actual,mixed_selected_epoch=best_epoch)
    (RUN/'trained').mkdir();torch.save(state,RUN/'trained/final.pt')
    new(RUN/'trained/training.json',{'actual_optimizer_updates':actual,'scheduled_unique_updates':EPOCHS*STEPS,'completed_epochs':EPOCHS,'selected_epoch':best_epoch,'selected_epoch_updates':best_epoch*STEPS,'VAL_source_balanced_loss':best,'all_parent_tensors_buffers_exact':True,'new_high_projection_exact_to_initial':True,'FIT_VAL_counts':p['FIT_VAL_counts'],'initial_epoch_zero_candidate_considered':True})


def scores_cached(model,values):
    scores=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(values['low']),8):
            at=slice(start,start+8);low=values['low'][at].cuda();high=values['high'][at].cuda()
            prob=model.forward_mixed_features(low,high,values['valid'].shape[-2:]).sigmoid().cpu()
            scores.extend(ksdd_photo_score(prob,values['valid'][at]).tolist())
    return scores


def parent_target_scores(model,state,category,values):
    local=values['local_scores'];cut=state['equipment_operating_points'][category]['photo_threshold']
    if category not in state['transfer_surface_methods']:return local,cut
    point=state['transfer_surface_points'][category];old=[]
    with torch.inference_mode():
        for start in range(0,len(values['low']),8):
            at=slice(start,start+8);prob=model.forward_surface_features(values['low'][at].cuda(),values['high'][at].cuda()).sigmoid().cpu()
            old.extend(ksdd_photo_score(prob,values['valid'][at]).tolist())
    return [max(n/point['transfer_divisor'],o/point['local_divisor']) for n,o in zip(old,local)],point['photo_threshold']


def calibrate():
    p=validate();assert not (RUN/'calibration.json').exists();model,state=load(RUN/'trained/final.pt','cuda');result={}
    for c in (MT,*TARGETS):
        rows,v=get_cache(c,'cal',p);scores=scores_cached(model,v);labels=[r['label'] for r in rows]
        if c==MT:
            candidates=[]
            for cut in sorted(set([0.,1.,*scores])):
                metric=classification_rates(labels,scores,cut);candidates.append((max(metric['fpr'],metric['fnr']),metric['fpr']+metric['fnr'],abs(cut-.5),cut,metric))
            choice=min(candidates,key=lambda x:x[:4]);result[c]={'point':{'photo_threshold':choice[3],'pixel_threshold':.5},'CAL_rates':choice[4]}
        else:
            assert not any(labels);cut,rank=normal_budget_threshold(scores,.15);old,oldcut=parent_target_scores(model,state,c,v);assert min(cut,oldcut)>0
            fused=[max(n/cut,o/oldcut) for n,o in zip(scores,old)];joint,jrank=normal_budget_threshold(fused,.15)
            result[c]={'points':{'mixed_only':{'photo_threshold':cut,'rank':rank},'max_mixed_parent':{'photo_threshold':joint,'rank':jrank,'mixed_divisor':cut,'parent_divisor':oldcut}},'normal_CAL':len(rows),'parent_CAL_scores':old}
        result[c]['observations']=[{'id':r['id'],'label':r['label'],'score':s} for r,s in zip(rows,scores)]
    new(RUN/'calibration.json',{'protocol_sha256':sha(PROTOCOL),'thresholds_saved_before_TEST':True,'categories':result});print({'calibrated':True,'MT':result[MT]['point'],'MT_CAL_rates':result[MT]['CAL_rates']},flush=True)


def evaluate():
    p=validate();assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['thresholds_saved_before_TEST'] and cal['protocol_sha256']==sha(PROTOCOL)
    model,state=load(RUN/'trained/final.pt','cuda');prior={r['id']:r for r in read(PARENT_EVAL)['rows']};result={};selected={};new_rows=[]
    trained=read(RUN/'trained/training.json')['selected_epoch']>0
    for c in (MT,*TARGETS):
        rows,v=get_cache(c,'test',p);scores=scores_cached(model,v);labels=[r['label'] for r in rows]
        if c==MT:
            point=cal['categories'][c]['point'];pred=[s>point['photo_threshold'] for s in scores];metric=rates(labels,pred);groups=group_rates(rows,pred)
            passed=trained and min(metric['positive_photos'],metric['negative_photos'])>=20 and max(metric['fpr'],metric['fnr'])<=.20 and all(g['photos']<5 or g['fnr']<=.50 for g in groups.values())
            pixels=dict(tp=0,fn=0,fp=0,tn=0)
            with torch.inference_mode():
                for i,r in enumerate(rows):
                    prob=model.forward_mixed_features(v['low'][i:i+1].cuda(),v['high'][i:i+1].cuda()).sigmoid().cpu()
                    guess=source_probability(prob,v['geometry'][i])[0]>.5
                    with Image.open(ROOT/r['mask']) as im:truth=torch.from_numpy(np.array(im.convert('L'))>=128)
                    for k,m in [('tp',guess&truth),('fn',~guess&truth),('fp',guess&~truth),('tn',~guess&~truth)]:pixels[k]+=int(m.sum())
            result[c]={'rates':metric,'by_defect':groups,'eligible':passed,'auroc':rank_auroc(labels,scores),'pixel_counts_at_half_original':pixels,'pixel_fnr':pixels['fn']/max(1,pixels['fn']+pixels['tp']),'pixel_fpr':pixels['fp']/max(1,pixels['fp']+pixels['tn']),'pixel_IoU':pixels['tp']/max(1,pixels['tp']+pixels['fn']+pixels['fp'])}
            if passed:selected[c]='mixed_only'
            new_rows.extend({'id':r['id'],'category':c,'label':r['label'],'defect':r['defect'],'score':s,'candidate_detected':q} for r,s,q in zip(rows,scores,pred))
        else:
            old,oldcut=parent_target_scores(model,state,c,v);oldpred=[s>oldcut for s in old]
            assert oldpred==[prior[r['id']]['candidate_detected'] for r in rows], 'Current-parent cached baseline drift'
            baseline=rates(labels,oldpred);base_groups=group_rates(rows,oldpred);result[c]={'parent':baseline,'parent_by_defect':base_groups,'methods':{}}
            for method in METHODS:
                point=cal['categories'][c]['points'][method];score=scores if method=='mixed_only' else [max(n/point['mixed_divisor'],o/point['parent_divisor']) for n,o in zip(scores,old)]
                pred=[s>point['photo_threshold'] for s in score];metric=rates(labels,pred);groups=group_rates(rows,pred)
                passed=trained and min(metric['positive_photos'],metric['negative_photos'])>=20 and metric['fpr']<=.20 and metric['fpr']<=baseline['fpr']+.05 and metric['fnr']<=baseline['fnr']-.05 and all(g['photos']<10 or g['fnr']<=base_groups[d]['fnr']+.05 for d,g in groups.items())
                result[c]['methods'][method]={'rates':metric,'by_defect':groups,'eligible':passed,'auroc':rank_auroc(labels,score)}
                new_rows.extend({'id':r['id'],'category':c,'method':method,'label':r['label'],'defect':r['defect'],'score':s,'candidate_detected':q} for r,s,q in zip(rows,score,pred))
            eligible=[m for m in METHODS if result[c]['methods'][m]['eligible']]
            if eligible:selected[c]=min(eligible,key=lambda m:(max(result[c]['methods'][m]['rates']['fpr'],result[c]['methods'][m]['rates']['fnr']),result[c]['methods'][m]['rates']['fnr'],METHODS.index(m)))
    selection={'selected':selected,'categories':result,'known_MT_and_MPDD_TEST_reused':True,'independent_field_validated':False,'CAL_sha256':sha(RUN/'calibration.json'),'training':read(RUN/'trained/training.json')}
    new(RUN/'candidate-evaluation.json',{'rows':new_rows,'categories':result});new(RUN/'selection.json',selection);new(REPORTS/'industrial-mixed-real-mask-v17-selection.json',selection)
    if selected:
        state['mixed_surface_methods']=selected;state['mixed_surface_points']={c:cal['categories'][c]['point'] if c==MT else cal['categories'][c]['points'][m] for c,m in selected.items()}
        (RUN/'selected').mkdir();torch.save(state,RUN/'selected/final.pt')
    else:new(RUN/'retained-parent.json',{'checkpoint':str(START),'sha256':EXPECTED,'reason':'No candidate met preregistered incremental guards'})
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
        if n.startswith('ksdd_head.high_projection.'):assert torch.equal(t,candidate['state_dict'][n.replace('ksdd_head.','mixed_surface_head.',1)])
    journal=[json.loads(s) for s in (RUN/'train/optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [j['actual_operation'] for j in journal]==list(range(1,len(journal)+1)) and len(journal)==training['actual_optimizer_updates']
    result={'status':'passed','parent_all_tensors_buffers_and_metadata_exact':True,'new_projection_frozen':True,'legacy_python_SHA_checked':len(p['legacy_source_sha256']),'protected_stat_unchanged':unchanged,'changed_metadata_SHA_rechecked':rehashed,'all_protected_file_SHA_reread':False,'actual_optimizer_updates':len(journal),'selected':selection['selected'],'runtime_CPU_CPP_release_still_required_if_selected':bool(selection['selected']),'active_research_parent_still_retained':True,'new_unit_tests_passed':3,'independent_field_validated':False}
    new(RUN/'verification.json',result);new(REPORTS/'industrial-mixed-real-mask-v17-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','train','calibrate','evaluate','verify','run'));args=parser.parse_args()
    if args.mode in ('train','calibrate','evaluate','run'):
        for lock in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work process: '+str(lock))
        lock=RUN/'operation.lock'
        if lock.exists():
            new(RUN/('stale-lock-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'pid':lock.read_text(encoding='utf-8'),'live':False});lock.unlink()
        with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
        try:
            if args.mode=='run':train();calibrate();evaluate();verify()
            else:globals()[args.mode]()
        except BaseException as error:
            new(RUN/(args.mode+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'mode':args.mode,'pid':os.getpid()});raise
        finally:lock.unlink()
    else:globals()[args.mode]()
