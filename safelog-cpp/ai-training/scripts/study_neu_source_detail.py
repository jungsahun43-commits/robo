"""Source-only actual-mask detail discrimination, separate from stopped v22."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.tile_surface import VERSION, tile_boxes, tile_view, per_tile_loss, DetailSurfaceMain
from safelog_ai.fine_target_surface import trainable_fine_head
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.native_component_views import normalized
from safelog_ai.magnetic_surface import surface_view
from safelog_ai.ksdd_supervised import ksdd_photo_score
from safelog_ai.belt_normal_roi import fingerprints, distance
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_unified_belt_tiles import snapshot
from scripts.prepare_magnetic_tiles import Bands
from scripts import study_neu_detail as preflight
from safelog_ai.source_detail_study import source_only_rows, photo_bucket_means
from scripts.study_mixed_real_mask import get_cache as old_cache, parent_target_scores, draw_group_balanced
from scripts.study_real_mask_transfer import group_rates

VERSION = 'safelog-neu-source-detail-v23'
RUN = ROOT / 'runs/industrial-neu-source-detail-v23'
DATA = ROOT / 'data/industrial-neu-source-detail-v23'
NEU = ROOT / 'data/industrial-neu-micro-v22'
REPORTS = ROOT / 'reports'
PROTOCOL = REPORTS / 'industrial-neu-source-detail-v23-protocol.json'
MANIFEST = DATA / 'manifest.json'
NORMAL = ROOT / 'data/industrial-neighbor-surfaces-v15/manifest.json'
START = ROOT / 'runs/industrial-magnetic-tiles-v16/released/final.pt'
EXPECTED = '5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e'
WARM = ROOT / 'runs/industrial-balanced-targets-v21/trained/final.pt'
WARM_SHA = '43df7c83403d65cbe0669e59c30c9c1085c0936de2e6c8b611ace436923dabdc'
PARENT_EVAL = START.parent.parent / 'evaluation.json'
TARGETS = ('mpdd_bracket_black', 'mpdd_tubes')
METHODS = ('detail_only', 'max_detail_parent')
SOURCES = ('safelog_ai/source_detail_study.py', 'scripts/study_neu_source_detail.py', 'tests/test_source_detail_study.py')
EPOCHS, STEPS, BATCH, SEED = 30, 64, 8, 2323
KEYS = ('low', 'high', 'fine', 'masks', 'valid')
HELDOUT = ('data/equipment-expansion-v1/manifest.json', 'data/industrial-ksdd-supervised-v10/manifest.json',
           'data/industrial-bsclass-v1/manifest.json', 'data/industrial-magnetic-tiles-v16/manifest.json',
           'data/industrial-ksdd1-v18/manifest-r1.json')


def freeze():
    assert not RUN.exists() and not DATA.exists() and not PROTOCOL.exists()
    assert sha(START)==EXPECTED and sha(WARM)==WARM_SHA
    assert read(preflight.RUN/'final-audit.json')['status']=='v22_stopped_at_preregistered_data_gate_no_training'
    assert not (preflight.RUN/'train').exists() and shutil.disk_usage(ROOT).free>4*1024**3
    RUN.mkdir();DATA.mkdir()
    ledger=dict(read(preflight.RUN/'protected-inputs.json'))
    for folder in (preflight.RUN,preflight.DATA,REPORTS):
        for path in folder.rglob('*'):
            name=path.relative_to(ROOT).as_posix()
            if path.is_file() and not path.name.endswith(('.lock','.log')) and name not in ledger:ledger[name]=snapshot(path)
    new(RUN/'protected-inputs.json',ledger)
    p={'schema':VERSION,'UTC':datetime.now(timezone.utc).isoformat(),
       'fixed_before_new_features_gradients_CAL_TEST_model_scores':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
       'legacy_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for d in ('safelog_ai','scripts','tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
       'parent_sha256':EXPECTED,'warm_start_sha256':WARM_SHA,'parent_evaluation_sha256':sha(PARENT_EVAL),
       'stopped_v22_manifest_sha256':sha(preflight.MANIFEST),
       'stopped_v22_audit_sha256':sha(REPORTS/'industrial-neu-detail-v22-data-audit.json'),
       'NEU_candidate_manifest_sha256':sha(NEU/'candidate-manifest.json'),'NEU_FIT_review_sha256':sha(NEU/'FIT-review-r1.json'),
       'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),
       'design':'source-only real annotated crop discrimination; no MPDD target normal in gradient training or VAL selection. Positive and annotation-negative crops come from same real source photos to reduce source/domain shortcut',
       'split':'same pre-reserved FIT35 SDI20/SPDI15 and VAL10 OilSmall4/Wipe6; excluded OilScratch6 never used; v22 preflight crop exact/near group exclusions inherited unchanged',
       'new_source_normal_whole_photos':0,'new_app_category_claimed':False,
       'tile_policy':'same nine half-width/half-height native crops at start/middle/end each axis, aspect-preserving512 bicubicRGB/nearestmask/grey127letterbox; ALL9 at CAL/TEST with no GT-based crop selection',
       'crop_labels':'source>=32 positive pixels at512 retained positive;0 retained annotation-negative, not healthy whole equipment;1..31 excluded, not normal',
       'data_gate':'FIT positive original photos>=20,>=5 each SDI/SPDI; each source annotation-negative crops>0; VAL positive original photos>=8 and annotation-negative original photos>=8; source FIT/VAL ids disjoint and all inherited heldout crop/group exclusions preserved',
       'features':'same original frozen shared encoder fine24/low40/high960 FP16 roundtrip; no new pretrained encoder and no synthesis',
       'head':'same89channel FineTargetHead at128grid, v21 VAL weights copied to detail_surface_head; residual channel always0 for ALL source/target/inference; no category/source embeddings',
       'trainable':'fine projection/fusion only; high projection and all parent tensors/buffers/metadata fixed',
       'loss':'per-tile positive/negative mean BCE .5/.5 for positives, negative mean for annotation-negative, +softDice+.5 top32valid photoBCE; no area weight',
       'sampling':'8 per step:2SDI positive+2SPDI positive+2SDI annotation-negative+2SPDI annotation-negative, uniform original photo then eligible tile via inverse tile multiplicity, replacement',
       'augmentation':'paired cached low/high/fine/mask/valid H/V .5; not fresh RGB flipped encoder',
       'optimizer':'AdamW lr1e-4 wd1e-4 cosine30 min1e-6 clip5 FP32 TF32off',
       'epochs':EPOCHS,'steps':STEPS,'batch':BATCH,'seed':SEED,'planned_optimizer_updates':EPOCHS*STEPS,
       'VAL_selection':'epoch0+each epoch, per original-photo mean tile loss within real-positive/annotation-negative buckets then equal2bucket mean; earliest strict minimum. 10source heldout photos not independent field',
       'resume':'per-epoch optimizer/scheduler/RNG, journal each actual optimizer op including replay, completed never rerun',
       'photo_score':'max of9 top32valid sigmoid tile scores, count original photo once',
       'CAL':'same target normal30/23,85percent orderceil((n+1)*.85),strict>; detail_only and max(new/newcut,parent/parentcut) own jointCAL cut saved beforeTEST',
       'methods':list(METHODS),'guards':['each TESTlabel>=20','FPR<=.20','FPR<=v16+.05','FNR<=v16-.05','subtype n>=10 FNR<=v16+.05','selected epoch>0'],
       'selection':'eligible min maxerror then FNR then method order; no TEST threshold/guard changes',
       'location':'all old target primary maps/pixel thresholds unchanged; actual source masks do not prove target localization',
       'known_public_TEST_reused_for_adoption':True,'independent_field_validated':False,'physical_capture_groups_verified':False,
       'inherited_parent_past_near_exposure_undone':False,'new_source_masks_team_exhaustively_verified':False,
       'new_unit_tests_passed':2,'same_turn_v22_geometry_and_dispatch_tests_passed':4,
       'source_use':'NEU original images/abin24 added masks academic research only/no commercial/cite both, no invented SPDX; preserve raw+failed studies',
       'release_gate':'fresh packaged CPU/GPU/allaliases/C++ and original location PNG before selected candidate adoption'}
    new(PROTOCOL,p);print({'frozen':True,'planned_optimizer_updates':EPOCHS*STEPS},flush=True)


def validate(committed=False,data=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in ((START,EXPECTED),(WARM,WARM_SHA),(PARENT_EVAL,p['parent_evaluation_sha256']),
        (preflight.MANIFEST,p['stopped_v22_manifest_sha256']),(REPORTS/'industrial-neu-detail-v22-data-audit.json',p['stopped_v22_audit_sha256']),
        (NEU/'candidate-manifest.json',p['NEU_candidate_manifest_sha256']),(NEU/'FIT-review-r1.json',p['NEU_FIT_review_sha256']),
        (RUN/'protected-inputs.json',p['protected_inputs_sha256'])):assert sha(path)==h,path
    if data:
        assert sha(MANIFEST)==read(RUN/'data-ready.json')['manifest_sha256'] and read(MANIFEST)['audit']['ready']
    if committed:
        repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
        names=[*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()]
        if data:names.append('reports/industrial-neu-source-detail-v23-data-audit.json')
        for n in names:assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def prepare():
    p=validate(committed=True);assert not MANIFEST.exists()
    old=read(preflight.MANIFEST);parts=source_only_rows(old['rows'])
    fit,val=parts['fit'],parts['validation']
    fitphotos={r['photo_id'] for r in fit if r['label']};valphotos={r['photo_id'] for r in val if r['label']}
    valnegative={r['photo_id'] for r in val if not r['label']}
    by_source={c:len({r['photo_id'] for r in fit if r['label'] and r['source_folder']==c}) for c in ('NEU_SDI','NEU_SPDI')}
    ready=(len(fitphotos)>=20 and min(by_source.values())>=5 and len(valphotos)>=8 and len(valnegative)>=8
        and all(any(not r['label'] and r['source_folder']==c for r in fit) for c in by_source))
    assert not fitphotos&valphotos
    excluded=set(old['audit']['excluded_original_groups'])
    assert all(r['group_id'] not in excluded for r in fit)
    counts={phase:dict(Counter(r['bucket'] for r in rows)) for phase,rows in parts.items()}
    a={'UTC':datetime.now(timezone.utc).isoformat(),'ready':ready,'protocol_sha256':sha(PROTOCOL),
       'stopped_v22_manifest_sha256':sha(preflight.MANIFEST),'stopped_v22_duplicate_audit_sha256':p['stopped_v22_audit_sha256'],
       'counts':counts,'training_crops':len(fit),'validation_crops':len(val),'FIT_real_mask_original_photos':len(fitphotos),
       'FIT_real_by_source':by_source,'VAL_real_mask_original_photos':len(valphotos),'VAL_annotation_negative_original_photos':len(valnegative),
       'target_normal_photos_in_training_or_VAL':0,'new_source_normal_whole_photos':0,
       'inherited_retained_TEST_full_checked':5746,'inherited_reference_full_and_crops_checked':old['audit']['reference_full_and_crops_checked'],
       'source_training_heldout_near_groups_remaining':0,'new_crop_coordinate_generation':0,
       'source_only_design_not_relaxation_of_stopped_v22_normal_minimum':True,
       'TEST_masks_used_for_preparation':False,'new_features_scores_optimizer_updates':0,'physical_capture_groups_verified':False,
       'annotation_negative_is_not_whole_equipment_health':True,'new_unit_tests_passed':2}
    ids={r['photo_id'] for r in fit+val}
    new(MANIFEST,{'schema':VERSION,'photos':[r for r in old['photos'] if r['id'] in ids],'rows':fit+val,'audit':a})
    new(REPORTS/'industrial-neu-source-detail-v23-data-audit.json',a)
    new(RUN/'data-ready.json',{'manifest_sha256':sha(MANIFEST),'audit':a})
    print(a,flush=True);assert ready,'Source-only fixed data guard failed'


def load(path=START,device='cpu',initialize=False):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=DetailSurfaceMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and (not missing or initialize and all(n.startswith('detail_surface_head.') for n in missing))
    if initialize:
        warm=torch.load(WARM,map_location='cpu',weights_only=True);prefix='fine_target_head.'
        model.detail_surface_head.load_state_dict({n[len(prefix):]:v for n,v in warm['state_dict'].items() if n.startswith(prefix)},strict=True)
    return model.eval().requires_grad_(False).to(device),state


def image_batch(rows):
    images=[];masks=[];valids=[]
    for r in rows:
        assert sha(ROOT/r['image'])==r['image_sha256']
        with Image.open(ROOT/r['image']) as im:rgb=im.convert('RGB')
        truth=None
        if r['is_NEU']:
            assert sha(ROOT/r['mask'])==r['mask_sha256']
            with Image.open(ROOT/r['mask']) as im:truth=im.convert('L')
        view,mask,valid,geom=tile_view(rgb,tuple(r['box']),truth)
        assert geom==r['geometry'] and hashlib.sha256(view.tobytes()).hexdigest()==r['view_rgb_sha256']
        assert int(mask.sum())==r['positive_pixels'] and not bool(mask[~valid].any())
        images.append(normalized(view));masks.append(mask);valids.append(valid)
    return torch.stack(images),torch.stack(masks),torch.stack(valids)


def cache(phase,model):
    active='training_active' if phase=='fit' else 'validation_active'
    rows=[r for r in read(MANIFEST)['rows'] if r[active]];path=RUN/(phase+'-features.pt')
    if path.exists():
        assert sha(path)==read(RUN/(phase+'-cache.json'))['sha256']
        values=torch.load(path,map_location='cpu',weights_only=True)
        assert values['ids']==[r['id'] for r in rows] and values['manifest_sha256']==sha(MANIFEST)
        return rows,values
    values={k:[] for k in KEYS}
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH):
            images,masks,valid=image_batch(rows[start:start+BATCH])
            f=forward_with_fine(model.backbone,images.cuda())
            for k in ('low','high','fine'):values[k].append(f[k].half().cpu())
            values['masks'].append(masks);values['valid'].append(valid)
            if start%256==0:print({'cache':phase,'crops':min(start+BATCH,len(rows)),'total':len(rows)},flush=True)
    saved={k:torch.cat(v) for k,v in values.items()};saved.update(ids=[r['id'] for r in rows],manifest_sha256=sha(MANIFEST),protocol_sha256=sha(PROTOCOL))
    partial=path.with_suffix('.partial.pt')
    with partial.open('xb') as out:torch.save(saved,out)
    partial.rename(path);new(RUN/(phase+'-cache.json'),{'sha256':sha(path),'bytes':path.stat().st_size,'crops':len(rows),'manifest_sha256':sha(MANIFEST)})
    return rows,saved


def logits(head,b):
    zero=b['low'].new_zeros((len(b['low']),1,*b['high'].shape[-2:]))
    return torch.nn.functional.interpolate(head(b['low'],b['high'],zero,b['fine']),size=(512,512),mode='bilinear',align_corners=False)


def validation(head,rows,values):
    losses=[];head.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH):
            at=slice(start,start+BATCH);b={k:values[k][at].cuda() for k in KEYS}
            losses.extend(per_tile_loss(logits(head,b),b['masks'][:,None],b['valid'][:,None]).cpu().tolist())
    means=photo_bucket_means(rows,losses)
    assert set(means)=={'real_positive','annotation_negative'}
    return {'group_photo_mean_losses':means,'balanced_loss':float(np.mean(list(means.values()))),'source':'actual source family-heldout10photos with annotation-positive/negative crops'}


def train():
    p=validate(committed=True,data=True)
    assert not (RUN/'trained/training.json').exists() and shutil.disk_usage(ROOT).free>3*1024**3
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);model,parent=load(device='cuda',initialize=True);head=model.detail_surface_head
    fr,fit=cache('fit',model);vr,val=cache('validation',model);params=trainable_fine_head(head)
    assert all(n.startswith('detail_surface_head.') for n,t in model.named_parameters() if t.requires_grad)
    draws=RUN/'draws.npz'
    if not draws.exists():
        rng=np.random.default_rng(SEED);pools=[]
        for folder,label,count in [('NEU_SDI',1,2),('NEU_SPDI',1,2),('NEU_SDI',0,2),('NEU_SPDI',0,2)]:
            ids=[i for i,r in enumerate(fr) if r['is_NEU'] and r['source_folder']==folder and r['label']==label]
            pools.extend([ids]*count)
        assert len(pools)==8 and all(pools)
        sample_rows=[{**r,'group_id':r['photo_id']} for r in fr]
        indices=[[draw_group_balanced(sample_rows,ids,rng) for ids in pools] for _ in range(EPOCHS*STEPS)]
        np.savez_compressed(draws,indices=np.array(indices).reshape(EPOCHS,STEPS,BATCH),flips=rng.random((EPOCHS,STEPS,BATCH,2))<.5)
        new(RUN/'draws.json',{'sha256':sha(draws),'manifest_sha256':sha(MANIFEST)})
    assert sha(draws)==read(RUN/'draws.json')['sha256'];draws=np.load(draws)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)
    folder=RUN/'train';folder.mkdir(exist_ok=True);progress=folder/'progress.pt';journal=folder/'optimizer.jsonl';epoch0=0;actual=0;best_epoch=0
    if progress.exists():
        s=torch.load(progress,map_location='cpu',weights_only=True);assert s['protocol_sha256']==sha(PROTOCOL)
        head.load_state_dict(s['head']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);epoch0=s['epoch'];best=s['best'];best_epoch=s['best_epoch'];torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state(s['cuda_rng'])
    else:
        initial=validation(head,vr,val);best=initial['balanced_loss'];new(folder/'initial-VAL.json',initial)
        torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':0,'VAL_loss':best},folder/'best.pt')
        torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':0,'best':best,'best_epoch':0,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        print({'initial_VAL':initial},flush=True)
    if journal.exists():actual=sum(1 for _ in journal.open(encoding='utf-8'))
    for epoch in range(epoch0,EPOCHS):
        head.train();losses=[]
        for step in range(STEPS):
            at=draws['indices'][epoch,step];b={k:fit[k][at].clone().cuda() for k in KEYS}
            for i in range(BATCH):
                axes=tuple(axis for enabled,axis in zip(draws['flips'][epoch,step,i],(-1,-2)) if enabled)
                if axes:
                    for k in b:b[k][i]=b[k][i].flip(axes)
            optimizer.zero_grad(set_to_none=True);loss=per_tile_loss(logits(head,b),b['masks'][:,None],b['valid'][:,None]).mean();assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,5);optimizer.step();actual+=1;losses.append(float(loss.detach()))
            with journal.open('a',encoding='utf-8') as out:out.write(json.dumps({'actual_operation':actual,'epoch':epoch+1,'step':step+1,'loss':losses[-1]})+'\n')
        vl=validation(head,vr,val)
        if vl['balanced_loss']<best:
            best=vl['balanced_loss'];best_epoch=epoch+1;torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':best_epoch,'VAL_loss':best},folder/'best.pt')
        scheduler.step();torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        item={'epoch':epoch+1,'actual_optimizer_updates':actual,'training_loss':float(np.mean(losses)),'VAL':vl,'selected_epoch':best_epoch}
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(item)+'\n')
        print(item,flush=True)
    chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);state=deepcopy(parent)
    state['state_dict'].update({'detail_surface_head.'+n:t for n,t in chosen['head'].items()})
    state.update(detail_surface_version=VERSION,detail_surface_parent_sha256=EXPECTED,detail_surface_warm_sha256=WARM_SHA,detail_surface_protocol_sha256=sha(PROTOCOL),detail_surface_actual_optimizer_updates=actual,detail_surface_selected_epoch=best_epoch)
    (RUN/'trained').mkdir();torch.save(state,RUN/'trained/final.pt')
    new(RUN/'trained/training.json',{'actual_optimizer_updates':actual,'scheduled_unique_updates':EPOCHS*STEPS,'completed_epochs':EPOCHS,'selected_epoch':best_epoch,'selected_epoch_updates':best_epoch*STEPS,'VAL_balanced_loss':best,'FIT_crops':len(fr),'VAL_crops':len(vr),'FIT_real_original_photos':read(MANIFEST)['audit']['FIT_real_mask_original_photos'],'manifest_sha256':sha(MANIFEST)})


def photo_scores(model,rows,device='cuda'):
    result=[]
    with torch.inference_mode():
        for row in rows:
            assert sha(ROOT/row['image'])==row['image_sha256']
            with Image.open(ROOT/row['image']) as im:rgb=im.convert('RGB')
            images=[];valid=[]
            for box in tile_boxes(rgb.size):
                view,_,known,_=tile_view(rgb,box);images.append(normalized(view));valid.append(known)
            crop_scores=[]
            for start in range(0,len(images),BATCH):
                prob=model.forward_detail_view(torch.stack(images[start:start+BATCH]).to(device)).sigmoid().cpu()
                crop_scores.extend(ksdd_photo_score(prob,torch.stack(valid[start:start+BATCH])).tolist())
            assert len(crop_scores)==9
            result.append({'id':row['id'],'score':max(crop_scores),'crop_scores':crop_scores})
    return result


def calibrate():
    validate(data=True);assert not (RUN/'calibration.json').exists();model,state=load(RUN/'trained/final.pt','cuda');result={}
    for c in TARGETS:
        rows,values=old_cache(c,'cal');assert not any(r['label'] for r in rows)
        observations=photo_scores(model,rows);scores=[r['score'] for r in observations];cut,rank=normal_budget_threshold(scores,.15)
        old,oldcut=parent_target_scores(model,state,c,values);assert min(cut,oldcut)>0
        joint,jrank=normal_budget_threshold([max(n/cut,o/oldcut) for n,o in zip(scores,old)],.15)
        result[c]={'points':{'detail_only':{'photo_threshold':cut,'rank':rank},'max_detail_parent':{'photo_threshold':joint,'rank':jrank,'detail_divisor':cut,'parent_divisor':oldcut}},'normal_CAL_photos':len(rows),'observations':[{**r,'parent_score':o} for r,o in zip(observations,old)]}
    new(RUN/'calibration.json',{'protocol_sha256':sha(PROTOCOL),'thresholds_saved_before_TEST':True,'categories':result})
    print({'calibrated':True,'points':{c:result[c]['points'] for c in TARGETS}},flush=True)


def evaluate():
    validate(data=True);assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json')
    assert cal['thresholds_saved_before_TEST'] and cal['protocol_sha256']==sha(PROTOCOL)
    model,state=load(RUN/'trained/final.pt','cuda');prior={r['id']:r for r in read(PARENT_EVAL)['rows']};result={};selected={};out=[]
    trained=read(RUN/'trained/training.json')['selected_epoch']>0
    for c in TARGETS:
        rows,values=old_cache(c,'test');labels=[r['label'] for r in rows];observations=photo_scores(model,rows);scores=[r['score'] for r in observations]
        old,oldcut=parent_target_scores(model,state,c,values);oldpred=[s>oldcut for s in old]
        assert oldpred==[prior[r['id']]['candidate_detected'] for r in rows]
        base=classification_rates(labels,oldpred,.5);bg=group_rates(rows,oldpred);result[c]={'parent':base,'parent_by_defect':bg,'methods':{}}
        for method in METHODS:
            point=cal['categories'][c]['points'][method]
            score=scores if method=='detail_only' else [max(n/point['detail_divisor'],o/point['parent_divisor']) for n,o in zip(scores,old)]
            pred=[s>point['photo_threshold'] for s in score];r=classification_rates(labels,pred,.5);groups=group_rates(rows,pred)
            eligible=trained and min(r['positive_photos'],r['negative_photos'])>=20 and r['fpr']<=.20 and r['fpr']<=base['fpr']+.05 and r['fnr']<=base['fnr']-.05 and all(g['photos']<10 or g['fnr']<=bg[d]['fnr']+.05 for d,g in groups.items())
            result[c]['methods'][method]={'rates':r,'by_defect':groups,'eligible':eligible,'auroc':rank_auroc(labels,score)}
            out.extend({'id':row['id'],'category':c,'method':method,'label':row['label'],'defect':row['defect'],'score':s,'candidate_detected':q,'detail_crop_scores':obs['crop_scores']} for row,s,q,obs in zip(rows,score,pred,observations))
        eligible=[m for m in METHODS if result[c]['methods'][m]['eligible']]
        if eligible:selected[c]=min(eligible,key=lambda m:(max(result[c]['methods'][m]['rates']['fpr'],result[c]['methods'][m]['rates']['fnr']),result[c]['methods'][m]['rates']['fnr'],METHODS.index(m)))
    s={'selected':selected,'categories':result,'training':read(RUN/'trained/training.json'),'CAL_sha256':sha(RUN/'calibration.json'),'known_public_TEST_reused':True,'independent_field_validated':False,'original_photos_not_crops_counted':True,'primary_location_unchanged':True}
    new(RUN/'candidate-evaluation.json',{'rows':out,'categories':result});new(RUN/'selection.json',s);new(REPORTS/'industrial-neu-source-detail-v23-selection.json',s)
    if selected:
        state['detail_surface_methods']=selected;state['detail_surface_points']={c:cal['categories'][c]['points'][m] for c,m in selected.items()};(RUN/'selected').mkdir();torch.save(state,RUN/'selected/final.pt')
    else:new(RUN/'retained-parent.json',{'checkpoint':str(START),'sha256':EXPECTED,'reason':'No preregistered detail candidate eligible'})
    print({'selection':s},flush=True)


def verify():
    p=validate(data=True);s=read(RUN/'selection.json');t=read(RUN/'trained/training.json')
    for n,h in p['legacy_source_sha256'].items():assert sha(ROOT/n)==h,n
    unchanged=0;rehashed=0
    for n,before in read(RUN/'protected-inputs.json').items():
        path=ROOT/n;st=path.stat()
        if st.st_size==before['size_bytes'] and st.st_mtime_ns==before['mtime_ns']:unchanged+=1
        else:assert sha(path)==before['sha256'],n;rehashed+=1
    parent=torch.load(START,map_location='cpu',weights_only=True);candidate=torch.load(RUN/'trained/final.pt',map_location='cpu',weights_only=True);warm=torch.load(WARM,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,candidate['state_dict'][n]) for n,v in parent['state_dict'].items())
    assert all(candidate[k]==v for k,v in parent.items() if k!='state_dict')
    for n,v in warm['state_dict'].items():
        if n.startswith('fine_target_head.high_projection.'):assert torch.equal(v,candidate['state_dict'][n.replace('fine_target_head.','detail_surface_head.',1)])
    journal=[json.loads(line) for line in (RUN/'train/optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['actual_operation'] for r in journal]==list(range(1,len(journal)+1)) and len(journal)==t['actual_optimizer_updates']
    for phase in ('fit','validation'):assert sha(RUN/(phase+'-features.pt'))==read(RUN/(phase+'-cache.json'))['sha256']
    result={'status':'passed','parent_all_tensors_buffers_metadata_exact':True,'high_projection_exact_to_v21':True,
        'legacy_python_SHA_checked':len(p['legacy_source_sha256']),'protected_stat_unchanged':unchanged,'changed_metadata_SHA_rechecked':rehashed,'all_protected_SHA_reread':False,
        'actual_optimizer_updates':len(journal),'new_unit_tests_passed':2,'same_turn_geometry_dispatch_tests_passed':4,'selected':s['selected'],'original_primary_locations_unchanged':True,
        'CPU_GPU_CPP_release_checks_still_required_if_selected':bool(s['selected']),'active_v16_retained':True,'independent_field_validated':False}
    new(RUN/'verification.json',result);new(REPORTS/'industrial-neu-source-detail-v23-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','prepare','train','calibrate','evaluate','verify','run'));args=parser.parse_args()
    if args.mode in ('prepare','train','calibrate','evaluate','run'):
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work process: '+str(path))
        lock=RUN/'operation.lock'
        if lock.exists():new(RUN/('stale-lock-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'pid':lock.read_text(encoding='utf-8'),'live':False});lock.unlink()
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:
            if args.mode=='run':train();calibrate();evaluate();verify()
            else:globals()[args.mode]()
        except BaseException as error:
            new(RUN/(args.mode+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'pid':os.getpid(),'mode':args.mode});raise
        finally:lock.unlink()
    else:globals()[args.mode]()
