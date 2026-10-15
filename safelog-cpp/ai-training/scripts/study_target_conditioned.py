"""One fixed target-domain synthetic/residual-conditioned experiment."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.conditioned_surface import VERSION,CATEGORIES,ConditionedSurfaceMain,warm_start,frozen_residual
from safelog_ai.target_synthetic_r1 import prepare_background,synthesize_prepared,source_residual
from safelog_ai.mixed_surface import freeze_projection
from safelog_ai.commutator_surface import fine_mask_loss
from safelog_ai.magnetic_surface import surface_view
from safelog_ai.ksdd_supervised import ksdd_photo_score
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_mixed_real_mask import get_cache as old_cache,parent_target_scores,draw_group_balanced
from scripts.study_real_mask_transfer import group_rates

RUN=ROOT/'runs/industrial-target-conditioned-v19';DATA=ROOT/'data/industrial-target-synthetic-v19-r1'
REPORTS=ROOT/'reports';PROTOCOL=REPORTS/'industrial-target-conditioned-v19-protocol.json'
NORMAL=ROOT/'data/industrial-neighbor-surfaces-v15/manifest.json';SHAPES=ROOT/'data/industrial-ksdd1-v18/manifest-r1.json'
START=ROOT/'runs/industrial-magnetic-tiles-v16/released/final.pt';PARENT_EVAL=START.parent.parent/'evaluation.json'
EXPECTED='5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e'
WARM=ROOT/'runs/industrial-ksdd1-v18/trained/final.pt';WARM_SHA='b2616342b511d143b3bf5f06eedbc03bd3cba650c98573b47d5d384160ec641b'
SOURCES=('safelog_ai/target_synthetic_r1.py','safelog_ai/conditioned_surface.py','scripts/study_target_conditioned.py','tests/test_conditioned_surface.py')
METHODS=('conditioned_only','max_conditioned_parent');EPOCHS,STEPS,BATCH,SEED=30,64,8,1919


def rates(labels,predictions):return classification_rates(labels,predictions,.5)
def source_rows(path,phase):return [r for r in read(path)['rows'] if r['phase']==phase]


def freeze():
    assert not RUN.exists() and not DATA.exists() and not PROTOCOL.exists()
    assert sha(START)==EXPECTED and sha(WARM)==WARM_SHA
    assert read(ROOT/'runs/industrial-ksdd1-v18/final-audit.json')['status']=='completed_v18_source_improved_limited_sample_targets_rejected_v16_retained'
    assert shutil.disk_usage(ROOT).free>8*1024**3
    RUN.mkdir();DATA.mkdir()
    ledger=dict(read(ROOT/'runs/industrial-ksdd1-v18/protected-inputs.json'))
    for folder in (ROOT/'runs/industrial-ksdd1-v18',ROOT/'data/industrial-target-synthetic-v19',REPORTS):
        for path in folder.rglob('*'):
            name=path.relative_to(ROOT).as_posix()
            if path.is_file() and not path.name.endswith(('.lock','.log')) and name not in ledger:ledger[name]=snapshot(path)
    new(RUN/'protected-inputs.json',ledger)
    target_hashes={}
    for c in CATEGORIES:
        for phase in ('cal','test'):
            path=ROOT/'runs/industrial-magnetic-tiles-v16'/f'{c}-{phase}-features.pt';target_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    p={'schema':VERSION,'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_r1_data_generation_features_training_CAL_TEST_scores':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'legacy_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for d in ('safelog_ai','scripts','tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
       'parent_sha256':EXPECTED,'warm_start_sha256':WARM_SHA,'normal_manifest_sha256':sha(NORMAL),'shape_manifest_sha256':sha(SHAPES),'parent_evaluation_sha256':sha(PARENT_EVAL),
       'target_cache_sha256':target_hashes,'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),
       'source_normal_counts':{'fit':{'mpdd_bracket_black':70,'mpdd_tubes':61},'validation':{'mpdd_bracket_black':18,'mpdd_tubes':15}},
       'real_shape_assets':{'fit':21,'validation':5},'new_natural_target_fault_GT':0,
       'synthesis':'same-phase target normal and KS1 source-mask assets only;4slots per normal,8attempts per slot,each256placement attempts; independent fixed SHA seed per base/slot/attempt',
       'shape_selection':'uniform source appearance_object_group then uniform real mask asset within group; inpaint residual is estimate not captured clean GT',
       'r1_generator':'extent32..96,shortdim>=5,angles0/45/90/135,amplitude24..64,original colour foreground eroded3x3,98percent shape overlap,actual RGB edit>=32pixels and mean absolute RGBchange>=12',
       'failure_handling':'failed synthesis excluded, never converted into normal; original normal view retained as its own publisher-normal sample independent of synthesis success',
       'minimum_data_guard':'per target FIT normal>=50 and synthetic>=2*normal, VAL normal>=10 and synthetic>=2*normal, synthetic covers>=ceil(.6*normal) distinct background groups; otherwise stop before gradients',
       'split':'existing normal FIT70/61 VAL18/15 and KS1 shape FIT21 VAL5 disjoint; do not use CAL/TEST or near-linked assets for synthesis/learning',
       'view':'512square isotropic RGB bicubic letterbox127; nearest procedural support, valid only original source rectangle; not an estimated foreground restriction of inference or loss',
       'features':'same frozen common backbone low40/high960 FP16 cache/live roundtrip; frozen equipment_head on same normalized raw low/high yields category-conditioned cosine residual clipped0..2 then FP16',
       'head':'high960to32/upsample + low40 + residual1 =73 channels; conv73to64/relu/dropout.1/conv64to32/relu/conv32to1, bilinear512',
       'initialization':'copy v18 commutator head into new target_conditioned_head; first72 input weights identical,new residual channel zero; freeze high projection and all parent tensors; train new fusion only',
       'not_DRAEM_reproduction':True,'one_candidate_one_model_one_common_backbone':True,
       'sampling':'each step2normal+2synthetic per target=8,total131unique FIT normal backgrounds at most; inverse base-group multiplicity inside category/label,replacement,not new photos',
       'feature_augmentation':'paired cached low/high/residual/mask/valid H/V flips at.5 per photo, not fresh RGB flip encoder passes',
       'loss':'inherited v18 valid BCE + softDice +.5 top32valid photoBCE; procedural positive weight clamp(sqrt(FIT_valid_negative/FIT_positive),10,100) fixed from generated FIT only',
       'optimizer':'AdamW lr1e-4 wd1e-4 cosine30 minimum1e-6 clip5 FP32 TF32off, only new fusion',
       'epochs':EPOCHS,'steps':STEPS,'batch':BATCH,'seed':SEED,'planned_optimizer_updates':EPOCHS*STEPS,
       'VAL_selection':'epoch0 and each completed epoch; equal four category/normal-or-synthetic group per-photo loss means, earliest minimum; natural defect VAL absent',
       'resume':'per-epoch optimizer/scheduler/RNG and per-step actual operation journal; partial replay counts as actual work, completed study never rerun',
       'CAL':'normalCAL30/23 unchanged,rankceil((n+1)*.85); conditioned_only or max(new/newCALcut,current_parent/current_parent_cut), own joint normalCALcut; frozen before TEST',
       'methods':list(METHODS),'target_guards':['each TEST label>=20','FPR<=.20','FPR<=current_v16+.05','FNR<=current_v16-.05','each defect subtype>=10 FNR<=current_v16+.05','selected epoch>0'],
       'selection':'eligible minimum maxFPRFNR then FNR then declared method order; no TEST threshold retuning',
       'location_policy':'original target location PNG and pixel threshold unchanged; synthetic-trained map not promoted as natural defect localization',
       'known_public_TEST_reused_for_adoption':True,'inherited_parent_past_near_training_exposure_undone':False,'physical_capture_groups_verified':False,
       'independent_field_validated':False,'synthetic_physical_fault_realism_validated':False,'unreviewed_v14_relations_used':False,'app_default_promoted':False,
       'new_unit_tests_passed':3,'source_use_conditions':'existing MPDD and KS1 research permissions, KS1 CC BY-NC-SA4.0 and DOI10.1007/s10845-019-01476-x retained'}
    new(PROTOCOL,p);print({'frozen':True,'planned_optimizer_updates':EPOCHS*STEPS,'planned_requested_synthetics':{'fit':524,'validation':132}},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in [(START,EXPECTED),(WARM,WARM_SHA),(NORMAL,p['normal_manifest_sha256']),(SHAPES,p['shape_manifest_sha256']),(PARENT_EVAL,p['parent_evaluation_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256'])]:assert sha(path)==h,path
    if committed:
        repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def save_image(im,path):
    buf=io.BytesIO();im.save(buf,format='PNG');raw=buf.getvalue();digest=hashlib.sha256(raw).hexdigest()
    if path.exists():assert sha(path)==digest,path
    else:
        with path.open('xb') as f:f.write(raw)
    return path.relative_to(ROOT).as_posix(),digest


def prepare():
    p=validate(committed=True)
    if (RUN/'data-ready.json').exists():
        assert sha(DATA/'manifest.json')==read(RUN/'data-ready.json')['manifest_sha256'];return
    rows=[];failures=[];shape_failures=[];counts={}
    for phase in ('fit','validation'):
        normals=[r for r in source_rows(NORMAL,phase) if r['category'] in CATEGORIES];shapes=[r for r in source_rows(SHAPES,phase) if r['label']]
        assert len(shapes)==p['real_shape_assets'][phase]
        assets={};groups={}
        for r in shapes:
            assert sha(ROOT/r['image'])==r['image_sha256'] and sha(ROOT/r['mask'])==r['mask_sha256']
            with Image.open(ROOT/r['image']) as im:rgb=im.convert('RGB')
            with Image.open(ROOT/r['mask']) as im:mask=im.convert('L')
            try:assets[r['id']]=source_residual(rgb,mask)
            except ValueError as error:shape_failures.append({'phase':phase,'source_id':r['id'],'reason':str(error)});continue
            groups.setdefault(r['appearance_object_group'],[]).append(r)
        assert len(assets)>=2
        for ni,nr in enumerate(normals):
            assert nr['label']==0 and nr['publisher_split']=='train' and not nr['masks'];assert sha(ROOT/nr['image'])==nr['image_sha256']
            with Image.open(ROOT/nr['image']) as im:rgb=im.convert('RGB')
            view,_,valid,geometry=surface_view(rgb)
            key=hashlib.sha256(nr['id'].encode()).hexdigest()[:16];base_id=nr['category']+'/'+phase+'/'+key
            image,digest=save_image(view,DATA/(key+'-normal.png'));mask,mdigest=save_image(Image.new('L',(512,512),0),DATA/(key+'-normal-mask.png'))
            common={'category':nr['category'],'phase':phase,'base_normal_id':nr['id'],'group_id':nr['appearance_group_proxy'],'original_normal_image':nr['image'],'original_normal_sha256':nr['image_sha256'],'base_view_image':image,'geometry':geometry}
            rows.append({**common,'id':base_id+'/normal','label':0,'kind':'publisher_normal_surface_view','image':image,'image_sha256':digest,'mask':mask,'mask_sha256':mdigest,'mask_pixels':0,'new_natural_fault_label':False})
            try:
                background=prepare_background(rgb);assert np.array_equal(background[0],np.asarray(view))
            except ValueError as error:
                failures.extend({'phase':phase,'base_id':nr['id'],'slot':v,'reason':'background:'+str(error),'attempts':0} for v in range(4));continue
            for slot in range(4):
                success=None
                for attempt in range(8):
                    seed=int(hashlib.sha256((f'target19-r1:{phase}:{nr["id"]}:{slot}:{attempt}').encode()).hexdigest()[:8],16)
                    rng=np.random.default_rng(seed);group=sorted(groups)[int(rng.integers(len(groups)))];pool=groups[group];sr=pool[int(rng.integers(len(pool)))]
                    try:augmented,label,meta=synthesize_prepared(background,assets[sr['id']],seed)
                    except ValueError as error:
                        failures.append({'phase':phase,'base_id':nr['id'],'slot':slot,'attempt':attempt+1,'source_id':sr['id'],'reason':str(error)});continue
                    success=(augmented,label,meta,sr,attempt+1);break
                if success is None:continue
                augmented,label,meta,sr,attempt=success
                image,digest=save_image(augmented,DATA/(key+f'-synthetic{slot}.png'));mask,mdigest=save_image(label,DATA/(key+f'-synthetic{slot}-mask.png'))
                rows.append({**common,'id':base_id+f'/synthetic{slot}','label':1,'kind':'procedural_visual_mark_not_natural_defect','image':image,'image_sha256':digest,'mask':mask,'mask_sha256':mdigest,
                    'mask_pixels':meta['changed_pixels'],'shape_source_id':sr['id'],'shape_source_image':sr['image'],'shape_source_image_sha256':sr['image_sha256'],'shape_source_mask_sha256':sr['mask_sha256'],'shape_group_id':sr['appearance_object_group'],
                    'source_phase':phase,'slot':slot,'attempts_used':attempt,'procedure':meta,'new_natural_fault_label':False})
            if ni%25==0:print({'prepared_phase':phase,'base_photos':ni+1,'total':len(normals)},flush=True)
        counts[phase]={}
        for c in CATEGORIES:
            local=[r for r in rows if r['phase']==phase and r['category']==c];normal_count=sum(not r['label'] for r in local);positive=[r for r in local if r['label']]
            assert normal_count==p['source_normal_counts'][phase][c]
            enough=normal_count>=(50 if phase=='fit' else 10) and len(positive)>=2*normal_count and len({r['group_id'] for r in positive})>=math.ceil(.6*normal_count)
            counts[phase][c]={'normal':normal_count,'synthetic':len(positive),'positive_background_groups':len({r['group_id'] for r in positive}),'ready':enough}
    for field in ('group_id','shape_group_id'):
        a={r[field] for r in rows if r['phase']=='fit' and field in r};b={r[field] for r in rows if r['phase']=='validation' and field in r};assert not a&b,field
    audit={'UTC':datetime.now(timezone.utc).isoformat(),'counts':counts,'ready':all(v['ready'] for d in counts.values() for v in d.values()),'synthetic_failure_attempts':len(failures),'source_asset_failures':shape_failures,
        'new_natural_target_fault_labels':0,'CAL_TEST_images_or_masks_used':False,'source_and_target_FIT_VAL_groups_disjoint':True,'input_foreground_is_part_or_defect_GT':False,'physical_synthetic_fault_realism_verified':False,'protocol_sha256':sha(PROTOCOL)}
    if not (DATA/'manifest.json').exists():new(DATA/'manifest.json',{'schema':VERSION,'protocol_sha256':sha(PROTOCOL),'rows':rows,'audit':audit})
    if not (DATA/'generation-failures.json').exists():new(DATA/'generation-failures.json',{'attempt_failures':failures,'source_asset_failures':shape_failures})
    if not (REPORTS/'industrial-target-conditioned-v19-data-audit.json').exists():new(REPORTS/'industrial-target-conditioned-v19-data-audit.json',audit)
    assert audit['ready'],'Insufficient generated examples under fixed constraints; stop before gradients'
    new(RUN/'data-ready.json',{'manifest_sha256':sha(DATA/'manifest.json'),'protocol_sha256':sha(PROTOCOL),'audit':audit});print(audit,flush=True)


def load(path=START,device='cpu',initialize=False):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=ConditionedSurfaceMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and (not missing or initialize and all(n.startswith('target_conditioned_head.') for n in missing))
    if initialize:
        warm=torch.load(WARM,map_location='cpu',weights_only=True);prefix='commutator_surface_head.'
        warm_start(model.target_conditioned_head,{n[len(prefix):]:t for n,t in warm['state_dict'].items() if n.startswith(prefix)})
    return model.eval().requires_grad_(False).to(device),state


def generated_rows(phase):
    assert sha(DATA/'manifest.json')==read(RUN/'data-ready.json')['manifest_sha256']
    a=read(DATA/'manifest.json');assert a['protocol_sha256']==sha(PROTOCOL)
    return [r for r in a['rows'] if r['phase']==phase]


def cache(phase,model):
    rows=generated_rows(phase);path=RUN/(phase+'-features.pt')
    if path.exists():
        assert sha(path)==read(RUN/(phase+'-cache-receipt.json'))['sha256']
        a=torch.load(path,map_location='cpu',weights_only=True);assert a['ids']==[r['id'] for r in rows] and a['protocol_sha256']==sha(PROTOCOL);return rows,a
    lows=[];highs=[];residuals=[];masks=[];valids=[]
    with torch.inference_mode():
        for start in range(0,len(rows),8):
            batch=rows[start:start+8];images=[]
            for r in batch:
                assert sha(ROOT/r['image'])==r['image_sha256'] and sha(ROOT/r['mask'])==r['mask_sha256']
                with Image.open(ROOT/r['image']) as im:rgb=im.convert('RGB')
                with Image.open(ROOT/r['mask']) as im:mask=np.array(im.convert('L'))>0
                assert bool(mask.any())==bool(r['label'])
                if r['label']:
                    with Image.open(ROOT/r['base_view_image']) as im:base=np.array(im.convert('RGB'))
                    assert np.array_equal(np.any(np.array(rgb)!=base,axis=2),mask)
                g=r['geometry'];x,y=g['offset'];w,h=g['resized_size'];valid=torch.zeros(512,512,dtype=torch.bool);valid[y:y+h,x:x+w]=True
                assert not bool(torch.from_numpy(mask)[~valid].any())
                images.append(normalized(rgb));masks.append(torch.from_numpy(mask.astype('uint8')));valids.append(valid)
            f=model.backbone(torch.stack(images).cuda());low,high=f['low'].half(),f['high'].half();residual=frozen_residual(model,low,high,[r['category'] for r in batch])
            lows.append(low.cpu());highs.append(high.cpu());residuals.append(residual.cpu())
            if start%128==0:print({'cache_phase':phase,'photos':min(start+8,len(rows)),'total':len(rows)},flush=True)
    a={'protocol_sha256':sha(PROTOCOL),'ids':[r['id'] for r in rows],'low':torch.cat(lows),'high':torch.cat(highs),'residual':torch.cat(residuals),'masks':torch.stack(masks),'valid':torch.stack(valids)}
    partial=path.with_suffix('.partial.pt')
    with partial.open('xb') as out:torch.save(a,out)
    partial.rename(path);new(RUN/(phase+'-cache-receipt.json'),{'sha256':sha(path),'bytes':path.stat().st_size,'photos':len(rows),'feature_and_residual_storage':'FP16','protocol_sha256':sha(PROTOCOL)})
    return rows,a


def head_logits(head,batch):return torch.nn.functional.interpolate(head(batch['low'],batch['high'],batch['residual']),size=batch['masks'].shape[-2:],mode='bilinear',align_corners=False)


def validation(head,v,rows,weight):
    groups={};head.eval()
    with torch.inference_mode():
        for c in CATEGORIES:
            for label in (0,1):
                ids=[i for i,r in enumerate(rows) if r['category']==c and r['label']==label];total=0
                for start in range(0,len(ids),BATCH):
                    at=ids[start:start+BATCH];b={k:v[k][at].cuda() for k in ('low','high','residual','masks','valid')};out=head_logits(head,b)
                    total+=float(fine_mask_loss(out,b['masks'][:,None],b['valid'][:,None],weight))*len(at)
                groups[c+':'+str(label)]=total/len(ids)
    return {'class_losses':groups,'balanced_loss':sum(groups.values())/len(groups),'positive_scope':'synthetic edits, not natural defects'}


def train():
    p=validate(committed=True);assert not (RUN/'trained/training.json').exists();assert read(RUN/'data-ready.json')['audit']['ready']
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);model,parent=load(device='cuda',initialize=True);head=model.target_conditioned_head
    fr,fit=cache('fit',model);vr,val=cache('validation',model);params=freeze_projection(head)
    positives=int(fit['masks'].sum());valid=int(fit['valid'].sum());weight=min(100.,max(10.,math.sqrt((valid-positives)/positives)))
    if not (RUN/'FIT-loss-weight.json').exists():new(RUN/'FIT-loss-weight.json',{'positive_pixels':positives,'valid_pixels':valid,'positive_weight':weight,'source':'generated FIT only'})
    draws=RUN/'draws.npz'
    if not draws.exists():
        rng=np.random.default_rng(SEED);indices=[]
        for _ in range(EPOCHS*STEPS):
            at=[]
            for c in CATEGORIES:
                for label in (0,0,1,1):at.append(draw_group_balanced(fr,[i for i,r in enumerate(fr) if r['category']==c and r['label']==label],rng))
            indices.append(at)
        np.savez_compressed(draws,indices=np.array(indices).reshape(EPOCHS,STEPS,BATCH),flips=rng.random((EPOCHS,STEPS,BATCH,2))<.5)
        new(RUN/'draws.json',{'sha256':sha(draws),'manifest_sha256':sha(DATA/'manifest.json')})
    assert sha(draws)==read(RUN/'draws.json')['sha256'];draws=np.load(draws)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)
    folder=RUN/'train';folder.mkdir(exist_ok=True);progress=folder/'progress.pt';journal=folder/'optimizer.jsonl';epoch0=0;actual=0;best=float('inf');best_epoch=0
    if progress.exists():
        s=torch.load(progress,map_location='cpu',weights_only=True);assert s['protocol_sha256']==sha(PROTOCOL)
        head.load_state_dict(s['head']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);epoch0=s['epoch'];best=s['best'];best_epoch=s['best_epoch'];torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state(s['cuda_rng'])
    else:
        initial=validation(head,val,vr,weight);best=initial['balanced_loss'];new(folder/'initial-VAL.json',initial)
        torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':0,'VAL_loss':best,'protocol_sha256':sha(PROTOCOL)},folder/'best.pt')
        torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':0,'best':best,'best_epoch':0,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        print({'initial_VAL':initial},flush=True)
    if journal.exists():actual=sum(1 for _ in journal.open(encoding='utf-8'))
    for epoch in range(epoch0,EPOCHS):
        head.train();losses=[]
        for step in range(STEPS):
            at=draws['indices'][epoch,step];b={k:fit[k][at].clone().cuda() for k in ('low','high','residual','masks','valid')}
            for i in range(BATCH):
                axes=tuple(a for enabled,a in zip(draws['flips'][epoch,step,i],(-1,-2)) if enabled)
                if axes:
                    for k in b:b[k][i]=b[k][i].flip(axes)
            optimizer.zero_grad(set_to_none=True);out=head_logits(head,b);loss=fine_mask_loss(out,b['masks'][:,None],b['valid'][:,None],weight);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,5);optimizer.step();actual+=1;losses.append(float(loss.detach()))
            with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'actual_operation':actual,'epoch':epoch+1,'step':step+1,'loss':losses[-1]})+'\n')
        vl=validation(head,val,vr,weight)
        if vl['balanced_loss']<best:
            best=vl['balanced_loss'];best_epoch=epoch+1;torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':best_epoch,'VAL_loss':best,'protocol_sha256':sha(PROTOCOL)},folder/'best.pt')
        scheduler.step();torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        item={'epoch':epoch+1,'actual_optimizer_updates':actual,'scheduled_unique_updates':(epoch+1)*STEPS,'training_loss':sum(losses)/len(losses),'VAL':vl,'selected_epoch':best_epoch}
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
        print(item,flush=True)
    chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);state=deepcopy(parent)
    state['state_dict'].update({'target_conditioned_head.'+n:t for n,t in chosen['head'].items()});assert all(torch.equal(t,state['state_dict'][n]) for n,t in parent['state_dict'].items())
    state.update(target_conditioned_version=VERSION,target_conditioned_parent_sha256=EXPECTED,target_conditioned_warm_start_sha256=WARM_SHA,target_conditioned_protocol_sha256=sha(PROTOCOL),target_conditioned_actual_optimizer_updates=actual,target_conditioned_selected_epoch=best_epoch)
    (RUN/'trained').mkdir();torch.save(state,RUN/'trained/final.pt')
    new(RUN/'trained/training.json',{'actual_optimizer_updates':actual,'scheduled_unique_updates':EPOCHS*STEPS,'completed_epochs':EPOCHS,'selected_epoch':best_epoch,'selected_epoch_updates':best_epoch*STEPS,'VAL_balanced_loss':best,'positive_weight':weight,'FIT_photos_including_procedural_variants':len(fr),'VAL_photos_including_procedural_variants':len(vr),'new_natural_target_fault_GT':0,'all_parent_tensors_exact':True,'data_manifest_sha256':sha(DATA/'manifest.json')})


def get_target(c,phase,p):
    path=ROOT/'runs/industrial-magnetic-tiles-v16'/f'{c}-{phase}-features.pt';assert sha(path)==p['target_cache_sha256'][path.relative_to(ROOT).as_posix()]
    return old_cache(c,phase)


def scores_cached(model,v,c):
    scores=[]
    with torch.inference_mode():
        for start in range(0,len(v['low']),BATCH):
            at=slice(start,start+BATCH);low,high=v['low'][at].cuda(),v['high'][at].cuda();residual=frozen_residual(model,low,high,[c]*len(low))
            prob=model.forward_conditioned_features(low,high,residual).sigmoid().cpu();scores.extend(ksdd_photo_score(prob,v['valid'][at]).tolist())
    return scores


def calibrate():
    p=validate();assert not (RUN/'calibration.json').exists();model,state=load(RUN/'trained/final.pt','cuda');result={}
    for c in CATEGORIES:
        rows,v=get_target(c,'cal',p);assert not any(r['label'] for r in rows)
        score=scores_cached(model,v,c);cut,rank=normal_budget_threshold(score,.15);old,oldcut=parent_target_scores(model,state,c,v);assert min(cut,oldcut)>0
        fusion=[max(n/cut,o/oldcut) for n,o in zip(score,old)];joint,jrank=normal_budget_threshold(fusion,.15)
        result[c]={'points':{'conditioned_only':{'photo_threshold':cut,'rank':rank},'max_conditioned_parent':{'photo_threshold':joint,'rank':jrank,'conditioned_divisor':cut,'parent_divisor':oldcut}},'normal_CAL':len(rows),'observations':[{'id':r['id'],'score':s,'parent_score':o} for r,s,o in zip(rows,score,old)]}
    new(RUN/'calibration.json',{'protocol_sha256':sha(PROTOCOL),'thresholds_saved_before_TEST':True,'categories':result});print({'calibrated':True,'points':{c:result[c]['points'] for c in CATEGORIES}},flush=True)


def evaluate():
    p=validate();assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['thresholds_saved_before_TEST'] and cal['protocol_sha256']==sha(PROTOCOL)
    model,state=load(RUN/'trained/final.pt','cuda');prior={r['id']:r for r in read(PARENT_EVAL)['rows']};result={};selected={};all_rows=[];trained=read(RUN/'trained/training.json')['selected_epoch']>0
    for c in CATEGORIES:
        rows,v=get_target(c,'test',p);labels=[r['label'] for r in rows];scores=scores_cached(model,v,c);old,oldcut=parent_target_scores(model,state,c,v);oldpred=[s>oldcut for s in old]
        assert oldpred==[prior[r['id']]['candidate_detected'] for r in rows],'Parent baseline drift'
        base=rates(labels,oldpred);bg=group_rates(rows,oldpred);result[c]={'parent':base,'parent_by_defect':bg,'methods':{}}
        for method in METHODS:
            point=cal['categories'][c]['points'][method];score=scores if method=='conditioned_only' else [max(n/point['conditioned_divisor'],o/point['parent_divisor']) for n,o in zip(scores,old)]
            pred=[s>point['photo_threshold'] for s in score];r=rates(labels,pred);groups=group_rates(rows,pred)
            eligible=trained and min(r['positive_photos'],r['negative_photos'])>=20 and r['fpr']<=.20 and r['fpr']<=base['fpr']+.05 and r['fnr']<=base['fnr']-.05 and all(g['photos']<10 or g['fnr']<=bg[d]['fnr']+.05 for d,g in groups.items())
            result[c]['methods'][method]={'rates':r,'by_defect':groups,'eligible':eligible,'auroc':rank_auroc(labels,score)}
            all_rows.extend({'id':row['id'],'category':c,'method':method,'label':row['label'],'defect':row['defect'],'score':s,'candidate_detected':q} for row,s,q in zip(rows,score,pred))
        eligible=[m for m in METHODS if result[c]['methods'][m]['eligible']]
        if eligible:selected[c]=min(eligible,key=lambda m:(max(result[c]['methods'][m]['rates']['fpr'],result[c]['methods'][m]['rates']['fnr']),result[c]['methods'][m]['rates']['fnr'],METHODS.index(m)))
    selection={'selected':selected,'categories':result,'training':read(RUN/'trained/training.json'),'CAL_sha256':sha(RUN/'calibration.json'),'known_public_TEST_reused':True,'independent_field_validated':False,'new_natural_target_fault_GT':0,'primary_location_unchanged':True}
    new(RUN/'candidate-evaluation.json',{'rows':all_rows,'categories':result});new(RUN/'selection.json',selection);new(REPORTS/'industrial-target-conditioned-v19-selection.json',selection)
    if selected:
        state['target_conditioned_methods']=selected;state['target_conditioned_points']={c:cal['categories'][c]['points'][m] for c,m in selected.items()};(RUN/'selected').mkdir();torch.save(state,RUN/'selected/final.pt')
    else:new(RUN/'retained-parent.json',{'checkpoint':str(START),'sha256':EXPECTED,'reason':'No preregistered target candidate eligible'})
    print({'selection':selection},flush=True)


def verify():
    p=validate();s=read(RUN/'selection.json');t=read(RUN/'trained/training.json')
    for n,h in p['legacy_source_sha256'].items():assert sha(ROOT/n)==h,n
    unchanged=0;rehashed=0
    for n,before in read(RUN/'protected-inputs.json').items():
        path=ROOT/n;st=path.stat()
        if st.st_size==before['size_bytes'] and st.st_mtime_ns==before['mtime_ns']:unchanged+=1
        else:assert sha(path)==before['sha256'],n;rehashed+=1
    parent=torch.load(START,map_location='cpu',weights_only=True);candidate=torch.load(RUN/'trained/final.pt',map_location='cpu',weights_only=True);warm=torch.load(WARM,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,candidate['state_dict'][n]) for n,v in parent['state_dict'].items()) and all(candidate[k]==v for k,v in parent.items() if k!='state_dict')
    for n,v in warm['state_dict'].items():
        if n.startswith('commutator_surface_head.high_projection.'):assert torch.equal(v,candidate['state_dict'][n.replace('commutator_surface_head.','target_conditioned_head.',1)])
    journal=[json.loads(line) for line in (RUN/'train/optimizer.jsonl').read_text(encoding='utf-8').splitlines()];assert [r['actual_operation'] for r in journal]==list(range(1,len(journal)+1)) and len(journal)==t['actual_optimizer_updates']
    for phase in ('fit','validation'):assert sha(RUN/(phase+'-features.pt'))==read(RUN/(phase+'-cache-receipt.json'))['sha256']
    result={'status':'passed','parent_all_tensors_buffers_metadata_exact':True,'projection_exact_to_v18_warm_start':True,'legacy_python_SHA_checked':len(p['legacy_source_sha256']),'protected_stat_unchanged':unchanged,'changed_metadata_SHA_rechecked':rehashed,'all_protected_SHA_reread':False,'actual_optimizer_updates':len(journal),'new_unit_tests_passed':3,'selected':s['selected'],'new_natural_target_fault_GT':0,'synthetic_physical_realism_verified':False,'new_CPU_CPP_release_required_if_selected':bool(s['selected']),'active_parent_still_retained':True,'independent_field_validated':False}
    new(RUN/'verification.json',result);new(REPORTS/'industrial-target-conditioned-v19-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','prepare','train','calibrate','evaluate','verify','run'));args=parser.parse_args()
    if args.mode in ('prepare','train','calibrate','evaluate','run'):
        for lock in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work process: '+str(lock))
        lock=RUN/'operation.lock'
        if lock.exists():new(RUN/('stale-lock-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'pid':lock.read_text(encoding='utf-8'),'live':False});lock.unlink()
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:
            if args.mode=='run':prepare();train();calibrate();evaluate();verify()
            else:globals()[args.mode]()
        except BaseException as error:
            new(RUN/(args.mode+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'mode':args.mode,'pid':os.getpid()});raise
        finally:lock.unlink()
    else:globals()[args.mode]()
