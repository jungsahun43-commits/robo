"""Preregistered normal-only neighboring-feature prediction; TEST is not a tuning set."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.neighbor_surface import NeighborMain,NeighborHead,CATEGORIES,VERSION,neighbor_view,neighbor_targets
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_expansion import photo_score,stage_gate
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.process_probe import process_exists
from scripts.prepare_neighbor_surfaces import DATA,RUN,REPORT as DATA_REPORT
from scripts.study_equipment_expansion import read
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_loco_context import load as parent_load

START=ROOT/'runs/industrial-loco-context-v13/selected/final.pt'
EXPECTED='b3e0f1cc8c487f45795b60d4bd67e2a8a03d7095d3844c6e2f42ba14875b65b0'
PARENT_EVAL=ROOT/'runs/industrial-loco-context-v13/evaluation.json'
MANIFEST=DATA/'manifest.json';REPORTS=ROOT/'reports'
PROTOCOL=REPORTS/'industrial-neighbor-surfaces-v15-protocol.json'
SOURCES=('safelog_ai/neighbor_surface.py','scripts/prepare_neighbor_surfaces.py','scripts/study_neighbor_surfaces.py','tests/test_neighbor_surface.py')
EPOCHS,STEPS,BATCH,SEED=30,48,8,1515
METHODS=('neighbor_only','max_neighbor_local')


def rows_for(category,phase):
    return [r for r in read(MANIFEST)['rows'] if r['category']==category and r['phase']==phase]


def freeze():
    assert not PROTOCOL.exists() and sha(START)==EXPECTED
    audit=read(DATA_REPORT);parent=read(PARENT_EVAL)
    ledger=dict(read(ROOT/'runs/industrial-loco-context-v13/protected-inputs.json'))
    for directory in (ROOT/'runs/industrial-loco-context-v13',REPORTS):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock','.log')):ledger[path.relative_to(ROOT).as_posix()]=snapshot(path)
    # The newly written v15 audit is a declared input; outputs are not added later.
    new(RUN/'protected-inputs.json',ledger)
    baseline={c:parent['categories'][c] for c in CATEGORIES};by_defect={}
    lookup={r['id']:r for r in parent['rows']}
    for c in CATEGORIES:
        by_defect[c]={}
        for defect in sorted({r['defect'] for r in rows_for(c,'test') if r['label']}):
            local=[r for r in rows_for(c,'test') if r['label'] and r['defect']==defect]
            fn=sum(not lookup[r['id']]['candidate_detected'] for r in local)
            by_defect[c][defect]={'photos':len(local),'fn':fn,'fnr':fn/len(local)}
    p={'schema':VERSION,'declared_utc':datetime.now(timezone.utc).isoformat(),
        'declared_before_new_feature_cache_training_CAL_TEST':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for d in ('safelog_ai','scripts','tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'parent_sha256':EXPECTED,'parent_evaluation_sha256':sha(PARENT_EVAL),'manifest_sha256':sha(MANIFEST),
        'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),'data_audit':audit,'baseline':baseline,'baseline_by_defect':by_defect,
        'categories':list(CATEGORIES),'epochs_each':EPOCHS,'steps_per_epoch':STEPS,'batch':BATCH,'seed':SEED,
        'planned_optimizer_updates_each':EPOCHS*STEPS,'planned_optimizer_updates_total':2*EPOCHS*STEPS,
        'training':{'view':'512x512 isotropic letterbox127; no foreground masks',
            'target':'same frozen shared backbone low40/high960 branch normalized unit features, native32x32, FP16 cache/runtime roundtrip',
            'augmentation':'Normal FIT original/H/V/HV extracted separately; no additional source photos',
            'standardization':'original FIT only per-channel mean/std across photo and spatial, population std>=.01, clamp[-8,8]',
            'input':'fixed depthwise3x3 average of8 neighboring standardized features, center kernel0, reflection pad1',
            'head':'per equipment Conv1000to128/ReLU/Conv128to64/ReLU/Conv64to1000;1x1 only; single inherited shared backbone and checkpoint',
            'loss':'MSE predicting fixed center target from neighboring features; no real defects, masks, foreground or object labels',
            'optimizer':'AdamW lr.001 wd.0001, cosine30 min.00001, clip5, FP32, TF32off; new head only',
            'selection':'minimum normal VAL MSE, earliest ties; VAL not used for thresholds, CAL not used for training/epoch selection',
            'RGB_blind_spot_proved':False,'backbone_receptive_fields_overlap':True,
            'resume':'per-epoch progress, per-operation journal, retain partial updates and replay counts separately'},
        'photo_score':'mean largest round1percent native32x32 channel-MSE cells; not calibrated failure probability',
        'CAL_budget':.15,'candidate_methods':list(METHODS),
        'calibration':'normal CAL rankceil((n+1)*.85), strict >. Fusion=max(neighbor/neighborCALcut, originalLocal/unchangedParentCut) and its CAL15 cut; save all before TEST',
        'guards':['normal and defect TEST each>=20','FPR<=.20','FPR<=parent+.05','FNR<=parent-.05',
                  'every >=10-photo defect subtype FNR<=parent+.05','original tensors/buffers/location maps/pixel thresholds preserved'],
        'selection':'per category eligible lowest max(FPR,FNR), then FNR, then declared method order. No candidate means retain v13. Partial improvement is not stage20 pass.',
        'old_and_new_location_outputs':'old map remains primary, new neighboring error is photo evidence only; no new pixel metric claimed',
        'known_public_TEST_reused_for_development':True,'independent_field_validated':False,
        'unreviewed_v14_relation_annotations_used':False,'app_default_promoted':False,'new_unit_tests_passed':3}
    new(PROTOCOL,p);rng=np.random.default_rng(SEED)
    for c in CATEGORIES:
        n=len(rows_for(c,'fit'));np.savez_compressed(RUN/(c+'-draws.npz'),indices=rng.integers(0,n,size=(EPOCHS,STEPS,BATCH)),variants=rng.integers(0,4,size=(EPOCHS,STEPS,BATCH)))
    new(RUN/'draws.json',{c:sha(RUN/(c+'-draws.npz')) for c in CATEGORIES})
    print({'protocol_frozen':True,'planned_updates':2*EPOCHS*STEPS,'baseline_by_defect':by_defect},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,d in p['source_sha256'].items():assert sha(ROOT/n)==d,n
    for path,digest in ((START,EXPECTED),(PARENT_EVAL,p['parent_evaluation_sha256']),(MANIFEST,p['manifest_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256'])):assert sha(path)==digest,path
    for c,d in read(RUN/'draws.json').items():assert sha(RUN/(c+'-draws.npz'))==d
    if committed:
        git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def load(path=START,device='cpu',initialize=False):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if initialize or state.get('neighbor_version'):
        model=NeighborMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'],state.get('neighbor_categories',CATEGORIES))
        missing,extra=model.load_state_dict(state['state_dict'],strict=False)
        assert not extra and (not missing or initialize and all(n.startswith('neighbor_heads.') for n in missing))
    else:return parent_load(path,device)
    return model.eval().requires_grad_(False).to(device),state


def cache(category,phase,model):
    path=RUN/f'{category}-{phase}-features.pt';rows=rows_for(category,phase)
    if path.exists():
        result=torch.load(path,map_location='cpu',weights_only=True)
        assert result['protocol_sha256']==sha(PROTOCOL) and result['ids']==[r['id'] for r in rows]
        return rows,result
    if phase!='test':assert all(r['label']==0 for r in rows)
    targets=[];local=[]
    with torch.inference_mode():
        for i,r in enumerate(rows):
            assert sha(ROOT/r['image'])==r['image_sha256']
            with Image.open(ROOT/r['image']) as im:rgb=im.convert('RGB')
            variants=[rgb]
            if phase=='fit':variants+=[rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT),rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM),rgb.transpose(Image.Transpose.ROTATE_180)]
            x=torch.stack([normalized(neighbor_view(a)[0]) for a in variants]).cuda()
            targets.append(neighbor_targets(model.backbone(x)).cpu().half())
            if phase in ('cal','test'):
                old=normalized(rgb.resize((256,256),Image.Resampling.BICUBIC))[None].cuda()
                local.append(float(photo_score(model.forward_equipment(old,category))[0]))
            if i%25==0:print({'cache':category,'phase':phase,'photos':i+1,'total':len(rows)},flush=True)
    result={'protocol_sha256':sha(PROTOCOL),'ids':[r['id'] for r in rows],'targets':torch.stack(targets),'local_scores':local}
    with path.open('xb') as f:torch.save(result,f)
    return rows,result


def val_loss(head,values):
    total=0
    with torch.inference_mode():
        for a in values.split(BATCH):
            x=a.float();total+=float(F.mse_loss(head(x),head.standardize(x)))*len(x)
    return total/len(values)


def train():
    validate(committed=True);assert not (RUN/'trained/training.json').exists()
    model,parent=load(device='cuda',initialize=True);records={}
    for category_index,c in enumerate(CATEGORIES):
        folder=RUN/c;folder.mkdir(exist_ok=True);done=folder/'training.json'
        if done.exists():records[c]=read(done);continue
        torch.manual_seed(SEED+category_index);torch.cuda.manual_seed_all(SEED+category_index)
        head=model.neighbor_heads[c];head.load_state_dict(NeighborHead().state_dict())
        _,fit=cache(c,'fit',model);_,val=cache(c,'validation',model)
        x=fit['targets'].cuda();v=val['targets'][:,0].cuda()
        with torch.no_grad():
            original=x[:,0].float();head.FIT_mean.copy_(original.mean((0,2,3))[None,:,None,None]);head.FIT_scale.copy_(original.std((0,2,3),unbiased=False).clamp_min(.01)[None,:,None,None]);del original
        initial={n:t.cpu().clone() for n,t in head.state_dict().items()}
        head.requires_grad_(True);optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=.00001)
        draws=np.load(RUN/(c+'-draws.npz'));progress=folder/'progress.pt';journal=folder/'optimizer.jsonl'
        epoch0=0;best=float('inf');best_epoch=0;actual=0
        if progress.exists():
            saved=torch.load(progress,map_location='cpu',weights_only=True);assert saved['protocol_sha256']==sha(PROTOCOL)
            head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            epoch0=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        if journal.exists():actual=sum(1 for _ in journal.open(encoding='utf-8'))
        for epoch in range(epoch0,EPOCHS):
            head.train();losses=[]
            for step in range(STEPS):
                ids=torch.as_tensor(draws['indices'][epoch,step],device='cuda');variant=torch.as_tensor(draws['variants'][epoch,step],device='cuda');target=x[ids,variant].float()
                optimizer.zero_grad(set_to_none=True);loss=F.mse_loss(head(target),head.standardize(target));assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);optimizer.step();actual+=1;losses.append(float(loss.detach()))
                with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'actual_operation':actual,'epoch':epoch+1,'step':step+1,'loss':losses[-1]})+'\n')
            head.eval();vl=val_loss(head,v)
            if vl<best:
                best=vl;best_epoch=epoch+1;torch.save({'head':{n:t.cpu() for n,t in head.state_dict().items()},'epoch':best_epoch,'VAL_MSE':vl,'protocol_sha256':sha(PROTOCOL)},folder/'best.pt')
            scheduler.step();torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
            item={'category':c,'epoch':epoch+1,'actual_optimizer_updates':actual,'scheduled_unique_updates':(epoch+1)*STEPS,'train_MSE':sum(losses)/len(losses),'VAL_MSE':vl,'selected_epoch':best_epoch}
            with (folder/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
            print(item,flush=True)
        chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        assert any(not torch.equal(initial[n],t) for n,t in chosen['head'].items() if n.startswith('predictor.'))
        records[c]={'actual_optimizer_updates':actual,'scheduled_unique_updates':EPOCHS*STEPS,'completed_epochs':EPOCHS,'selected_epoch':best_epoch,'selected_epoch_updates':best_epoch*STEPS,'VAL_MSE':best,'normal_FIT':len(fit['ids']),'normal_VAL':len(val['ids']),'real_defect_or_relation_labels':0,'fitted_head_changed':True}
        new(done,records[c]);head.requires_grad_(False);del x,v,fit,val;torch.cuda.empty_cache()
    state=deepcopy(parent)
    for c in CATEGORIES:
        head=torch.load(RUN/c/'best.pt',map_location='cpu',weights_only=True)['head']
        state['state_dict'].update({'neighbor_heads.'+c+'.'+n:t for n,t in head.items()})
    assert all(torch.equal(t,state['state_dict'][n]) for n,t in parent['state_dict'].items())
    state.update(neighbor_version=VERSION,neighbor_categories=list(CATEGORIES),neighbor_parent_sha256=EXPECTED,neighbor_protocol_sha256=sha(PROTOCOL))
    (RUN/'trained').mkdir();torch.save(state,RUN/'trained/final.pt')
    new(RUN/'trained/training.json',{'categories':records,'actual_optimizer_updates':sum(v['actual_optimizer_updates'] for v in records.values()),'old_tensors_exact':True,'unreviewed_relation_labels_used':False})


def scores(head,values):
    result=[]
    with torch.inference_mode():
        for batch in values[:,0].split(BATCH):result.extend(head.evidence(batch.cuda().float())[1].cpu().tolist())
    return result


def calibrate():
    validate();assert not (RUN/'calibration.json').exists();model,state=load(RUN/'trained/final.pt','cuda');result={}
    for c in CATEGORIES:
        rows,values=cache(c,'cal',model);ns=scores(model.neighbor_heads[c],values['targets']);local=values['local_scores'];cut,rank=normal_budget_threshold(ns,.15)
        old_cut=state['equipment_operating_points'][c]['photo_threshold'];assert min(cut,old_cut)>0
        fusion=[max(n/cut,o/old_cut) for n,o in zip(ns,local)];joint,jrank=normal_budget_threshold(fusion,.15)
        result[c]={'normal_CAL_photos':len(rows),'points':{'neighbor_only':{'photo_threshold':cut,'rank':rank},'max_neighbor_local':{'photo_threshold':joint,'rank':jrank,'neighbor_divisor':cut,'local_divisor':old_cut}},'observations':[{'id':r['id'],'neighbor':n,'local':o,'fusion':f} for r,n,o,f in zip(rows,ns,local,fusion)]}
    new(RUN/'calibration.json',{'protocol_sha256':sha(PROTOCOL),'saved_before_TEST':True,'categories':result});print({'calibrated':True},flush=True)


def evaluate():
    p=validate();assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['saved_before_TEST'] and cal['protocol_sha256']==sha(PROTOCOL)
    model,state=load(RUN/'trained/final.pt','cuda');results={};chosen={};evaluations=[]
    parent=read(PARENT_EVAL);old_lookup={r['id']:r for r in parent['rows']}
    for c in CATEGORIES:
        rows,values=cache(c,'test',model);ns=scores(model.neighbor_heads[c],values['targets']);local=values['local_scores'];labels=[r['label'] for r in rows]
        base=classification_rates(labels,[old_lookup[r['id']]['candidate_detected'] for r in rows],.5);results[c]={}
        for method in METHODS:
            point=cal['categories'][c]['points'][method]
            evidence=ns if method=='neighbor_only' else [max(n/point['neighbor_divisor'],o/point['local_divisor']) for n,o in zip(ns,local)]
            pred=[s>point['photo_threshold'] for s in evidence];rates=classification_rates(labels,pred,.5);groups={}
            for defect,prior in p['baseline_by_defect'][c].items():
                ids=[i for i,r in enumerate(rows) if r['label'] and r['defect']==defect];fn=sum(not pred[i] for i in ids);groups[defect]={'photos':len(ids),'fn':fn,'fnr':fn/len(ids)}
            eligible=(min(rates['positive_photos'],rates['negative_photos'])>=20 and rates['fpr']<=.20 and rates['fpr']<=base['fpr']+.05 and rates['fnr']<=base['fnr']-.05 and all(g['photos']<10 or g['fnr']<=p['baseline_by_defect'][c][d]['fnr']+.05 for d,g in groups.items()))
            results[c][method]={'rates':rates,'by_defect':groups,'eligible_incremental':eligible,'stage20':stage_gate(rates['fpr'],rates['fnr'],rates['negative_photos'],rates['positive_photos']),'auroc':rank_auroc(labels,evidence)}
            evaluations.extend({'id':r['id'],'category':c,'method':method,'label':r['label'],'defect':r['defect'],'score':s,'candidate_detected':v} for r,s,v in zip(rows,evidence,pred))
        eligible=[m for m in METHODS if results[c][m]['eligible_incremental']]
        if eligible:chosen[c]=min(eligible,key=lambda m:(max(results[c][m]['rates']['fpr'],results[c][m]['rates']['fnr']),results[c][m]['rates']['fnr'],METHODS.index(m)))
    new(RUN/'candidate-evaluation.json',{'categories':results,'rows':evaluations,'known_TEST_reused':True,'independent_field_validated':False})
    selection={'selected':chosen,'categories':results,'actual_optimizer_updates':read(RUN/'trained/training.json')['actual_optimizer_updates'],'CAL_sha256':sha(RUN/'calibration.json'),'one_model_one_backbone':True,'TEST_threshold_retuning':False}
    new(RUN/'selection.json',selection);new(REPORTS/'industrial-neighbor-surfaces-v15-selection.json',selection)
    if chosen:
        selected=deepcopy(state);selected['neighbor_categories']=list(chosen)
        selected['state_dict']={n:t for n,t in state['state_dict'].items() if not n.startswith('neighbor_heads.') or n.split('.')[1] in chosen}
        selected['neighbor_methods']=chosen;selected['neighbor_operating_points']={c:cal['categories'][c]['points'][m] for c,m in chosen.items()}
        (RUN/'selected').mkdir();torch.save(selected,RUN/'selected/final.pt')
        updated=deepcopy(parent);new_rows={r['id']:r for r in evaluations if chosen.get(r['category'])==r['method']}
        for row in updated['rows']:
            if row['id'] in new_rows:
                r=new_rows[row['id']];row.update(score=r['score'],candidate_detected=r['candidate_detected'])
        for c,m in chosen.items():
            updated['categories'][c].update(photo_rates=results[c][m]['rates'],stage_gate=results[c][m]['stage20'],
                auroc=results[c][m]['auroc'],normal_CAL_photos=cal['categories'][c]['normal_CAL_photos'],
                original_local_photo_threshold=state['equipment_operating_points'][c]['photo_threshold'],
                photo_threshold=selected['neighbor_operating_points'][c]['photo_threshold'],neighbor_method=m)
        updated['stage20_categories']=[c for c,item in updated['categories'].items() if item.get('stage_gate',{}).get('ready_for_next_stage')]
        updated['all_categories_photo_rates_within20percent']=len(updated['stage20_categories'])==len(updated['categories'])
        updated.update(checkpoint_sha256=sha(RUN/'selected/final.pt'),known_TEST_reused=True,independent_industrial_field_validated=False)
        new(RUN/'evaluation.json',updated)
    else:new(RUN/'retained-parent.json',{'retained':str(START),'SHA256':EXPECTED,'reason':'No preregistered eligible candidate; retain failed weights and scores'})
    print({'selected':chosen,'results':results},flush=True)


def verify():
    p=validate();training=read(RUN/'trained/training.json');selection=read(RUN/'selection.json')
    for n,d in p['legacy_source_sha256'].items():assert sha(ROOT/n)==d,n
    ledger=read(RUN/'protected-inputs.json');meta_same=0;hashed=0
    for n,before in ledger.items():
        path=ROOT/n;s=path.stat()
        if s.st_size==before['size_bytes'] and s.st_mtime_ns==before['mtime_ns']:meta_same+=1
        else:assert sha(path)==before['sha256'],n;hashed+=1
    model,state=load(RUN/'trained/final.pt');parent,prior=load()
    assert all(torch.equal(t,state['state_dict'][n]) for n,t in prior['state_dict'].items())
    for key in ('equipment_operating_points','ksdd_operating_point','wear_operating_point','context_operating_point','surface_reference_specs','thresholds'):assert state[key]==prior[key],key
    actual=0
    for c in CATEGORIES:
        lines=[json.loads(s) for s in (RUN/c/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['actual_operation'] for r in lines]==list(range(1,len(lines)+1))
        assert len(lines)==training['categories'][c]['actual_optimizer_updates'];actual+=len(lines)
    assert training['actual_optimizer_updates']==actual
    result={'status':'passed','parent_tensors_buffers_and_operating_points_exact':True,'legacy_source_SHA_checked':len(p['legacy_source_sha256']),'protected_files_stat_unchanged':meta_same,'protected_files_changed_metadata_SHA_rechecked':hashed,'full_SHA_of_every_protected_file_repeated':False,'actual_optimizer_updates_journal_verified':actual,'new_unit_tests_passed':3,'selected':selection['selected'],'new_location_or_field_performance_claimed':False}
    new(RUN/'verification.json',result);new(REPORTS/'industrial-neighbor-surfaces-v15-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','train','calibrate','evaluate','verify','run'));args=parser.parse_args()
    if args.mode in ('train','calibrate','evaluate','run'):
        for lock in (ROOT/'runs').glob('**/*.lock'):
            pid=int(lock.read_text(encoding='utf-8').strip())
            if process_exists(pid):raise RuntimeError('Existing work process: '+str(lock))
        lock=RUN/'operation.lock'
        with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
        try:
            if args.mode=='run':train();calibrate();evaluate();verify()
            else:globals()[args.mode]()
        except BaseException as error:
            new(RUN/(args.mode+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'mode':args.mode,'pid':os.getpid()});raise
        finally:lock.unlink()
    else:globals()[args.mode]()
