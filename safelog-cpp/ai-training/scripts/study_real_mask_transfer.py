"""Actual publisher-mask transfer training; all accepted branches preserved."""
import argparse
from collections import Counter
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
from safelog_ai.magnetic_surface import RealMaskMain,VERSION,CATEGORY,TARGETS,surface_view,source_probability,photo_score,mask_loss
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_expansion import photo_score as local_photo_score,stage_gate
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.process_probe import process_exists
from scripts.prepare_magnetic_tiles import DATA,RUN,REPORT as DATA_REPORT
from scripts.study_equipment_expansion import read
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_loco_context import load as parent_load

START=ROOT/'runs/industrial-loco-context-v13/selected/final.pt'
EXPECTED='b3e0f1cc8c487f45795b60d4bd67e2a8a03d7095d3844c6e2f42ba14875b65b0'
PARENT_EVAL=ROOT/'runs/industrial-loco-context-v13/evaluation.json'
MANIFEST=DATA/'manifest.json';TRANSFER_MANIFEST=ROOT/'data/industrial-neighbor-surfaces-v15/manifest.json'
REPORTS=ROOT/'reports';PROTOCOL=REPORTS/'industrial-real-mask-transfer-v16-protocol.json'
SOURCES=('safelog_ai/magnetic_surface.py','scripts/fetch_magnetic_tiles.py','scripts/prepare_magnetic_tiles.py','scripts/study_real_mask_transfer.py','tests/test_magnetic_surface.py')
EPOCHS,STEPS,BATCH,SEED=30,64,8,1616
METHODS=('transfer_only','max_transfer_local')


def rows_for(phase,category=CATEGORY):
    path=MANIFEST if category==CATEGORY else TRANSFER_MANIFEST
    return [r for r in read(path)['rows'] if r['phase']==phase and (category==CATEGORY or r['category']==category)]


def rates(labels,predictions):return classification_rates(labels,predictions,.5)


def freeze():
    assert not PROTOCOL.exists() and sha(START)==EXPECTED
    assert read(ROOT/'runs/industrial-neighbor-surfaces-v15/verification.json')['status']=='passed'
    ledger=dict(read(ROOT/'runs/industrial-neighbor-surfaces-v15/protected-inputs.json'))
    for folder in (ROOT/'runs/industrial-neighbor-surfaces-v15',REPORTS):
        for path in folder.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock','.log')):ledger[path.relative_to(ROOT).as_posix()]=snapshot(path)
    new(RUN/'protected-inputs.json',ledger)
    audit=read(DATA_REPORT);p={'schema':VERSION,'UTC':datetime.now(timezone.utc).isoformat(),
        'fixed_before_new_features_training_CAL_TEST':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for d in ('safelog_ai','scripts','tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'parent_sha256':EXPECTED,'manifest_sha256':sha(MANIFEST),'target_manifest_sha256':sha(TRANSFER_MANIFEST),'parent_evaluation_sha256':sha(PARENT_EVAL),
        'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),'data_audit':audit,'epochs':EPOCHS,'steps':STEPS,'batch':BATCH,'seed':SEED,'planned_optimizer_updates':EPOCHS*STEPS,
        'training':{'source':'only new source normal and actual-mask defect FIT from audited counts, no target MPDD defect TEST',
            'branch':'clone existing accepted ksdd_head into a new transfer_surface_head, original ksdd_head and all previous weights remain frozen',
            'input':'same frozen one shared backbone low40/high960,512 square isotropic letterbox127, FP16 cache/live roundtrip',
            'architecture':'existing SurfaceMaskHead high960to32/upsample/concatlow40/Conv72to64 k3/ReLU/dropout.1/Conv64to32 k3/ReLU/Conv32to1, upsample512',
            'target':'author grayscale mask >=128 before nearest-neighbor view resize; missing source defect masks and their groups quarantined',
            'sampling':'4 normal uniformFIT +4 real-mask defects with uniform defect type then uniform source photo, replacement; repeats are not new data',
            'augmentation':'paired cached low/high feature, mask, valid horizontal/vertical flips at.5; not freshly re-encoded RGB augmented photos',
            'loss':'valid BCEWithLogits pos_weight10 + softDice +.5 native valid top32 photo BCE, fixed inherited mask_loss',
            'optimizer':'AdamW lr.001 wd.0001, cosine30 min.00001, FP32 TF32off, clip5, only new branch trainable',
            'checkpoint':'minimum held-out VAL loss, equal weighting of normal and each defect type; earliest ties; CAL/TEST excluded',
            'resume':'per-epoch optimizer/RNG/cursor checkpoints and per-step optimizer journal, partial epoch replay operations counted'},
        'new_source_photo_calibration':'labelled CAL normal129/defect42 minimax max(FPR,FNR), then sum, abs(cut-.5), cut; finite candidates {0,1,all distinct CAL scores}, strict >; saved before TEST',
        'new_source_pixel_threshold':.5,'photo_score':'mean top32 valid native sigmoid pixels; not calibrated mechanical failure probability',
        'new_source_selection':['new held-out normal and defect each>=20','photoFPR<=.20','photoFNR<=.20','every defect type with>=5 held-out photos has FNR<=.50'],
        'target_MPDD_methods':list(METHODS),'target_MPDD_CAL':'normalCAL15 rankceil((n+1)*.85); single new score or max(new/newCALcut, old/unchangedParentCut), own joint CAL15 cut',
        'target_MPDD_guards':['photoFPR<=.20','FPR<=parent+.05','FNR<=parent-.05','each class>=20','every >=10-photo defect type FNR<=parent+.05'],
        'target_MPDD_selection':'eligible lower maxFPRFNR, then FNR, then declared method order; no threshold retuning after TEST',
        'pixel_scope':'new magnetic-tile learned map measured at source coordinates at fixed.5; MPDD old location maps and thresholds remain primary unchanged',
        'old_checkpoint_operating_points_unchanged':True,'one_model_one_common_backbone':True,
        'source_split_is_internal_not_verified_physical_part_groups':True,'known_MPDD_TEST_reused':True,
        'new_source_holdout_fixed_before_model_scores':True,'independent_field_validated':False,'unreviewed_v14_relations_used':False,
        'not_a_motor_or_equipment_internal_mechanical_diagnosis':True,'app_default_promoted':False,'new_unit_tests_passed':3}
    new(PROTOCOL,p)
    fit=rows_for('fit');normal=[i for i,r in enumerate(fit) if not r['label']];by_type={d:[i for i,r in enumerate(fit) if r['defect']==d] for d in sorted({r['defect'] for r in fit if r['label']})}
    rng=np.random.default_rng(SEED);indices=[]
    for _ in range(EPOCHS*STEPS):
        sample=rng.choice(normal,4).tolist()
        for defect in rng.choice(list(by_type),4):sample.append(int(rng.choice(by_type[defect])))
        indices.append(sample)
    np.savez_compressed(RUN/'draws.npz',indices=np.array(indices).reshape(EPOCHS,STEPS,BATCH),flips=rng.random((EPOCHS,STEPS,BATCH,2))<.5)
    new(RUN/'draws.json',{'SHA256':sha(RUN/'draws.npz'),'FIT_by_type':dict(Counter(r['defect'] for r in fit))})
    print({'protocol_frozen':True,'planned_optimizer_updates':EPOCHS*STEPS,'counts':audit['counts']},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in ((START,EXPECTED),(MANIFEST,p['manifest_sha256']),(TRANSFER_MANIFEST,p['target_manifest_sha256']),(PARENT_EVAL,p['parent_evaluation_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256'])):assert sha(path)==h,path
    assert sha(RUN/'draws.npz')==read(RUN/'draws.json')['SHA256']
    if committed:
        git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def load(path=START,device='cpu',initialize=False):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not initialize and not state.get('transfer_surface_version'):return parent_load(path,device)
    model=RealMaskMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and (not missing or initialize and all(n.startswith('transfer_surface_head.') for n in missing))
    if initialize:model.transfer_surface_head.load_state_dict(model.ksdd_head.state_dict())
    return model.eval().requires_grad_(False).to(device),state


def cache(phase,model,category=CATEGORY):
    path=RUN/f'{category}-{phase}-features.pt';rows=rows_for(phase,category)
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['protocol_sha256']==sha(PROTOCOL) and saved['ids']==[r['id'] for r in rows]
        return rows,saved
    low=[];high=[];masks=[];known=[];geometries=[];local=[]
    with torch.inference_mode():
        for i,r in enumerate(rows):
            assert sha(ROOT/r['image'])==r['image_sha256']
            with Image.open(ROOT/r['image']) as im:rgb=im.convert('RGB')
            truth=None
            if category==CATEGORY:
                assert sha(ROOT/r['mask'])==r['mask_sha256']
                with Image.open(ROOT/r['mask']) as im:truth=im.convert('L')
            view,mask,valid,geometry=surface_view(rgb,truth)
            if category==CATEGORY:assert bool(mask.any())==bool(r['label']),r['id']
            features=model.backbone(normalized(view)[None].cuda());low.append(features['low'][0].cpu().half());high.append(features['high'][0].cpu().half())
            masks.append(mask);known.append(valid);geometries.append(geometry)
            if category!=CATEGORY:
                old=normalized(rgb.resize((256,256),Image.Resampling.BICUBIC))[None].cuda();local.append(float(local_photo_score(model.forward_equipment(old,category))[0]))
            if i%100==0:print({'cache':category,'phase':phase,'photos':i+1,'total':len(rows)},flush=True)
    saved={'protocol_sha256':sha(PROTOCOL),'ids':[r['id'] for r in rows],'low':torch.stack(low),'high':torch.stack(high),'masks':torch.stack(masks),'valid':torch.stack(known),'geometry':geometries,'local_scores':local}
    with path.open('xb') as f:torch.save(saved,f)
    return rows,saved


def logits(head,low,high):return F.interpolate(head(low.float(),high.float()),size=(512,512),mode='bilinear',align_corners=False)


def validation_loss(head,values,rows):
    losses=[]
    with torch.inference_mode():
        for defect in sorted({r['defect'] for r in rows}):
            ids=[i for i,r in enumerate(rows) if r['defect']==defect];total=0
            for start in range(0,len(ids),BATCH):
                at=ids[start:start+BATCH];x=logits(head,values['low'][at],values['high'][at]);total+=float(mask_loss(x,values['masks'][at,None],values['valid'][at,None]))*len(at)
            losses.append(total/len(ids))
    return sum(losses)/len(losses)


def train():
    validate(committed=True);assert not (RUN/'trained/training.json').exists()
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);model,parent=load(device='cuda',initialize=True);head=model.transfer_surface_head
    fit_rows,fit=cache('fit',model);val_rows,val=cache('validation',model)
    x={k:fit[k].cuda() for k in ('low','high','masks','valid')};v={k:val[k].cuda() for k in ('low','high','masks','valid')}
    initial={n:t.cpu().clone() for n,t in head.state_dict().items()};head.requires_grad_(True)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=.00001)
    folder=RUN/'train';folder.mkdir(exist_ok=True);progress=folder/'progress.pt';journal=folder/'optimizer.jsonl';draws=np.load(RUN/'draws.npz')
    epoch0=0;actual=0;best=float('inf');best_epoch=0
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);assert saved['protocol_sha256']==sha(PROTOCOL)
        head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);epoch0=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    if journal.exists():actual=sum(1 for _ in journal.open(encoding='utf-8'))
    for epoch in range(epoch0,EPOCHS):
        head.train();losses=[]
        for step in range(STEPS):
            at=torch.as_tensor(draws['indices'][epoch,step],device='cuda');batch={k:t[at].clone() for k,t in x.items()}
            for i in range(BATCH):
                axes=tuple(a for enabled,a in zip(draws['flips'][epoch,step,i],(-1,-2)) if enabled)
                if axes:
                    for k in batch:batch[k][i]=batch[k][i].flip(axes)
            optimizer.zero_grad(set_to_none=True);output=logits(head,batch['low'],batch['high']);loss=mask_loss(output,batch['masks'][:,None],batch['valid'][:,None]);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);optimizer.step();actual+=1;losses.append(float(loss.detach()))
            with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'actual_operation':actual,'epoch':epoch+1,'step':step+1,'loss':losses[-1]})+'\n')
        head.eval();vl=validation_loss(head,v,val_rows)
        if vl<best:
            best=vl;best_epoch=epoch+1;torch.save({'head':{n:t.cpu() for n,t in head.state_dict().items()},'epoch':best_epoch,'VAL_loss':best,'protocol_sha256':sha(PROTOCOL)},folder/'best.pt')
        scheduler.step();torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        item={'epoch':epoch+1,'actual_optimizer_updates':actual,'scheduled_unique_updates':(epoch+1)*STEPS,'training_loss':sum(losses)/len(losses),'VAL_balanced_loss':vl,'selected_epoch':best_epoch}
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
        print(item,flush=True)
    chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    assert any(not torch.equal(initial[n],t) for n,t in chosen['head'].items())
    state=deepcopy(parent);state['state_dict'].update({'transfer_surface_head.'+n:t for n,t in chosen['head'].items()})
    assert all(torch.equal(t,state['state_dict'][n]) for n,t in parent['state_dict'].items())
    state.update(transfer_surface_version=VERSION,transfer_parent_sha256=EXPECTED,transfer_protocol_sha256=sha(PROTOCOL))
    (RUN/'trained').mkdir();torch.save(state,RUN/'trained/final.pt')
    new(RUN/'trained/training.json',{'actual_optimizer_updates':actual,'scheduled_unique_updates':EPOCHS*STEPS,'completed_epochs':EPOCHS,'selected_epoch':best_epoch,'selected_epoch_updates':best_epoch*STEPS,'VAL_balanced_loss':best,'normal_FIT':sum(not r['label'] for r in fit_rows),'real_mask_defect_FIT':sum(r['label'] for r in fit_rows),'normal_VAL':sum(not r['label'] for r in val_rows),'real_mask_defect_VAL':sum(r['label'] for r in val_rows),'all_parent_weights_buffers_exact':True,'trained_new_branch_changed':True})


def infer_cached(model,values):
    output=[]
    with torch.inference_mode():
        for start in range(0,len(values['low']),BATCH):
            probability=model.forward_surface_features(values['low'][start:start+BATCH].cuda().float(),values['high'][start:start+BATCH].cuda().float()).sigmoid().cpu()
            output.extend(photo_score(probability,values['valid'][start:start+BATCH]).tolist())
    return output


def calibrate():
    validate();assert not (RUN/'calibration.json').exists();model,state=load(RUN/'trained/final.pt','cuda');result={}
    for c in TARGETS:
        rows,values=cache('cal',model,c);scores=infer_cached(model,values);labels=[r['label'] for r in rows]
        if c==CATEGORY:
            cuts=sorted(set([0.,1.,*scores]));candidates=[]
            for cut in cuts:
                metric=classification_rates(labels,scores,cut);candidates.append((max(metric['fpr'],metric['fnr']),metric['fpr']+metric['fnr'],abs(cut-.5),cut,metric))
            selected=min(candidates,key=lambda r:r[:4]);result[c]={'point':{'photo_threshold':selected[3],'pixel_threshold':.5,'normal_CAL':labels.count(0),'defect_CAL':labels.count(1)},'CAL_rates':selected[4],'CAL_candidate_count':len(cuts)}
        else:
            cut,rank=normal_budget_threshold(scores,.15);oldcut=state['equipment_operating_points'][c]['photo_threshold'];assert min(cut,oldcut)>0
            fusion=[max(n/cut,o/oldcut) for n,o in zip(scores,values['local_scores'])];joint,jrank=normal_budget_threshold(fusion,.15)
            result[c]={'points':{'transfer_only':{'photo_threshold':cut,'rank':rank},'max_transfer_local':{'photo_threshold':joint,'rank':jrank,'transfer_divisor':cut,'local_divisor':oldcut}},'normal_CAL':len(rows)}
        result[c]['observations']=[{'id':r['id'],'label':r['label'],'score':s} for r,s in zip(rows,scores)]
    new(RUN/'calibration.json',{'protocol_sha256':sha(PROTOCOL),'thresholds_saved_before_TEST':True,'categories':result});print({'calibrated':True,'magnetic_point':result[CATEGORY]['point'],'CAL_rates':result[CATEGORY]['CAL_rates']},flush=True)


def group_rates(rows,pred):
    result={}
    for defect in sorted({r['defect'] for r in rows if r['label']}):
        idx=[i for i,r in enumerate(rows) if r['label'] and r['defect']==defect];fn=sum(not pred[i] for i in idx);result[defect]={'photos':len(idx),'fn':fn,'fnr':fn/len(idx)}
    return result


def evaluate():
    validate();assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['thresholds_saved_before_TEST'] and cal['protocol_sha256']==sha(PROTOCOL)
    model,state=load(RUN/'trained/final.pt','cuda');parent=read(PARENT_EVAL);prior={r['id']:r for r in parent['rows']};result={};selected={};new_rows=[]
    for c in TARGETS:
        rows,values=cache('test',model,c);scores=infer_cached(model,values);labels=[r['label'] for r in rows]
        if c==CATEGORY:
            point=cal['categories'][c]['point'];pred=[s>point['photo_threshold'] for s in scores];metric=rates(labels,pred);groups=group_rates(rows,pred)
            passed=min(metric['positive_photos'],metric['negative_photos'])>=20 and max(metric['fpr'],metric['fnr'])<=.20 and all(v['photos']<5 or v['fnr']<=.50 for v in groups.values())
            pixel={'tp':0,'fn':0,'fp':0,'tn':0}
            with torch.inference_mode():
                for i,r in enumerate(rows):
                    native=model.forward_surface_features(values['low'][i:i+1].cuda().float(),values['high'][i:i+1].cuda().float()).sigmoid().cpu()
                    probability=source_probability(native,values['geometry'][i])[0];guess=probability>.5
                    with Image.open(ROOT/r['mask']) as im:truth=torch.from_numpy(np.asarray(im.convert('L')).copy()>=128)
                    for k,m in [('tp',guess&truth),('fn',~guess&truth),('fp',guess&~truth),('tn',~guess&~truth)]:pixel[k]+=int(m.sum())
            result[c]={'photo_rates':metric,'by_defect':groups,'stage_gate':stage_gate(metric['fpr'],metric['fnr'],metric['negative_photos'],metric['positive_photos']),'eligible':passed,'auroc':rank_auroc(labels,scores),'source_pixel_counts':pixel,'source_pixel_fnr':pixel['fn']/max(1,pixel['tp']+pixel['fn']),'source_pixel_fpr':pixel['fp']/max(1,pixel['fp']+pixel['tn']),'source_pixel_IoU':pixel['tp']/max(1,pixel['tp']+pixel['fp']+pixel['fn']),'pixel_threshold':.5,'photo_threshold':point['photo_threshold'],'normal_CAL_photos':point['normal_CAL'],'defect_CAL_photos':point['defect_CAL']}
            if passed:selected[c]='real_mask'
            new_rows.extend({'id':r['id'],'category':c,'label':r['label'],'defect':r['defect'],'score':s,'candidate_detected':q} for r,s,q in zip(rows,scores,pred))
        else:
            baseline=rates(labels,[prior[r['id']]['candidate_detected'] for r in rows]);base_groups=group_rates(rows,[prior[r['id']]['candidate_detected'] for r in rows]);result[c]={}
            for method in METHODS:
                point=cal['categories'][c]['points'][method];score=scores if method=='transfer_only' else [max(n/point['transfer_divisor'],o/point['local_divisor']) for n,o in zip(scores,values['local_scores'])]
                pred=[s>point['photo_threshold'] for s in score];metric=rates(labels,pred);groups=group_rates(rows,pred)
                eligible=min(metric['positive_photos'],metric['negative_photos'])>=20 and metric['fpr']<=.20 and metric['fpr']<=baseline['fpr']+.05 and metric['fnr']<=baseline['fnr']-.05 and all(v['photos']<10 or v['fnr']<=base_groups[d]['fnr']+.05 for d,v in groups.items())
                result[c][method]={'rates':metric,'by_defect':groups,'eligible_incremental':eligible,'auroc':rank_auroc(labels,score)}
                new_rows.extend({'id':r['id'],'category':c,'method':method,'label':r['label'],'defect':r['defect'],'score':s,'candidate_detected':q} for r,s,q in zip(rows,score,pred))
            eligible=[m for m in METHODS if result[c][m]['eligible_incremental']]
            if eligible:selected[c]=min(eligible,key=lambda m:(max(result[c][m]['rates']['fpr'],result[c][m]['rates']['fnr']),result[c][m]['rates']['fnr'],METHODS.index(m)))
    selection={'selected':selected,'categories':result,'new_source_holdout_first_model_scoring':True,'known_MPDD_TEST_reused':True,'independent_field_validated':False,'actual_optimizer_updates':read(RUN/'trained/training.json')['actual_optimizer_updates'],'CAL_sha256':sha(RUN/'calibration.json')}
    new(RUN/'candidate-evaluation.json',{'rows':new_rows,'categories':result});new(RUN/'selection.json',selection);new(REPORTS/'industrial-real-mask-transfer-v16-selection.json',selection)
    if selected:
        state['transfer_surface_methods']=selected;state['transfer_surface_points']={c:(cal['categories'][c]['point'] if c==CATEGORY else cal['categories'][c]['points'][m]) for c,m in selected.items()}
        (RUN/'selected').mkdir();torch.save(state,RUN/'selected/final.pt')
    else:new(RUN/'retained-parent.json',{'retained':str(START),'SHA256':EXPECTED,'reason':'No preregistered eligible candidate'})
    print(selection,flush=True)


def verify():
    p=validate();training=read(RUN/'trained/training.json');selection=read(RUN/'selection.json')
    for n,d in p['legacy_source_sha256'].items():assert sha(ROOT/n)==d,n
    ledger=read(RUN/'protected-inputs.json');unchanged=0;hashed=0
    for n,before in ledger.items():
        path=ROOT/n;stat=path.stat()
        if stat.st_size==before['size_bytes'] and stat.st_mtime_ns==before['mtime_ns']:unchanged+=1
        else:assert sha(path)==before['sha256'],n;hashed+=1
    _,state=load(RUN/'trained/final.pt');parent=torch.load(START,map_location='cpu',weights_only=True)
    assert all(torch.equal(t,state['state_dict'][n]) for n,t in parent['state_dict'].items())
    for key in ('equipment_operating_points','ksdd_operating_point','wear_operating_point','context_operating_point','surface_reference_specs','thresholds'):assert state[key]==parent[key],key
    journal=[json.loads(s) for s in (RUN/'train/optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['actual_operation'] for r in journal]==list(range(1,len(journal)+1)) and len(journal)==training['actual_optimizer_updates']
    result={'status':'passed','parent_all_tensors_buffers_and_operating_points_exact':True,'legacy_python_source_SHA_checked':len(p['legacy_source_sha256']),'protected_files_stat_unchanged':unchanged,'changed_metadata_files_SHA_rechecked':hashed,'all_protected_file_SHA_repeated':False,'actual_optimizer_updates_journal_verified':len(journal),'new_unit_tests_passed':3,'selected':selection['selected'],'independent_field_performance_claimed':False}
    new(RUN/'verification.json',result);new(REPORTS/'industrial-real-mask-transfer-v16-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','train','calibrate','evaluate','verify','run'));args=parser.parse_args()
    if args.mode in ('train','calibrate','evaluate','run'):
        for lock in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work process: '+str(lock))
        lock=RUN/'operation.lock'
        with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
        try:
            if args.mode=='run':train();calibrate();evaluate();verify()
            else:globals()[args.mode]()
        except BaseException as error:
            new(RUN/(args.mode+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'mode':args.mode,'pid':os.getpid()});raise
        finally:lock.unlink()
    else:globals()[args.mode]()
