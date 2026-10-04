"""Fixed-budget paired stride4 detail architecture study; source gold unchanged."""
import os
import argparse
from pathlib import Path
import random
import sys
import time
import hashlib
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision.transforms import ColorJitter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCH_HOME',str(ROOT/'.config/torch'))
from safelog_ai.spatial_classifier import SpatialClassifier, spatial_loss, ARCH
from safelog_ai.presence_classifier import image_transform,MEAN,STD
from scripts.train_facility_target import read,save,sha,split_supplemental,dacl_items,FacilityPhotos,predict,masked_focal,DOMAINS,TARGETS
from scripts.facility_error_target import operating_point
from scripts.train_facility_presence import average_precision
from scripts.facility_hard_sampling import hard_multipliers
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier,AUX_CLASSES,ARCH as AUX_ARCH
from scripts.facility_spatial_manifest import validate_replacement
from scripts.facility_photo_supplement import validate_photo_supplement
from scripts.facility_target_ranking import target_ranking_loss
from safelog_ai.detail_classifier import DetailClassifier, load_auxiliary_initializer, ARCH as DETAIL_ARCH

SPATIAL_DOMAINS={**DOMAINS,'s2ds':3}

class SpatialPhotos(Dataset):
    def __init__(self,items,auxiliary=None,full_count=None,domain_map=None):self.items=items;self.auxiliary=auxiliary;self.full_count=full_count;self.domain_map=domain_map or SPATIAL_DOMAINS;self.transform=image_transform(640);self.jitter=ColorJitter(.1,.1,.1,.01)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        item=self.items[i]
        with Image.open(ROOT/item['image']) as handle:image=handle.convert('RGB')
        with np.load(ROOT/item['pixel_target']) as data:mask=data['mask'].copy();pixel_known=data['known'].copy()
        if random.random()<.5:image=ImageOps.mirror(image);mask=mask[:,:,::-1].copy()
        targets=torch.tensor(item['targets'],dtype=torch.float32)
        result=(self.transform(self.jitter(image)),targets.clamp_min(0),(targets>=0).float(),self.domain_map[item['domain']],torch.from_numpy(mask).float(),torch.from_numpy(pixel_known).float())
        if self.auxiliary is not None:
            aux=torch.tensor(self.auxiliary.get(item['image'],[-1]*len(AUX_CLASSES)),dtype=torch.float32)
            result+=aux.clamp_min(0),(aux>=0).float()
        if self.full_count is not None:result+=(int(i>=self.full_count),i)
        return result


class MiningPhotos(Dataset):
    """Deterministic scoring of verified TRAIN records, with no pixel loading."""
    def __init__(self,items):self.items=items;self.transform=image_transform(640)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        item=self.items[i]
        with Image.open(ROOT/item['image']) as image:inputs=self.transform(image.convert('RGB'))
        targets=torch.tensor(item['targets'],dtype=torch.float32)
        return inputs,targets.clamp_min(0),(targets>=0).float()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--seed',type=int,default=54)
    parser.add_argument('--model-variant',choices=('control','detail'),required=True)
    parser.add_argument('--epochs',type=int,default=8);parser.add_argument('--patience',type=int,default=8)
    parser.add_argument('--initial',type=Path);parser.add_argument('--supplement-spatial',type=Path)
    parser.add_argument('--spatial-manifest',type=Path,help='Audited TRAIN small-region crop replacement; core full rows and splits unchanged')
    parser.add_argument('--photo-supplement',type=Path,help='Bounded verified building photo labels; source boxes/folders never become pixel masks')
    parser.add_argument('--draws-per-epoch',type=int,help='Fixed sampling budget for a matched supplement/control comparison')
    parser.add_argument('--backbone-lr',type=float,default=.00008);parser.add_argument('--head-lr',type=float,default=.0005)
    parser.add_argument('--hard-mining-strength',type=float,default=0.,help='Bounded TRAIN-only residual sampling from frozen initializer (0 disables)')
    parser.add_argument('--auxiliary-manifest',type=Path)
    parser.add_argument('--auxiliary-weight',type=float,default=.5)
    parser.add_argument('--target-ranking-weight',type=float,default=0.,help='TRAIN original full-photo within-source known positive/negative ranking loss (0 disables)')
    parser.add_argument('--study-protocol',type=Path,required=True,help='Immutable predeclared architecture pair')
    args=parser.parse_args()
    if args.photo_supplement and args.supplement_spatial:raise ValueError('Compare one supplemental source intervention at a time')
    if args.draws_per_epoch is not None and args.draws_per_epoch<1:raise ValueError('Sampling budget must be positive')
    name,seed,epochs,patience=args.name,args.seed,args.epochs,args.patience
    if Path(name).name!=name or min(epochs,patience)<1:raise ValueError('Invalid spatial run options')
    if not np.isfinite(args.hard_mining_strength) or not 0<=args.hard_mining_strength<=4 or (args.hard_mining_strength and not args.initial):
        raise ValueError('Hard mining needs an initializer and finite strength in [0,4]')
    if not np.isfinite(args.auxiliary_weight) or args.auxiliary_weight<=0:raise ValueError('Auxiliary loss weight must be finite and positive')
    if not np.isfinite(args.target_ranking_weight) or not 0<=args.target_ranking_weight<=1:
        raise ValueError('Ranking loss weight must be finite and in [0,1]')
    if args.target_ranking_weight and (not args.initial or args.photo_supplement or args.supplement_spatial or args.spatial_manifest or args.hard_mining_strength):
        raise ValueError('Compare ranking from an initializer using unchanged core supervision and sampling only')
    protocol=None
    if args.study_protocol:
        args.study_protocol=args.study_protocol.resolve();protocol=read(args.study_protocol)
        if protocol.get('schema')!='facility_detail_architecture_protocol_v1' or name not in (protocol.get('control'),protocol.get('treatment')):
            raise ValueError('Protocol schema or declared paired run changed')
        for key in ('seed','requested_epochs','patience','draws_per_epoch','backbone_lr','head_lr','auxiliary_weight'):
            actual=epochs if key=='requested_epochs' else getattr(args,key)
            if actual!=protocol[key]:raise ValueError(f'Protocol fixed argument differs: {key}')
        expected_weight=0.
        if args.model_variant!=('control' if name==protocol['control'] else 'detail'):
            raise ValueError('Architecture variant differs from declared paired run')
        if args.target_ranking_weight!=expected_weight or not args.initial or sha(args.initial)!=protocol['initial_weights_sha256']:
            raise ValueError('Protocol ranking intervention or initializer changed')
        if not args.auxiliary_manifest or any((args.photo_supplement,args.supplement_spatial,args.spatial_manifest,args.hard_mining_strength)):
            raise ValueError('Protocol requires unchanged core and original auxiliary supervision')
    output=ROOT/'runs'/name
    if (output/'best.pt').exists():raise ValueError('Preserve existing experiment')
    output.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    core_path=ROOT/'data/facility-spatial-training/train.json';manifest_path=core_path
    manifest=read(core_path)
    if args.spatial_manifest:
        manifest_path=args.spatial_manifest.resolve()
        manifest=validate_replacement(manifest,read(manifest_path),sha(core_path))
    classes=manifest['classes'];items=manifest['items'];full_count=manifest['full_count']
    if manifest['split']!='train':raise ValueError('Spatial supervision must be training only')
    split=split_supplemental();save(output/'SPLIT.json',split)
    original,_=dacl_items('train',640)
    code_train=read(ROOT/'data/codebrim-training/train.json');code_val=read(ROOT/'data/codebrim-training/val.json');code_test=read(ROOT/'data/codebrim-training/test.json')
    for expected,data in zip(('train','val','test'),(code_train,code_val,code_test)):
        if data['split']!=expected or data['classes']!=classes:raise ValueError('CODEBRIM mismatch')
    groups=[{i['group_id'] for i in data['items']} for data in (code_train,code_val,code_test)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):raise ValueError('CODEBRIM group leakage')
    base=original+[{**i,'domain':'damsegment'} for i in split['train']]+code_train['items']
    if [i['image'] for i in base]!=[i['image'] for i in items[:full_count]]:raise ValueError('Spatial manifest training split/order changed')
    allowed={i['image'] for i in base}
    if any(i['parent_image'] not in allowed or i['parent_split']!='train' for i in items[full_count:]):raise ValueError('Detail parent leakage')
    supplemental=None;domain_map=DOMAINS;proportions=(.7,.1,.2)
    pixel_counts={c:dict(v) for c,v in manifest['audit']['per_label_pixel_cells'].items()}
    core_full_count=full_count;core_items=list(items);photo_supplement=None
    if args.supplement_spatial:
        args.supplement_spatial=args.supplement_spatial.resolve();supplemental=read(args.supplement_spatial)
        if supplemental['split']!='train' or supplemental['classes']!=classes or supplemental['audit']['status']!='prepared':
            raise ValueError('Spatial supplement class/split/audit mismatch')
        if sha(ROOT/'reports/facility-target-s2ds-crop-audit.json')!=supplemental['audit']['crop_audit_sha256']:
            raise ValueError('Supplement local-overlap audit changed')
        extras=supplemental['items']
        if not extras or any(i['domain']!='s2ds' or i['source_split']!='train' for i in extras):
            raise ValueError('Only screened S2DS author TRAIN may supplement')
        if {i['image'] for i in extras}&allowed:raise ValueError('Supplement repeats core image paths')
        items=items[:full_count]+extras+items[full_count:];full_count+=len(extras)
        domain_map=SPATIAL_DOMAINS;proportions=(.6,.1,.2,.1)
        for c in classes:
            for k in ('positive','negative'):pixel_counts[c][k]+=supplemental['audit']['per_label_pixel_cells'][c][k]
    if args.photo_supplement:
        args.photo_supplement=args.photo_supplement.resolve();photo_supplement=read(args.photo_supplement)
        extras=validate_photo_supplement(photo_supplement,classes,ROOT,allowed)
        items=items[:full_count]+extras+items[full_count:];full_count+=len(extras)
        domain_map={**DOMAINS,extras[0]['domain']:3};proportions=(.63,.09,.18,.10)
    labels=torch.tensor([i['targets'] for i in items]);domains=torch.tensor([domain_map[i['domain']] for i in items])
    auxiliary=None;auxiliary_manifest=None;auxiliary_weights=None;architecture=ARCH
    if args.auxiliary_manifest:
        args.auxiliary_manifest=args.auxiliary_manifest.resolve();auxiliary_manifest=read(args.auxiliary_manifest)
        aux_items=auxiliary_manifest['items']
        if auxiliary_manifest['split']!='train' or auxiliary_manifest['classes']!=list(AUX_CLASSES) or auxiliary_manifest['audit']['status']!='prepared':
            raise ValueError('Auxiliary manifest classes/split/audit changed')
        if len(aux_items)!=len(original) or {i['image'] for i in aux_items}!={i['image'] for i in original}:
            raise ValueError('Auxiliary labels must cover exactly original DACL TRAIN full photos')
        if any(i['split']!='train' or i['domain']!='dacl' or sha(ROOT/i['annotation'])!=i['annotation_sha256'] for i in aux_items):
            raise ValueError('Auxiliary source split or annotation changed')
        auxiliary={i['image']:i['targets'] for i in aux_items};aux_labels=torch.tensor(list(auxiliary.values()))
        if aux_labels.shape!=(len(original),len(AUX_CLASSES)) or not ((aux_labels==0)|(aux_labels==1)).all():
            raise ValueError('Auxiliary targets must be original known photo tags')
        pos=(aux_labels==1).sum(0);neg=(aux_labels==0).sum(0)
        auxiliary_weights=(neg/pos.clamp_min(1)).clamp(.2,6).float().to('cuda');architecture=AUX_ARCH
    if sha(core_path)!=protocol['core_spatial_manifest_sha256'] or sha(args.auxiliary_manifest)!=protocol['auxiliary_manifest_sha256']:
        raise ValueError('Declared original TRAIN or native auxiliary supervision changed')
    architecture=AUX_ARCH if args.model_variant=='control' else DETAIL_ARCH
    model=(AuxiliaryClassifier if args.model_variant=='control' else DetailClassifier)(len(classes),pretrained=False).to('cuda')
    checkpoint=torch.load(args.initial,map_location='cpu',weights_only=True)
    if (checkpoint['architecture']!=AUX_ARCH or checkpoint['classes']!=classes
            or checkpoint['split_sha256']!=sha(output/'SPLIT.json')
            or checkpoint.get('auxiliary_classes')!=list(AUX_CLASSES)):
        raise ValueError('Architecture study requires the frozen seven-class auxiliary initializer')
    initial_transfer=load_auxiliary_initializer(model,checkpoint['state_dict'])
    # Model construction must not change subsequent global or worker RNG streams.
    torch.manual_seed(seed+2)
    hard_audit=None
    if args.hard_mining_strength:
        print('Scoring verified TRAIN records only for bounded hard-example sampling',flush=True)
        mining_loader=DataLoader(MiningPhotos(items),batch_size=16,num_workers=4,pin_memory=True)
        mining_targets,mining_scores=predict(model,mining_loader,'cuda')
        if not np.array_equal(mining_targets,labels.numpy()):raise ValueError('Training mining item order/labels changed')
        multiplier=hard_multipliers(mining_targets,mining_scores,[classes.index(c) for c in TARGETS],args.hard_mining_strength)
        save(output/'TRAIN-MINING.json',{'split':'train','initial_weights_sha256':sha(args.initial),
             'images':[i['image'] for i in items],'targets':mining_targets.tolist(),'probabilities':mining_scores.tolist(),
             'multipliers':multiplier.tolist(),'strength':args.hard_mining_strength})
        hard_audit={'strength':args.hard_mining_strength,'records':len(items),'min_multiplier':float(multiplier.min()),
                    'max_multiplier':float(multiplier.max()),'mean_multiplier':float(multiplier.mean()),
                    'cache_sha256':sha(output/'TRAIN-MINING.json'),'helper_sha256':sha(ROOT/'scripts/facility_hard_sampling.py'),
                    'policy':'Only verified TRAIN full photos and TRAIN-parent crops; no validation/test scores. Frozen initializer residual, max 1+strength; fixed domain mass retained.'}
        print(__import__('json').dumps(hard_audit),flush=True)
    sampling=torch.ones(len(items),dtype=torch.float64)
    balance_labels=torch.tensor([i['targets'] for i in core_items]) if photo_supplement else labels
    for label in TARGETS:
        k=classes.index(label);ratio=min(3.,max(1.,len(balance_labels)/(2*max(1,int((balance_labels[:,k]==1).sum())))))
        sampling=torch.maximum(sampling,torch.where(labels[:,k]==1,ratio,1.).double())
    crop_balance_full=core_full_count if photo_supplement else full_count
    sampling[full_count:]*=crop_balance_full/(len(items)-full_count)*.25/.75
    if hard_audit:sampling*=multiplier
    weights=[]
    for d,exposure in enumerate(proportions):
        sampling[domains==d]*=exposure/sampling[domains==d].sum()
        w=sampling*(domains==d);pos=((labels==1)*w[:,None]).sum(0);neg=((labels==0)*w[:,None]).sum(0)
        weights.append((neg/pos.clamp_min(1e-8)).clamp(.2,6).float())
    weights=torch.stack(weights).to('cuda')
    pixel_weights=torch.tensor([min(20.,max(1.,pixel_counts[c]['negative']/max(1,pixel_counts[c]['positive']))) for c in classes],device='cuda')
    draws_per_epoch=args.draws_per_epoch or full_count
    sampler=WeightedRandomSampler(sampling,draws_per_epoch,replacement=True,generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(SpatialPhotos(items,auxiliary,full_count,domain_map),batch_size=8,sampler=sampler,num_workers=4,persistent_workers=True,pin_memory=True,generator=torch.Generator().manual_seed(seed+1))
    val,_=dacl_items('val',640)
    loaders={domain:DataLoader(FacilityPhotos(records,640),batch_size=16,num_workers=4,persistent_workers=True,generator=torch.Generator().manual_seed(seed+100+DOMAINS[domain]))
             for domain,records in {'dacl':val,'damsegment':split['val'],'codebrim':code_val['items']}.items()}
    backbone=list(model.backbone.parameters());backbone_ids={id(p) for p in backbone}
    head=[p for p in model.parameters() if id(p) not in backbone_ids]
    if min(args.backbone_lr,args.head_lr)<=0:raise ValueError('Learning rates must be positive')
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':args.backbone_lr},{'params':head,'lr':args.head_lr}],weight_decay=.0002)
    update_counter={'count':0}
    def record_optimizer_step(optimizer,args,kwargs):
        update_counter['count']+=1
    step_hook=optimizer.register_step_post_hook(record_optimizer_step)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.000005);scaler=torch.amp.GradScaler('cuda')
    emphasis=torch.ones(len(classes),device='cuda')
    for label in TARGETS:emphasis[classes.index(label)]=2.
    training={'status':'running','architecture':architecture,'classes':classes,'imgsz':640,'batch_size':8,'seed':seed,'requested_epochs':epochs,'patience':patience,
              'train_dacl':len(original),'train_damsegment':len(split['train']),'train_codebrim':len(code_train['items']),
              'val_dacl':len(val),'val_damsegment':len(split['val']),'validation_domains':list(loaders),'draws_per_epoch':draws_per_epoch,
              'spatial_manifest_sha256':sha(manifest_path),'spatial_manifest_path':manifest_path.relative_to(ROOT).as_posix(),
              'core_spatial_manifest_sha256':sha(core_path),'spatial_manifest_audit':manifest['audit'],
              'training_script_sha256':sha(Path(__file__)),
              'source_parent_training_script_sha256':sha(ROOT/'scripts/train_facility_spatial.py'),
              'model_variant':args.model_variant,'initial_state_transfer':initial_transfer,
              'detail_model_source_sha256':sha(ROOT/'safelog_ai/detail_classifier.py'),
              'model_factory_source_sha256':sha(ROOT/'safelog_ai/presence_classifier.py'),
              'loader_randomness':{'sampler_seed':seed,'training_worker_seed':seed+1,'post_model_seed':seed+2,
                                   'validation_worker_seeds':{d:seed+100+k for d,k in DOMAINS.items()}},
              'detail_architecture':protocol['detail_architecture'] if args.model_variant=='detail' else None,
              'photo_supplement_helper_sha256':sha(ROOT/'scripts/facility_photo_supplement.py'),
              'model_source_sha256':sha(ROOT/'safelog_ai/spatial_classifier.py'),'split_sha256':sha(output/'SPLIT.json'),
              'pretrained_backbone_sha256':sha(ROOT/'.config/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth') if args.initial is None else None,
              'additional_validation':{'path':'data/codebrim-training/val.json','sha256':sha(ROOT/'data/codebrim-training/val.json')},
              'additional_test':{'path':'data/codebrim-training/test.json','sha256':sha(ROOT/'data/codebrim-training/test.json')},
              'initial':'Shared original auxiliary checkpoint transferred tensor-for-tensor; treatment adds zero-initialized stride4 residual branch',
              'loss':'Masked weighted focal photo presence + masked focal 8x8-cell source segmentation + .5 positive-class Dice',
              'domain_proportions':list(proportions),'backbone_lr':args.backbone_lr,'head_lr':args.head_lr,
              'selection':'Minimax per-class FNR/FPR in all three validation domains; no pixel accuracy or test used for selection'}
    training['photo_positive_weights']={d:weights[k].detach().cpu().tolist() for d,k in domain_map.items()}
    training['pixel_positive_weights']=pixel_weights.detach().cpu().tolist()
    training['target_ranking']={'weight':args.target_ranking_weight,'classes':list(TARGETS),
                               'helper_sha256':sha(ROOT/'scripts/facility_target_ranking.py'),
                               'scope':'Original verified TRAIN full photos only; same-source same-class asserted positive versus negative pairs; crops and unknowns excluded',
                               'formula':'Mean softplus(negative_logit-positive_logit) within each contributing source-class group, then equal group mean; float32',
                               'sampling_changed':False,'new_labels_asserted':0,'public_outputs_changed':False}
    training['total_loss_formula']='Recorded base photo/pixel/auxiliary loss + target_ranking.weight * within-source full-photo ranking loss'
    if protocol:
        if architecture!=protocol['control_architecture' if args.model_variant=='control' else 'treatment_architecture']:
            raise ValueError('Declared architecture does not match actual model')
        for key in ('classes','imgsz','batch_size','domain_proportions','loader_randomness'):
            if training[key]!=protocol[key]:raise ValueError(f'Protocol model/input condition differs: {key}')
        training['study_protocol_sha256']=sha(args.study_protocol)
        training['study_protocol_path']=args.study_protocol.relative_to(ROOT).as_posix()
    if auxiliary_weights is not None:training['auxiliary_positive_weights']=auxiliary_weights.detach().cpu().tolist()
    training['expected_sampling']={d:{'full':float(sampling[(domains==k)&(torch.arange(len(items))<full_count)].sum()/sampling.sum()),
                                     'crop':float(sampling[(domains==k)&(torch.arange(len(items))>=full_count)].sum()/sampling.sum())}
                                   for d,k in domain_map.items()}
    training['expected_label_sampling']={label:{'positive':float(sampling[labels[:,k]==1].sum()/sampling.sum()),
                                               'negative':float(sampling[labels[:,k]==0].sum()/sampling.sum()),
                                               'unknown':float(sampling[labels[:,k]<0].sum()/sampling.sum())} for k,label in enumerate(classes)}
    if args.initial:training.update(initial=str(args.initial.resolve()),initial_weights_sha256=sha(args.initial))
    if hard_audit:training['hard_training_sampling']=hard_audit
    if auxiliary is not None:
        training['loss']+=f' + {args.auxiliary_weight} masked original-DACL photo-tag focal'
        training.update(auxiliary_classes=list(AUX_CLASSES),auxiliary_weight=args.auxiliary_weight,
                        auxiliary_manifest_sha256=sha(args.auxiliary_manifest),auxiliary_manifest_path=args.auxiliary_manifest.relative_to(ROOT).as_posix(),
                        auxiliary_model_source_sha256=sha(ROOT/'safelog_ai/auxiliary_classifier.py'),auxiliary_audit=auxiliary_manifest['audit'],
                        auxiliary_scope='Original DACL TRAIN full-photo tags only; all other source/crop auxiliary labels unknown; inference still seven facility logits')
    if supplemental:training.update(train_s2ds=len(supplemental['items']),supplement_spatial_path=args.supplement_spatial.relative_to(ROOT).as_posix(),
                                    supplement_spatial_sha256=sha(args.supplement_spatial),supplement_audit=supplemental['audit'],
                                    supplement_scope='Author TRAIN only; original scene IDs unavailable, heuristic crop screen does not prove field independence; S2DS val/test never selected')
    if photo_supplement:training.update({f"train_{photo_supplement['items'][0]['domain']}":len(photo_supplement['items'])},photo_supplement_path=args.photo_supplement.relative_to(ROOT).as_posix(),
                                       photo_supplement_sha256=sha(args.photo_supplement),photo_supplement_audit=photo_supplement['audit'],
                                       photo_supplement_scope='Verified source files TRAIN-only; facility type unverified, scene IDs unavailable, overlap screening is heuristic, no source/industrial test or new pixel truth',
                                       core_sampling_policy='Original source item weights and crop balance preserved; original domain mass scaled to .90 and .10 assigned to building supplement')
    save(output/'TRAINING.json',training);history=[];best=None;bad=0;start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,epochs+1):
        freeze=epoch==1 and args.initial is None
        for p in backbone:p.requires_grad=not freeze
        model.train()
        if freeze:model.backbone.eval()
        total=0.;draws=torch.zeros(len(domain_map),dtype=torch.long);row_types=torch.zeros(2,dtype=torch.long)
        row_hash=hashlib.sha256()
        epoch_update_start=update_counter['count'];attempted_batches=0
        ranking_total=0.;ranking_pairs=0;ranking_batches=0;ranking_groups={}
        full_target_states={d:{state:0 for state in ('00','10','01','11','unknown')} for d in domain_map}
        for batch_number,batch in enumerate(loader,1):
            attempted_batches+=1
            image,label,known,domain,mask,pixel_known=batch[:6]
            row_types+=torch.bincount(batch[-2],minlength=2)
            row_hash.update(batch[-1].numpy().astype('<i8',copy=False).tobytes())
            for d,k in domain_map.items():
                eligible=(domain==k)&(batch[-2]==0)
                target_known=known[:,[classes.index(c) for c in TARGETS]].all(1)
                full_target_states[d]['unknown']+=int((eligible&~target_known).sum())
                for state in ('00','10','01','11'):
                    chosen=eligible&target_known
                    for j,c in enumerate(TARGETS):chosen&=(label[:,classes.index(c)]==int(state[j]))
                    full_target_states[d][state]+=int(chosen.sum())
            draws+=torch.bincount(domain,minlength=len(domain_map));optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda'):
                if auxiliary is None:scores,maps=model.forward_details(image.to('cuda'))
                else:scores,maps,auxiliary_scores=model.forward_training(image.to('cuda'))
                loss=masked_focal(scores,label.to('cuda'),known.to('cuda'),weights[domain],1.,emphasis)+spatial_loss(maps,mask.to('cuda'),pixel_known.to('cuda'),pixel_weights)
                if auxiliary is not None:
                    loss+=args.auxiliary_weight*masked_focal(auxiliary_scores,batch[6].to('cuda'),batch[7].to('cuda'),auxiliary_weights,1.)
                if args.target_ranking_weight:
                    ranking,ranking_audit=target_ranking_loss(scores,label.to('cuda'),known.to('cuda'),domain.to('cuda'),
                                                            (batch[-2]==0).to('cuda'),[classes.index(c) for c in TARGETS])
                    loss+=args.target_ranking_weight*ranking
                    ranking_total+=float(ranking.detach())*len(image)
                    ranking_pairs+=ranking_audit['pair_count'];ranking_batches+=int(ranking_audit['contributing_groups']>0)
                    for group in ranking_audit['groups']:
                        key=f"{group['domain']}:{group['target_index']}"
                        ranking_groups[key]=ranking_groups.get(key,0)+group['pair_count']
            if not torch.isfinite(loss):raise ValueError('Nonfinite spatial training loss')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();total+=loss.item()*len(image)
            if batch_number%400==0:print(__import__('json').dumps({'epoch':epoch,'sampled_so_far':int(draws.sum()),'budget':draws_per_epoch}),flush=True)
        predictions={domain:predict(model,validation_loader,'cuda') for domain,validation_loader in loaders.items()}
        points={label:operating_point({d:(t[:,classes.index(label)].astype(bool),p[:,classes.index(label)]) for d,(t,p) in predictions.items()}) for label in TARGETS}
        worst=max(p['worst_error'] for p in points.values());sum_errors=sum(r[k] for p in points.values() for r in p['domains'].values() for k in ('fnr','fpr'))
        row={'epoch':epoch,'elapsed_minutes':(time.perf_counter()-start)/60,'train_loss':total/draws_per_epoch,'worst_target_error':worst,'sum_target_errors':sum_errors,
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
             'optimizer_step_diagnostics':{'attempted_batches':attempted_batches,
                 'actual_optimizer_steps':update_counter['count']-epoch_update_start,
                 'amp_skipped_steps':attempted_batches-(update_counter['count']-epoch_update_start)},
             'target_passed':all(p['target_passed'] for p in points.values()),'operating_points':points,'sampled_domain_counts':{d:int(draws[k]) for d,k in domain_map.items()},
             'sampled_row_type_counts':{'full':int(row_types[0]),'crop':int(row_types[1])},
             'sampled_full_target_joint_counts':full_target_states,'sampled_row_indices_sha256':row_hash.hexdigest(),
             'target_ranking_audit':{'unweighted_mean_batch_loss':ranking_total/draws_per_epoch,'pair_count':ranking_pairs,
                                     'contributing_batches':ranking_batches,'pair_counts_by_domain_target':ranking_groups},
             'ranking_ap':{d:{label:average_precision(t[:,k],p[:,k]) for k,label in enumerate(classes) if (t[:,k]>=0).all()} for d,(t,p) in predictions.items()}}
        history.append(row);print(__import__('json').dumps({k:row[k] for k in ('epoch','elapsed_minutes','train_loss','worst_target_error','target_passed')}),flush=True)
        key=(worst,sum_errors)
        if best is None or key<best:
            best=key;bad=0
            torch.save({'architecture':architecture,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'classes':classes,'imgsz':640,'mean':MEAN,'std':STD,
                        'auxiliary_classes':list(AUX_CLASSES) if auxiliary is not None else None,
                        'epoch':epoch,'selection_split':'val','split_sha256':training['split_sha256'],'worst_target_error':worst},output/'best.pt')
            save(output/'VALIDATION.json',row)
            for d,(t,p) in predictions.items():save(output/f'validation-{d}.json',{'split':'val','targets':t.tolist(),'probabilities':p.tolist(),'weights_sha256':sha(output/'best.pt')})
        else:bad+=1
        save(output/'history.json',history);scheduler.step()
        if bad>=patience:break
    training.update(status='complete',actual_epochs=len(history),best_worst_target_error=best[0],weights_sha256=sha(output/'best.pt'),
                    elapsed_training_minutes=(time.perf_counter()-start)/60,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    optimizer_step_diagnostics={'attempted_batches':sum(r['optimizer_step_diagnostics']['attempted_batches'] for r in history),
                        'actual_optimizer_steps':update_counter['count'],
                        'amp_skipped_steps':sum(r['optimizer_step_diagnostics']['amp_skipped_steps'] for r in history)})
    step_hook.remove()
    save(output/'TRAINING.json',training)
    # Detailed source-annotation audits stay in the ignored run metadata.
    print(__import__('json').dumps({k:training[k] for k in ('status','actual_epochs','best_worst_target_error','weights_sha256')},indent=2))


if __name__=='__main__':main()
