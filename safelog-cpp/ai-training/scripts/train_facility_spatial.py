"""Joint source-pixel and photo-label training; selection stays on validation only."""
import os
import argparse
from pathlib import Path
import random
import sys
import time
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

SPATIAL_DOMAINS={**DOMAINS,'s2ds':3}

class SpatialPhotos(Dataset):
    def __init__(self,items,auxiliary=None):self.items=items;self.auxiliary=auxiliary;self.transform=image_transform(640);self.jitter=ColorJitter(.1,.1,.1,.01)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        item=self.items[i]
        with Image.open(ROOT/item['image']) as handle:image=handle.convert('RGB')
        with np.load(ROOT/item['pixel_target']) as data:mask=data['mask'].copy();pixel_known=data['known'].copy()
        if random.random()<.5:image=ImageOps.mirror(image);mask=mask[:,:,::-1].copy()
        targets=torch.tensor(item['targets'],dtype=torch.float32)
        result=(self.transform(self.jitter(image)),targets.clamp_min(0),(targets>=0).float(),SPATIAL_DOMAINS[item['domain']],torch.from_numpy(mask).float(),torch.from_numpy(pixel_known).float())
        if self.auxiliary is not None:
            aux=torch.tensor(self.auxiliary.get(item['image'],[-1]*len(AUX_CLASSES)),dtype=torch.float32)
            result+=aux.clamp_min(0),(aux>=0).float()
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
    parser.add_argument('--name',default='facility-presence-target-spatial');parser.add_argument('--seed',type=int,default=47)
    parser.add_argument('--epochs',type=int,default=18);parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--initial',type=Path);parser.add_argument('--supplement-spatial',type=Path)
    parser.add_argument('--backbone-lr',type=float,default=.00008);parser.add_argument('--head-lr',type=float,default=.0005)
    parser.add_argument('--hard-mining-strength',type=float,default=0.,help='Bounded TRAIN-only residual sampling from frozen initializer (0 disables)')
    parser.add_argument('--auxiliary-manifest',type=Path)
    parser.add_argument('--auxiliary-weight',type=float,default=.5)
    args=parser.parse_args()
    name,seed,epochs,patience=args.name,args.seed,args.epochs,args.patience
    if Path(name).name!=name or min(epochs,patience)<1:raise ValueError('Invalid spatial run options')
    if not np.isfinite(args.hard_mining_strength) or not 0<=args.hard_mining_strength<=4 or (args.hard_mining_strength and not args.initial):
        raise ValueError('Hard mining needs an initializer and finite strength in [0,4]')
    if not np.isfinite(args.auxiliary_weight) or args.auxiliary_weight<=0:raise ValueError('Auxiliary loss weight must be finite and positive')
    output=ROOT/'runs'/name
    if (output/'best.pt').exists():raise ValueError('Preserve existing experiment')
    output.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    manifest=read(ROOT/'data/facility-spatial-training/train.json');classes=manifest['classes'];items=manifest['items'];full_count=manifest['full_count']
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
    model=(AuxiliaryClassifier if auxiliary is not None else SpatialClassifier)(len(classes),pretrained=args.initial is None).to('cuda')
    if args.initial:
        checkpoint=torch.load(args.initial,map_location='cpu',weights_only=True)
        if checkpoint['architecture'] not in ((ARCH,AUX_ARCH) if auxiliary is not None else (ARCH,)) or checkpoint['classes']!=classes or checkpoint['split_sha256']!=sha(output/'SPLIT.json'):
            raise ValueError('Spatial initializer architecture/classes/split mismatch')
        if checkpoint['architecture']==AUX_ARCH and checkpoint.get('auxiliary_classes')!=list(AUX_CLASSES):
            raise ValueError('Auxiliary initializer tag order changed')
        if auxiliary is not None and checkpoint['architecture']==ARCH:
            mismatch=model.load_state_dict(checkpoint['state_dict'],strict=False)
            if set(mismatch.missing_keys)!={'auxiliary_head.weight','auxiliary_head.bias'} or mismatch.unexpected_keys:
                raise ValueError('Only new auxiliary head may be missing in base initializer')
        else:model.load_state_dict(checkpoint['state_dict'])
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
    for label in TARGETS:
        k=classes.index(label);ratio=min(3.,max(1.,len(items)/(2*max(1,int((labels[:,k]==1).sum())))))
        sampling=torch.maximum(sampling,torch.where(labels[:,k]==1,ratio,1.).double())
    sampling[full_count:]*=full_count/(len(items)-full_count)*.25/.75
    if hard_audit:sampling*=multiplier
    weights=[]
    for d,exposure in enumerate(proportions):
        sampling[domains==d]*=exposure/sampling[domains==d].sum()
        w=sampling*(domains==d);pos=((labels==1)*w[:,None]).sum(0);neg=((labels==0)*w[:,None]).sum(0)
        weights.append((neg/pos.clamp_min(1e-8)).clamp(.2,6).float())
    weights=torch.stack(weights).to('cuda')
    pixel_weights=torch.tensor([min(20.,max(1.,pixel_counts[c]['negative']/max(1,pixel_counts[c]['positive']))) for c in classes],device='cuda')
    sampler=WeightedRandomSampler(sampling,full_count,replacement=True,generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(SpatialPhotos(items,auxiliary),batch_size=8,sampler=sampler,num_workers=4,persistent_workers=True,pin_memory=True)
    val,_=dacl_items('val',640)
    loaders={name:DataLoader(FacilityPhotos(records,640),batch_size=16,num_workers=4,persistent_workers=True)
             for name,records in {'dacl':val,'damsegment':split['val'],'codebrim':code_val['items']}.items()}
    backbone=list(model.backbone.parameters());backbone_ids={id(p) for p in backbone}
    head=[p for p in model.parameters() if id(p) not in backbone_ids]
    if min(args.backbone_lr,args.head_lr)<=0:raise ValueError('Learning rates must be positive')
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':args.backbone_lr},{'params':head,'lr':args.head_lr}],weight_decay=.0002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.000005);scaler=torch.amp.GradScaler('cuda')
    emphasis=torch.ones(len(classes),device='cuda')
    for label in TARGETS:emphasis[classes.index(label)]=2.
    training={'status':'running','architecture':architecture,'classes':classes,'imgsz':640,'seed':seed,'requested_epochs':epochs,'patience':patience,
              'train_dacl':len(original),'train_damsegment':len(split['train']),'train_codebrim':len(code_train['items']),
              'val_dacl':len(val),'val_damsegment':len(split['val']),'validation_domains':list(loaders),'draws_per_epoch':full_count,
              'spatial_manifest_sha256':sha(ROOT/'data/facility-spatial-training/train.json'),'training_script_sha256':sha(Path(__file__)),
              'model_source_sha256':sha(ROOT/'safelog_ai/spatial_classifier.py'),'split_sha256':sha(output/'SPLIT.json'),
              'pretrained_backbone_sha256':sha(ROOT/'.config/torch/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth') if args.initial is None else None,
              'additional_validation':{'path':'data/codebrim-training/val.json','sha256':sha(ROOT/'data/codebrim-training/val.json')},
              'additional_test':{'path':'data/codebrim-training/test.json','sha256':sha(ROOT/'data/codebrim-training/test.json')},
              'initial':'Official LRASPP COCO/VOC pretrained backbone/semantic features; new seven facility heads, no source facility validation training',
              'loss':'Masked weighted focal photo presence + masked focal 8x8-cell source segmentation + .5 positive-class Dice',
              'domain_proportions':list(proportions),'backbone_lr':args.backbone_lr,'head_lr':args.head_lr,
              'selection':'Minimax per-class FNR/FPR in all three validation domains; no pixel accuracy or test used for selection'}
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
    save(output/'TRAINING.json',training);history=[];best=None;bad=0;start=time.perf_counter()
    for epoch in range(1,epochs+1):
        freeze=epoch==1 and args.initial is None
        for p in backbone:p.requires_grad=not freeze
        model.train()
        if freeze:model.backbone.eval()
        total=0.;draws=torch.zeros(len(domain_map),dtype=torch.long)
        for batch in loader:
            image,label,known,domain,mask,pixel_known=batch[:6]
            draws+=torch.bincount(domain,minlength=len(domain_map));optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda'):
                if auxiliary is None:scores,maps=model.forward_details(image.to('cuda'))
                else:scores,maps,auxiliary_scores=model.forward_training(image.to('cuda'))
                loss=masked_focal(scores,label.to('cuda'),known.to('cuda'),weights[domain],1.,emphasis)+spatial_loss(maps,mask.to('cuda'),pixel_known.to('cuda'),pixel_weights)
                if auxiliary is not None:
                    loss+=args.auxiliary_weight*masked_focal(auxiliary_scores,batch[6].to('cuda'),batch[7].to('cuda'),auxiliary_weights,1.)
            if not torch.isfinite(loss):raise ValueError('Nonfinite spatial training loss')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();total+=loss.item()*len(image)
        predictions={domain:predict(model,validation_loader,'cuda') for domain,validation_loader in loaders.items()}
        points={label:operating_point({d:(t[:,classes.index(label)].astype(bool),p[:,classes.index(label)]) for d,(t,p) in predictions.items()}) for label in TARGETS}
        worst=max(p['worst_error'] for p in points.values());sum_errors=sum(r[k] for p in points.values() for r in p['domains'].values() for k in ('fnr','fpr'))
        row={'epoch':epoch,'elapsed_minutes':(time.perf_counter()-start)/60,'train_loss':total/full_count,'worst_target_error':worst,'sum_target_errors':sum_errors,
             'target_passed':all(p['target_passed'] for p in points.values()),'operating_points':points,'sampled_domain_counts':{d:int(draws[k]) for d,k in domain_map.items()},
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
    training.update(status='complete',actual_epochs=len(history),best_worst_target_error=best[0],weights_sha256=sha(output/'best.pt'))
    save(output/'TRAINING.json',training);print(__import__('json').dumps(training,indent=2))


if __name__=='__main__':main()
