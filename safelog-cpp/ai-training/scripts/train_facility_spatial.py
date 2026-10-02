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


class SpatialPhotos(Dataset):
    def __init__(self,items):self.items=items;self.transform=image_transform(640);self.jitter=ColorJitter(.1,.1,.1,.01)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        item=self.items[i]
        with Image.open(ROOT/item['image']) as handle:image=handle.convert('RGB')
        with np.load(ROOT/item['pixel_target']) as data:mask=data['mask'].copy();pixel_known=data['known'].copy()
        if random.random()<.5:image=ImageOps.mirror(image);mask=mask[:,:,::-1].copy()
        targets=torch.tensor(item['targets'],dtype=torch.float32)
        return self.transform(self.jitter(image)),targets.clamp_min(0),(targets>=0).float(),DOMAINS[item['domain']],torch.from_numpy(mask).float(),torch.from_numpy(pixel_known).float()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='facility-presence-target-spatial');parser.add_argument('--seed',type=int,default=47)
    parser.add_argument('--epochs',type=int,default=18);parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--initial',type=Path);args=parser.parse_args()
    name,seed,epochs,patience=args.name,args.seed,args.epochs,args.patience
    if Path(name).name!=name or min(epochs,patience)<1:raise ValueError('Invalid spatial run options')
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
    labels=torch.tensor([i['targets'] for i in items]);domains=torch.tensor([DOMAINS[i['domain']] for i in items])
    sampling=torch.ones(len(items),dtype=torch.float64)
    for label in TARGETS:
        k=classes.index(label);ratio=min(3.,max(1.,len(items)/(2*max(1,int((labels[:,k]==1).sum())))))
        sampling=torch.maximum(sampling,torch.where(labels[:,k]==1,ratio,1.).double())
    sampling[full_count:]*=full_count/(len(items)-full_count)*.25/.75
    weights=[]
    for d,exposure in enumerate((.7,.1,.2)):
        sampling[domains==d]*=exposure/sampling[domains==d].sum()
        w=sampling*(domains==d);pos=((labels==1)*w[:,None]).sum(0);neg=((labels==0)*w[:,None]).sum(0)
        weights.append((neg/pos.clamp_min(1e-8)).clamp(.2,6).float())
    weights=torch.stack(weights).to('cuda')
    pixel_counts=manifest['audit']['per_label_pixel_cells']
    pixel_weights=torch.tensor([min(20.,max(1.,pixel_counts[c]['negative']/max(1,pixel_counts[c]['positive']))) for c in classes],device='cuda')
    sampler=WeightedRandomSampler(sampling,full_count,replacement=True,generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(SpatialPhotos(items),batch_size=8,sampler=sampler,num_workers=4,persistent_workers=True,pin_memory=True)
    val,_=dacl_items('val',640)
    loaders={name:DataLoader(FacilityPhotos(records,640),batch_size=16,num_workers=4,persistent_workers=True)
             for name,records in {'dacl':val,'damsegment':split['val'],'codebrim':code_val['items']}.items()}
    model=SpatialClassifier(len(classes),pretrained=args.initial is None).to('cuda')
    if args.initial:
        checkpoint=torch.load(args.initial,map_location='cpu',weights_only=True)
        if checkpoint['architecture']!=ARCH or checkpoint['classes']!=classes or checkpoint['split_sha256']!=sha(output/'SPLIT.json'):
            raise ValueError('Spatial initializer architecture/classes/split mismatch')
        model.load_state_dict(checkpoint['state_dict'])
    backbone=list(model.backbone.parameters());backbone_ids={id(p) for p in backbone}
    head=[p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':.00008},{'params':head,'lr':.0005}],weight_decay=.0002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.000005);scaler=torch.amp.GradScaler('cuda')
    emphasis=torch.ones(len(classes),device='cuda')
    for label in TARGETS:emphasis[classes.index(label)]=2.
    training={'status':'running','architecture':ARCH,'classes':classes,'imgsz':640,'seed':seed,'requested_epochs':epochs,'patience':patience,
              'train_dacl':len(original),'train_damsegment':len(split['train']),'train_codebrim':len(code_train['items']),
              'val_dacl':len(val),'val_damsegment':len(split['val']),'validation_domains':list(loaders),'draws_per_epoch':full_count,
              'spatial_manifest_sha256':sha(ROOT/'data/facility-spatial-training/train.json'),'training_script_sha256':sha(Path(__file__)),
              'model_source_sha256':sha(ROOT/'safelog_ai/spatial_classifier.py'),'split_sha256':sha(output/'SPLIT.json'),
              'additional_validation':{'path':'data/codebrim-training/val.json','sha256':sha(ROOT/'data/codebrim-training/val.json')},
              'additional_test':{'path':'data/codebrim-training/test.json','sha256':sha(ROOT/'data/codebrim-training/test.json')},
              'initial':'Official LRASPP COCO/VOC pretrained backbone/semantic features; new seven facility heads, no source facility validation training',
              'loss':'Masked weighted focal photo presence + masked focal 8x8-cell source segmentation + .5 positive-class Dice',
              'selection':'Minimax per-class FNR/FPR in all three validation domains; no pixel accuracy or test used for selection'}
    if args.initial:training.update(initial=str(args.initial.resolve()),initial_weights_sha256=sha(args.initial))
    save(output/'TRAINING.json',training);history=[];best=None;bad=0;start=time.perf_counter()
    for epoch in range(1,epochs+1):
        freeze=epoch==1 and args.initial is None
        for p in backbone:p.requires_grad=not freeze
        model.train()
        if freeze:model.backbone.eval()
        total=0.;draws=torch.zeros(3,dtype=torch.long)
        for image,label,known,domain,mask,pixel_known in loader:
            draws+=torch.bincount(domain,minlength=3);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda'):
                scores,maps=model.forward_details(image.to('cuda'))
                loss=masked_focal(scores,label.to('cuda'),known.to('cuda'),weights[domain],1.,emphasis)+spatial_loss(maps,mask.to('cuda'),pixel_known.to('cuda'),pixel_weights)
            if not torch.isfinite(loss):raise ValueError('Nonfinite spatial training loss')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();total+=loss.item()*len(image)
        predictions={domain:predict(model,validation_loader,'cuda') for domain,validation_loader in loaders.items()}
        points={label:operating_point({d:(t[:,classes.index(label)].astype(bool),p[:,classes.index(label)]) for d,(t,p) in predictions.items()}) for label in TARGETS}
        worst=max(p['worst_error'] for p in points.values());sum_errors=sum(r[k] for p in points.values() for r in p['domains'].values() for k in ('fnr','fpr'))
        row={'epoch':epoch,'elapsed_minutes':(time.perf_counter()-start)/60,'train_loss':total/full_count,'worst_target_error':worst,'sum_target_errors':sum_errors,
             'target_passed':all(p['target_passed'] for p in points.values()),'operating_points':points,'sampled_domain_counts':{d:int(draws[k]) for d,k in DOMAINS.items()},
             'ranking_ap':{d:{label:average_precision(t[:,k],p[:,k]) for k,label in enumerate(classes) if (t[:,k]>=0).all()} for d,(t,p) in predictions.items()}}
        history.append(row);print(__import__('json').dumps({k:row[k] for k in ('epoch','elapsed_minutes','train_loss','worst_target_error','target_passed')}),flush=True)
        key=(worst,sum_errors)
        if best is None or key<best:
            best=key;bad=0
            torch.save({'architecture':ARCH,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'classes':classes,'imgsz':640,'mean':MEAN,'std':STD,
                        'epoch':epoch,'selection_split':'val','split_sha256':training['split_sha256'],'worst_target_error':worst},output/'best.pt')
            save(output/'VALIDATION.json',row)
            for d,(t,p) in predictions.items():save(output/f'validation-{d}.json',{'split':'val','targets':t.tolist(),'probabilities':p.tolist(),'weights_sha256':sha(output/'best.pt')})
        else:bad+=1
        save(output/'history.json',history);scheduler.step()
        if bad>=patience:break
    training.update(status='complete',actual_epochs=len(history),best_worst_target_error=best[0],weights_sha256=sha(output/'best.pt'))
    save(output/'TRAINING.json',training);print(__import__('json').dumps(training,indent=2))


if __name__=='__main__':main()
