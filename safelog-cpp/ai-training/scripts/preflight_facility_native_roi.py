"""Actual paired TRAIN replay and disposable CUDA step before native ROI training."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import default_collate

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier
from safelog_ai.spatial_classifier import spatial_loss
from safelog_ai.presence_classifier import PresenceClassifier, MEAN, STD
from scripts.facility_resolution_study import ResolutionPhotos
from scripts.facility_native_roi_study import (PROTOCOL,NAMES,REFERENCE,validate_protocol,
    prepare_data,protected_hashes,require,write)
from scripts.train_facility_native_roi import load_initial_state
from scripts.train_facility_target import read,sha,DOMAINS,masked_focal

OUTPUT='runs/facility-native-roi-preflight.json'


def replay_indices(base,full):
    changed=[i for i,(a,b) in enumerate(zip(base['original_items'],base['items'])) if a['image']!=b['image']]
    indices=[next(i for i,r in enumerate(base['items'][:full]) if r['domain']==domain) for domain in DOMAINS]
    for predicate in (lambda t:t[0]==0 and t[1]==0,
                      lambda t:t[0]==1 or t[1]==1,
                      lambda t:-1 in t[:2]):
        candidate=next((i for i in changed if predicate(base['items'][i]['targets']) and i not in indices),None)
        if candidate is not None:indices.append(candidate)
    for i in range(full,len(base['items'])):
        if i not in changed and i not in indices:
            indices.append(i)
        if len(indices)==8:break
    require(len(indices)==8 and any(i in changed for i in indices),'Need actual original and changed TRAIN replay')
    return indices


def batch(data,indices):
    random.seed(57);np.random.seed(57);torch.manual_seed(57)
    dataset=ResolutionPhotos(data['items'],data['auxiliary'],data['full_count'],DOMAINS,imgsz=640)
    return default_collate([dataset[i] for i in indices])


def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    require(torch.cuda.is_available(),'Actual CUDA preflight requires GPU')
    output=ROOT/OUTPUT;require(not output.exists(),'Preserve the existing preflight evidence')
    directory=ROOT/'runs/facility-native-roi-preflight';directory.mkdir(exist_ok=False)
    p=validate_protocol(read(ROOT/PROTOCOL),ROOT);before=protected_hashes(ROOT)
    data={v:prepare_data(ROOT,p,v) for v in ('control','native')}
    original=read(ROOT/'data/facility-spatial-training/train.json')['items']
    for d in data.values():d['original_items']=original
    indices=replay_indices(data['control'],data['control']['full_count'])
    batches={v:batch(d,indices) for v,d in data.items()}
    for left,right in zip(batches['control'][1:],batches['native'][1:]):
        require(torch.equal(left,right),'Paired labels/masks/known/domain/aux/row replay differs')
    unchanged=[j for j,i in enumerate(indices) if original[i]['image']==data['control']['items'][i]['image']]
    require(unchanged and torch.equal(batches['control'][0][unchanged],batches['native'][0][unchanged]),
            'Unchanged TRAIN photo tensors differ')
    weights=ROOT/f'runs/{REFERENCE}/best.pt'
    checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
    split_sha=checkpoint['split_sha256'];proofs={}
    for variant,values in batches.items():
        torch.manual_seed(56)
        model=AuxiliaryClassifier(7,pretrained=False)
        transfer=load_initial_state(model,checkpoint,p['classes'],split_sha)
        model.to('cuda');model.eval();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            public=model(values[0].cuda())
            photo,maps,aux=model.forward_training(values[0].cuda())
            require(tuple(public.shape)==(8,7) and tuple(maps.shape)==(8,7,80,80)
                    and tuple(aux.shape)==(8,19) and torch.equal(public,photo)
                    and all(torch.isfinite(t).all() for t in (public,maps,aux)),
                    'Actual GPU output contract failed')
        optimizer=torch.optim.AdamW(model.parameters(),lr=4e-5,weight_decay=.0002)
        updates={'count':0}
        def updated(optimizer,args,kwargs):updates['count']+=1
        hook=optimizer.register_step_post_hook(updated)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.);model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda'):
            photo,maps,aux=model.forward_training(values[0].cuda())
            d=data[variant];emphasis=torch.tensor([2.,2.,1.,1.,1.,1.,1.],device='cuda')
            loss=masked_focal(photo,values[1].cuda(),values[2].cuda(),
                             d['sampling_data']['photo_weights'].cuda()[values[3]],1.,emphasis)
            loss+=spatial_loss(maps,values[4].cuda(),values[5].cuda(),
                               d['supervision_weights']['pixel_weights'].cuda())
            loss+=.5*masked_focal(aux,values[6].cuda(),values[7].cuda(),
                                  d['supervision_weights']['auxiliary_weights'].cuda(),1.)
        require(torch.isfinite(loss),'Actual replay training loss is nonfinite')
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();hook.remove()
        require(updates['count']==1,'Disposable AMP update was skipped')
        path=directory/f'{variant}.pt'
        torch.save({**{k:v for k,v in checkpoint.items() if k!='state_dict'},
                    'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},
                    'mean':MEAN,'std':STD},path)
        peak=torch.cuda.max_memory_allocated();del model,optimizer,scaler
        torch.cuda.empty_cache()
        classifier=PresenceClassifier(path,'cpu')
        with torch.inference_mode():
            public=classifier.model(values[0][:1]);photo,maps,aux=classifier.model.forward_training(values[0][:1])
        require(torch.equal(public,photo) and all(torch.isfinite(t).all() for t in (photo,maps,aux)),
                'Actual saved checkpoint CPU reload failed')
        proofs[variant]={'actual_batch_size':8,'actual_optimizer_steps':1,'amp_skipped_steps':0,'scaler_initial_scale':1024.,
                         'public_shape':[8,7],'pixel_shape':[8,7,80,80],'auxiliary_shape':[8,19],
                         'parameter_count':sum(t.numel() for t in classifier.model.parameters()),
                         'all_outputs_finite':True,'cpu_reload_verified':True,'strict_factory_reload_verified':True,
                         'public_output_equals_training_photo_output':True,
                         'disposable_checkpoint_sha256':sha(path),'peak_cuda_allocated_bytes':peak,
                         'initial_state_transfer':transfer}
        del classifier
    require(protected_hashes(ROOT)==before,'Preflight changed original/paired/app files')
    validate_protocol(p,ROOT)
    result={'schema':'facility_native_roi_preflight_v1','status':'passed',
            'verified_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(ROOT/PROTOCOL),
            'source_sha256':p['source_sha256'],'protected_file_sha256':before,'pairs':proofs,
            'actual_labels_masks_known_aux_domains_row_indices_equal':True,
            'mask_label_known_aux_index_replay_equal':True,
            'unchanged_image_tensors_equal':True,'unique_train_rows':len(indices),
            'replay_row_indices_sha256':hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest(),
            'new_training_epochs':0,'source_test_inference_executed':False,'app_model_promoted':False,
            'accuracy_measured':False,'label_changes':0,'new_masks':0}
    write(output,result)
    print({'status':'passed','actual_train_rows':8,'disposable_amp_steps':2,'new_epochs':0})


if __name__=='__main__':main()
