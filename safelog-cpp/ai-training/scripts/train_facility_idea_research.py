"""Actually train one fixed building IDEA photo data/exposure candidate."""
from __future__ import annotations
from collections import defaultdict
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_idea_research import NAME,PROTOCOL,INITIAL,EPOCHS,SEED,DRAW_COUNT,REPLACEMENTS,NEW_DOMAINS,MANIFEST,load_data,dataset,validate_protocol,read,sha,require
from scripts.train_facility_rc_positive import initial_models,optimizer_for
from scripts.facility_subtype_study import FixedEpochSampler
from scripts.train_facility_target import FacilityPhotos,dacl_items,predict,masked_focal,save,TARGETS,DOMAINS
from scripts.train_facility_batchnorm import photo_target_counts
from scripts.train_facility_presence import average_precision
from scripts.facility_error_target import operating_point
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier,AUX_CLASSES,ARCH
from safelog_ai.presence_classifier import MEAN,STD
from safelog_ai.spatial_classifier import spatial_loss
from safelog_ai.frozen_batchnorm import apply_frozen_batchnorm,batchnorm_state_sha256,batchnorm_configuration
from safelog_ai.retention_distillation import masked_bernoulli_kl,state_sha256

def losses(model,teacher,batch,data):
    images,labels,known,domains,masks,pixel_known,auxiliary,aux_known=batch[:8]
    images=images.to("cuda");scores,maps,tags=model.forward_training(images)
    with torch.no_grad():reference=teacher(images)
    weights=torch.cat((data['sampling_data']['photo_weights'],torch.ones(1,7))).to('cuda')
    components={
        'photo':masked_focal(scores,labels.to('cuda'),known.to('cuda'),weights[domains.to('cuda')],1.,torch.tensor([2.,2.,1.,1.,1.,1.,1.],device='cuda')),
        'spatial':spatial_loss(maps,masks.to('cuda'),pixel_known.to('cuda'),data['supervision_weights']['pixel_weights'].to('cuda')),
        'auxiliary':.5*masked_focal(tags,auxiliary.to('cuda'),aux_known.to('cuda'),data['supervision_weights']['auxiliary_weights'].to('cuda'),1.),
        'retention':4.*masked_bernoulli_kl(scores,reference,known.to('cuda'),temperature=2.)}
    total=sum(components.values());require(torch.isfinite(total),'Nonfinite candidate loss')
    return total,components

def metadata(data,split_sha):
    return {"architecture":ARCH,"classes":data["classes"],"auxiliary_classes":list(AUX_CLASSES),
        "imgsz":640,"mean":MEAN,"std":STD,"split_sha256":split_sha}

def preflight(protocol,data):
    dest=ROOT/"runs/facility-idea-preflight";require(not dest.exists(),"Preserve actual preflight")
    dest.mkdir()
    dam=next(i for i,r in enumerate(data["items"][:data["full_count"]])if r["domain"]=="damsegment")
    positive=next(i for i,r in enumerate(data['supplement'])if r['stratum']=='rc_spalling')
    normal=next(i for i,r in enumerate(data['supplement'])if r['stratum']=='author_no_damage')
    count=len(data['items']);indices=[0,1,dam,dam+1,count+positive,count+positive,count+normal,count+normal]
    batch=next(iter(DataLoader(dataset(data),batch_size=8,sampler=indices,num_workers=0)))
    require(not batch[5][4:].any()and not batch[7][4:].any()and not batch[2][4:,2:].any(),'Supplement invented pixel/auxiliary/other5 labels')
    model,teacher,transfer,split_sha=initial_models(data)
    original,teacher_sha,bn_sha=state_sha256(model.state_dict()),state_sha256(teacher.state_dict()),batchnorm_state_sha256(model)
    model.train();apply_frozen_batchnorm(model);optimizer=optimizer_for(model)
    loss,components=losses(model,teacher,batch,data);loss.backward()
    gradients=[p.grad for p in model.parameters()if p.grad is not None]
    require(gradients and all(torch.isfinite(g).all()for g in gradients),"Actual gradients invalid")
    optimizer.step();require(original!=state_sha256(model.state_dict())and bn_sha==batchnorm_state_sha256(model),"Actual update/frozen BN invalid")
    require(teacher_sha==state_sha256(teacher.state_dict())and all(p.grad is None for p in teacher.parameters()),"Teacher changed")
    saved={**metadata(data,split_sha),"state_dict":{k:v.cpu()for k,v in model.state_dict().items()}}
    torch.save(saved,dest/"disposable.pt");cpu=AuxiliaryClassifier(7,pretrained=False)
    cpu.load_state_dict(torch.load(dest/"disposable.pt",map_location="cpu",weights_only=True)["state_dict"],strict=True);cpu.eval();model.eval()
    previous=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        with torch.no_grad():
            gpu=[x.cpu()for x in model.forward_training(batch[0][:2].to("cuda"))];outputs=cpu.forward_training(batch[0][:2])
        require([list(x.shape)for x in outputs]==[[2,7],[2,7,80,80],[2,19]]and all(torch.isfinite(x).all()and torch.allclose(x,y,atol=.002,rtol=.002)for x,y in zip(outputs,gpu)),"Strict CPU/GPU reload differs")
    finally:torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32=previous
    save(dest/"preflight.json",{"status":"passed","protocol_sha256":sha(ROOT/PROTOCOL),"actual_disposable_fp32_updates":1,
        "actual_training_epochs":0,"initial_transfer":transfer,"batch_indices":indices,"loss":float(loss.detach()),
        "loss_components":{k:float(v.detach())for k,v in components.items()},"strict_cpu_324_reload":True,
        "teacher_and_bn_preserved":True,"strict_fp32_diagnostic_flags_restored":True,"human_feedback_used":False,
        "core_samples":4,"source_positive_samples":2,"source_normal_samples":2,"new_pixel_known_cells":int(batch[5][4:].sum()),"new_aux_known_cells":int(batch[7][4:].sum()),"new_other5_known_targets":int(batch[2][4:,2:].sum())})
    print("TRAIN-only actual disposable update passed; zero completed epochs",flush=True)

def train(protocol,data):
    dest=ROOT/"runs"/NAME;require(not dest.exists(),"Preserve existing actual run")
    require(read(ROOT/"runs/facility-idea-preflight/preflight.json")["protocol_sha256"]==sha(ROOT/PROTOCOL),"Preflight binding changed")
    before=read(ROOT/"runs/facility-idea-before.json")
    require(before["status"]=="passed"and before["protocol_sha256"]==sha(ROOT/PROTOCOL),"Committed source and preserved inputs required")
    dest.mkdir();save(dest/"SPLIT.json",data["split"]);model,teacher,transfer,split_sha=initial_models(data)
    require(sha(dest/"SPLIT.json")==split_sha,"Original split changed")
    teacher_sha,bn_sha,bn_config=state_sha256(teacher.state_dict()),batchnorm_state_sha256(model),batchnorm_configuration(model)
    torch.manual_seed(SEED+2);sampler=FixedEpochSampler(data["candidate_draws"])
    loader=DataLoader(dataset(data),batch_size=8,sampler=sampler,num_workers=4,persistent_workers=True,pin_memory=True,generator=torch.Generator().manual_seed(SEED+1))
    val,classes=dacl_items("val",640);require(classes==data["classes"],"Public ontology changed")
    records={"dacl":val,"damsegment":data["split"]["val"],"codebrim":data["code_val"]["items"]}
    loaders={d:DataLoader(FacilityPhotos(rows,640),batch_size=16,num_workers=4,persistent_workers=True,generator=torch.Generator().manual_seed(SEED+100+DOMAINS[d]))for d,rows in records.items()}
    optimizer=optimizer_for(model);updates={"steps":0}
    def updated(*args):updates["steps"]+=1
    hook=optimizer.register_step_post_hook(updated);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=.000005);scaler=torch.amp.GradScaler("cuda")
    training={**metadata(data,split_sha),"status":"running","batch_size":8,"seed":SEED,"requested_epochs":EPOCHS,
        "draws_per_epoch":DRAW_COUNT,"initial_state_transfer":transfer,"initial_weights_sha256":sha(ROOT/INITIAL),
        "source_sha256":protocol["source_sha256"],"source_git_commit":before["source_git_commit"],"study_protocol_path":PROTOCOL,
        "study_protocol_sha256":sha(ROOT/PROTOCOL),"additional_validation":{"path":"data/codebrim-training/val.json","sha256":sha(ROOT/"data/codebrim-training/val.json")},
        "validation_domains":list(records),"validation_counts":{d:len(r)for d,r in records.items()},"backbone_lr":4e-5,"head_lr":1e-4,
        "distillation_weight":4.,"source_test_used":False,"historical_control_retrained":False,"human_feedback_used":False,
        "ai_pseudo_labels_created":0,"original_labels_changed":False,"primary_photo_loss_recipe":protocol["loss_recipe"],
        "original_sampling_unchanged":False,"supplement_manifest_sha256":sha(ROOT/MANIFEST),"local_noncommercial_research_only":True,"new_domain_photo_weights":[1.]*7,"ai_native_observations_used_as_labels":False,"original_photo_positive_weights":data["sampling_data"]["photo_weights"].tolist(),
        "original_pixel_positive_weights":data["supervision_weights"]["pixel_weights"].tolist(),"original_auxiliary_positive_weights":data["supervision_weights"]["auxiliary_weights"].tolist()}
    save(dest/"TRAINING.json",training);history=[];best=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,EPOCHS+1):
        validate_protocol(protocol);sampler.set_epoch(epoch);model.train();apply_frozen_batchnorm(model)
        require(batchnorm_state_sha256(model)==bn_sha and batchnorm_configuration(model)==bn_config,"BN changed")
        counts=torch.zeros((7,3),dtype=torch.long);domains=torch.zeros(4,dtype=torch.long);digest=hashlib.sha256()
        supplement_positive=0;supplement_normal=0
        attempted=0;steps_before=updates["steps"];sum_loss=0.;sums=defaultdict(float);rates=[g["lr"]for g in optimizer.param_groups]
        for batch in loader:
            attempted+=1;digest.update(batch[9].numpy().astype("<i8",copy=False).tobytes());counts+=photo_target_counts(batch[1],batch[2]);domains+=torch.bincount(batch[3],minlength=4)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):loss,components=losses(model,teacher,batch,data)
            supplement_positive+=int(((batch[3]==3)&(batch[1][:,1]==1)).sum());supplement_normal+=int(((batch[3]==3)&(batch[1][:,1]==0)).sum())
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();sum_loss+=float(loss.detach())*len(batch[0])
            for k,v in components.items():sums[k]+=float(v.detach())
            if attempted%200==0:print(json.dumps({"epoch":epoch,"draws":int(domains.sum()),"budget":DRAW_COUNT}),flush=True)
        require(digest.hexdigest()==hashlib.sha256(data["candidate_draws"][epoch-1].astype("<i8").tobytes()).hexdigest()and int(domains.sum())==DRAW_COUNT,"Observed draw order differs")
        require(supplement_positive==supplement_normal==REPLACEMENTS//2 and int(domains[3])==REPLACEMENTS,"Fixed balanced source exposure differs")
        bn_ok=batchnorm_state_sha256(model)==bn_sha and batchnorm_configuration(model)==bn_config
        teacher_ok=state_sha256(teacher.state_dict())==teacher_sha and not teacher.training and all(p.grad is None for p in teacher.parameters());require(bn_ok and teacher_ok,"Frozen state changed")
        predictions={d:predict(model,l,"cuda")for d,l in loaders.items()};require(all(not(t[:,:2]<0).any()for t,p in predictions.values()),"Unknown primary VAL")
        points={c:operating_point({d:(t[:,k].astype(bool),p[:,k])for d,(t,p)in predictions.items()})for k,c in enumerate(TARGETS)}
        worst=max(p["worst_error"]for p in points.values());total_errors=sum(r[k]for p in points.values()for r in p["domains"].values()for k in("fnr","fpr"))
        row={"epoch":epoch,"elapsed_minutes":(time.perf_counter()-start)/60,"train_loss":sum_loss/DRAW_COUNT,"loss_components_mean_per_batch":{k:v/attempted for k,v in sums.items()},
            "worst_target_error":worst,"sum_target_errors":total_errors,"target_passed":all(p["target_passed"]for p in points.values()),"operating_points":points,
            "optimizer_learning_rates":rates,"optimizer_step_diagnostics":{"attempted_batches":attempted,"actual_optimizer_steps":updates["steps"]-steps_before,"amp_skipped_steps":attempted-updates["steps"]+steps_before},
            "sampled_domain_counts":{d:int(domains[k])for d,k in NEW_DOMAINS.items()},"sampled_row_indices_sha256":digest.hexdigest(),
            "sampled_photo_target_counts":{c:{"positive":int(counts[k,0]),"negative":int(counts[k,1]),"unknown":int(counts[k,2])}for k,c in enumerate(classes)},
            "supplement_exposure":{"positive_draws":supplement_positive,"normal_draws":supplement_normal},"teacher_forward_batches":attempted,"teacher_state_unchanged":teacher_ok,"bn_buffers_unchanged":bn_ok,
            "ranking_ap":{d:{c:average_precision(t[:,k],p[:,k])for k,c in enumerate(classes)if(t[:,k]>=0).all()}for d,(t,p)in predictions.items()}}
        history.append(row)
        if best is None or(worst,total_errors)<best:
            best=(worst,total_errors);saved={**metadata(data,split_sha),"epoch":epoch,"selection_split":"val","worst_target_error":worst,"study_protocol_sha256":sha(ROOT/PROTOCOL),"state_dict":{k:v.cpu()for k,v in model.state_dict().items()}}
            torch.save(saved,dest/"best.pt");save(dest/"VALIDATION.json",row)
            for d,(t,p)in predictions.items():save(dest/f"validation-{d}.json",{"split":"val","targets":t.tolist(),"probabilities":p.tolist(),"weights_sha256":sha(dest/"best.pt")})
        save(dest/"history.json",history);print(json.dumps({k:row[k]for k in("epoch","elapsed_minutes","train_loss","worst_target_error","target_passed")}),flush=True);scheduler.step()
    hook.remove();training.update(status="complete",actual_epochs=len(history),weights_sha256=sha(dest/"best.pt"),best_worst_target_error=best[0],
        elapsed_training_minutes=(time.perf_counter()-start)/60,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),initial_bn_buffer_sha256=bn_sha,
        final_bn_buffer_sha256=batchnorm_state_sha256(model),teacher_state_sha256=teacher_sha,final_teacher_state_sha256=state_sha256(teacher.state_dict()),
        attempted_batches=sum(r["optimizer_step_diagnostics"]["attempted_batches"]for r in history),actual_optimizer_steps=updates["steps"],actual_teacher_forward_batches=sum(r["teacher_forward_batches"]for r in history))
    save(dest/"TRAINING.json",training);print(json.dumps({k:training[k]for k in("status","actual_epochs","weights_sha256","best_worst_target_error")}),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--preflight",action="store_true");args=parser.parse_args()
    require(torch.cuda.is_available(),"GPU required");torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    protocol=validate_protocol(read(ROOT/PROTOCOL));data=load_data()
    arrays=np.load(ROOT/'data/idea-research/draws.npz',allow_pickle=False)
    require(np.array_equal(data['candidate_draws'],arrays['candidate'])and np.array_equal(data['epoch_draws'],arrays['control']),'Declared sampling differs')
    (preflight if args.preflight else train)(protocol,data)

if __name__=="__main__":main()
