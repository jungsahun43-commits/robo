"""One fixed six-epoch native19 spatial-supervision package."""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.facility_dense_auxiliary import (NAME, PROTOCOL, INITIAL, EPOCHS, SEED, DRAW_COUNT,
    DENSE_WEIGHT, DensePhotos, load_data, validate_protocol, read, sha, require)
from scripts.facility_subtype_study import FixedEpochSampler
from scripts.train_facility_target import FacilityPhotos, dacl_items, predict, masked_focal, save, TARGETS, DOMAINS
from scripts.train_facility_batchnorm import load_initial_state, photo_target_counts
from scripts.train_facility_presence import average_precision
from scripts.facility_error_target import operating_point
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH as AUX_ARCH
from safelog_ai.dense_auxiliary_classifier import DenseAuxiliaryClassifier, ARCH, load_original_state, original_state, model_inventory
from safelog_ai.presence_classifier import MEAN, STD
from safelog_ai.spatial_classifier import spatial_loss
from safelog_ai.frozen_batchnorm import apply_frozen_batchnorm, batchnorm_state_sha256, batchnorm_configuration
from safelog_ai.retention_distillation import masked_bernoulli_kl, state_sha256


def losses(model, teacher, batch, data, device="cuda"):
    images, labels, known, domains, masks, pixel_known, auxiliary, aux_known = batch[:8]
    scores, maps, tags, dense = model.forward_dense_training(images.to(device))
    with torch.no_grad(): reference = teacher(images.to(device))
    components = {
        "photo": masked_focal(scores, labels.to(device), known.to(device), data["sampling_data"]["photo_weights"].to(device)[domains.to(device)], 1., torch.tensor([2., 2., 1., 1., 1., 1., 1.], device=device)),
        "spatial": spatial_loss(maps, masks.to(device), pixel_known.to(device), data["supervision_weights"]["pixel_weights"].to(device)),
        "auxiliary": .5 * masked_focal(tags, auxiliary.to(device), aux_known.to(device), data["supervision_weights"]["auxiliary_weights"].to(device), 1.),
        "retention": 4. * masked_bernoulli_kl(scores, reference, known.to(device), temperature=2.),
        "dense_auxiliary": DENSE_WEIGHT * spatial_loss(dense, batch[10].to(device), batch[11].to(device), data["dense_weights"].to(device))}
    total = sum(components.values())
    require(torch.isfinite(total), "Nonfinite native19 loss")
    return total, components


def initial_models(data, device="cuda"):
    checkpoint = torch.load(ROOT / INITIAL, map_location="cpu", weights_only=True)
    split_sha = sha(ROOT / "runs/facility-presence-target-head-lr-low/SPLIT.json")
    model = DenseAuxiliaryClassifier(7, pretrained=False, dense_seed=SEED)
    transfer = load_original_state(model, checkpoint, data["classes"], split_sha)
    teacher = AuxiliaryClassifier(7, pretrained=False)
    load_initial_state(teacher, checkpoint, data["classes"], split_sha)
    require(state_sha256(original_state(model)) == state_sha256(teacher.state_dict()), "Original state differs")
    teacher.eval().requires_grad_(False)
    return model.to(device), teacher.to(device), transfer, split_sha


def optimizer_for(model):
    backbone = list(model.backbone.parameters()); ids = {id(p) for p in backbone}
    return torch.optim.AdamW([{"params": backbone, "lr": .00004},
        {"params": [p for p in model.parameters() if id(p) not in ids], "lr": .0001}], weight_decay=.0002)


def checkpoint_metadata(data, split_sha):
    return {"classes": data["classes"], "auxiliary_classes": list(AUX_CLASSES), "imgsz": 640,
            "mean": MEAN, "std": STD, "split_sha256": split_sha}


def save_selected(model, data, split_sha, row, predictions, destination):
    common = checkpoint_metadata(data, split_sha)
    common.update(epoch=row["epoch"], selection_split="val", worst_target_error=row["worst_target_error"],
                  study_protocol_sha256=sha(ROOT / PROTOCOL))
    full = {**common, "architecture": ARCH, "dense_classes": list(AUX_CLASSES), "dense_initialization_seed": SEED,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}
    torch.save(full, destination / "dense-best.pt")
    exported = {**common, "architecture": AUX_ARCH, "state_dict": {k: v.cpu() for k, v in original_state(model).items()},
                "derived_from_dense_weights_sha256": sha(destination / "dense-best.pt"), "training_only_head_removed": True}
    require(len(exported["state_dict"]) == 324, "Export must preserve original324 names")
    torch.save(exported, destination / "best.pt")
    save(destination / "VALIDATION.json", row)
    for domain, (targets, scores) in predictions.items():
        save(destination / f"validation-{domain}.json", {"split": "val", "targets": targets.tolist(),
            "probabilities": scores.tolist(), "weights_sha256": sha(destination / "best.pt")})


def preflight(protocol, data):
    destination = ROOT / "runs/facility-dense-auxiliary-preflight"
    require(not destination.exists(), "Preserve completed preflight")
    destination.mkdir()
    eligible = [i for i, r in enumerate(data["items"][:data["full_count"]]) if r["image"] in data["dense"]]
    others = [i for i, r in enumerate(data["items"]) if r["domain"] != "dacl"]
    crops = list(range(data["full_count"], len(data["items"])))
    indices = eligible[:4] + others[:2] + crops[:2]
    batch = next(iter(DataLoader(DensePhotos(data), batch_size=8, sampler=indices, num_workers=0)))
    require(batch[11][4:].count_nonzero() == 0 and batch[12].tolist() == [1]*4+[0]*4, "Unknown dense samples fabricated")
    model, teacher, transfer, split_sha = initial_models(data)
    initial = state_sha256(model.state_dict()); teacher_sha = state_sha256(teacher.state_dict()); bn_sha = batchnorm_state_sha256(model)
    model.train(); apply_frozen_batchnorm(model); optimizer = optimizer_for(model)
    loss, components = losses(model, teacher, batch, data); loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    require(gradients and all(torch.isfinite(g).all() for g in gradients), "Preflight gradient invalid")
    require(model.dense_auxiliary_head.projection.weight.grad.count_nonzero() > 0, "Dense projection gradient missing")
    optimizer.step()
    require(initial != state_sha256(model.state_dict()) and bn_sha == batchnorm_state_sha256(model), "Preflight update/BN invalid")
    require(teacher_sha == state_sha256(teacher.state_dict()) and all(p.grad is None for p in teacher.parameters()), "Teacher changed")
    saved = {**checkpoint_metadata(data, split_sha), "architecture": ARCH,
             "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}
    torch.save(saved, destination / "disposable.pt")
    cpu = DenseAuxiliaryClassifier(7, pretrained=False); cpu.load_state_dict(torch.load(destination / "disposable.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True); cpu.eval(); model.eval()
    # Strict-FP32 cross-device diagnostic only. Actual training uses unchanged
    # installed flags in a separate process, matching the existing baseline.
    previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            gpu = [x.cpu() for x in model.forward_dense_training(batch[0][:2].to("cuda"))]
            outputs = cpu.forward_dense_training(batch[0][:2])
        require([list(x.shape) for x in outputs] == [[2,7],[2,7,80,80],[2,19],[2,19,80,80]]
                and all(torch.isfinite(x).all() and torch.allclose(x, y, atol=.002, rtol=.002) for x, y in zip(outputs, gpu)), "CPU/GPU strict reload differs")
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous
    save(destination / "preflight.json", {"status": "passed", "protocol_sha256": sha(ROOT / PROTOCOL),
        "actual_disposable_fp32_updates": 1, "actual_training_epochs": 0, "initial_transfer": transfer,
        "batch_indices": indices, "loss": float(loss.detach()), "loss_components": {k:float(v.detach()) for k,v in components.items()},
        "strict_cpu_328_state_reload": True, "dense_projection_actual_gradient": True, "unknown_dense_supervision_preserved": True,
        "frozen_teacher_and_bn_preserved": True, "strict_fp32_diagnostic_only_flags_restored": True,
        "output_shapes": [list(x.shape) for x in outputs]})
    print("TRAIN-only one disposable update passed; zero completed epochs", flush=True)


def train(protocol, data):
    destination = ROOT / "runs" / NAME
    require(not destination.exists(), "Preserve existing run")
    require(read(ROOT / "runs/facility-dense-auxiliary-preflight/preflight.json")["protocol_sha256"] == sha(ROOT / PROTOCOL), "Preflight binding changed")
    before = read(ROOT / "runs/facility-dense-auxiliary-before.json")
    require(before["status"] == "passed" and before["protocol_sha256"] == sha(ROOT / PROTOCOL), "Committed-source preservation proof required")
    destination.mkdir(); save(destination / "SPLIT.json", data["split"])
    model, teacher, transfer, split_sha = initial_models(data)
    require(sha(destination / "SPLIT.json") == split_sha, "Original split changed")
    teacher_sha, bn_sha, bn_config = state_sha256(teacher.state_dict()), batchnorm_state_sha256(model), batchnorm_configuration(model)
    inventory = model_inventory(model); torch.manual_seed(SEED+2)
    sampler = FixedEpochSampler(data["epoch_draws"])
    loader = DataLoader(DensePhotos(data), batch_size=8, sampler=sampler, num_workers=4, persistent_workers=True,
        pin_memory=True, generator=torch.Generator().manual_seed(SEED+1))
    val, classes = dacl_items("val",640)
    require(classes == data["classes"], "Public ontology changed")
    records = {"dacl":val, "damsegment":data["split"]["val"], "codebrim":data["code_val"]["items"]}
    loaders = {d:DataLoader(FacilityPhotos(rows,640), batch_size=16, num_workers=4, persistent_workers=True,
        generator=torch.Generator().manual_seed(SEED+100+DOMAINS[d])) for d,rows in records.items()}
    optimizer = optimizer_for(model); count = {"steps":0}
    def updated(*args): count["steps"] += 1
    hook = optimizer.register_step_post_hook(updated)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=.000005)
    scaler = torch.amp.GradScaler("cuda")
    training = {"status":"running", "architecture":AUX_ARCH, "training_model_architecture":ARCH,
        "classes":classes, "auxiliary_classes":list(AUX_CLASSES), "dense_classes":list(AUX_CLASSES), "imgsz":640,
        "batch_size":8, "seed":SEED, "requested_epochs":EPOCHS, "draws_per_epoch":DRAW_COUNT,
        "initial_state_transfer":transfer, "initial_weights_sha256":sha(ROOT/INITIAL), "model_inventory":inventory,
        "source_sha256":protocol["source_sha256"], "source_git_commit":before["source_git_commit"],
        "study_protocol_path":PROTOCOL, "study_protocol_sha256":sha(ROOT/PROTOCOL), "split_sha256":split_sha,
        "additional_validation":{"path":"data/codebrim-training/val.json", "sha256":sha(ROOT/"data/codebrim-training/val.json")},
        "validation_domains":["dacl","damsegment","codebrim"], "validation_counts":{d:len(r) for d,r in records.items()},
        "backbone_lr":.00004, "head_lr":.0001, "distillation_weight":4., "dense_spatial_loss_weight":DENSE_WEIGHT,
        "dense_positive_weights":data["dense_weights"].tolist(), "source_test_used":False, "historical_control_retrained":False,
        "original_sampling_and_targets_unchanged":True, "original_photo_positive_weights":data["sampling_data"]["photo_weights"].tolist(),
        "original_pixel_positive_weights":data["supervision_weights"]["pixel_weights"].tolist(),
        "original_auxiliary_positive_weights":data["supervision_weights"]["auxiliary_weights"].tolist()}
    save(destination/"TRAINING.json",training); history=[]; best=None
    start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,EPOCHS+1):
        validate_protocol(protocol); sampler.set_epoch(epoch); model.train(); apply_frozen_batchnorm(model)
        require(batchnorm_state_sha256(model)==bn_sha and batchnorm_configuration(model)==bn_config,"BN changed")
        counts=torch.zeros((7,3),dtype=torch.long); domains=torch.zeros(3,dtype=torch.long)
        dense_positive=torch.zeros(19,dtype=torch.long); dense_known=torch.zeros(19,dtype=torch.long)
        digest=hashlib.sha256(); attempted=0; eligible_count=0; steps_before=count["steps"]; sum_loss=0.; sums=defaultdict(float)
        learning_rates=[g["lr"] for g in optimizer.param_groups]
        for batch in loader:
            attempted+=1; digest.update(batch[9].numpy().astype("<i8",copy=False).tobytes())
            counts+=photo_target_counts(batch[1],batch[2]); domains+=torch.bincount(batch[3],minlength=3)
            eligible_count+=int(batch[12].sum()); dense_known+=batch[11].sum(0).long()
            dense_positive+=(batch[10].sum((2,3))*batch[11]).sum(0).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"): loss,components=losses(model,teacher,batch,data)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            sum_loss+=float(loss.detach())*len(batch[0])
            for k,v in components.items(): sums[k]+=float(v.detach())
            if attempted%200==0: print(json.dumps({"epoch":epoch,"draws":int(domains.sum()),"budget":DRAW_COUNT}),flush=True)
        require(digest.hexdigest()==hashlib.sha256(data["epoch_draws"][epoch-1].astype("<i8").tobytes()).hexdigest()
                and int(domains.sum())==DRAW_COUNT and eligible_count==protocol["eligible_draws_by_epoch"][epoch-1],"Draw order/dense exposure differs")
        bn_ok=batchnorm_state_sha256(model)==bn_sha and batchnorm_configuration(model)==bn_config
        teacher_ok=state_sha256(teacher.state_dict())==teacher_sha and not teacher.training and all(p.grad is None for p in teacher.parameters())
        require(bn_ok and teacher_ok,"Frozen state changed")
        predictions={d:predict(model,loader,"cuda") for d,loader in loaders.items()}
        require(all(not(t[:,:2]<0).any() for t,p in predictions.values()),"Unknown primaryVAL")
        points={c:operating_point({d:(t[:,k].astype(bool),p[:,k]) for d,(t,p) in predictions.items()}) for k,c in enumerate(TARGETS)}
        worst=max(p["worst_error"] for p in points.values()); sum_errors=sum(r[k] for p in points.values() for r in p["domains"].values() for k in("fnr","fpr"))
        row={"epoch":epoch,"elapsed_minutes":(time.perf_counter()-start)/60,"train_loss":sum_loss/DRAW_COUNT,
            "loss_components_mean_per_batch":{k:v/attempted for k,v in sums.items()},"worst_target_error":worst,"sum_target_errors":sum_errors,
            "target_passed":all(p["target_passed"] for p in points.values()),"operating_points":points,"optimizer_learning_rates":learning_rates,
            "optimizer_step_diagnostics":{"attempted_batches":attempted,"actual_optimizer_steps":count["steps"]-steps_before,"amp_skipped_steps":attempted-count["steps"]+steps_before},
            "sampled_domain_counts":{d:int(domains[k]) for d,k in DOMAINS.items()},"sampled_row_indices_sha256":digest.hexdigest(),
            "sampled_photo_target_counts":{c:{"positive":int(counts[k,0]),"negative":int(counts[k,1]),"unknown":int(counts[k,2])} for k,c in enumerate(classes)},
            "sampled_dense_eligible_rows":eligible_count,"sampled_dense_known_class_rows":dense_known.tolist(),"sampled_dense_positive_cells":dense_positive.tolist(),
            "teacher_forward_batches":attempted,"teacher_state_unchanged":teacher_ok,"bn_buffers_unchanged":bn_ok,
            "ranking_ap":{d:{c:average_precision(t[:,k],p[:,k]) for k,c in enumerate(classes) if(t[:,k]>=0).all()} for d,(t,p) in predictions.items()}}
        history.append(row)
        if best is None or(worst,sum_errors)<best:
            best=(worst,sum_errors); save_selected(model,data,split_sha,row,predictions,destination)
        save(destination/"history.json",history); print(json.dumps({k:row[k] for k in("epoch","elapsed_minutes","train_loss","worst_target_error","target_passed")}),flush=True)
        scheduler.step()
    hook.remove()
    training.update(status="complete",actual_epochs=len(history),weights_sha256=sha(destination/"best.pt"),dense_weights_sha256=sha(destination/"dense-best.pt"),
        best_worst_target_error=best[0],elapsed_training_minutes=(time.perf_counter()-start)/60,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        initial_bn_buffer_sha256=bn_sha,final_bn_buffer_sha256=batchnorm_state_sha256(model),teacher_state_sha256=teacher_sha,final_teacher_state_sha256=state_sha256(teacher.state_dict()),
        attempted_batches=sum(r["optimizer_step_diagnostics"]["attempted_batches"] for r in history),actual_optimizer_steps=count["steps"],
        actual_teacher_forward_batches=sum(r["teacher_forward_batches"] for r in history))
    save(destination/"TRAINING.json",training); print(json.dumps({k:training[k] for k in("status","actual_epochs","weights_sha256","best_worst_target_error")}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--preflight",action="store_true"); args=parser.parse_args()
    require(torch.cuda.is_available(),"GPU required"); torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    protocol=validate_protocol(read(ROOT/PROTOCOL)); data=load_data()
    require(data["dense_weights"].tolist()==protocol["dense_positive_weights"]
            and data["dense_counts"]==protocol["dense_train_pixel_counts"],"Declared dense TRAIN weights/counts changed")
    (preflight if args.preflight else train)(protocol,data)


if __name__=="__main__": main()
