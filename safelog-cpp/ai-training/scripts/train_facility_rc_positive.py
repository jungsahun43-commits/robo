"""One declared six-epoch RC positive-region package; old runs stay immutable."""
from __future__ import annotations

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
from scripts.facility_rc_positive import (NAME, PROTOCOL, INITIAL, EPOCHS, SEED, DRAW_COUNT,
    NEW_DOMAINS, REPLACEMENTS, StudyPhotos, load_data, validate_protocol, read, sha)
from scripts.facility_subtype_study import FixedEpochSampler
from scripts.train_facility_target import FacilityPhotos, dacl_items, predict, masked_focal, save, TARGETS
from scripts.train_facility_batchnorm import load_initial_state, photo_target_counts
from scripts.train_facility_presence import average_precision
from scripts.facility_error_target import operating_point
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH
from safelog_ai.presence_classifier import MEAN, STD
from safelog_ai.spatial_classifier import spatial_loss
from safelog_ai.positive_region_loss import positive_region_loss
from safelog_ai.frozen_batchnorm import apply_frozen_batchnorm, batchnorm_state_sha256, batchnorm_configuration
from safelog_ai.retention_distillation import masked_bernoulli_kl, state_sha256


def losses(model, teacher, batch, data):
    images, labels, known, domains, masks, pixel_known, auxiliary, aux_known = batch[:8]
    images = images.to("cuda")
    scores, maps, tags = model.forward_training(images)
    with torch.no_grad(): reference = teacher(images)
    weights = torch.cat((data["sampling_data"]["photo_weights"], torch.ones(1, 7))).to("cuda")
    emphasis = torch.tensor([2., 2., 1., 1., 1., 1., 1.], device="cuda")
    components = {
        "photo": masked_focal(scores, labels.to("cuda"), known.to("cuda"), weights[domains.to("cuda")], 1., emphasis),
        "spatial": spatial_loss(maps, masks.to("cuda"), pixel_known.to("cuda"), data["supervision_weights"]["pixel_weights"].to("cuda")),
        "auxiliary": .5 * masked_focal(tags, auxiliary.to("cuda"), aux_known.to("cuda"), data["supervision_weights"]["auxiliary_weights"].to("cuda"), 1.),
        "retention": 4. * masked_bernoulli_kl(scores, reference, known.to("cuda"), temperature=2.),
        "positive_regions": positive_region_loss(maps, batch[10].to("cuda"), (domains == 3).to("cuda"), weight=.25)}
    total = sum(components.values())
    if not torch.isfinite(total): raise ValueError("Nonfinite candidate loss")
    return total, components


def initial_models(data):
    checkpoint = torch.load(ROOT / INITIAL, map_location="cpu", weights_only=True)
    model = AuxiliaryClassifier(7, pretrained=False)
    split_sha = sha(ROOT / "runs/facility-presence-target-head-lr-low/SPLIT.json")
    transfer = load_initial_state(model, checkpoint, data["classes"], split_sha)
    teacher = AuxiliaryClassifier(7, pretrained=False)
    load_initial_state(teacher, checkpoint, data["classes"], split_sha)
    teacher.eval().requires_grad_(False)
    if state_sha256(model.state_dict()) != state_sha256(teacher.state_dict()): raise ValueError("Original full state transfer failed")
    return model.to("cuda"), teacher.to("cuda"), transfer, split_sha


def optimizer_for(model):
    backbone = list(model.backbone.parameters()); ids = {id(p) for p in backbone}
    heads = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{"params": backbone, "lr": .00004}, {"params": heads, "lr": .0001}], weight_decay=.0002)


def preflight(protocol, data):
    destination = ROOT / "runs/facility-rc-positive-preflight"
    if destination.exists(): raise ValueError("Preserve completed preflight")
    destination.mkdir()
    dataset = StudyPhotos(data)
    indices = data["epoch_draws"][0, :4].tolist() + [len(data["items"]) + i for i in range(4)]
    batch = next(iter(DataLoader(dataset, batch_size=8, sampler=indices, num_workers=0)))
    if batch[2][4:, 2:].any() or (batch[1][4:] == 0).logical_and(batch[2][4:].bool()).any():
        raise ValueError("New dataset fabricated a negative/other-class label")
    if batch[5][4:].any() or batch[10][:4].any(): raise ValueError("Core/new spatial supervision mixed")
    model, teacher, transfer, split_sha = initial_models(data)
    original_sha = state_sha256(model.state_dict()); teacher_sha = state_sha256(teacher.state_dict())
    bn_sha = batchnorm_state_sha256(model)
    model.train(); apply_frozen_batchnorm(model)
    optimizer = optimizer_for(model)
    loss, components = losses(model, teacher, batch, data)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients): raise ValueError("Preflight gradients are not finite")
    optimizer.step()
    if state_sha256(model.state_dict()) == original_sha or batchnorm_state_sha256(model) != bn_sha:
        raise ValueError("Preflight did not update trainable weights or changed frozen BN buffers")
    if state_sha256(teacher.state_dict()) != teacher_sha or any(p.grad is not None for p in teacher.parameters()):
        raise ValueError("Frozen teacher changed")
    model.eval()
    with torch.no_grad(): gpu_output = [x.cpu() for x in model.forward_training(batch[0][:2].to("cuda"))]
    saved = {"architecture": ARCH, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
             "classes": data["classes"], "imgsz": 640, "mean": MEAN, "std": STD,
             "auxiliary_classes": list(AUX_CLASSES), "split_sha256": split_sha}
    torch.save(saved, destination / "disposable.pt")
    cpu = AuxiliaryClassifier(7, pretrained=False)
    cpu.load_state_dict(torch.load(destination / "disposable.pt", weights_only=True, map_location="cpu")["state_dict"], strict=True); cpu.eval()
    with torch.no_grad(): cpu_output = cpu.forward_training(batch[0][:2])
    if [list(x.shape) for x in cpu_output] != [[2, 7], [2, 7, 80, 80], [2, 19]]:
        raise ValueError("CPU reload output contract changed")
    if not all(torch.isfinite(x).all() and torch.allclose(x, y, atol=.002, rtol=.002) for x, y in zip(cpu_output, gpu_output)):
        raise ValueError("CPU/GPU disposable reload differs")
    save(destination / "preflight.json", {"status": "passed", "protocol_sha256": sha(ROOT / PROTOCOL),
        "core_samples": 4, "new_samples": 4, "actual_disposable_fp32_updates": 1,
        "actual_training_epochs": 0, "initial_transfer": transfer,
        "new_negative_photo_labels": 0, "new_background_negative_pixels": 0,
        "batch_indices": indices, "loss": float(loss.detach()),
        "loss_components": {k: float(v.detach()) for k, v in components.items()},
        "all_observed_gradients_finite": True, "frozen_bn_preserved": True, "frozen_teacher_preserved": True,
        "cpu_strict_324_state_reload": True, "output_shapes": [list(x.shape) for x in cpu_output]})
    print("TRAIN-only disposable preflight passed; no completed training epoch", flush=True)


def train(protocol, data):
    destination = ROOT / "runs" / NAME
    if destination.exists(): raise ValueError("Preserve existing training run")
    if read(ROOT / "runs/facility-rc-positive-preflight/preflight.json")["protocol_sha256"] != sha(ROOT / PROTOCOL):
        raise ValueError("Preflight/protocol binding changed")
    before = read(ROOT / "runs/facility-rc-positive-before.json")
    if before["protocol_sha256"] != sha(ROOT / PROTOCOL) or before["status"] != "passed":
        raise ValueError("Source/data preservation snapshot required before training")
    destination.mkdir()
    save(destination / "SPLIT.json", data["split"])
    model, teacher, transfer, split_sha = initial_models(data)
    if sha(destination / "SPLIT.json") != split_sha: raise ValueError("Original validation split changed")
    teacher_sha = state_sha256(teacher.state_dict()); bn_sha = batchnorm_state_sha256(model)
    bn_configuration = batchnorm_configuration(model)
    torch.manual_seed(SEED + 2)
    sampler = FixedEpochSampler(data["candidate_draws"])
    loader = DataLoader(StudyPhotos(data), batch_size=8, sampler=sampler, num_workers=4,
                        persistent_workers=True, pin_memory=True, generator=torch.Generator().manual_seed(SEED + 1))
    val, classes = dacl_items("val", 640)
    if classes != data["classes"]: raise ValueError("Original validation ontology changed")
    records = {"dacl": val, "damsegment": data["split"]["val"], "codebrim": data["code_val"]["items"]}
    loaders = {d: DataLoader(FacilityPhotos(rows, 640), batch_size=16, num_workers=4, persistent_workers=True,
                generator=torch.Generator().manual_seed(SEED + 100 + NEW_DOMAINS[d])) for d, rows in records.items()}
    optimizer = optimizer_for(model); count = {"steps": 0}
    def updated(*args): count["steps"] += 1
    hook = optimizer.register_step_post_hook(updated)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=.000005)
    scaler = torch.amp.GradScaler("cuda")
    training = {"status": "running", "architecture": ARCH, "classes": classes, "auxiliary_classes": list(AUX_CLASSES),
        "imgsz": 640, "batch_size": 8, "seed": SEED, "requested_epochs": EPOCHS,
        "draws_per_epoch": DRAW_COUNT, "initial_state_transfer": transfer, "initial_weights_sha256": sha(ROOT / INITIAL),
        "source_sha256": protocol["source_sha256"], "study_protocol_path": PROTOCOL, "study_protocol_sha256": sha(ROOT / PROTOCOL),
        "source_git_commit": before["source_git_commit"], "split_sha256": split_sha,
        "core_spatial_manifest_sha256": sha(data["manifest_path"]), "supplement_manifest_sha256": protocol["input_sha256"]["data/rc2119-training/positive-study/train.json"],
        "backbone_lr": .00004, "head_lr": .0001, "distillation_weight": 4., "positive_region_loss_weight": .25,
        "historical_control_reused": True, "historical_control_retrained": False,
        "data_and_foreground_loss_package": True, "new_background_negative_pixels": 0,
        "new_negative_photo_targets": 0, "new_auxiliary_labels": 0, "original_photo_pixel_auxiliary_targets_changed": False,
        "selected_supplement_photos": len(data["supplement"]), "source_test_used": False,
        "new_photo_weights": [1.] * 7, "original_photo_positive_weights": data["sampling_data"]["photo_weights"].tolist(),
        "original_pixel_positive_weights": data["supervision_weights"]["pixel_weights"].tolist(),
        "original_auxiliary_positive_weights": data["supervision_weights"]["auxiliary_weights"].tolist()}
    save(destination / "TRAINING.json", training)
    history, best = [], None
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    all_items = data["items"] + data["supplement"]
    for epoch in range(1, EPOCHS + 1):
        validate_protocol(protocol)
        sampler.set_epoch(epoch); model.train(); apply_frozen_batchnorm(model)
        if batchnorm_state_sha256(model) != bn_sha or batchnorm_configuration(model) != bn_configuration:
            raise ValueError("Original BN state/configuration changed")
        counts = torch.zeros((7, 3), dtype=torch.long); domains = torch.zeros(4, dtype=torch.long)
        digest = hashlib.sha256(); attempted = 0; steps_before = count["steps"]
        components_sum = defaultdict_float(); positive_cells = 0; sum_loss = 0.
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        for batch in loader:
            attempted += 1
            indices = batch[9].numpy(); digest.update(indices.astype("<i8", copy=False).tobytes())
            counts += photo_target_counts(batch[1], batch[2]); domains += torch.bincount(batch[3], minlength=4)
            positive_cells += int(batch[10].sum())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                loss, components = losses(model, teacher, batch, data)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            sum_loss += float(loss.detach()) * len(batch[0])
            for key, value in components.items(): components_sum[key] += float(value.detach())
            if attempted % 200 == 0:
                print(json.dumps({"epoch": epoch, "draws": int(domains.sum()), "budget": DRAW_COUNT}), flush=True)
        expected_sha = hashlib.sha256(data["candidate_draws"][epoch - 1].astype("<i8").tobytes()).hexdigest()
        if digest.hexdigest() != expected_sha or int(domains.sum()) != DRAW_COUNT or int(domains[3]) != REPLACEMENTS:
            raise ValueError("Actual fixed source exposure/order differs")
        bn_ok = batchnorm_state_sha256(model) == bn_sha and batchnorm_configuration(model) == bn_configuration
        teacher_ok = state_sha256(teacher.state_dict()) == teacher_sha and not teacher.training and all(p.grad is None for p in teacher.parameters())
        if not bn_ok or not teacher_ok: raise ValueError("Frozen BN/teacher invariants failed")
        predictions = {d: predict(model, loader, "cuda") for d, loader in loaders.items()}
        if any((t[:, :2] < 0).any() for t, p in predictions.values()): raise ValueError("Unknown primary source-VAL label")
        points = {label: operating_point({d: (t[:, k].astype(bool), p[:, k]) for d, (t, p) in predictions.items()})
                  for k, label in enumerate(TARGETS)}
        worst = max(p["worst_error"] for p in points.values())
        sum_errors = sum(row[key] for p in points.values() for row in p["domains"].values() for key in ("fnr", "fpr"))
        row = {"epoch": epoch, "elapsed_minutes": (time.perf_counter() - start) / 60,
            "train_loss": sum_loss / DRAW_COUNT, "loss_components_mean_per_batch": {k: v / attempted for k, v in components_sum.items()},
            "worst_target_error": worst, "sum_target_errors": sum_errors,
            "target_passed": all(p["target_passed"] for p in points.values()), "operating_points": points,
            "optimizer_learning_rates": learning_rates, "optimizer_step_diagnostics": {"attempted_batches": attempted,
                "actual_optimizer_steps": count["steps"] - steps_before, "amp_skipped_steps": attempted - (count["steps"] - steps_before)},
            "sampled_domain_counts": {d: int(domains[k]) for d, k in NEW_DOMAINS.items()},
            "sampled_photo_target_counts": {label: {"positive": int(counts[k, 0]), "negative": int(counts[k, 1]), "unknown": int(counts[k, 2])} for k, label in enumerate(classes)},
            "asserted_new_positive_cells_sampled": positive_cells, "sampled_row_indices_sha256": digest.hexdigest(),
            "teacher_forward_batches": attempted, "teacher_state_unchanged": teacher_ok, "bn_buffers_unchanged": bn_ok,
            "ranking_ap": {d: {label: average_precision(t[:, k], p[:, k]) for k, label in enumerate(classes) if (t[:, k] >= 0).all()}
                           for d, (t, p) in predictions.items()}}
        history.append(row)
        if best is None or (worst, sum_errors) < best:
            best = (worst, sum_errors)
            torch.save({"architecture": ARCH, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": classes, "auxiliary_classes": list(AUX_CLASSES), "imgsz": 640, "mean": MEAN, "std": STD,
                "epoch": epoch, "selection_split": "val", "split_sha256": split_sha,
                "worst_target_error": worst, "study_protocol_sha256": sha(ROOT / PROTOCOL)}, destination / "best.pt")
            save(destination / "VALIDATION.json", row)
            for d, (targets, scores) in predictions.items():
                save(destination / f"validation-{d}.json", {"split": "val", "targets": targets.tolist(),
                    "probabilities": scores.tolist(), "weights_sha256": sha(destination / "best.pt")})
        save(destination / "history.json", history)
        print(json.dumps({k: row[k] for k in ("epoch", "elapsed_minutes", "train_loss", "worst_target_error", "target_passed")}), flush=True)
        scheduler.step()
    hook.remove()
    training.update(status="complete", actual_epochs=len(history), weights_sha256=sha(destination / "best.pt"),
        best_worst_target_error=best[0], elapsed_training_minutes=(time.perf_counter() - start) / 60,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), final_bn_buffer_sha256=batchnorm_state_sha256(model),
        initial_bn_buffer_sha256=bn_sha, teacher_state_sha256=teacher_sha, final_teacher_state_sha256=state_sha256(teacher.state_dict()),
        attempted_batches=sum(r["optimizer_step_diagnostics"]["attempted_batches"] for r in history), actual_optimizer_steps=count["steps"],
        actual_teacher_forward_batches=sum(r["teacher_forward_batches"] for r in history))
    save(destination / "TRAINING.json", training)
    print(json.dumps({k: training[k] for k in ("status", "actual_epochs", "weights_sha256", "best_worst_target_error")}), flush=True)


def defaultdict_float():
    from collections import defaultdict
    return defaultdict(float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise ValueError("Declared CUDA experiment requires GPU")
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    protocol = validate_protocol(read(ROOT / PROTOCOL))
    data = load_data()
    if args.preflight: preflight(protocol, data)
    else: train(protocol, data)


if __name__ == "__main__":
    main()
