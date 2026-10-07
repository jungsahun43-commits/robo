"""Learn zero-initialized corrections from fixed pretrained semantic features."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".config/torch"))

from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH as AUX_ARCH
from safelog_ai.semantic_residual_classifier import (SemanticResidualClassifier, ARCH as SEMANTIC_ARCH,
    load_original_state, load_semantic_pretrained, semantic_encoder_state_sha256, model_inventory)
from safelog_ai.presence_classifier import MEAN, STD
from safelog_ai.spatial_classifier import spatial_loss
from scripts.train_facility_target import (DOMAINS, TARGETS, FacilityPhotos, dacl_items,
                                           masked_focal, predict, read, save, sha)
from scripts.train_facility_presence import average_precision
from scripts.facility_error_target import operating_point
from scripts.facility_resolution_study import ResolutionPhotos, loss_grid
from scripts.facility_semantic_study import prepare_data, validate_protocol, RECIPE, FixedEpochSampler
from scripts.train_facility_batchnorm import load_initial_state, photo_target_counts
from safelog_ai.frozen_batchnorm import (apply_frozen_batchnorm, batchnorm_configuration,
    batchnorm_inventory, batchnorm_modules, batchnorm_state_sha256)
from safelog_ai.retention_distillation import masked_bernoulli_kl, state_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--variant", choices=("semantic",), required=True)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--auxiliary-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=56)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--draws-per-epoch", type=int, default=14248)
    parser.add_argument("--backbone-lr", type=float, default=.00004)
    parser.add_argument("--head-lr", type=float, default=.0001)
    parser.add_argument("--auxiliary-weight", type=float, default=.5)
    args = parser.parse_args(argv)
    args.study_protocol = args.study_protocol.resolve()
    args.initial = args.initial.resolve(); args.auxiliary_manifest = args.auxiliary_manifest.resolve()
    if args.study_protocol != (ROOT / "reports/facility-semantic-study-protocol.json").resolve():
        raise ValueError("Use the predeclared local semantic-feature protocol")
    protocol = read(args.study_protocol)
    recipe = validate_protocol(protocol, ROOT, args)
    size, architecture = recipe["imgsz"], recipe["architecture"]
    if architecture != SEMANTIC_ARCH or size != 640:
        raise ValueError("Use the declared semantic feature model with 640 input")
    output = ROOT / "runs" / args.name
    if output.exists():
        raise ValueError("Preserve existing study evidence; do not overwrite a run")
    if not torch.cuda.is_available():
        raise ValueError("The declared actual AMP retention study requires CUDA")
    torch.set_num_threads(4)
    print(__import__("json").dumps({"variant":args.variant,"event":"validating_original_inputs"}), flush=True)
    data = prepare_data(ROOT, protocol, args.variant, args.auxiliary_manifest)
    items, classes, full = data["items"], data["classes"], data["full_count"]
    seed, epochs = args.seed, args.epochs
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    output.mkdir(parents=True, exist_ok=False)
    save(output / "SPLIT.json", data["split"])
    split_sha = sha(output / "SPLIT.json")
    checkpoint = torch.load(args.initial, map_location="cpu", weights_only=True)
    model = SemanticResidualClassifier(len(classes), pretrained=False)
    initial_transfer = load_original_state(model, checkpoint, classes, split_sha)
    semantic_transfer = load_semantic_pretrained(model, ROOT / protocol["semantic_pretrained_weights_path"])
    semantic_initial_sha = semantic_encoder_state_sha256(model)
    if semantic_initial_sha != protocol["semantic_encoder_state_sha256"]:
        raise ValueError("Pinned pretrained feature encoder differs")
    semantic_inventory = model_inventory(model)
    batchnorm_initial_sha = batchnorm_state_sha256(model)
    if batchnorm_initial_sha != protocol["initial_batchnorm_buffers_sha256"]:
        raise ValueError("Student BN buffers differ from the original initializer")
    batchnorm_initial_configuration = batchnorm_configuration(model)
    teacher = AuxiliaryClassifier(len(classes), pretrained=False)
    load_initial_state(teacher, checkpoint, classes, split_sha)
    teacher.eval().requires_grad_(False)
    teacher_initial_sha = state_sha256(teacher.state_dict())
    if teacher_initial_sha != protocol["teacher_state_sha256"]:
        raise ValueError("Teacher tensor state differs from the predeclared original checkpoint")
    del checkpoint
    model = model.to("cuda"); teacher = teacher.to("cuda")
    distillation_weight = RECIPE["weight_by_variant"][args.variant]
    torch.manual_seed(seed + 2)
    sampling_data = data["sampling_data"]
    labels, domains, sampling = (sampling_data[key] for key in ("labels", "domains", "sampling"))
    photo_weights = sampling_data["photo_weights"].to("cuda")
    pixel_weights = data["supervision_weights"]["pixel_weights"].to("cuda")
    auxiliary_weights = data["supervision_weights"]["auxiliary_weights"].to("cuda")
    sampler = FixedEpochSampler(data["epoch_draws"])
    loader = DataLoader(ResolutionPhotos(items, data["auxiliary"], full, DOMAINS, imgsz=size),
                        batch_size=args.batch, sampler=sampler, num_workers=4,
                        persistent_workers=True, pin_memory=True,
                        generator=torch.Generator().manual_seed(seed + 1))
    val, val_classes = dacl_items("val", size)
    if val_classes != classes:
        raise ValueError("Frozen DACL validation class order changed")
    loaders = {domain: DataLoader(FacilityPhotos(records, size), batch_size=16,
                    num_workers=4, persistent_workers=True,
                    generator=torch.Generator().manual_seed(seed + 100 + DOMAINS[domain]))
               for domain, records in {"dacl": val, "damsegment": data["split"]["val"],
                                        "codebrim": data["code_val"]["items"]}.items()}
    backbone = list(model.backbone.parameters()); backbone_ids = {id(p) for p in backbone}
    head = [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.backbone_lr},
                                  {"params": head, "lr": args.head_lr}], weight_decay=.0002)
    updates = {"count": 0}
    def record_optimizer_step(optimizer, args, kwargs):
        updates["count"] += 1
    step_hook = optimizer.register_step_post_hook(record_optimizer_step)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.000005)
    scaler = torch.amp.GradScaler("cuda")
    emphasis = torch.ones(len(classes), device="cuda")
    target_indices = [classes.index(label) for label in TARGETS]
    for index in target_indices:
        emphasis[index] = 2.
    training = {"status": "running", "architecture": architecture, "classes": classes,
        "imgsz": size, "batch_size": args.batch, "validation_batch_size": 16,
        "seed": seed, "requested_epochs": epochs, "patience": args.patience,
        "model_variant": args.variant, "initial_state_transfer": initial_transfer,
        "sampling_intervention_applied": False,
        "distillation_weight": distillation_weight,
        "distillation_recipe": dict(RECIPE),
        "paired_draws_key": "control",
        "original_sampling_weights_changed": False,
        "train_dacl": len(data["original"]), "train_damsegment": len(data["split"]["train"]),
        "train_codebrim": len(data["code_train"]["items"]), "val_dacl": len(val),
        "val_damsegment": len(data["split"]["val"]), "validation_domains": list(loaders),
        "draws_per_epoch": args.draws_per_epoch, "split_sha256": split_sha,
        "initial_weights_sha256": sha(args.initial), "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr, "auxiliary_weight": args.auxiliary_weight,
        "photo_positive_weights": photo_weights.detach().cpu().tolist(),
        "pixel_positive_weights": pixel_weights.detach().cpu().tolist(),
        "auxiliary_positive_weights": auxiliary_weights.detach().cpu().tolist(),
        "domain_proportions": protocol["domain_proportions"],
        "loader_randomness": protocol["loader_randomness"],
        "spatial_manifest_sha256": sha(data["manifest_path"]),
        "core_spatial_manifest_sha256": protocol["core_spatial_manifest_sha256"],
        "spatial_manifest_path": data["manifest_path"].relative_to(ROOT).as_posix(),
        "spatial_manifest_audit": data["manifest"]["audit"],
        "auxiliary_manifest_sha256": sha(args.auxiliary_manifest),
        "auxiliary_manifest_path": args.auxiliary_manifest.relative_to(ROOT).as_posix(),
        "auxiliary_manifest_audit": data["auxiliary_manifest"]["audit"],
        "additional_validation": {"path": "data/codebrim-training/val.json",
                                  "sha256": sha(ROOT / "data/codebrim-training/val.json")},
        "additional_test": {"path": "data/codebrim-training/test.json",
                            "sha256": sha(ROOT / "data/codebrim-training/test.json")},
        "auxiliary_classes": list(AUX_CLASSES), "auxiliary_source_class_count": len(AUX_CLASSES),
        "training_script_sha256": sha(Path(__file__)),
        "source_parent_training_script_sha256": sha(ROOT / "scripts/train_facility_context.py"),
        "model_source_sha256": sha(ROOT / "safelog_ai/spatial_classifier.py"),
        "auxiliary_model_source_sha256": sha(ROOT / "safelog_ai/auxiliary_classifier.py"),
        "retention_study": True,
        "retention_strength_study": False,
        "batchnorm_study": True,
        "head_lr_study": False,
        "semantic_feature_study": True,
        "semantic_features_policy": dict(protocol["semantic_features_policy"]),
        "semantic_model_inventory": semantic_inventory,
        "semantic_pretrained_transfer": semantic_transfer,
        "semantic_model_source_sha256": sha(ROOT / "safelog_ai/semantic_residual_classifier.py"),
        "optimizer_schedule": dict(protocol["optimizer_schedule"]),
        "batchnorm_policy": dict(protocol["batchnorm_policy"]),
        "historical_control_reused": True,
        "reused_control_retrained": False,
        "sampler_plan_sha256": protocol["sampler_plan_sha256"],
        "private_draw_archive_sha256": protocol["private_draw_archive_sha256"],
        "paired_label_preservation": data["pair_proof"],
        "model_factory_source_sha256": sha(ROOT / "safelog_ai/research_presence.py"),
        "source_sha256": dict(protocol["source_sha256"]),
        "study_protocol_sha256": sha(args.study_protocol),
        "study_protocol_path": args.study_protocol.relative_to(ROOT).as_posix(),
        "photo_pooling_grid": [80, 80],
        "source_pixel_target_grid": [80, 80],
        "raw_spatial_grid": [80, 80],
        "loss": "Original masked photo/spatial/auxiliary losses + known-other-five frozen-teacher Bernoulli KL",
        "total_loss_formula": "masked_focal(photo,gamma=1,crack/spall emphasis=2)+spatial_loss(original80grid)+auxiliary_weight*masked_focal(native19,gamma=1)+distillation_weight*T^2*masked_Bernoulli_KL(teacher||student,known_other5,T=2)",
        "target_ranking_weight": 0.,
        "target_ranking": {"weight": 0., "classes": list(TARGETS), "sampling_changed": False,
                           "new_labels_asserted": 0, "public_outputs_changed": False},
        "selection": "Minimax crack/spall FNR/FPR across the same three VAL domains; no TEST selection",
        "budget_scope": "Same original initializer base state, images, labels, draw order, frozen BN, teacher/losses, learning rates, batch and six epochs; reuse completed head1e-4 control. Candidate adds fixed ImageNet ConvNeXt features and three learned zero-initialized logit heads. Epoch/draw budget is matched; parameter count, encoder forwards, wall time and FLOPs are not matched.",
        "retention_scope": "Frozen original teacher probabilities regularize known original other-five photo classes; teacher output is not new ground truth",
        "new_photo_targets": 0, "new_pixel_targets": 0, "app_model_promoted": False}
    training["expected_sampling"] = {d: {"full": float(sampling[(domains == k) & (torch.arange(len(items)) < full)].sum() / sampling.sum()),
                                            "crop": float(sampling[(domains == k) & (torch.arange(len(items)) >= full)].sum() / sampling.sum())}
                                     for d, k in DOMAINS.items()}
    training["expected_label_sampling_scope"] = "Identical original baseline weights and exact draw arrays for both arms"
    training["expected_label_sampling"] = {label: {"positive": float(sampling[labels[:, k] == 1].sum() / sampling.sum()),
                    "negative": float(sampling[labels[:, k] == 0].sum() / sampling.sum()),
                    "unknown": float(sampling[labels[:, k] < 0].sum() / sampling.sum())}
                    for k, label in enumerate(classes)}
    save(output / "TRAINING.json", training)
    history = []; best = None; bad = 0; start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        optimizer_learning_rates = {"backbone": float(optimizer.param_groups[0]["lr"]),
                                    "head": float(optimizer.param_groups[1]["lr"])}
        floor = protocol["optimizer_schedule"]["eta_min"]
        factor = (1 + math.cos(math.pi * (epoch - 1) / epochs)) / 2
        expected_rates = {"backbone": floor + (args.backbone_lr - floor) * factor,
            "head": floor + (args.head_lr - floor) * factor}
        if any(not math.isclose(optimizer_learning_rates[key], expected_rates[key],
                rel_tol=1e-12, abs_tol=1e-15) for key in expected_rates):
            raise ValueError("Observed optimizer LR differs from the declared candidate cosine curve")
        sampler.set_epoch(epoch)
        epoch_indices = data["epoch_draws"][epoch-1]
        expected_draw_sha = hashlib.sha256(epoch_indices.astype("<i8", copy=False).tobytes()).hexdigest()
        kd_total = 0.; teacher_batches = 0; kd_contributing = 0; kd_known_entries = 0
        teacher.eval()
        target_counts = torch.zeros((len(classes),3),dtype=torch.long)
        for parameter in backbone:
            parameter.requires_grad = True
        model.train()
        if (any(module.training for module in model.semantic_encoder.modules())
                or any(p.requires_grad for p in model.semantic_encoder.parameters())):
            raise ValueError("The semantic encoder must remain fixed/eval after model.train")
        semantic_forwards = {"count": 0}
        def semantic_forwarded(module, inputs, outputs): semantic_forwards["count"] += 1
        semantic_hook = model.semantic_encoder.register_forward_hook(semantic_forwarded)
        batchnorm_policy_inventory = apply_frozen_batchnorm(model)
        if (batchnorm_policy_inventory["layer_count"] != 47
                or batchnorm_configuration(model) != batchnorm_initial_configuration
                or batchnorm_state_sha256(model) != batchnorm_initial_sha):
            raise ValueError("Declared BN policy/configuration/buffers differ before training")
        total = 0.
        draws = torch.zeros(len(DOMAINS), dtype=torch.long); row_types = torch.zeros(2, dtype=torch.long)
        row_hash = hashlib.sha256(); epoch_update_start = updates["count"]; attempted_batches = 0
        joint = {d: {state: 0 for state in ("00", "10", "01", "11", "unknown")} for d in DOMAINS}
        for number, batch in enumerate(loader, 1):
            attempted_batches += 1
            image, label, known, domain, mask, pixel_known = batch[:6]
            row_types += torch.bincount(batch[-2], minlength=2)
            batch_indices = batch[-1].numpy()
            row_hash.update(batch_indices.astype("<i8", copy=False).tobytes())
            target_counts += photo_target_counts(label, known)
            for d, k in DOMAINS.items():
                eligible = (domain == k) & (batch[-2] == 0)
                target_known = known[:, target_indices].all(1)
                joint[d]["unknown"] += int((eligible & ~target_known).sum())
                for state in ("00", "10", "01", "11"):
                    chosen = eligible & target_known
                    for j, index in enumerate(target_indices):
                        chosen &= label[:, index] == int(state[j])
                    joint[d][state] += int(chosen.sum())
            draws += torch.bincount(domain, minlength=len(DOMAINS))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                image_cuda = image.to("cuda")
                scores, maps, tag_scores = model.forward_training(image_cuda)
                with torch.no_grad():
                    teacher_scores = teacher(image_cuda)
                teacher_batches += 1
                kd = masked_bernoulli_kl(scores, teacher_scores, known.to("cuda"), temperature=2.)
                entries = int(known[:, 2:].sum())
                kd_known_entries += entries; kd_contributing += int(entries > 0)
                kd_total += float(kd.detach())
                if tuple(maps.shape) != (len(image), len(classes), 80, 80):
                    raise ValueError("Both photo pooling and spatial supervision must keep the declared 80-cell grid")
                maps = loss_grid(maps, mask.shape)
                loss = masked_focal(scores, label.to("cuda"), known.to("cuda"), photo_weights[domain], 1., emphasis)
                loss += spatial_loss(maps, mask.to("cuda"), pixel_known.to("cuda"), pixel_weights)
                loss += args.auxiliary_weight * masked_focal(tag_scores, batch[6].to("cuda"), batch[7].to("cuda"), auxiliary_weights, 1.)
                loss += distillation_weight * kd
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite retention training loss")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += loss.item() * len(image)
            if number % 200 == 0:
                print(__import__("json").dumps({"variant": args.variant, "epoch": epoch,
                        "sampled_so_far": int(draws.sum()), "budget": args.draws_per_epoch}), flush=True)
        semantic_hook.remove()
        semantic_heads = {"map": model.semantic_map_head, "photo": model.semantic_photo_head,
            "aux": model.semantic_auxiliary_head}
        semantic_gradients = [p.grad for module in semantic_heads.values() for p in module.parameters()]
        semantic_epoch_audit = {
            "encoder_state_sha256": semantic_encoder_state_sha256(model),
            "encoder_state_unchanged": semantic_encoder_state_sha256(model) == semantic_initial_sha,
            "encoder_all_eval": all(not module.training for module in model.semantic_encoder.modules()),
            "encoder_parameters_frozen": all(not p.requires_grad for p in model.semantic_encoder.parameters()),
            "encoder_gradients_absent": all(p.grad is None for p in model.semantic_encoder.parameters()),
            "semantic_head_gradient_tensors": sum(g is not None for g in semantic_gradients),
            "semantic_head_gradient_nonzero_tensors": sum(g is not None and bool((g != 0).any()) for g in semantic_gradients),
            "semantic_head_gradients_finite": all(g is not None and bool(torch.isfinite(g).all()) for g in semantic_gradients),
            "semantic_heads_changed_from_zero": {key: any(bool((p != 0).any()) for p in module.parameters())
                for key, module in semantic_heads.items()},
            "encoder_forward_batches": semantic_forwards["count"],
        }
        if (not all(semantic_epoch_audit[k] for k in ("encoder_state_unchanged", "encoder_all_eval",
                    "encoder_parameters_frozen", "encoder_gradients_absent", "semantic_head_gradients_finite"))
                or semantic_epoch_audit["semantic_head_gradient_tensors"] != 6
                or semantic_epoch_audit["semantic_head_gradient_nonzero_tensors"] < 1
                or not all(semantic_epoch_audit["semantic_heads_changed_from_zero"].values())
                or semantic_forwards["count"] != attempted_batches):
            raise ValueError("Fixed semantic features and learned-head TRAIN proof failed")
        # Audit the final batch's unscaled gradients. Early AMP skips remain
        # recorded separately and do not assert every preceding gradient finite.
        batchnorm_epoch_inventory = batchnorm_inventory(model)
        affine_gradients = [p.grad for _, module in batchnorm_modules(model) for p in (module.weight, module.bias)]
        batchnorm_epoch_audit = {"buffer_sha256": batchnorm_state_sha256(model),
            "buffers_unchanged": batchnorm_state_sha256(model) == batchnorm_initial_sha,
            "actual_eval_layers": sum(not module.training for _, module in batchnorm_modules(model)),
            "affine_trainable_tensors": sum(p.requires_grad for _, module in batchnorm_modules(model) for p in (module.weight, module.bias)),
            "affine_gradient_tensors": sum(gradient is not None for gradient in affine_gradients),
            "affine_gradient_nonzero_tensors": sum(gradient is not None and bool((gradient != 0).any()) for gradient in affine_gradients),
            "affine_gradients_finite": all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in affine_gradients),
            "policy_applied_after_model_train": True}
        if (not batchnorm_epoch_audit["buffers_unchanged"] or batchnorm_epoch_audit["actual_eval_layers"] != 47
                or batchnorm_epoch_audit["affine_trainable_tensors"] != 94
                or batchnorm_epoch_audit["affine_gradient_tensors"] != 94
                or batchnorm_epoch_audit["affine_gradient_nonzero_tensors"] < 1
                or not batchnorm_epoch_audit["affine_gradients_finite"]
                or batchnorm_configuration(model) != batchnorm_initial_configuration):
            raise ValueError("Student frozen BN buffers/configuration/trainable affine audit failed")
        predictions = {d: predict(model, validation_loader, "cuda") for d, validation_loader in loaders.items()}
        points = {label: operating_point({d: (t[:, classes.index(label)].astype(bool), p[:, classes.index(label)])
                         for d, (t, p) in predictions.items()}) for label in TARGETS}
        worst = max(point["worst_error"] for point in points.values())
        sum_errors = sum(row[key] for point in points.values() for row in point["domains"].values() for key in ("fnr", "fpr"))
        row = {"epoch": epoch, "elapsed_minutes": (time.perf_counter() - start) / 60,
            "optimizer_learning_rates": optimizer_learning_rates,
            "train_loss": total / args.draws_per_epoch, "worst_target_error": worst,
            "sum_target_errors": sum_errors, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "optimizer_step_diagnostics": {"attempted_batches": attempted_batches,
                  "actual_optimizer_steps": updates["count"] - epoch_update_start,
                  "amp_skipped_steps": attempted_batches - (updates["count"] - epoch_update_start)},
            "target_passed": all(point["target_passed"] for point in points.values()),
            "operating_points": points,
            "sampled_domain_counts": {d: int(draws[k]) for d, k in DOMAINS.items()},
            "sampled_row_type_counts": {"full": int(row_types[0]), "crop": int(row_types[1])},
            "sampled_full_target_joint_counts": joint, "sampled_row_indices_sha256": row_hash.hexdigest(),
            "batchnorm_training": batchnorm_epoch_audit,
            "semantic_features_training": semantic_epoch_audit,
            "sampled_photo_target_counts": {label:{"positive":int(target_counts[k,0]),"negative":int(target_counts[k,1]),"unknown":int(target_counts[k,2])} for k,label in enumerate(classes)},
            "fixed_sampling": {"changed_positions_from_control": 0,
                "declared_draw_sha256": expected_draw_sha, "draw_hash_matches_prepared": row_hash.hexdigest()==expected_draw_sha},
            "retention_distillation": {"weight": distillation_weight, "temperature": 2.,
                "unweighted_mean_batch_loss": kd_total / attempted_batches,
                "weighted_mean_batch_loss": distillation_weight * kd_total / attempted_batches,
                "teacher_forward_batches": teacher_batches, "contributing_batches": kd_contributing,
                "known_other_class_entries": kd_known_entries, "excluded_primary_classes": list(TARGETS),
                "teacher_eval": not teacher.training,
                "teacher_frozen": all(not p.requires_grad for p in teacher.parameters()),
                "teacher_gradients_absent": all(p.grad is None for p in teacher.parameters()),
                "teacher_state_unchanged": state_sha256(teacher.state_dict()) == teacher_initial_sha},
            "target_ranking_audit": {"unweighted_mean_batch_loss": 0., "pair_count": 0,
                                     "contributing_batches": 0, "pair_counts_by_domain_target": {}},
            "ranking_ap": {d: {label: average_precision(t[:, k], p[:, k]) for k, label in enumerate(classes)
                               if (t[:, k] >= 0).all()} for d, (t, p) in predictions.items()}}
        if int(draws.sum()) != args.draws_per_epoch or int(row_types.sum()) != args.draws_per_epoch:
            raise ValueError("Actual sampled exposure differs from the declared epoch budget")
        if row_hash.hexdigest() != expected_draw_sha:
            raise ValueError("Actual row order differs from fixed original control draws")
        if not all(row["retention_distillation"][key] for key in
                   ("teacher_eval", "teacher_frozen", "teacher_gradients_absent", "teacher_state_unchanged")):
            raise ValueError("Frozen teacher state or gradient invariant was violated")
        if teacher_batches != attempted_batches:
            raise ValueError("Both arms must perform one frozen teacher forward on every training batch")
        if attempted_batches != args.draws_per_epoch // args.batch:
            raise ValueError("Actual epoch batch count differs from fixed divisible budget")
        history.append(row)
        print(__import__("json").dumps({key: row[key] for key in ("epoch", "elapsed_minutes", "train_loss", "worst_target_error", "target_passed")}), flush=True)
        key = (worst, sum_errors)
        if best is None or key < best:
            best = key; bad = 0
            torch.save({"architecture": architecture, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "classes": classes, "imgsz": size, "mean": MEAN, "std": STD,
                        "auxiliary_classes": list(AUX_CLASSES), "epoch": epoch,
                        "selection_split": "val", "split_sha256": split_sha,
                        "worst_target_error": worst}, output / "best.pt")
            save(output / "VALIDATION.json", row)
            for d, (targets, scores) in predictions.items():
                save(output / f"validation-{d}.json", {"split": "val", "targets": targets.tolist(),
                     "probabilities": scores.tolist(), "weights_sha256": sha(output / "best.pt")})
        else:
            bad += 1
        save(output / "history.json", history); scheduler.step()
        if bad >= args.patience:
            break
    training.update(status="complete", actual_epochs=len(history), best_worst_target_error=best[0],
                    optimizer_final_learning_rates={"backbone": float(optimizer.param_groups[0]["lr"]),
                        "head": float(optimizer.param_groups[1]["lr"])},
                    frozen_semantic_state_preservation={
                        "initial_encoder_state_sha256": semantic_initial_sha,
                        "final_encoder_state_sha256": semantic_encoder_state_sha256(model),
                        "encoder_state_unchanged": semantic_encoder_state_sha256(model) == semantic_initial_sha,
                        "encoder_all_eval": all(not module.training for module in model.semantic_encoder.modules()),
                        "encoder_parameters_frozen": all(not p.requires_grad for p in model.semantic_encoder.parameters()),
                        "encoder_gradients_absent": all(p.grad is None for p in model.semantic_encoder.parameters()),
                        "head_parameters_trainable": all(p.requires_grad for module in semantic_heads.values() for p in module.parameters()),
                        "semantic_heads_changed_from_zero": semantic_epoch_audit["semantic_heads_changed_from_zero"],
                        "encoder_forward_batches": sum(row["semantic_features_training"]["encoder_forward_batches"] for row in history)},
                    weights_sha256=sha(output / "best.pt"), elapsed_training_minutes=(time.perf_counter() - start) / 60,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    batchnorm_state_preservation={"initial_buffers_sha256": batchnorm_initial_sha,
                        "final_buffers_sha256": batchnorm_state_sha256(model),
                        "buffers_unchanged": batchnorm_state_sha256(model) == batchnorm_initial_sha,
                        "layer_count": batchnorm_inventory(model)["layer_count"],
                        "channel_count": batchnorm_inventory(model)["channel_count"],
                        "buffer_tensor_count": batchnorm_inventory(model)["buffer_tensor_count"],
                        "affine_parameter_tensor_count": batchnorm_inventory(model)["affine_parameter_tensor_count"],
                        "affine_parameters_trainable": batchnorm_inventory(model)["affine_parameters_trainable"],
                        "backbone_parameters_trainable": all(p.requires_grad for p in backbone),
                        "head_parameters_trainable": all(p.requires_grad for p in head),
                        "eps_and_momentum_unchanged": batchnorm_configuration(model) == batchnorm_initial_configuration,
                        "all_batchnorm_eval": batchnorm_inventory(model)["all_batchnorm_eval"],
                        "policy_applied_after_each_model_train": True},
                    actual_teacher_forward_batches=sum(row["retention_distillation"]["teacher_forward_batches"] for row in history),
                    teacher_state_preservation={"initial_state_sha256": teacher_initial_sha,
                        "final_state_sha256": state_sha256(teacher.state_dict()),
                        "initial_weights_sha256": training["initial_weights_sha256"],
                        "unchanged": state_sha256(teacher.state_dict()) == teacher_initial_sha,
                        "eval_mode": not teacher.training,
                        "all_parameters_frozen": all(not p.requires_grad for p in teacher.parameters()),
                        "no_parameter_gradients": all(p.grad is None for p in teacher.parameters()),
                        "student_parameter_count": sum(p.numel() for p in model.parameters()),
                        "teacher_parameter_count": sum(p.numel() for p in teacher.parameters()),
                        "teacher_state_tensor_count": len(teacher.state_dict()), "teacher_forward_both_arms": True},
                    optimizer_step_diagnostics={"attempted_batches": sum(row["optimizer_step_diagnostics"]["attempted_batches"] for row in history),
                        "actual_optimizer_steps": updates["count"],
                        "amp_skipped_steps": sum(row["optimizer_step_diagnostics"]["amp_skipped_steps"] for row in history)})
    step_hook.remove(); save(output / "TRAINING.json", training)
    print(__import__("json").dumps({key: training[key] for key in ("status", "actual_epochs", "best_worst_target_error", "weights_sha256")}, indent=2))


if __name__ == "__main__":
    main()
