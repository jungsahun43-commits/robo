"""Disposable TRAIN-only CUDA replay for fixed other-class retention loss.

The actual study starts from the unchanged initializer after these disposable
updates. No VAL/TEST accuracy is measured here.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import random
import statistics
import sys
import time

import numpy as np
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier
from safelog_ai.presence_classifier import MEAN, STD
from safelog_ai.spatial_classifier import spatial_loss
from safelog_ai.retention_distillation import masked_bernoulli_kl, state_sha256
from safelog_ai.frozen_batchnorm import apply_frozen_batchnorm, batchnorm_inventory, batchnorm_state_sha256
from safelog_ai.semantic_residual_classifier import (SemanticResidualClassifier, load_original_state,
    load_semantic_pretrained, semantic_encoder_state_sha256, model_inventory)
from safelog_ai.research_presence import ResearchPresenceClassifier
from scripts.facility_resolution_study import ResolutionPhotos
from scripts.facility_semantic_study import (PROTOCOL, REFERENCE, prepare_data,
    protected_hashes, require, validate_protocol, write)
from scripts.train_facility_target import DOMAINS, masked_focal
from scripts.train_facility_head_lr import load_initial_state
from scripts.verify_facility_resolution import local_path, read, sha

OUTPUT = 'runs/facility-semantic-preflight.json'
DIRECTORY = 'runs/facility-semantic-preflight'
LEDGER = DIRECTORY + '/protected-inputs.json'
PREVIOUS_LEDGER = 'runs/facility-head-lr-preflight/protected-inputs.json'
REUSED_PREFLIGHT = 'runs/facility-head-lr-preflight.json'


def snapshot_inputs(root, paths):
    """Private full original TRAIN images/masks and frozen evidence inventory."""
    result = {}
    for relative in sorted(set(paths)):
        path = local_path(root, relative); stat = path.stat()
        result[relative] = {'sha256': sha(path), 'size_bytes': stat.st_size,
                            'mtime_ns': stat.st_mtime_ns}
    return result


def input_paths(data, protected):
    paths = set(protected)
    paths.update(row[key] for row in data['items'] for key in ('image', 'pixel_target'))
    paths.update(row['annotation'] for row in data['auxiliary_manifest']['items'])
    # Include every prior protected input, even if the new metadata inventory
    # contains additional completed-study evidence paths.
    paths.update(read(ROOT / PREVIOUS_LEDGER)['snapshots_before'])
    return sorted(paths)


def replay_positions(data):
    """Cover all sources, crops and known/unknown targets with original draws."""
    left = data['epoch_draws'][0]
    rows, full = data['items'], data['full_count']
    chosen = []
    def add(predicate):
        position = next((p for p, i in enumerate(left) if p not in chosen and predicate(p, rows[int(i)], int(i))), None)
        require(position is not None, 'Actual paired TRAIN replay coverage is incomplete')
        chosen.append(position)
    for domain in DOMAINS:
        add(lambda p, row, i, domain=domain: row['domain'] == domain and i < full)
    add(lambda p, row, i: i >= full)
    add(lambda p, row, i: row['domain'] == 'dacl' and i < full and row['targets'][:2] == [0, 0])
    add(lambda p, row, i: row['targets'][0] == 1 and row['targets'][1] == 1)
    add(lambda p, row, i: i >= full and -1 in row['targets'][:2])
    add(lambda p, row, i: row['domain'] == 'dacl' and i < full and row['targets'][:2] == [1, 0])
    require(len(chosen) == len(set(chosen)) == 8, 'Eight paired replay positions are required')
    return chosen


def batch(data, indices):
    random.seed(57); np.random.seed(57); torch.manual_seed(57)
    dataset = ResolutionPhotos(data['items'], data['auxiliary'], data['full_count'], DOMAINS, imgsz=640)
    return default_collate([dataset[int(i)] for i in indices])


def benchmark_forward(baseline, candidate, inputs):
    """Actual same TRAIN tensor, float32 CUDA forwards; excludes transfers."""
    sample = inputs[:1].to('cuda'); baseline.eval(); candidate.eval()
    result = {'device': 'cuda', 'device_name': torch.cuda.get_device_name(), 'torch_version': str(torch.__version__),
        'dtype': str(sample.dtype), 'batch_size': 1, 'input_shape': list(sample.shape),
        'warmup_forwards_each': 3, 'timed_forwards_each': 10, 'synchronization': 'CUDA synchronize before and after each timed forward',
        'input_tensor_sha256': hashlib.sha256(sample.detach().cpu().numpy().tobytes()).hexdigest(),
        'same_input_tensor_both_models': True, 'transfer_excluded': True, 'model_forward_only': True,
        'accuracy_measured': False, 'android_latency_measured': False,
        'scope': 'Initial zero-residual candidate and original architecture on one existing augmented TRAIN tensor'}
    with torch.inference_mode():
        for name, model in (('original', baseline), ('semantic', candidate)):
            for _ in range(3): model(sample)
            torch.cuda.synchronize(); durations = []
            for _ in range(10):
                torch.cuda.synchronize(); start = time.perf_counter()
                model(sample); torch.cuda.synchronize()
                durations.append((time.perf_counter() - start) * 1000.)
            result[name] = {'mean_ms': statistics.mean(durations), 'median_ms': statistics.median(durations),
                            'min_ms': min(durations), 'max_ms': max(durations)}
    return result


def main():
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    require(torch.cuda.is_available(), 'Actual paired preflight requires CUDA')
    require(not (ROOT / OUTPUT).exists() and not (ROOT / DIRECTORY).exists(),
            'Preserve existing preflight evidence')
    p = validate_protocol(read(ROOT / PROTOCOL), ROOT)
    protected = protected_hashes(ROOT)
    original_data = prepare_data(ROOT, p, 'semantic')
    data = {'semantic': original_data}
    paths = input_paths(original_data, protected)
    print({'phase': 'hash-original-train-inputs', 'files': len(paths)}, flush=True)
    before = snapshot_inputs(ROOT, paths)
    previous = read(ROOT / PREVIOUS_LEDGER)
    require(previous['snapshots_before'] == previous['snapshots_after']
            and all(before[name] == value for name, value in previous['snapshots_before'].items()),
            'Established original input SHA/size/mtime changed since the completed study')
    positions = replay_positions(original_data)
    indices = {v: d['epoch_draws'][0, positions] for v, d in data.items()}
    batches = {v: batch(data[v], indices[v]) for v in data}
    reused = read(ROOT / REUSED_PREFLIGHT)
    require(reused.get('schema') == 'facility_head_lr_preflight_v1' and reused.get('status') == 'passed'
            and reused.get('source_sha256') == read(ROOT / 'reports/facility-head-lr-study-protocol.json')['source_sha256']
            and reused['candidates']['low_lr']['actual_optimizer_steps'] == 1,
            'Completed control preflight proof must be reused')
    replay_hash = hashlib.sha256(np.asarray(positions, dtype='<i8').tobytes()).hexdigest()
    require(reused['replay_positions_sha256'] == replay_hash,
            'The single candidate must replay the same original TRAIN positions as the reused control')
    checkpoint = torch.load(ROOT / 'runs' / REFERENCE / 'best.pt', map_location='cpu', weights_only=True)
    proofs = {}; (ROOT / DIRECTORY).mkdir(exist_ok=False)
    for variant, values in batches.items():
        torch.manual_seed(56); model = SemanticResidualClassifier(7, pretrained=False)
        transfer = load_original_state(model, checkpoint, p['classes'], checkpoint['split_sha256'])
        semantic_transfer = load_semantic_pretrained(model, ROOT / p['semantic_pretrained_weights_path'])
        inventory = model_inventory(model)
        encoder_initial_sha = semantic_encoder_state_sha256(model)
        require(encoder_initial_sha == p['semantic_encoder_state_sha256'], 'Declared frozen semantic encoder state differs')
        baseline = AuxiliaryClassifier(7, pretrained=False)
        load_initial_state(baseline, checkpoint, p['classes'], checkpoint['split_sha256'])
        baseline.to('cuda').eval()
        require(all(torch.equal(value.detach().cpu(), model.state_dict()[name].detach().cpu())
                for name, value in checkpoint['state_dict'].items()), 'Original 324 state tensors were not exactly transferred')
        initial_bn_inventory = batchnorm_inventory(model)
        initial_bn_sha = batchnorm_state_sha256(model)
        require(initial_bn_sha == p['initial_batchnorm_buffers_sha256'], 'Original BN buffer state hash differs')
        teacher = AuxiliaryClassifier(7, pretrained=False)
        load_initial_state(teacher, checkpoint, p['classes'], checkpoint['split_sha256'])
        teacher.eval().requires_grad_(False)
        teacher_initial_hash = state_sha256(teacher)
        require(teacher_initial_hash == p['teacher_state_sha256'], 'Original teacher tensor state hash differs')
        teacher.to('cuda'); torch.manual_seed(58)
        model.to('cuda'); model.eval(); torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            public = model(values[0].cuda()); photo, maps, aux = model.forward_training(values[0].cuda())
            original_photo, original_maps, original_aux = baseline.forward_training(values[0].cuda())
            require(torch.equal(public, original_photo) and torch.equal(photo, original_photo)
                    and torch.equal(maps, original_maps) and torch.equal(aux, original_aux),
                    'Zero semantic heads changed actual initial public/map/aux outputs')
            require(tuple(public.shape) == (8, 7) and tuple(maps.shape) == (8, 7, 80, 80)
                    and tuple(aux.shape) == (8, 19) and torch.equal(public, photo)
                    and all(torch.isfinite(t).all() for t in (public, maps, aux)),
                    'Actual CUDA output contract failed')
        latency = benchmark_forward(baseline, model, values[0])
        del baseline
        backbone = list(model.backbone.parameters()); ids = {id(v) for v in backbone}
        head = [v for v in model.parameters() if v.requires_grad and id(v) not in ids]
        require(not {id(v) for v in model.semantic_encoder.parameters()} & {id(v) for v in backbone + head},
                'Frozen semantic encoder must be excluded from both optimizer groups')
        optimizer = torch.optim.AdamW([{'params': backbone, 'lr': p['backbone_lr']},
                                      {'params': head, 'lr': p['head_lr']}], weight_decay=.0002)
        actual_lrs = {'backbone': float(optimizer.param_groups[0]['lr']), 'head': float(optimizer.param_groups[1]['lr'])}
        actual_weight_decay = {'backbone': float(optimizer.param_groups[0]['weight_decay']),
                               'head': float(optimizer.param_groups[1]['weight_decay'])}
        require(actual_lrs == {'backbone': .00004, 'head': .0001}, 'Actual lower head-LR preflight groups differ')
        updates = {'count': 0}
        def updated(optimizer, args, kwargs): updates['count'] += 1
        hook = optimizer.register_step_post_hook(updated)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        model.train(); apply_frozen_batchnorm(model); optimizer.zero_grad(set_to_none=True)
        encoder_observed = []
        def observe_encoder(module, args): encoder_observed.append(all(not child.training for child in module.modules()))
        encoder_hook = model.semantic_encoder.register_forward_pre_hook(observe_encoder)
        bn_modules = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        observed_bn = []
        def observe_bn(module, args): observed_bn.append(not module.training)
        bn_hooks = [module.register_forward_pre_hook(observe_bn) for module in bn_modules]
        with torch.autocast(device_type='cuda'):
            image = values[0].cuda()
            photo, maps, aux = model.forward_training(image)
            # This forward happens in both arms on this same augmented tensor.
            with torch.no_grad(): teacher_photo = teacher(image)
            require(not teacher_photo.requires_grad, 'Teacher output unexpectedly requires gradients')
            d = data[variant]; emphasis = torch.tensor([2., 2., 1., 1., 1., 1., 1.], device='cuda')
            loss = masked_focal(photo, values[1].cuda(), values[2].cuda(),
                                d['sampling_data']['photo_weights'].cuda()[values[3]], 1., emphasis)
            loss += spatial_loss(maps, values[4].cuda(), values[5].cuda(),
                                 d['supervision_weights']['pixel_weights'].cuda())
            loss += .5 * masked_focal(aux, values[6].cuda(), values[7].cuda(),
                                     d['supervision_weights']['auxiliary_weights'].cuda(), 1.)
            retention = masked_bernoulli_kl(photo, teacher_photo, values[2].cuda(), temperature=2.)
            loss += p['distillation_weight_by_variant'][variant] * retention
        for observer in bn_hooks: observer.remove()
        encoder_hook.remove()
        require(encoder_observed == [True], 'Actual semantic encoder TRAIN forward was not entirely eval')
        require(len(observed_bn) == 47 and all(observed_bn), 'Actual TRAIN forward did not use eval in every BN layer')
        probe = photo.detach().float().requires_grad_()
        probe_loss = masked_bernoulli_kl(probe, teacher_photo, values[2].cuda(), temperature=2.)
        gradients = torch.autograd.grad(probe_loss, probe)[0]
        require(torch.equal(gradients[:, :2], torch.zeros_like(gradients[:, :2]))
                and torch.equal(gradients[:, 2:][values[2][:, 2:].cuda() == 0],
                                torch.zeros_like(gradients[:, 2:][values[2][:, 2:].cuda() == 0]))
                and bool(torch.isfinite(gradients).all()), 'Actual known-other-five gradient mask differs')
        require(torch.isfinite(loss), 'Actual disposable loss is nonfinite')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        affine = [parameter for module in bn_modules for parameter in (module.weight, module.bias)]
        require(len(affine) == 94 and all(parameter.requires_grad and parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all()) for parameter in affine),
                'Every BN affine parameter must remain trainable with finite actual gradients')
        nonzero = sum(int(bool(torch.count_nonzero(parameter.grad))) for parameter in affine)
        require(nonzero > 0, 'Frozen BN affine gradients must include an actual nonzero value')
        require(all(parameter.requires_grad for parameter in backbone + head),
                'Backbone/head parameters must remain learnable')
        semantic_heads = {'map': model.semantic_map_head, 'photo': model.semantic_photo_head,
                          'aux': model.semantic_auxiliary_head}
        semantic_parameters = [parameter for module in semantic_heads.values() for parameter in module.parameters()]
        require(len(semantic_parameters) == 6 and all(parameter.requires_grad and parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all()) and bool(torch.count_nonzero(parameter.grad))
                for parameter in semantic_parameters), 'Actual semantic head gradients must be finite and nonzero')
        scaler.step(optimizer); scaler.update(); hook.remove()
        require(updates['count'] == 1, 'Disposable AMP update was skipped')
        changed_heads = {name: any(bool(torch.count_nonzero(parameter.detach())) for parameter in module.parameters())
                         for name, module in semantic_heads.items()}
        require(all(changed_heads.values()), 'All three semantic heads must change from actual zero initialization')
        require(semantic_encoder_state_sha256(model) == encoder_initial_sha
                and all(not child.training for child in model.semantic_encoder.modules())
                and all(not parameter.requires_grad and parameter.grad is None for parameter in model.semantic_encoder.parameters()),
                'Frozen semantic encoder state/eval/no-gradient invariant changed')
        encoder_proof = {'initial_encoder_state_sha256': encoder_initial_sha,
            'final_encoder_state_sha256': semantic_encoder_state_sha256(model), 'encoder_state_unchanged': True,
            'encoder_all_eval': True, 'encoder_parameters_frozen': True, 'encoder_gradients_absent': True,
            'head_parameters_trainable': True, 'semantic_heads_changed_from_zero': changed_heads,
            'encoder_forward_batches': 1}
        require(batchnorm_state_sha256(model) == initial_bn_sha
                and batchnorm_inventory(model)['configuration'] == initial_bn_inventory['configuration'],
                'BN buffers or eps/momentum changed during the actual update')
        bn_proof = {'initial_buffers_sha256': initial_bn_sha, 'final_buffers_sha256': batchnorm_state_sha256(model),
            'buffers_unchanged': True, 'layer_count': 47, 'channel_count': 12328, 'buffer_tensor_count': 141,
            'affine_parameter_tensor_count': 94, 'affine_parameters_trainable': True,
            'backbone_parameters_trainable': True, 'head_parameters_trainable': True,
            'eps_and_momentum_unchanged': True, 'all_batchnorm_eval': True,
            'policy_applied_after_each_model_train': True}
        require(all(not module.training for module in teacher.modules())
                and all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
                and state_sha256(teacher) == teacher_initial_hash, 'Teacher parameters/buffers/eval/frozen/no-grad changed')
        teacher_proof = {'initial_state_sha256': teacher_initial_hash, 'final_state_sha256': state_sha256(teacher),
            'initial_weights_sha256': p['initial_weights_sha256'], 'unchanged': True, 'eval_mode': True,
            'all_parameters_frozen': True, 'no_parameter_gradients': True,
            'student_parameter_count': sum(t.numel() for t in model.parameters()),
            'teacher_parameter_count': sum(t.numel() for t in teacher.parameters()),
            'teacher_state_tensor_count': len(teacher.state_dict()), 'teacher_forward_both_arms': True}
        path = ROOT / DIRECTORY / f'{variant}.pt'
        torch.save({**{k: v for k, v in checkpoint.items() if k != 'state_dict'},
                    'architecture': p['architecture_by_variant'][variant],
                    'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'mean': MEAN, 'std': STD}, path)
        peak = torch.cuda.max_memory_allocated(); del model, optimizer, scaler, teacher, probe, gradients
        torch.cuda.empty_cache(); classifier = ResearchPresenceClassifier(path, 'cpu')
        with torch.inference_mode():
            public = classifier.model(values[0][:1]); photo, maps, aux = classifier.model.forward_training(values[0][:1])
        require(torch.equal(public, photo) and all(torch.isfinite(t).all() for t in (photo, maps, aux)),
                'Actual strict CPU factory reload failed')
        proofs[variant] = {'actual_batch_size': 8, 'actual_optimizer_steps': 1, 'amp_skipped_steps': 0,
            'scaler_initial_scale': 1024., 'public_shape': [8, 7], 'pixel_shape': [8, 7, 80, 80],
            'auxiliary_shape': [8, 19], 'parameter_count': sum(t.numel() for t in classifier.model.parameters()),
            'all_outputs_finite': True, 'cpu_reload_verified': True, 'strict_factory_reload_verified': True,
            'public_output_equals_training_photo_output': True, 'disposable_checkpoint_sha256': sha(path),
            'peak_cuda_allocated_bytes': peak, 'initial_state_transfer': transfer,
            'original_state_transfer': transfer, 'semantic_pretrained_transfer': semantic_transfer,
            'semantic_model_inventory': inventory, 'frozen_semantic_state_preservation': encoder_proof,
            'zero_initial_outputs_identical_to_original': True, 'original_state_tensors_exactly_equal': 324,
            'semantic_head_gradient_tensors': 6, 'semantic_head_gradient_nonzero_tensors': 6,
            'semantic_head_gradients_finite': True, 'actual_semantic_encoder_forward_eval_verified': True,
            'actual_cuda_forward_latency': latency,
            'teacher_state_preservation': teacher_proof, 'teacher_forward_batches': 1,
            'actual_distillation_gradient_mask_verified': True,
            'teacher_and_student_same_augmented_image_tensor': True,
            'distillation_weight': p['distillation_weight_by_variant'][variant],
            'temperature': 2., 'known_other_class_entries': int(values[2][:, 2:].sum()),
            'unweighted_distillation_loss': float(retention.detach().cpu())}
        require(batchnorm_state_sha256(classifier.model) == initial_bn_sha,
                'Serialized candidate CPU reload changed original BN buffers')
        proofs[variant].update(batchnorm_state_preservation=bn_proof,
            actual_bn_forward_eval_layers=len(observed_bn), bn_affine_gradient_tensors=len(affine),
            bn_affine_gradient_nonzero_tensors=nonzero, bn_affine_gradients_finite=True,
            bn_buffer_cpu_reload_verified=True, optimizer_learning_rates=actual_lrs,
            optimizer_weight_decay=actual_weight_decay)
        del classifier
    print({'phase': 'rehash-original-train-inputs', 'files': len(paths)}, flush=True)
    after = snapshot_inputs(ROOT, paths)
    require(before == after and protected_hashes(ROOT) == protected,
            'Preflight changed original TRAIN or protected experiment bytes')
    validate_protocol(p, ROOT)
    write(ROOT / LEDGER, {'schema': 'facility_semantic_private_input_ledger_v1', 'local_only': True,
          'split': 'train', 'snapshots_before': before, 'snapshots_after': after,
          'core_manifest_sha256': p['core_spatial_manifest_sha256'], 'all_inputs_preserved': True,
          'previous_ledger_sha256': sha(ROOT / PREVIOUS_LEDGER),
          'previous_input_file_count': len(previous['snapshots_before']), 'previous_inputs_preserved': True})
    result = {'schema': 'facility_semantic_preflight_v1', 'status': 'passed',
        'verified_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(ROOT / PROTOCOL),
        'source_sha256': p['source_sha256'], 'protected_file_sha256': protected, 'candidates': proofs,
        'reused_control_preflight_path': REUSED_PREFLIGHT, 'reused_control_preflight_sha256': sha(ROOT / REUSED_PREFLIGHT),
        'reused_control_preflight_status': 'passed', 'same_original_replay_positions_as_reused_control': True,
        'actual_disposable_optimizer_steps': 1, 'new_candidate_count': 1, 'control_training_repeated': False,
        'unique_paired_draw_positions': 8, 'changed_replay_positions': 0,
        'candidate_teacher_unchanged_eval_frozen_no_grad': True,
        'candidate_original_batchnorm_buffers_preserved_affine_learnable': True,
        'replay_positions_sha256': replay_hash,
        'protected_input_ledger_path': LEDGER, 'protected_input_ledger_sha256': sha(ROOT / LEDGER),
        'protected_input_file_count': len(before), 'all_original_sha_size_mtime_preserved': True,
        'previous_input_file_count': len(previous['snapshots_before']), 'previous_inputs_preserved': True,
        'new_training_epochs': 0, 'source_test_inference_executed': False, 'app_model_promoted': False,
        'accuracy_measured': False, 'label_changes': 0, 'new_masks': 0}
    write(ROOT / OUTPUT, result)
    print({'status': 'passed', 'candidate_train_draw_positions': 8, 'disposable_amp_steps': 1,
           'protected_original_files': len(before), 'new_epochs': 0}, flush=True)


if __name__ == '__main__': main()
