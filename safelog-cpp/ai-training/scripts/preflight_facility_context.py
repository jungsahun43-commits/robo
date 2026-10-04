"""Disposable TRAIN-only GPU checks for the fixed narrow/broad pooling pair."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH as AUX_ARCH
from safelog_ai.context_classifier import ContextClassifier, load_auxiliary_initializer, ARCH
from safelog_ai.spatial_classifier import spatial_loss
from safelog_ai.presence_classifier import image_transform
from scripts.train_facility_context import SpatialPhotos
from scripts.train_facility_target import DOMAINS, TARGETS, masked_focal, read, save, sha
from scripts.preflight_facility_detail import (
    select_train_replay, loss_weights, seed_model, batch_fingerprint, diagnostic_latency,
)

PROTOCOL = ROOT / 'reports/facility-pool-context-protocol.json'
OUTPUT = ROOT / 'runs/facility-context-preflight.json'
SOURCE_PATHS = {
    'training_script_sha256': 'scripts/train_facility_context.py',
    'source_parent_training_script_sha256': 'scripts/train_facility_detail.py',
    'context_model_source_sha256': 'safelog_ai/context_classifier.py',
    'model_source_sha256': 'safelog_ai/spatial_classifier.py',
    'auxiliary_model_source_sha256': 'safelog_ai/auxiliary_classifier.py',
    'model_factory_source_sha256': 'safelog_ai/presence_classifier.py',
}


def augmented_replay(items, auxiliary, seed):
    torch.manual_seed(seed + 2)
    sampler = WeightedRandomSampler(torch.ones(len(items), dtype=torch.float64), 24,
                                    replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(SpatialPhotos(items, auxiliary, len(items), DOMAINS),
                        batch_size=8, sampler=sampler, num_workers=4, persistent_workers=True,
                        pin_memory=True, generator=torch.Generator().manual_seed(seed + 1))
    iterator = iter(loader)
    replay, first = [], None
    try:
        for _ in range(3):
            batch = next(iterator)
            if first is None:
                first = batch
            replay.append({'augmented_batch_sha256': batch_fingerprint(batch),
                           'sampled_indices_sha256': hashlib.sha256(
                               batch[-1].numpy().astype('<i8', copy=False).tobytes()).hexdigest()})
    finally:
        iterator._shutdown_workers()
    return replay, first


def run():
    if not torch.cuda.is_available():
        raise ValueError('An actual CUDA AMP backward is required')
    torch.set_num_threads(4)
    protocol = read(PROTOCOL)
    if protocol['schema'] != 'facility_pool_context_protocol_v1' or protocol['seed'] != 55:
        raise ValueError('Declared pooling protocol changed')
    paths = {'initial': ROOT / 'runs' / protocol['reference'] / 'best.pt',
             'core': ROOT / 'data/facility-spatial-training/train.json',
             'auxiliary': ROOT / 'data/facility-auxiliary-training/train.json',
             'protocol': PROTOCOL, 'app': ROOT / 'reports/facility-inference-profile.json'}
    before = {k: sha(v) for k, v in paths.items()}
    for key, field in (('initial', 'initial_weights_sha256'), ('core', 'core_spatial_manifest_sha256'),
                       ('auxiliary', 'auxiliary_manifest_sha256')):
        if before[key] != protocol[field]:
            raise ValueError('Original initializer or TRAIN supervision changed')
    manifest, auxiliary = read(paths['core']), read(paths['auxiliary'])
    selected, aux = select_train_replay(manifest, auxiliary)
    checkpoint = torch.load(paths['initial'], map_location='cpu', weights_only=True)
    if (checkpoint['architecture'] != AUX_ARCH or checkpoint['classes'] != protocol['classes']
            or checkpoint['auxiliary_classes'] != list(AUX_CLASSES) or checkpoint['imgsz'] != 640):
        raise ValueError('Seven public and nineteen private class contract changed')
    models, transfer, rng, replay, batches = {}, {}, {}, {}, {}
    for variant, constructor in (('control', AuxiliaryClassifier), ('context', ContextClassifier)):
        seed_model(protocol['seed'])
        model = constructor(7, pretrained=False)
        transfer[variant] = load_auxiliary_initializer(model, checkpoint['state_dict'])
        rng[variant] = torch.get_rng_state().clone()
        models[variant] = model
        replay[variant], batches[variant] = augmented_replay(selected, aux, protocol['seed'])
    if not torch.equal(rng['control'], rng['context']) or replay['control'] != replay['context']:
        raise ValueError('Constructor RNG or bounded augmented TRAIN replay differs')
    transform = image_transform(640)
    images = []
    for item in selected:
        with Image.open(ROOT / item['image']) as handle:
            images.append(transform(handle.convert('RGB')))
    image = torch.stack(images).to('cuda')
    shapes, latency, control = {}, {}, None
    for variant, model in models.items():
        model.to('cuda').eval()
        calls = []
        hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
        with torch.inference_mode():
            values = model.forward_training(image)
        hook.remove()
        if (len(calls) != 1 or [list(v.shape) for v in values] != [[8, 7], [8, 7, 80, 80], [8, 19]]
                or not all(bool(torch.isfinite(v).all()) for v in values)):
            raise ValueError('One-backbone finite output contract failed')
        if variant == 'control':
            control = tuple(v.detach().cpu().clone() for v in values)
        elif not all(torch.equal(v.detach().cpu(), old) for v, old in zip(values, control)):
            raise ValueError('Zero pooling gates changed initial predictions')
        shapes[variant] = {'public_shape': [8, 7], 'pixel_shape': [8, 7, 80, 80],
                           'auxiliary_shape': [8, 19],
                           'parameter_count': sum(p.numel() for p in model.parameters())}
        latency[variant] = diagnostic_latency(model, image)
    del values, control
    model = models['context'].train()
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone}
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': protocol['backbone_lr']},
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': protocol['head_lr']},
    ], weight_decay=.0002)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    photo_weights, pixel_weights, tag_weights = (v.to('cuda') for v in loss_weights(manifest, auxiliary))
    batch = batches['control']
    emphasis = torch.ones(7, device='cuda')
    for label in TARGETS:
        emphasis[manifest['classes'].index(label)] = 2.
    steps = []
    torch.cuda.reset_peak_memory_stats()
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda'):
            photo, maps, tags = model.forward_training(batch[0].to('cuda'))
            loss = masked_focal(photo, batch[1].to('cuda'), batch[2].to('cuda'),
                                photo_weights[batch[3]], 1., emphasis)
            loss += spatial_loss(maps, batch[4].to('cuda'), batch[5].to('cuda'), pixel_weights)
            loss += protocol['auxiliary_weight'] * masked_focal(
                tags, batch[6].to('cuda'), batch[7].to('cuda'), tag_weights, 1.)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Disposable AMP loss is nonfinite')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        gate = model.context_gate.grad
        if (not gradients or not all(bool(torch.isfinite(v).all()) for v in gradients)
                or gate is None or not bool(torch.count_nonzero(gate))):
            raise ValueError('Context gate gradient is disconnected or nonfinite')
        scaler.step(optimizer)
        scaler.update()
        if (not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
                or not bool(torch.count_nonzero(model.context_gate))):
            raise ValueError('Disposable optimizer did not learn finite context gates')
        steps.append({'step': step, 'loss': float(loss.detach()), 'all_gradients_finite': True,
                      'gate_gradient_nonzero': True, 'gate_gradient_l2': float(torch.linalg.vector_norm(gate)),
                      'learned_gate_nonzero': True})
    torch.cuda.synchronize()
    if before != {k: sha(v) for k, v in paths.items()}:
        raise ValueError('Preflight modified a frozen input')
    return {'status': 'passed', 'schema': 'facility_context_preflight_v1',
            'train_photo_count': 8, 'train_joint_tag_counts': {s: 2 for s in ('00', '10', '01', '11')},
            'public_shape': [8, 7], 'pixel_shape': [8, 7, 80, 80], 'auxiliary_shape': [8, 19],
            'backbone_forward_calls': 1, 'initial_fp32_outputs_equal': True,
            'initial_state_transfer': transfer, 'constructor_cpu_rng_equal': True,
            'loader_randomness': protocol['loader_randomness'], 'replay_num_workers': 4, 'replay_batches': 3,
            'augmented_batch_replay_equal': True, 'bounded_replay_digests': replay['control'],
            'finite_loss_and_gradients': True, 'learned_gate_gradient_nonzero': True,
            'disposable_optimizer_steps': steps, 'amp_initial_scale': 1024., 'checkpoint_saved': False,
            'source_checkpoint_unchanged': True, 'frozen_inputs_unchanged': True,
            'initial_weights_sha256': before['initial'], 'core_spatial_manifest_sha256': before['core'],
            'auxiliary_manifest_sha256': before['auxiliary'], 'study_protocol_sha256': before['protocol'],
            'app_profile_sha256': before['app'],
            'executed_sources_sha256': {k: sha(ROOT / p) for k, p in SOURCE_PATHS.items()},
            'preflight_script_sha256': sha(Path(__file__)),
            'reused_preflight_helper_sha256': sha(ROOT / 'scripts/preflight_facility_detail.py'),
            'model_shapes_and_parameters': shapes, 'latency_diagnostic': latency,
            'amp_two_step_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'python': platform.python_version(), 'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
            'accuracy_measured_by_this_check': False, 'independent_field_safety_verified': False}


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save(OUTPUT, {'status': 'running', 'schema': 'facility_context_preflight_v1'})
    try:
        result = run()
    except Exception as error:
        save(OUTPUT, {'status': 'failed', 'schema': 'facility_context_preflight_v1',
                      'error_type': type(error).__name__, 'error': str(error), 'checkpoint_saved': False})
        raise
    save(OUTPUT, result)
    print(json.dumps({'status': result['status'], 'initial_outputs_equal': True, 'learned_gate': True}))


if __name__ == '__main__':
    main()
