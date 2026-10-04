"""Disposable TRAIN-only integration checks for the declared detail architecture.

No checkpoint is written. The eight source photos are selected by TRAIN tags,
never by validation scores. Replay proves the bounded loader pipeline only;
the completed trainer's per-epoch index digest proves the full sampling pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH as AUX_ARCH
from safelog_ai.detail_classifier import DetailClassifier, load_auxiliary_initializer, ARCH as DETAIL_ARCH
from safelog_ai.spatial_classifier import spatial_loss
from safelog_ai.presence_classifier import image_transform
from scripts.train_facility_detail import SpatialPhotos
from scripts.train_facility_target import DOMAINS, TARGETS, masked_focal, read, save, sha

PROTOCOL = ROOT / 'reports/facility-detail-architecture-protocol.json'
OUTPUT = ROOT / 'runs/facility-detail-preflight.json'
SOURCE_PATHS = {
    'training_script_sha256': 'scripts/train_facility_detail.py',
    'source_parent_training_script_sha256': 'scripts/train_facility_spatial.py',
    'detail_model_source_sha256': 'safelog_ai/detail_classifier.py',
    'model_source_sha256': 'safelog_ai/spatial_classifier.py',
    'auxiliary_model_source_sha256': 'safelog_ai/auxiliary_classifier.py',
    'model_factory_source_sha256': 'safelog_ai/presence_classifier.py',
}


def seed_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_train_replay(manifest, auxiliary):
    """First two native DACL full rows per known crack/spall joint tag."""
    if manifest['split'] != 'train' or auxiliary['split'] != 'train':
        raise ValueError('Only original TRAIN metadata may be replayed')
    if auxiliary['classes'] != list(AUX_CLASSES):
        raise ValueError('Original auxiliary class order changed')
    targets = [manifest['classes'].index(c) for c in TARGETS]
    aux = {item['image']: item['targets'] for item in auxiliary['items']}
    groups = {state: [] for state in ('00', '10', '01', '11')}
    for item in manifest['items'][:manifest['full_count']]:
        if item['domain'] != 'dacl':
            continue
        if 'parent_image' in item or any(item['targets'][k] not in (0, 1) for k in targets):
            raise ValueError('Replay requires native known DACL full-photo targets')
        state = ''.join(str(item['targets'][k]) for k in targets)
        if len(groups[state]) < 2:
            if item['image'] not in aux or len(aux[item['image']]) != len(AUX_CLASSES):
                raise ValueError('Replay photo has no native auxiliary tags')
            groups[state].append(item)
    if any(len(items) != 2 for items in groups.values()):
        raise ValueError('Two TRAIN originals per joint target state are required')
    selected = [item for items in groups.values() for item in items]
    return selected, aux


def loss_weights(manifest, auxiliary):
    """Reproduce the unchanged trainer's core photo/pixel/auxiliary weights."""
    items = manifest['items']
    classes = manifest['classes']
    labels = torch.tensor([item['targets'] for item in items])
    domains = torch.tensor([DOMAINS[item['domain']] for item in items])
    sampling = torch.ones(len(items), dtype=torch.float64)
    for label in TARGETS:
        k = classes.index(label)
        ratio = min(3., max(1., len(items) / (2 * max(1, int((labels[:, k] == 1).sum())))))
        sampling = torch.maximum(sampling, torch.where(labels[:, k] == 1, ratio, 1.).double())
    full = manifest['full_count']
    sampling[full:] *= full / (len(items) - full) * .25 / .75
    weights = []
    for domain, exposure in enumerate((.7, .1, .2)):
        sampling[domains == domain] *= exposure / sampling[domains == domain].sum()
        w = sampling * (domains == domain)
        pos = ((labels == 1) * w[:, None]).sum(0)
        neg = ((labels == 0) * w[:, None]).sum(0)
        weights.append((neg / pos.clamp_min(1e-8)).clamp(.2, 6).float())
    counts = manifest['audit']['per_label_pixel_cells']
    pixel = torch.tensor([min(20., max(1., counts[c]['negative'] / max(1, counts[c]['positive'])))
                          for c in classes])
    tags = torch.tensor([item['targets'] for item in auxiliary['items']])
    auxiliary_weights = ((tags == 0).sum(0) / (tags == 1).sum(0).clamp_min(1)).clamp(.2, 6).float()
    return torch.stack(weights), pixel, auxiliary_weights


def batch_fingerprint(batch):
    digest = hashlib.sha256()
    for value in batch:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(json.dumps(list(array.shape)).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def augmented_replay(items, auxiliary, seed, batches=3):
    # Four workers and persistent workers match the actual trainer. The bounded
    # eight-photo sampler is deliberately not a full-epoch sampling assertion.
    torch.manual_seed(seed + 2)
    sampler = WeightedRandomSampler(torch.ones(len(items), dtype=torch.float64),
                                    batches * 8, replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(SpatialPhotos(items, auxiliary, len(items), DOMAINS),
                        batch_size=8, sampler=sampler, num_workers=4,
                        persistent_workers=True, pin_memory=True,
                        generator=torch.Generator().manual_seed(seed + 1))
    iterator = iter(loader)
    replay = []
    first_batch = None
    try:
        for _ in range(batches):
            batch = next(iterator)
            if first_batch is None:
                first_batch = batch
            replay.append({'augmented_batch_sha256': batch_fingerprint(batch),
                           'sampled_indices_sha256': hashlib.sha256(
                               batch[-1].numpy().astype('<i8', copy=False).tobytes()).hexdigest()})
    finally:
        # Closing persistent workers prevents a second replay from retaining
        # worker processes or pinned prefetch buffers on Windows.
        iterator._shutdown_workers()
    return replay, first_batch


def gradient_evidence(model):
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError('AMP loss produced missing or nonfinite gradient evidence')
    first = model.detail_head[0].weight.grad
    projection = model.detail_head[-1].weight.grad
    if first is None or projection is None:
        raise ValueError('The new detail branch is disconnected from the loss')
    return {'gradient_tensors': len(gradients), 'all_gradients_finite': True,
            'projection_gradient_nonzero': bool(torch.count_nonzero(projection)),
            'upstream_gradient_nonzero': bool(torch.count_nonzero(first)),
            'projection_gradient_l2': float(torch.linalg.vector_norm(projection.float())),
            'upstream_gradient_l2': float(torch.linalg.vector_norm(first.float()))}


def diagnostic_latency(model, inputs):
    """One warmup and three synchronized FP32 batch timings, not app p95."""
    model.eval()
    with torch.inference_mode():
        model.forward_training(inputs)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(3):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model.forward_training(inputs)
            end.record()
            torch.cuda.synchronize()
            timings.append(float(start.elapsed_time(end)))
    return {'precision': 'FP32', 'warmup_batches': 1, 'measured_batches': 3,
            'batch_size': int(inputs.shape[0]), 'batch_milliseconds': timings,
            'mean_batch_milliseconds': sum(timings) / len(timings),
            'diagnostic_photos_per_second': inputs.shape[0] * 1000 * len(timings) / sum(timings),
            'baseline_allocated_bytes': baseline,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_increment_bytes': max(0, torch.cuda.max_memory_allocated() - baseline),
            'production_latency_or_p95_measured': False}


def run(initial):
    if not torch.cuda.is_available():
        raise ValueError('This preflight requires an actual CUDA AMP backward; no CPU substitute')
    torch.set_num_threads(4)
    protocol = read(PROTOCOL)
    if protocol['schema'] != 'facility_detail_architecture_protocol_v1':
        raise ValueError('Architecture protocol changed')
    paths = {'core': ROOT / 'data/facility-spatial-training/train.json',
             'auxiliary': ROOT / 'data/facility-auxiliary-training/train.json',
             'initial': initial, 'protocol': PROTOCOL,
             'app_profile': ROOT / 'reports/facility-inference-profile.json'}
    before = {key: sha(path) for key, path in paths.items()}
    if (before['initial'] != protocol['initial_weights_sha256']
            or before['core'] != protocol['core_spatial_manifest_sha256']
            or before['auxiliary'] != protocol['auxiliary_manifest_sha256']):
        raise ValueError('Frozen initializer or original TRAIN metadata changed')
    manifest, auxiliary = read(paths['core']), read(paths['auxiliary'])
    if manifest['classes'] != protocol['classes'] or len(manifest['classes']) != 7:
        raise ValueError('Public facility class contract changed')
    selected, aux = select_train_replay(manifest, auxiliary)
    checkpoint = torch.load(initial, map_location='cpu', weights_only=True)
    if (checkpoint['architecture'] != AUX_ARCH or checkpoint['classes'] != protocol['classes']
            or checkpoint['auxiliary_classes'] != list(AUX_CLASSES) or checkpoint['imgsz'] != 640):
        raise ValueError('Original auxiliary initializer contract changed')
    seed = protocol['seed']
    randomness = {'sampler_seed': seed, 'training_worker_seed': seed + 1,
                  'post_model_seed': seed + 2,
                  'validation_worker_seeds': {d: seed + 100 + k for d, k in DOMAINS.items()}}
    if randomness != protocol['loader_randomness'] or (seed, seed + 1, seed + 2) != (54, 55, 56):
        raise ValueError('Declared sampler/worker/post-model randomness changed')
    models, transfer, construction_rng = {}, {}, {}
    for variant, constructor in (('control', AuxiliaryClassifier), ('detail', DetailClassifier)):
        seed_model(seed)
        model = constructor(7, pretrained=False)
        transfer[variant] = load_auxiliary_initializer(model, checkpoint['state_dict'])
        construction_rng[variant] = torch.get_rng_state().clone()
        models[variant] = model
    if not torch.equal(construction_rng['control'], construction_rng['detail']):
        raise ValueError('Additional architecture construction consumed the shared CPU RNG stream')
    replays, batches = {}, {}
    for variant in ('control', 'detail'):
        seed_model(seed)
        replays[variant], batches[variant] = augmented_replay(selected, aux, seed)
    if replays['control'] != replays['detail']:
        raise ValueError('Same four-worker loader seeds did not replay identical augmentations')
    transform = image_transform(640)
    originals = []
    for item in selected:
        with Image.open(ROOT / item['image']) as handle:
            originals.append(transform(handle.convert('RGB')))
    image = torch.stack(originals).to('cuda')
    shapes, control_outputs, feature_shapes = {}, None, []
    latency = {}
    for variant, model in models.items():
        model.to('cuda').eval()
        handle = None
        if variant == 'detail':
            handle = model.backbone['3'].register_forward_hook(
                lambda _module, _args, output: feature_shapes.append(list(output.shape)))
        with torch.inference_mode():
            values = model.forward_training(image)
        if handle is not None:
            handle.remove()
        if ([list(v.shape) for v in values] != [[8, 7], [8, 7, 80, 80], [8, 19]]
                or not all(bool(torch.isfinite(v).all()) for v in values)):
            raise ValueError('Finite public/map/auxiliary shape contract failed')
        if variant == 'control':
            control_outputs = tuple(v.detach().cpu().clone() for v in values)
        elif not all(torch.equal(v.detach().cpu(), original) for v, original in zip(values, control_outputs)):
            raise ValueError('Zero-initialized residual changed the initial FP32 predictions')
        with torch.inference_mode():
            if not torch.equal(model(image), values[0]):
                raise ValueError('Public forward differs from the training photo logits')
        shapes[variant] = {'public_shape': list(values[0].shape), 'pixel_shape': list(values[1].shape),
                           'auxiliary_shape': list(values[2].shape),
                           'parameter_count': sum(p.numel() for p in model.parameters())}
        latency[variant] = diagnostic_latency(model, image)
    if feature_shapes != [[8, 24, 160, 160]]:
        raise ValueError('Stride4 early feature is not produced once at the declared shape')
    del control_outputs, values
    # The source checkpoint and control model remain untouched. Only this
    # disposable treatment copy takes two steps, then is discarded.
    model = models['detail'].train()
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone}
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': protocol['backbone_lr']},
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': protocol['head_lr']},
    ], weight_decay=.0002)
    # A modest scale avoids disposable integration checks depending on a first
    # dynamic-scale overflow. The actual trainer retains its own GradScaler.
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    photo_weights, pixel_weights, auxiliary_weights = loss_weights(manifest, auxiliary)
    photo_weights, pixel_weights, auxiliary_weights = (value.to('cuda') for value in
                                                      (photo_weights, pixel_weights, auxiliary_weights))
    batch = batches['control']
    amp_image = batch[0].to('cuda')
    emphasis = torch.ones(7, device='cuda')
    for label in TARGETS:
        emphasis[manifest['classes'].index(label)] = 2.
    steps = []
    torch.cuda.reset_peak_memory_stats()
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda'):
            scores, maps, tags = model.forward_training(amp_image)
            loss = masked_focal(scores, batch[1].to('cuda'), batch[2].to('cuda'),
                                photo_weights[batch[3]], 1., emphasis)
            loss += spatial_loss(maps, batch[4].to('cuda'), batch[5].to('cuda'), pixel_weights)
            loss += protocol['auxiliary_weight'] * masked_focal(
                tags, batch[6].to('cuda'), batch[7].to('cuda'), auxiliary_weights, 1.)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Disposable CUDA AMP loss is nonfinite')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        proof = gradient_evidence(model)
        if not proof['projection_gradient_nonzero'] or (step == 2 and not proof['upstream_gradient_nonzero']):
            raise ValueError('Residual projection or subsequent upstream gradient cannot learn')
        scaler.step(optimizer)
        scaler.update()
        if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise ValueError('Disposable AMP optimizer step produced nonfinite states')
        steps.append({'step': step, 'loss': float(loss.detach()), **proof})
    torch.cuda.synchronize()
    after = {key: sha(path) for key, path in paths.items()}
    if before != after:
        raise ValueError('Preflight changed a frozen input or app profile')
    return {
        'status': 'passed', 'schema': 'facility_detail_preflight_v1',
        'scope': 'Eight original TRAIN DACL full photos, two per known crack/spall joint tag; disposable checks only',
        'train_photo_count': 8, 'train_joint_tag_counts': {state: 2 for state in ('00', '10', '01', '11')},
        'public_shape': [8, 7], 'pixel_shape': [8, 7, 80, 80], 'auxiliary_shape': [8, 19],
        'early_feature_shape': feature_shapes[0], 'early_feature_forward_calls': len(feature_shapes),
        'initial_fp32_outputs_equal': True, 'initial_state_transfer': transfer,
        'constructor_cpu_rng_equal': True, 'loader_randomness': randomness,
        'replay_num_workers': 4, 'replay_batches': 3, 'augmented_batch_replay_equal': True,
        'bounded_replay_digests': replays['control'],
        'replay_scope': 'Same eight-photo loader pipeline only; completed epoch digests must independently match full sampling',
        'finite_loss_and_gradients': True, 'detail_upstream_gradient_after_projection_step': True,
        'disposable_optimizer_steps': steps, 'amp_initial_scale': 1024.,
        'checkpoint_saved': False, 'source_checkpoint_unchanged': True, 'frozen_inputs_unchanged': True,
        'initial_weights_sha256': before['initial'], 'core_spatial_manifest_sha256': before['core'],
        'auxiliary_manifest_sha256': before['auxiliary'], 'study_protocol_sha256': before['protocol'],
        'app_profile_sha256': before['app_profile'],
        'executed_sources_sha256': {key: sha(ROOT / relative) for key, relative in SOURCE_PATHS.items()},
        'preflight_script_sha256': sha(Path(__file__)), 'architecture': DETAIL_ARCH,
        'model_shapes_and_parameters': shapes, 'latency_diagnostic': latency,
        'amp_two_step_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'device': torch.cuda.get_device_name(), 'accuracy_measured_by_this_check': False,
        'independent_field_safety_verified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path)
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    initial = (args.initial or ROOT / 'runs' / protocol['reference'] / 'best.pt').resolve()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save(OUTPUT, {'status': 'running', 'schema': 'facility_detail_preflight_v1'})
    try:
        result = run(initial)
    except Exception as error:
        save(OUTPUT, {'status': 'failed', 'schema': 'facility_detail_preflight_v1',
                      'error_type': type(error).__name__, 'error': str(error), 'checkpoint_saved': False})
        raise
    save(OUTPUT, result)
    print(json.dumps({'status': result['status'], 'initial_fp32_outputs_equal': True,
                      'augmented_batch_replay_equal': True, 'disposable_steps': 2,
                      'checkpoint_saved': False}))


if __name__ == '__main__':
    main()
