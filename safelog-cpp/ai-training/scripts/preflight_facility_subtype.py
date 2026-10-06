"""Disposable TRAIN-only CUDA replay for the fixed paired subtype sampler.

The actual study starts from the unchanged initializer after these disposable
updates. No VAL/TEST accuracy is measured here.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import random
import sys

import numpy as np
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier
from safelog_ai.presence_classifier import PresenceClassifier, MEAN, STD
from safelog_ai.spatial_classifier import spatial_loss
from scripts.facility_resolution_study import ResolutionPhotos
from scripts.facility_subtype_study import (PROTOCOL, REFERENCE, prepare_data,
    protected_hashes, require, validate_protocol, write)
from scripts.train_facility_target import DOMAINS, masked_focal
from scripts.train_facility_subtype import load_initial_state
from scripts.verify_facility_resolution import local_path, read, sha

OUTPUT = 'runs/facility-subtype-preflight.json'
DIRECTORY = 'runs/facility-subtype-preflight'
LEDGER = DIRECTORY + '/protected-inputs.json'


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
    return sorted(paths)


def replay_positions(data, negative_draws):
    """Cover all sources, crops and an actual changed subtype draw position."""
    left, right = data['epoch_draws'][0], negative_draws[0]
    rows, full = data['items'], data['full_count']
    chosen = []
    def add(predicate):
        position = next((p for p, i in enumerate(left) if p not in chosen and predicate(p, rows[int(i)], int(i))), None)
        require(position is not None, 'Actual paired TRAIN replay coverage is incomplete')
        chosen.append(position)
    for domain in DOMAINS:
        add(lambda p, row, i, domain=domain: row['domain'] == domain and i < full)
    add(lambda p, row, i: i >= full)
    add(lambda p, row, i: left[p] != right[p])
    add(lambda p, row, i: row['targets'][0] == 1 and row['targets'][1] == 1)
    add(lambda p, row, i: i >= full and -1 in row['targets'][:2])
    add(lambda p, row, i: left[p] == right[p])
    require(len(chosen) == len(set(chosen)) == 8, 'Eight paired replay positions are required')
    return chosen


def batch(data, indices):
    random.seed(57); np.random.seed(57); torch.manual_seed(57)
    dataset = ResolutionPhotos(data['items'], data['auxiliary'], data['full_count'], DOMAINS, imgsz=640)
    return default_collate([dataset[int(i)] for i in indices])


def main():
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    require(torch.cuda.is_available(), 'Actual paired preflight requires CUDA')
    require(not (ROOT / OUTPUT).exists() and not (ROOT / DIRECTORY).exists(),
            'Preserve existing preflight evidence')
    p = validate_protocol(read(ROOT / PROTOCOL), ROOT)
    protected = protected_hashes(ROOT)
    data = {v: prepare_data(ROOT, p, v) for v in ('control', 'negative')}
    paths = input_paths(data['control'], protected)
    print({'phase': 'hash-original-train-inputs', 'files': len(paths)}, flush=True)
    before = snapshot_inputs(ROOT, paths)
    positions = replay_positions(data['control'], data['negative']['epoch_draws'])
    indices = {v: d['epoch_draws'][0, positions] for v, d in data.items()}
    batches = {v: batch(data[v], indices[v]) for v in data}
    a, b = batches['control'], batches['negative']
    require(torch.equal(a[1][:, :2], b[1][:, :2]) and torch.equal(a[2][:, :2], b[2][:, :2])
            and torch.equal(a[3], b[3]) and torch.equal(a[-2], b[-2]),
            'Actual two-target/source/full-crop replay differs')
    unchanged = [j for j in range(8) if indices['control'][j] == indices['negative'][j]]
    require(unchanged and all(torch.equal(x[unchanged], y[unchanged]) for x, y in zip(a, b)),
            'Unchanged original TRAIN row replay differs')
    checkpoint = torch.load(ROOT / 'runs' / REFERENCE / 'best.pt', map_location='cpu', weights_only=True)
    proofs = {}; (ROOT / DIRECTORY).mkdir(exist_ok=False)
    for variant, values in batches.items():
        torch.manual_seed(56); model = AuxiliaryClassifier(7, pretrained=False)
        transfer = load_initial_state(model, checkpoint, p['classes'], checkpoint['split_sha256'])
        model.to('cuda'); model.eval(); torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            public = model(values[0].cuda()); photo, maps, aux = model.forward_training(values[0].cuda())
            require(tuple(public.shape) == (8, 7) and tuple(maps.shape) == (8, 7, 80, 80)
                    and tuple(aux.shape) == (8, 19) and torch.equal(public, photo)
                    and all(torch.isfinite(t).all() for t in (public, maps, aux)),
                    'Actual CUDA output contract failed')
        backbone = list(model.backbone.parameters()); ids = {id(v) for v in backbone}
        head = [v for v in model.parameters() if id(v) not in ids]
        optimizer = torch.optim.AdamW([{'params': backbone, 'lr': p['backbone_lr']},
                                      {'params': head, 'lr': p['head_lr']}], weight_decay=.0002)
        updates = {'count': 0}
        def updated(optimizer, args, kwargs): updates['count'] += 1
        hook = optimizer.register_step_post_hook(updated)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda'):
            photo, maps, aux = model.forward_training(values[0].cuda())
            d = data[variant]; emphasis = torch.tensor([2., 2., 1., 1., 1., 1., 1.], device='cuda')
            loss = masked_focal(photo, values[1].cuda(), values[2].cuda(),
                                d['sampling_data']['photo_weights'].cuda()[values[3]], 1., emphasis)
            loss += spatial_loss(maps, values[4].cuda(), values[5].cuda(),
                                 d['supervision_weights']['pixel_weights'].cuda())
            loss += .5 * masked_focal(aux, values[6].cuda(), values[7].cuda(),
                                     d['supervision_weights']['auxiliary_weights'].cuda(), 1.)
        require(torch.isfinite(loss), 'Actual disposable loss is nonfinite')
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); hook.remove()
        require(updates['count'] == 1, 'Disposable AMP update was skipped')
        path = ROOT / DIRECTORY / f'{variant}.pt'
        torch.save({**{k: v for k, v in checkpoint.items() if k != 'state_dict'},
                    'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'mean': MEAN, 'std': STD}, path)
        peak = torch.cuda.max_memory_allocated(); del model, optimizer, scaler
        torch.cuda.empty_cache(); classifier = PresenceClassifier(path, 'cpu')
        with torch.inference_mode():
            public = classifier.model(values[0][:1]); photo, maps, aux = classifier.model.forward_training(values[0][:1])
        require(torch.equal(public, photo) and all(torch.isfinite(t).all() for t in (photo, maps, aux)),
                'Actual strict CPU factory reload failed')
        proofs[variant] = {'actual_batch_size': 8, 'actual_optimizer_steps': 1, 'amp_skipped_steps': 0,
            'scaler_initial_scale': 1024., 'public_shape': [8, 7], 'pixel_shape': [8, 7, 80, 80],
            'auxiliary_shape': [8, 19], 'parameter_count': sum(t.numel() for t in classifier.model.parameters()),
            'all_outputs_finite': True, 'cpu_reload_verified': True, 'strict_factory_reload_verified': True,
            'public_output_equals_training_photo_output': True, 'disposable_checkpoint_sha256': sha(path),
            'peak_cuda_allocated_bytes': peak, 'initial_state_transfer': transfer}
        del classifier
    print({'phase': 'rehash-original-train-inputs', 'files': len(paths)}, flush=True)
    after = snapshot_inputs(ROOT, paths)
    require(before == after and protected_hashes(ROOT) == protected,
            'Preflight changed original TRAIN or protected experiment bytes')
    validate_protocol(p, ROOT)
    write(ROOT / LEDGER, {'schema': 'facility_subtype_private_input_ledger_v1', 'local_only': True,
          'split': 'train', 'snapshots_before': before, 'snapshots_after': after,
          'core_manifest_sha256': p['core_spatial_manifest_sha256'], 'all_inputs_preserved': True})
    result = {'schema': 'facility_subtype_preflight_v1', 'status': 'passed',
        'verified_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(ROOT / PROTOCOL),
        'source_sha256': p['source_sha256'], 'protected_file_sha256': protected, 'pairs': proofs,
        'actual_two_target_source_full_crop_replay_equal': True, 'unchanged_row_tensors_equal': True,
        'unique_paired_draw_positions': 8, 'changed_replay_positions': int(np.count_nonzero(indices['control'] != indices['negative'])),
        'replay_positions_sha256': hashlib.sha256(np.asarray(positions, dtype='<i8').tobytes()).hexdigest(),
        'protected_input_ledger_path': LEDGER, 'protected_input_ledger_sha256': sha(ROOT / LEDGER),
        'protected_input_file_count': len(before), 'all_original_sha_size_mtime_preserved': True,
        'new_training_epochs': 0, 'source_test_inference_executed': False, 'app_model_promoted': False,
        'accuracy_measured': False, 'label_changes': 0, 'new_masks': 0}
    write(ROOT / OUTPUT, result)
    print({'status': 'passed', 'paired_train_draw_positions': 8, 'disposable_amp_steps': 2,
           'protected_original_files': len(before), 'new_epochs': 0}, flush=True)


if __name__ == '__main__': main()
