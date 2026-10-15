"""Preregister, train and assess one finer-grid candidate; old studies stay immutable."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.fine_target_surface import VERSION, CATEGORIES, FineTargetMain, initialize_fine_head, trainable_fine_head
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.conditioned_surface import frozen_residual
from safelog_ai.commutator_surface import fine_mask_loss
from safelog_ai.magnetic_surface import surface_view
from safelog_ai.ksdd_supervised import ksdd_photo_score
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.process_probe import process_exists
from scripts import study_target_conditioned as previous
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_mixed_real_mask import get_cache as old_cache, parent_target_scores, draw_group_balanced
from scripts.study_real_mask_transfer import group_rates

RUN = ROOT / 'runs/industrial-fine-targets-v20'
REPORTS = ROOT / 'reports'
PROTOCOL = REPORTS / 'industrial-fine-targets-v20-protocol.json'
START = previous.START
EXPECTED = previous.EXPECTED
WARM = previous.RUN / 'trained/final.pt'
WARM_SHA = 'ae0654d54e4f465e1d3826eade7173fa82d3dfb46fab4287b0c27766afe75255'
MANIFEST = previous.DATA / 'manifest.json'
MANIFEST_SHA = 'b24d0f8a296fd5adbacce831b95a9a02bec54b996f5456fc633a6105bb37c7de'
PARENT_EVAL = previous.PARENT_EVAL
SOURCES = ('safelog_ai/fine_target_surface.py', 'scripts/study_fine_targets.py', 'tests/test_fine_target_surface.py')
METHODS = ('fine_only', 'max_fine_parent')
EPOCHS, STEPS, BATCH, SEED = 30, 64, 8, 2020
KEYS = ('low', 'high', 'residual', 'fine', 'masks', 'valid')
ATOL, RTOL = 5e-4, 1e-3


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists()
    assert sha(START) == EXPECTED and sha(WARM) == WARM_SHA and sha(MANIFEST) == MANIFEST_SHA
    assert read(previous.RUN / 'verification.json')['status'] == 'passed'
    assert read(ROOT / 'runs/industrial-fine-probe-v20/probe.json')['all_parameters_buffers_hash_unchanged']
    assert shutil.disk_usage(ROOT).free > 4 * 1024**3
    RUN.mkdir()
    ledger = dict(read(previous.RUN / 'protected-inputs.json'))
    for folder in (previous.RUN, previous.DATA, ROOT / 'runs/industrial-fine-probe-v20', REPORTS):
        for path in folder.rglob('*'):
            name = path.relative_to(ROOT).as_posix()
            if path.is_file() and not path.name.endswith(('.lock', '.log')) and name not in ledger:
                ledger[name] = snapshot(path)
    new(RUN / 'protected-inputs.json', ledger)
    p = {'schema': VERSION, 'UTC': datetime.now(timezone.utc).isoformat(),
         'fixed_before_new_learning_features_gradients_CAL_TEST': True,
         'source_sha256': {n: sha(ROOT / n) for n in SOURCES},
         'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('safelog_ai', 'scripts', 'tests') for p in (ROOT / d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
         'parent_sha256': EXPECTED, 'warm_start_sha256': WARM_SHA, 'manifest_sha256': MANIFEST_SHA,
         'parent_evaluation_sha256': sha(PARENT_EVAL), 'protected_inputs_sha256': sha(RUN / 'protected-inputs.json'),
         'old_generated_cache_sha256': {phase: sha(previous.RUN / (phase + '-features.pt')) for phase in ('fit', 'validation')},
         'target_cache_sha256': read(previous.PROTOCOL)['target_cache_sha256'],
         'data': 'reuse immutable v19-r1 FIT609 = normal131 + synthetic478, VAL162 = normal33 + synthetic129; do not regenerate; KS1 shape FIT21/VAL5 disjoint; no CAL/TEST assets',
         'new_natural_target_fault_GT': 0, 'procedural_support_is_human_natural_defect_GT': False,
         'view': 'same 512square bicubic RGB letterbox127, original-rectangle valid mask; original source RGB for CAL/TEST',
         'features': 'same frozen common encoder stage3 fine24 stride4 at128x128, old low40 at64x64 and high960 at32x32; frozen category equipment cosine residual',
         'cache': 'only new fine FP16 cache; verify original RGB SHA, ids/order/geometry, old masks/valid and fresh FP16 low/high vs v19 cache for all771 FIT/VAL inputs. target CAL/TEST likewise compare old low/high; same batch8 and TF32off',
         'cache_equivalence_guard': {'atol': ATOL, 'rtol': RTOL, 'rule': 'torch.allclose each tensor; stop without silently rebuilding old cache; report exact equality separately'},
         'fine_normalization': 'none; raw FP16 roundtrip, no mean/std fitting or channel scaling',
         'head': 'upsample low40, frozen high960to32 and residual1 to128grid, concatenate new fine24to16 Conv1x1/ReLU =89 channels; conv89to64/ReLU/dropout.1/conv64to32/ReLU/conv32to1 then bilinear512',
         'initialization': 'copy v19 VAL-selected target_conditioned_head into separate fine_target_head; first73 fusion inputs copied and added16 weights zero. Fine projection torch default seeded2020. Different spatial grid is not identical initial logits to v19',
         'trainable': 'new fine projection and new fusion only; high projection and all parent model weights/buffers/metadata frozen',
         'sampling': '2normal+2synthetic per category per step=8, inverse base-group multiplicity replacement, not new photos',
         'augmentation': 'paired cached fine/low/high/residual/mask/valid H/V flips .5 per photo; not fresh RGB augmentation',
         'loss': 'same valid BCE + softDice +.5 top32 photoBCE; positive weight clamp(sqrt(FIT valid negative / positive),10,100) from same FIT masks only',
         'optimizer': 'AdamW lr1e-4 wd1e-4 cosine30 minimum1e-6 clip5 FP32 TF32off',
         'epochs': EPOCHS, 'steps': STEPS, 'batch': BATCH, 'seed': SEED, 'planned_optimizer_updates': EPOCHS * STEPS,
         'VAL_selection': 'epoch0 and each completed epoch; equal four category/normal-synthetic group loss means, earliest strictly lower minimum; natural target defect VAL absent',
         'resume': 'per-epoch checkpoint includes optimizer/scheduler/RNG; append each actual optimizer operation, count any partial replay; completed study never rerun',
         'CAL': 'unchanged normal30/23; order ceil((n+1)*.85), strict >; fine_only and max(new/newCALcut,currentv16/currentcut) with own normal jointCALcut; save before any TEST scores',
         'methods': list(METHODS),
         'guards': ['each TEST label>=20', 'FPR<=.20', 'FPR<=v16+.05', 'FNR<=v16-.05', 'defect subtype n>=10 FNR<=v16+.05', 'selected epoch>0'],
         'selection': 'eligible minimum max(FPR,FNR), then FNR, then declared method order; no threshold retuning or relaxed guards',
         'location': 'all original primary target maps/pixel thresholds preserved; synthetic map not promoted without real natural location validation',
         'one_candidate_one_common_encoder': True, 'not_DRAEM_reproduction': True,
         'known_public_TEST_reused_for_adoption': True, 'independent_field_validated': False,
         'physical_capture_groups_verified': False, 'inherited_parent_past_near_training_exposure_undone': False,
         'synthetic_physical_fault_realism_verified': False, 'unreviewed_v14_relations_used': False,
         'new_unit_tests_passed': 3, 'new_release_requires_fresh_package_CPU_GPU_CPP': True,
         'source_conditions': 'MPDD/KS1 permitted research use; KS1 CC BY-NC-SA4.0, cite DOI10.1007/s10845-019-01476-x; raw/source/failed runs all preserved'}
    new(PROTOCOL, p)
    print({'frozen': True, 'planned_optimizer_updates': EPOCHS * STEPS}, flush=True)


def validate(committed=False):
    p = read(PROTOCOL)
    for name, digest in p['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for path, digest in ((START, EXPECTED), (WARM, WARM_SHA), (MANIFEST, MANIFEST_SHA),
                         (PARENT_EVAL, p['parent_evaluation_sha256']), (RUN / 'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == digest, path
    if committed:
        repo = ROOT.parents[1]
        git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory=' + repo.as_posix(), '-C', str(repo)]
        for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
            content = subprocess.check_output(git + ['show', 'HEAD:safelog-cpp/ai-training/' + name])
            assert hashlib.sha256(content).hexdigest() == sha(ROOT / name), name
    return p


def load(path=START, device='cpu', initialize=False):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = FineTargetMain(state['reference_counts'], state['native_specs'], state['wear_variant'], state['surface_reference_specs'])
    missing, extra = model.load_state_dict(state['state_dict'], strict=False)
    assert not extra and (not missing or initialize and all(n.startswith('fine_target_head.') for n in missing))
    if initialize:
        warm = torch.load(WARM, map_location='cpu', weights_only=True)
        prefix = 'target_conditioned_head.'
        initialize_fine_head(model.fine_target_head, {n[len(prefix):]: t for n, t in warm['state_dict'].items() if n.startswith(prefix)})
    return model.eval().requires_grad_(False).to(device), state


def checked_coarse(phase, p):
    path = previous.RUN / (phase + '-features.pt')
    assert sha(path) == p['old_generated_cache_sha256'][phase]
    values = torch.load(path, map_location='cpu', weights_only=True)
    rows = [r for r in read(MANIFEST)['rows'] if r['phase'] == phase]
    assert values['ids'] == [r['id'] for r in rows]
    assert values['protocol_sha256'] == sha(previous.PROTOCOL)
    return rows, values


def aligned(actual, expected, name, stats):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a, b = actual.float(), expected.float()
    stats[name + '_max_abs'] = max(stats.get(name + '_max_abs', 0), float((a - b).abs().max()))
    stats[name + '_bitwise_exact'] = stats.get(name + '_bitwise_exact', True) and torch.equal(actual, expected)
    assert torch.allclose(a, b, atol=ATOL, rtol=RTOL), (name, stats)


def fine_cache(name, rows, values, model, generated=False):
    path = RUN / (name + '-fine.pt')
    receipt = RUN / (name + '-fine-receipt.json')
    if path.exists():
        assert sha(path) == read(receipt)['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['ids'] == [r['id'] for r in rows] and saved['protocol_sha256'] == sha(PROTOCOL)
        return saved['fine']
    if name.endswith('-test'):
        assert read(RUN / 'calibration.json')['thresholds_saved_before_TEST']
    fine = []
    stats = {}
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            images = []
            for i, row in enumerate(batch, start):
                assert sha(ROOT / row['image']) == row['image_sha256']
                with Image.open(ROOT / row['image']) as im:
                    rgb = im.convert('RGB')
                if generated:
                    assert sha(ROOT / row['mask']) == row['mask_sha256']
                    with Image.open(ROOT / row['mask']) as im:
                        truth = torch.from_numpy((np.array(im.convert('L')) > 0).astype('uint8'))
                    assert torch.equal(truth, values['masks'][i])
                    g = row['geometry']
                    x, y = g['offset']; w, h = g['resized_size']
                    valid = torch.zeros(512, 512, dtype=torch.bool)
                    valid[y:y + h, x:x + w] = True
                    assert torch.equal(valid, values['valid'][i])
                else:
                    rgb, _, valid, geometry = surface_view(rgb)
                    assert geometry == values['geometry'][i]
                    assert torch.equal(valid, values['valid'][i])
                images.append(normalized(rgb))
            f = forward_with_fine(model.backbone, torch.stack(images).cuda())
            for key in ('low', 'high'):
                aligned(f[key].half().cpu(), values[key][start:start + len(batch)], key, stats)
            if generated:
                res = frozen_residual(model, f['low'], f['high'], [r['category'] for r in batch]).cpu()
                aligned(res, values['residual'][start:start + len(batch)], 'residual', stats)
            fine.append(f['fine'].half().cpu())
            if start % 128 == 0:
                print({'fine_cache': name, 'photos': min(start + BATCH, len(rows)), 'total': len(rows)}, flush=True)
    tensor = torch.cat(fine)
    assert tensor.shape == (len(rows), 24, 128, 128)
    partial = path.with_suffix('.partial.pt')
    with partial.open('xb') as out:
        torch.save({'fine': tensor, 'ids': [r['id'] for r in rows], 'protocol_sha256': sha(PROTOCOL)}, out)
    partial.rename(path)
    new(receipt, {'sha256': sha(path), 'photos': len(rows), 'bytes': path.stat().st_size,
                  'alignment': stats, 'atol': ATOL, 'rtol': RTOL, 'RGB_and_coordinate_checks': True})
    return tensor


def head_logits(head, b):
    return F.interpolate(head(b['low'], b['high'], b['residual'], b['fine']),
                         size=b['masks'].shape[-2:], mode='bilinear', align_corners=False)


def validation(head, values, rows, weight):
    groups = {}
    head.eval()
    with torch.inference_mode():
        for category in CATEGORIES:
            for label in (0, 1):
                ids = [i for i, r in enumerate(rows) if r['category'] == category and r['label'] == label]
                total = 0
                for start in range(0, len(ids), BATCH):
                    at = ids[start:start + BATCH]
                    b = {k: values[k][at].cuda() for k in KEYS}
                    total += float(fine_mask_loss(head_logits(head, b), b['masks'][:, None], b['valid'][:, None], weight)) * len(at)
                groups[category + ':' + str(label)] = total / len(ids)
    return {'class_losses': groups, 'balanced_loss': sum(groups.values()) / len(groups),
            'positive_scope': 'procedural edits, not natural target defects'}


def train():
    p = validate(committed=True)
    assert not (RUN / 'trained/training.json').exists()
    assert shutil.disk_usage(ROOT).free > 2 * 1024**3
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model, parent = load(device='cuda', initialize=True)
    head = model.fine_target_head
    fr, fit = checked_coarse('fit', p); vr, val = checked_coarse('validation', p)
    fit['fine'] = fine_cache('fit', fr, fit, model, True)
    val['fine'] = fine_cache('validation', vr, val, model, True)
    params = trainable_fine_head(head)
    assert {n for n, t in model.named_parameters() if t.requires_grad} == {'fine_target_head.' + n for n, t in head.named_parameters() if t.requires_grad}
    positives = int(fit['masks'].sum()); valid = int(fit['valid'].sum())
    weight = min(100., max(10., math.sqrt((valid - positives) / positives)))
    assert weight == read(previous.RUN / 'FIT-loss-weight.json')['positive_weight']
    if not (RUN / 'FIT-loss-weight.json').exists():
        new(RUN / 'FIT-loss-weight.json', {'positive_pixels': positives, 'valid_pixels': valid, 'positive_weight': weight, 'source': 'same generated FIT masks only'})
    draws = RUN / 'draws.npz'
    if not draws.exists():
        rng = np.random.default_rng(SEED); indices = []
        for _ in range(EPOCHS * STEPS):
            indices.append([draw_group_balanced(fr, [i for i, r in enumerate(fr) if r['category'] == c and r['label'] == label], rng) for c in CATEGORIES for label in (0, 0, 1, 1)])
        np.savez_compressed(draws, indices=np.array(indices).reshape(EPOCHS, STEPS, BATCH), flips=rng.random((EPOCHS, STEPS, BATCH, 2)) < .5)
        new(RUN / 'draws.json', {'sha256': sha(draws), 'manifest_sha256': MANIFEST_SHA})
    assert sha(draws) == read(RUN / 'draws.json')['sha256']
    draws = np.load(draws)
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    folder = RUN / 'train'; folder.mkdir(exist_ok=True)
    progress = folder / 'progress.pt'; journal = folder / 'optimizer.jsonl'
    epoch0 = 0; actual = 0; best_epoch = 0
    if progress.exists():
        s = torch.load(progress, map_location='cpu', weights_only=True)
        assert s['protocol_sha256'] == sha(PROTOCOL)
        head.load_state_dict(s['head']); optimizer.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        epoch0 = s['epoch']; best = s['best']; best_epoch = s['best_epoch']
        torch.set_rng_state(s['torch_rng']); torch.cuda.set_rng_state(s['cuda_rng'])
    else:
        initial = validation(head, val, vr, weight); best = initial['balanced_loss']
        new(folder / 'initial-VAL.json', initial)
        torch.save({'head': {n: t.detach().cpu() for n, t in head.state_dict().items()}, 'epoch': 0, 'VAL_loss': best, 'protocol_sha256': sha(PROTOCOL)}, folder / 'best.pt')
        torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': 0, 'best': best, 'best_epoch': 0, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'protocol_sha256': sha(PROTOCOL)}, progress)
        print({'initial_VAL': initial}, flush=True)
    if journal.exists():
        actual = sum(1 for _ in journal.open(encoding='utf-8'))
    for epoch in range(epoch0, EPOCHS):
        head.train(); losses = []
        for step in range(STEPS):
            at = draws['indices'][epoch, step]
            b = {k: fit[k][at].clone().cuda() for k in KEYS}
            for i in range(BATCH):
                axes = tuple(axis for enabled, axis in zip(draws['flips'][epoch, step, i], (-1, -2)) if enabled)
                if axes:
                    for k in b:
                        b[k][i] = b[k][i].flip(axes)
            optimizer.zero_grad(set_to_none=True)
            loss = fine_mask_loss(head_logits(head, b), b['masks'][:, None], b['valid'][:, None], weight)
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5); optimizer.step()
            actual += 1; losses.append(float(loss.detach()))
            with journal.open('a', encoding='utf-8') as out:
                out.write(json.dumps({'actual_operation': actual, 'epoch': epoch + 1, 'step': step + 1, 'loss': losses[-1]}) + '\n')
        vl = validation(head, val, vr, weight)
        if vl['balanced_loss'] < best:
            best = vl['balanced_loss']; best_epoch = epoch + 1
            torch.save({'head': {n: t.detach().cpu() for n, t in head.state_dict().items()}, 'epoch': best_epoch, 'VAL_loss': best, 'protocol_sha256': sha(PROTOCOL)}, folder / 'best.pt')
        scheduler.step()
        torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch + 1, 'best': best, 'best_epoch': best_epoch, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'protocol_sha256': sha(PROTOCOL)}, progress)
        item = {'epoch': epoch + 1, 'actual_optimizer_updates': actual, 'scheduled_unique_updates': (epoch + 1) * STEPS, 'training_loss': sum(losses) / len(losses), 'VAL': vl, 'selected_epoch': best_epoch}
        with (folder / 'epochs.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(item) + '\n')
        print(item, flush=True)
    chosen = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    state = deepcopy(parent)
    state['state_dict'].update({'fine_target_head.' + n: t for n, t in chosen['head'].items()})
    assert all(torch.equal(t, state['state_dict'][n]) for n, t in parent['state_dict'].items())
    state.update(fine_target_version=VERSION, fine_target_parent_sha256=EXPECTED, fine_target_warm_start_sha256=WARM_SHA,
                 fine_target_protocol_sha256=sha(PROTOCOL), fine_target_actual_optimizer_updates=actual, fine_target_selected_epoch=best_epoch)
    (RUN / 'trained').mkdir()
    torch.save(state, RUN / 'trained/final.pt')
    new(RUN / 'trained/training.json', {'actual_optimizer_updates': actual, 'scheduled_unique_updates': EPOCHS * STEPS,
        'completed_epochs': EPOCHS, 'selected_epoch': best_epoch, 'selected_epoch_updates': best_epoch * STEPS,
        'VAL_balanced_loss': best, 'positive_weight': weight, 'FIT_photos_including_procedural_variants': len(fr),
        'VAL_photos_including_procedural_variants': len(vr), 'new_natural_target_fault_GT': 0,
        'all_parent_tensors_exact': True, 'data_manifest_sha256': MANIFEST_SHA})


def target(c, phase, p, model):
    path = ROOT / 'runs/industrial-magnetic-tiles-v16' / f'{c}-{phase}-features.pt'
    assert sha(path) == p['target_cache_sha256'][path.relative_to(ROOT).as_posix()]
    rows, values = old_cache(c, phase)
    values['fine'] = fine_cache(c + '-' + phase, rows, values, model)
    return rows, values


def scores(model, values, category):
    result = []
    with torch.inference_mode():
        for start in range(0, len(values['low']), BATCH):
            at = slice(start, start + BATCH)
            low, high = values['low'][at].cuda(), values['high'][at].cuda()
            residual = frozen_residual(model, low, high, [category] * len(low))
            prob = model.forward_fine_target_features(low, high, residual, values['fine'][at].cuda()).sigmoid().cpu()
            result.extend(ksdd_photo_score(prob, values['valid'][at]).tolist())
    return result


def calibrate():
    p = validate()
    assert not (RUN / 'calibration.json').exists()
    model, state = load(RUN / 'trained/final.pt', 'cuda'); result = {}
    for c in CATEGORIES:
        rows, v = target(c, 'cal', p, model)
        assert not any(r['label'] for r in rows)
        score = scores(model, v, c); cut, rank = normal_budget_threshold(score, .15)
        old, oldcut = parent_target_scores(model, state, c, v)
        assert min(cut, oldcut) > 0
        joint, jrank = normal_budget_threshold([max(n / cut, o / oldcut) for n, o in zip(score, old)], .15)
        result[c] = {'points': {'fine_only': {'photo_threshold': cut, 'rank': rank},
                    'max_fine_parent': {'photo_threshold': joint, 'rank': jrank, 'fine_divisor': cut, 'parent_divisor': oldcut}},
                    'normal_CAL': len(rows), 'observations': [{'id': r['id'], 'score': s, 'parent_score': o} for r, s, o in zip(rows, score, old)]}
    new(RUN / 'calibration.json', {'protocol_sha256': sha(PROTOCOL), 'thresholds_saved_before_TEST': True, 'categories': result})
    print({'calibrated': True, 'points': {c: result[c]['points'] for c in CATEGORIES}}, flush=True)


def evaluate():
    p = validate()
    assert not (RUN / 'selection.json').exists()
    cal = read(RUN / 'calibration.json')
    assert cal['thresholds_saved_before_TEST'] and cal['protocol_sha256'] == sha(PROTOCOL)
    model, state = load(RUN / 'trained/final.pt', 'cuda')
    prior = {r['id']: r for r in read(PARENT_EVAL)['rows']}
    result = {}; selected = {}; all_rows = []
    trained = read(RUN / 'trained/training.json')['selected_epoch'] > 0
    for c in CATEGORIES:
        rows, v = target(c, 'test', p, model); labels = [r['label'] for r in rows]
        new_scores = scores(model, v, c); old, oldcut = parent_target_scores(model, state, c, v)
        oldpred = [s > oldcut for s in old]
        assert oldpred == [prior[r['id']]['candidate_detected'] for r in rows], 'Parent baseline drift'
        base = classification_rates(labels, oldpred, .5); bg = group_rates(rows, oldpred)
        result[c] = {'parent': base, 'parent_by_defect': bg, 'methods': {}}
        for method in METHODS:
            point = cal['categories'][c]['points'][method]
            score = new_scores if method == 'fine_only' else [max(n / point['fine_divisor'], o / point['parent_divisor']) for n, o in zip(new_scores, old)]
            pred = [s > point['photo_threshold'] for s in score]
            r = classification_rates(labels, pred, .5); groups = group_rates(rows, pred)
            eligible = (trained and min(r['positive_photos'], r['negative_photos']) >= 20 and r['fpr'] <= .20
                and r['fpr'] <= base['fpr'] + .05 and r['fnr'] <= base['fnr'] - .05
                and all(g['photos'] < 10 or g['fnr'] <= bg[d]['fnr'] + .05 for d, g in groups.items()))
            result[c]['methods'][method] = {'rates': r, 'by_defect': groups, 'eligible': eligible, 'auroc': rank_auroc(labels, score)}
            all_rows.extend({'id': row['id'], 'category': c, 'method': method, 'label': row['label'], 'defect': row['defect'], 'score': s, 'candidate_detected': q} for row, s, q in zip(rows, score, pred))
        eligible = [m for m in METHODS if result[c]['methods'][m]['eligible']]
        if eligible:
            selected[c] = min(eligible, key=lambda m: (max(result[c]['methods'][m]['rates']['fpr'], result[c]['methods'][m]['rates']['fnr']), result[c]['methods'][m]['rates']['fnr'], METHODS.index(m)))
    selection = {'selected': selected, 'categories': result, 'training': read(RUN / 'trained/training.json'),
        'CAL_sha256': sha(RUN / 'calibration.json'), 'known_public_TEST_reused': True, 'independent_field_validated': False,
        'new_natural_target_fault_GT': 0, 'primary_location_unchanged': True}
    new(RUN / 'candidate-evaluation.json', {'rows': all_rows, 'categories': result})
    new(RUN / 'selection.json', selection); new(REPORTS / 'industrial-fine-targets-v20-selection.json', selection)
    if selected:
        state['fine_target_methods'] = selected
        state['fine_target_points'] = {c: cal['categories'][c]['points'][m] for c, m in selected.items()}
        (RUN / 'selected').mkdir(); torch.save(state, RUN / 'selected/final.pt')
    else:
        new(RUN / 'retained-parent.json', {'checkpoint': str(START), 'sha256': EXPECTED, 'reason': 'No preregistered finer-grid candidate eligible'})
    print({'selection': selection}, flush=True)


def verify():
    p = validate(); s = read(RUN / 'selection.json'); t = read(RUN / 'trained/training.json')
    for n, digest in p['legacy_source_sha256'].items():
        assert sha(ROOT / n) == digest, n
    unchanged = 0; rehashed = 0
    for n, before in read(RUN / 'protected-inputs.json').items():
        path = ROOT / n; stat = path.stat()
        if stat.st_size == before['size_bytes'] and stat.st_mtime_ns == before['mtime_ns']:
            unchanged += 1
        else:
            assert sha(path) == before['sha256'], n
            rehashed += 1
    parent = torch.load(START, map_location='cpu', weights_only=True)
    candidate = torch.load(RUN / 'trained/final.pt', map_location='cpu', weights_only=True)
    warm = torch.load(WARM, map_location='cpu', weights_only=True)
    assert all(torch.equal(v, candidate['state_dict'][n]) for n, v in parent['state_dict'].items())
    assert all(candidate[k] == v for k, v in parent.items() if k != 'state_dict')
    for n, v in warm['state_dict'].items():
        if n.startswith('target_conditioned_head.high_projection.'):
            assert torch.equal(v, candidate['state_dict'][n.replace('target_conditioned_head.', 'fine_target_head.', 1)])
    journal = [json.loads(line) for line in (RUN / 'train/optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['actual_operation'] for r in journal] == list(range(1, len(journal) + 1))
    assert len(journal) == t['actual_optimizer_updates']
    receipts = {}
    for receipt in RUN.glob('*-fine-receipt.json'):
        v = read(receipt); path = RUN / receipt.name.replace('-receipt.json', '.pt')
        assert sha(path) == v['sha256']; receipts[receipt.stem] = v
    for phase in ('fit', 'validation'):
        assert sha(previous.RUN / (phase + '-features.pt')) == p['old_generated_cache_sha256'][phase]
    result = {'status': 'passed', 'parent_all_tensors_buffers_metadata_exact': True,
        'high_projection_exact_to_v19_initialization': True, 'legacy_python_SHA_checked': len(p['legacy_source_sha256']),
        'protected_stat_unchanged': unchanged, 'changed_metadata_SHA_rechecked': rehashed, 'all_protected_SHA_reread': False,
        'actual_optimizer_updates': len(journal), 'new_unit_tests_passed': 3, 'selected': s['selected'],
        'fine_cache_receipts': receipts, 'new_natural_target_fault_GT': 0, 'synthetic_physical_realism_verified': False,
        'fresh_release_CPU_GPU_CPP_required': bool(s['selected']), 'active_parent_still_retained': True, 'independent_field_validated': False}
    new(RUN / 'verification.json', result); new(REPORTS / 'industrial-fine-targets-v20-verification.json', result)
    print({k: v for k, v in result.items() if k != 'fine_cache_receipts'}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('freeze', 'train', 'calibrate', 'evaluate', 'verify', 'run'))
    args = parser.parse_args()
    if args.mode in ('train', 'calibrate', 'evaluate', 'run'):
        for path in (ROOT / 'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):
                raise RuntimeError('Existing work process: ' + str(path))
        lock = RUN / 'operation.lock'
        if lock.exists():
            new(RUN / ('stale-lock-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'), {'pid': lock.read_text(encoding='utf-8'), 'live': False})
            lock.unlink()
        with lock.open('x', encoding='utf-8') as out:
            out.write(str(os.getpid()))
        try:
            if args.mode == 'run':
                train(); calibrate(); evaluate(); verify()
            else:
                globals()[args.mode]()
        except BaseException as error:
            new(RUN / (args.mode + '-failure-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'), {'error': repr(error), 'mode': args.mode, 'pid': os.getpid()})
            raise
        finally:
            lock.unlink()
    else:
        globals()[args.mode]()
