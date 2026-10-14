"""Fixed real-mask KSDD refinement; original-coordinate joint adoption guards."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.ksdd_supervised import ARCH, CATEGORY, HEIGHT, WIDTH, SurfaceMaskHead, KSDDSupervisedMain, ksdd_view, restore_map, ksdd_photo_score, mask_loss
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.belt_segmentation import pixel_counts
from safelog_ai.equipment_expansion import CATEGORIES, stage_gate
from safelog_ai.retention_distillation import state_sha256
from scripts import study_industrial_surfaces as previous
from scripts.study_bsclass_supervised import calibrated_threshold
from scripts.study_equipment_expansion import read, images
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_unified_belt_tiles import snapshot

RUN = ROOT/'runs/industrial-ksdd-supervised-v10'; REPORTS = ROOT/'reports'
DATA = ROOT/'data/industrial-ksdd-supervised-v10'; MANIFEST = DATA/'manifest.json'
START = previous.RUN/'selected/final.pt'; EXPECTED = '422b9e0453b3e125a7fe75385c016a89c2870e58e3d492c2241bded938196e2c'
PARENT_EVALUATION = previous.RUN/'evaluation.json'; PROTOCOL = REPORTS/'industrial-ksdd-supervised-v10-protocol.json'
SOURCES = ('safelog_ai/ksdd_supervised.py', 'scripts/prepare_ksdd_supervised.py', 'scripts/study_ksdd_supervised.py', 'tests/test_ksdd_supervised.py')
EPOCHS, STEPS, BATCH, SEED = 40, 64, 8, 1471
PIXEL_THRESHOLDS = [i/10 for i in range(1, 10)]


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(START) == EXPECTED
    assert read(REPORTS/'industrial-night-v8-v9-final-audit.json')['status'] == 'completed_v8_v9_night_batch_continuation_enabled'
    manifest = read(MANIFEST); assert manifest['audit']['status'] == 'ready_for_supervised_protocol'
    for n, h in manifest['source_sha256'].items(): assert sha(ROOT/n) == h, n
    RUN.mkdir(); ledger = dict(read(previous.RUN/'protected-inputs.json'))
    for directory in (REPORTS, previous.RUN):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    ledger[MANIFEST.relative_to(ROOT).as_posix()] = snapshot(MANIFEST)
    new(RUN/'protected-inputs.json', ledger)
    new(PROTOCOL, {'schema': 'safelog-real-mask-ksdd-fixed-v10', 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_feature_cache_training_validation_calibration_and_new_TEST_scores': True,
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'tests', 'safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256': EXPECTED, 'manifest_sha256': sha(MANIFEST), 'parent_evaluation_sha256': sha(PARENT_EVALUATION), 'protected_inputs_sha256': sha(RUN/'protected-inputs.json'),
        'data_audit': manifest['audit'], 'source': 'Existing publisher KolektorSDD2 TRAIN images and real pixel masks only',
        'normal_CAL_reassigned_for_new_head_only': True, 'old_operating_points_retained': True,
        'input': 'RGB bicubic aspect-preserving256x704 letterbox127; GT nearest; valid only resized source rectangle; no GT-based crop',
        'architecture': 'Same frozen accepted encoder low40/high960; high960to32 conv1x1 then bilinear to low grid; concat low40; conv72to64 k3 ReLU Dropout2d.1 conv64to32 k3 ReLU conv32to1; bilinear to704x256',
        'feature_precision': 'FP16 cache and identical FP16 roundtrip in runtime, head FP32',
        'trainable': 'New ksdd_head only; original encoder/heads/banks exact', 'epochs': EPOCHS, 'steps_per_epoch': STEPS, 'batch': BATCH,
        'planned_optimizer_updates': EPOCHS*STEPS, 'seed': SEED, 'sampling': '4real-mask defects+4normal each step, replacement; inverse class-group multiplicity weights; repeats are not new photos',
        'feature_augmentation': 'Paired horizontal/vertical feature-grid, mask and valid flips at.5 each; not extra photographed examples',
        'optimizer': 'AdamW lr.001 wd.0001, cosine40 eta_min.00001, clip5, FP32 TF32off',
        'loss': 'Valid pixel BCEWithLogits pos_weight10 + per-photo softDice +.5 photoBCE',
        'photo_score': 'Mean top32 native valid pixel sigmoid probabilities; not calibrated internal failure probability',
        'checkpoint_selection': 'VAL only: prefer FPR/FNR each<=.20 at.5; eligible highest native pixelIoU at.5 then lower maxphotoerror, loss, epoch; otherwise lower maxphotoerror then higherIoU,loss,epoch',
        'photo_threshold': 'Labelled CAL only minimax FPR/FNR, then their sum, then distance to.5; fixed before TEST',
        'pixel_thresholds_registered': PIXEL_THRESHOLDS, 'pixel_threshold_selection': 'CAL original coordinates; if any FNR<=.15 choose highestIoU then lowerFNR then higher threshold; otherwise lowerFNR then higherIoU then higher threshold',
        'baseline_comparison': 'Recompute old256 coarse residual and new native head at original source coordinates against identical original masks, before visualization PNG quantization',
        'guards': ['photoFPR<=parent', 'photoFNR<=parent', 'each_photo_class>=20', 'original_pixelIoU>parent', 'original_pixelFNR<=parent', 'original_pixelFPR<=parent', 'all_parent_tensors_buffers_exact'],
        'no_TEST_training_or_threshold_or_epoch_selection': True, 'known_TEST_reused_for_final_candidate_adoption_guard': True,
        'numeric_filename_groups_are_proxy_not_physical_acquisition_ids': True, 'independent_field_validated': False,
        'one_model_one_shared_backbone': True, 'app_default_promoted': False, 'new_unit_tests_passed': 4})
    print(json.dumps({'frozen': True, 'updates_planned': EPOCHS*STEPS, 'counts': manifest['audit']['counts'], 'protected_files': len(ledger)}), flush=True)


def validate(committed=False):
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path, digest in ((START, EXPECTED), (MANIFEST, p['manifest_sha256']), (PARENT_EVALUATION, p['parent_evaluation_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == digest, str(path)
    if committed:
        repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
        for n in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
            assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+n])).hexdigest() == sha(ROOT/n), n
    return p


def original_truth(row):
    assert sha(ROOT/row['masks'][0]) == row['mask_sha256'][0]
    with Image.open(ROOT/row['masks'][0]) as im: return torch.from_numpy(np.array(im.convert('L')) > 0)


def cache(phase):
    rows = [r for r in read(MANIFEST)['rows'] if r['phase'] == phase]; path = RUN/(phase+'-features.pt')
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['protocol_sha256'] == sha(PROTOCOL) and saved['ids'] == [r['id'] for r in rows]; return rows, saved
    if phase == 'test': assert (RUN/'calibration.json').exists(), 'No TEST forward before final model/thresholds'
    model, _ = previous.load(START); encoder = model.backbone.cuda().eval(); low = []; high = []; masks = []; valid = []; geometries = []
    def prepare(row):
        assert sha(ROOT/row['image']) == row['image_sha256']
        with Image.open(ROOT/row['image']) as im: rgb = im.convert('RGB')
        truth = original_truth(row); truth_image = Image.fromarray(truth.numpy().astype('uint8')*255)
        view, mask, known, geometry = ksdd_view(rgb, truth_image)
        assert bool(mask.any()) == bool(row['label']), 'Source defect must remain visible after resizing'
        return normalized(view), mask, known, geometry
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for i in range(0, len(rows), 16):
            prepared = list(pool.map(prepare, rows[i:i+16])); batch = torch.stack([v[0] for v in prepared]).cuda()
            features = encoder(batch); low.append(features['low'].cpu().half()); high.append(features['high'].cpu().half())
            masks.extend(v[1] for v in prepared); valid.extend(v[2] for v in prepared); geometries.extend(v[3] for v in prepared)
            if i % 256 == 0: print(json.dumps({'cached_phase': phase, 'photos': min(i+16, len(rows)), 'total': len(rows)}), flush=True)
    saved = {'protocol_sha256': sha(PROTOCOL), 'ids': [r['id'] for r in rows], 'low': torch.cat(low), 'high': torch.cat(high),
        'mask': torch.stack(masks), 'valid': torch.stack(valid), 'geometries': geometries}
    with path.open('xb') as out: torch.save(saved, out)
    del model, encoder; torch.cuda.empty_cache(); return rows, saved


def up_logits(head, low, high):
    return F.interpolate(head(low.float(), high.float()), size=(HEIGHT, WIDTH), mode='bilinear', align_corners=False)


def summarize_pixels(counts):
    tp, fp, tn, fn = (counts[k] for k in ('tp', 'fp', 'tn', 'fn'))
    return {'counts': counts, 'iou': tp/(tp+fp+fn) if tp+fp+fn else None, 'fnr': fn/(tp+fn) if tp+fn else None, 'fpr': fp/(tn+fp) if tn+fp else None}


def add_counts(total, value):
    for key in total: total[key] += value[key]


def validation(head, rows, saved):
    head.eval(); scores = []; counts = dict.fromkeys(('tp', 'fp', 'tn', 'fn'), 0); losses = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            low = saved['low'][start:start+BATCH].cuda(); high = saved['high'][start:start+BATCH].cuda()
            mask = saved['mask'][start:start+BATCH].cuda()[:, None]; valid = saved['valid'][start:start+BATCH].cuda()[:, None]
            logits = up_logits(head, low, high); losses.append(float(mask_loss(logits, mask, valid))); probability = logits.sigmoid()
            scores.extend(ksdd_photo_score(probability[:, 0], valid[:, 0]).cpu().tolist())
            add_counts(counts, pixel_counts(probability, mask, valid, .5))
    rates = classification_rates([r['label'] for r in rows], scores, .5); pixels = summarize_pixels(counts)
    return {'photo_rates': rates, 'native_pixels_at_half': pixels, 'loss': sum(losses)/len(losses)}


def train():
    p = validate(committed=True); assert not (RUN/'trained/final.pt').exists()
    rows, fit = cache('fit'); val_rows, val = cache('validation')
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); head = SurfaceMaskHead().cuda(); initial_sha = state_sha256(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=.00001)
    rng = torch.Generator().manual_seed(SEED+1); draws = []
    for label in (0, 1):
        selected = [i for i, r in enumerate(rows) if r['label'] == label]; groups = Counter(rows[i]['group_id'] for i in selected)
        weights = torch.tensor([1/groups[rows[i]['group_id']] for i in selected])
        indices = torch.multinomial(weights, EPOCHS*STEPS*4, replacement=True, generator=rng)
        draws.append(torch.tensor(selected)[indices].reshape(EPOCHS, STEPS, 4))
    draws = torch.cat(draws, 2); folder = RUN/'train'; folder.mkdir(exist_ok=True); progress = folder/'progress.pt'
    cursor = 0; updates = 0; best = None
    if progress.exists():
        s = torch.load(progress, map_location='cpu', weights_only=True); assert s['protocol_sha256'] == sha(PROTOCOL)
        head.load_state_dict(s['head']); optimizer.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        cursor, updates, best = s['epochs_completed'], s['updates'], s['best']; torch.set_rng_state(s['cpu_rng']); torch.cuda.set_rng_state_all(s['cuda_rng'])
        assert len((folder/'epochs.jsonl').read_text().splitlines()) == cursor
    else:
        with (folder/'draws.pt').open('xb') as out: torch.save(draws, out)
    with (folder/'epochs.jsonl').open('a' if cursor else 'x', encoding='utf-8', newline='\n') as journal:
        for epoch in range(cursor, EPOCHS):
            head.train(); loss_sum = 0.
            for step in range(STEPS):
                ids = draws[epoch, step]; low = fit['low'][ids].cuda(); high = fit['high'][ids].cuda()
                mask = fit['mask'][ids].cuda()[:, None]; valid = fit['valid'][ids].cuda()[:, None]
                axes = tuple(dim for dim in (-2, -1) if torch.rand(()) < .5)
                if axes: low, high, mask, valid = (v.flip(axes) for v in (low, high, mask, valid))
                logits = up_logits(head, low, high); loss = mask_loss(logits, mask, valid); assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True); loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.); assert torch.isfinite(norm)
                optimizer.step(); updates += 1; loss_sum += float(loss.detach())
            scheduler.step(); measured = validation(head, val_rows, val); rates = measured['photo_rates']; err = max(rates['fpr'], rates['fnr']); iou = measured['native_pixels_at_half']['iou']
            key = [0, -iou, err, measured['loss'], epoch+1] if err <= .20 else [1, err, -iou, measured['loss'], epoch+1]
            if best is None or key < best['key']:
                best = {'key': key, 'epoch': epoch+1, 'validation': measured, 'head': {n: v.detach().cpu().clone() for n, v in head.state_dict().items()}}
            record = {'epoch': epoch+1, 'optimizer_updates': updates, 'train_loss': loss_sum/STEPS, 'validation': measured, 'best_epoch': best['epoch']}
            journal.write(json.dumps(record)+'\n'); journal.flush()
            temporary = folder/'progress-next.pt'
            torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epochs_completed': epoch+1, 'updates': updates,
                'best': best, 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'protocol_sha256': sha(PROTOCOL)}, temporary); temporary.replace(progress)
            if (epoch+1) % 2 == 0: print(json.dumps(record), flush=True)
    assert updates == EPOCHS*STEPS and initial_sha != state_sha256(head)
    source = torch.load(START, map_location='cpu', weights_only=True); state = deepcopy(source)
    state['state_dict'].update({'ksdd_head.'+n: v for n, v in best['head'].items()})
    state.update(architecture=ARCH, ksdd_protocol_sha256=sha(PROTOCOL), ksdd_parent_sha256=EXPECTED, ksdd_actual_optimizer_updates=updates,
        ksdd_selected_epoch_by_validation=best['epoch'], ksdd_real_mask_training=True)
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in source['state_dict'].items())
    output = RUN/'trained'; output.mkdir()
    with (output/'final.pt').open('xb') as out: torch.save(state, out)
    new(output/'training.json', {'epochs': EPOCHS, 'actual_optimizer_updates': updates, 'selected_epoch_by_validation': best['epoch'],
        'selected_validation': best['validation'], 'FIT_counts': dict(Counter(str(r['label']) for r in rows)), 'initial_head_sha': initial_sha, 'final_head_sha': state_sha256(head),
        'selected_head_sha': state_sha256(best['head']), 'all_parent_tensors_exact': True, 'TEST_or_CAL_used_for_gradient_or_epoch_selection': False,
        'feature_flips_are_not_new_photos': True})
    print(json.dumps({'real_mask_training_completed': True, 'updates': updates, 'selected_epoch': best['epoch']}), flush=True)


def load(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['architecture'] != ARCH: return previous.load(path, device)
    model = KSDDSupervisedMain(state['reference_counts'], state['native_specs'], state['wear_variant'], state['surface_reference_specs']).eval().requires_grad_(False)
    model.load_state_dict(state['state_dict'], strict=True); return model.to(device), state


def predict(head, rows, saved):
    probabilities = []; scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            logits = up_logits(head, saved['low'][start:start+BATCH].cuda(), saved['high'][start:start+BATCH].cuda())
            p = logits[:, 0].sigmoid(); valid = saved['valid'][start:start+BATCH].cuda()
            scores.extend(ksdd_photo_score(p, valid).cpu().tolist()); probabilities.extend(p.cpu().unbind(0))
    return scores, probabilities


def original_counts(rows, probabilities, geometries, thresholds):
    totals = {str(t): dict.fromkeys(('tp', 'fp', 'tn', 'fn'), 0) for t in thresholds}
    for row, probability, geometry in zip(rows, probabilities, geometries):
        restored = restore_map(probability, geometry); truth = original_truth(row); valid = torch.ones_like(truth)
        for threshold in thresholds: add_counts(totals[str(threshold)], pixel_counts(restored, truth, valid, threshold))
    return {key: summarize_pixels(value) for key, value in totals.items()}


def calibrate():
    validate(); assert not (RUN/'calibration.json').exists(); rows, saved = cache('calibration'); model, state = load(RUN/'trained/final.pt', 'cuda')
    scores, probabilities = predict(model.ksdd_head, rows, saved); labels = [r['label'] for r in rows]
    threshold = calibrated_threshold(labels, scores); pixels = original_counts(rows, probabilities, saved['geometries'], PIXEL_THRESHOLDS)
    eligible = [t for t in PIXEL_THRESHOLDS if pixels[str(t)]['fnr'] <= .15]
    if eligible: pixel_threshold = min(eligible, key=lambda t: (-pixels[str(t)]['iou'], pixels[str(t)]['fnr'], -t))
    else: pixel_threshold = min(PIXEL_THRESHOLDS, key=lambda t: (pixels[str(t)]['fnr'], -pixels[str(t)]['iou'], -t))
    point = {'photo_threshold': threshold, 'pixel_threshold': pixel_threshold, 'photo_score_method': 'real_mask_top32_valid_sigmoid',
        'input_size': [WIDTH, HEIGHT], 'labelled_CAL_photos': len(rows), 'thresholds_selected_on_CAL_only_before_TEST': True}
    new(RUN/'calibration.json', {'protocol_sha256': sha(PROTOCOL), 'point': point, 'photo_rates': classification_rates(labels, scores, threshold), 'pixel_candidates_original_coordinates': pixels})
    state['ksdd_operating_point'] = point; output = RUN/'calibrated'; output.mkdir()
    state['state_dict'] = {n: v.detach().cpu() for n, v in model.state_dict().items()}
    with (output/'final.pt').open('xb') as out: torch.save(state, out)
    print(json.dumps({'CAL_saved_before_TEST': True, 'photo_rates': classification_rates(labels, scores, threshold), 'pixel_threshold': pixel_threshold, 'original_pixels': pixels[str(pixel_threshold)]}), flush=True)


def evaluate():
    p = validate(); assert not (REPORTS/'industrial-ksdd-supervised-v10-validation.json').exists(); calibration = read(RUN/'calibration.json')
    assert calibration['protocol_sha256'] == sha(PROTOCOL) and calibration['point']['thresholds_selected_on_CAL_only_before_TEST']
    rows, saved = cache('test'); model, state = load(RUN/'calibrated/final.pt', 'cuda'); point = state['ksdd_operating_point']
    scores, probabilities = predict(model.ksdd_head, rows, saved); rates = classification_rates([r['label'] for r in rows], scores, point['photo_threshold'])
    pixels = original_counts(rows, probabilities, saved['geometries'], [point['pixel_threshold']])[str(point['pixel_threshold'])]
    parent, source = previous.load(START, 'cuda'); old_probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH): old_probabilities.extend(parent.forward_equipment(images(rows[start:start+BATCH]), CATEGORY).cpu().unbind(0))
    old_counts = dict.fromkeys(('tp', 'fp', 'tn', 'fn'), 0); old_threshold = source['equipment_operating_points'][CATEGORY]['pixel_threshold']
    for row, old in zip(rows, old_probabilities):
        width, height = row['size']; restored = F.interpolate(old[None, None], size=(height, width), mode='bilinear', align_corners=False)[0, 0]
        truth = original_truth(row); add_counts(old_counts, pixel_counts(restored, truth, torch.ones_like(truth), old_threshold))
    old_pixels = summarize_pixels(old_counts); old_photo = read(PARENT_EVALUATION)['categories'][CATEGORY]['photo_rates']
    guards = {'photo_FPR_no_worse': rates['fpr'] <= old_photo['fpr'], 'photo_FNR_no_worse': rates['fnr'] <= old_photo['fnr'],
        'each_photo_class_at_least20': min(rates['negative_photos'], rates['positive_photos']) >= 20, 'original_pixel_IoU_better': pixels['iou'] > old_pixels['iou'],
        'original_pixel_FNR_no_worse': pixels['fnr'] <= old_pixels['fnr'], 'original_pixel_FPR_no_worse': pixels['fpr'] <= old_pixels['fpr'],
        'all_parent_tensors_and_buffers_exact': all(torch.equal(v, state['state_dict'][n]) for n, v in source['state_dict'].items())}
    eligible = all(guards.values()); observations = [{'id': r['id'], 'label': r['label'], 'score': score, 'candidate_detected': score > point['photo_threshold']} for r, score in zip(rows, scores)]
    new(REPORTS/'industrial-ksdd-supervised-v10-validation.json', {'category': CATEGORY, 'photo_rates': rates, 'parent_photo_rates': old_photo, 'original_pixel_metrics': pixels,
        'parent_original_pixel_metrics_recomputed_same_coordinate': old_pixels, 'point': point, 'auroc': rank_auroc([r['label'] for r in rows], scores),
        'guards': guards, 'eligible': eligible, 'observations': observations, 'known_TEST_reused_for_adoption': True, 'independent_field_validated': False})
    checkpoint = START
    if eligible:
        import shutil
        output = RUN/'selected'; output.mkdir(); checkpoint = output/'final.pt'; shutil.copy2(RUN/'calibrated/final.pt', checkpoint)
    evaluation = read(PARENT_EVALUATION)
    if eligible:
        evaluation['categories'][CATEGORY].update(photo_rates=rates, photo_threshold=point['photo_threshold'], photo_score_source='real_mask_supervised_top32',
            pixel_threshold=point['pixel_threshold'], supervised_original_pixel_metrics=pixels, parent_original_pixel_metrics=old_pixels,
            stage_gate=stage_gate(rates['fpr'], rates['fnr'], rates['negative_photos'], rates['positive_photos']))
        lookup = {r['id']: r for r in observations}
        for row in evaluation['rows']:
            if row['category'] == CATEGORY: row.update(score=lookup[row['id']]['score'], candidate_detected=lookup[row['id']]['candidate_detected'])
    evaluation.update(checkpoint_sha256=sha(checkpoint), status='industrial_ksdd_real_mask_v10')
    new(RUN/'evaluation.json', evaluation)
    new(REPORTS/'industrial-ksdd-supervised-v10-selection.json', {'selected': 'real_mask_supervised_ksdd' if eligible else 'prior_prototype',
        'checkpoint_path': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(checkpoint), 'guards': guards,
        'actual_new_optimizer_updates': EPOCHS*STEPS, 'other11_equipment_categories_and_legacy_heads_preserved': True,
        'one_model_one_shared_backbone': True, 'app_default_promoted': False, 'independent_field_validated': False})
    print(json.dumps({'TEST_photo_rates': rates, 'TEST_original_pixels': pixels, 'parent_original_pixels': old_pixels, 'eligible': eligible, 'guards': guards}), flush=True)


def verify():
    p = validate(); choice = read(REPORTS/'industrial-ksdd-supervised-v10-selection.json')
    for n, h in p['legacy_source_sha256'].items(): assert sha(ROOT/n) == h, n
    ledger = read(RUN/'protected-inputs.json')
    def check(item):
        n, v = item; assert snapshot(ROOT/n) == v, n
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, ledger.items()))
    parent, source = previous.load(START); model, state = load(RUN/'calibrated/final.pt')
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in source['state_dict'].items())
    assert state['equipment_operating_points'] == source['equipment_operating_points'] and state['thresholds'] == source['thresholds']
    proofs = {}
    with torch.inference_mode():
        x = torch.zeros(1, 3, 256, 256)
        for category in CATEGORIES:
            a, b = model.forward_equipment_evidence(x, category), parent.forward_equipment_evidence(x, category)
            assert all(torch.equal(v, w) for v, w in zip(a, b)); proofs[category] = True
        assert torch.equal(model.forward_wear(x), parent.forward_wear(x))
        for task in ('facility_surface', 'belt_damage', 'conveyor_foreign'):
            a, b = model.forward_task(x, task), parent.forward_task(x, task)
            a, b = (a if isinstance(a, tuple) else (a,)), (b if isinstance(b, tuple) else (b,))
            assert all(torch.equal(v, w) for v, w in zip(a, b)); proofs[task] = True
    new(REPORTS/'industrial-ksdd-supervised-v10-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'all_old_bytes_mtimes_tensors_buffers_and_points_preserved': True, 'all_old_CPU_equipment_and_legacy_paths_exact': proofs,
        'BSClass_wear_CPU_output_exact': True, 'actual_optimizer_updates': EPOCHS*STEPS, 'selected': choice['selected']})
    print(json.dumps({'verified': True, 'protected_files': len(ledger), 'selected': choice['selected']}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'train', 'calibrate', 'evaluate', 'verify']); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    else:
        from safelog_ai.process_probe import process_exists
        lock = RUN/'operation.lock'
        if lock.exists():
            if process_exists(int(lock.read_text())): raise RuntimeError('Owned KSDD operation already running')
            lock.rename(RUN/('stale-operation-'+str(time.time_ns())+'.txt'))
        with lock.open('x') as out: out.write(str(os.getpid()))
        try: globals()[args.mode]()
        finally:
            if lock.exists() and lock.read_text() == str(os.getpid()): lock.unlink()
