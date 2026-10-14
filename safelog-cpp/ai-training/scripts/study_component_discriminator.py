"""Fixed actual gradient training of one additional shared component head."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.component_discriminator import ARCH, DiscriminatorMain, synthetic_features
from safelog_ai.component_reference import NEW_CATEGORIES
from safelog_ai.equipment_expansion import CATEGORIES, SIZE, photo_score, stage_gate
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from scripts import study_component_detail as detail
from scripts import study_component_reference as reference
from scripts.study_equipment_expansion import images, read, MANIFEST
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_equipment_reference_photo import counts

START, EXPECTED, PARENT_EVALUATION = reference.START, reference.EXPECTED, reference.PARENT_EVALUATION
DONOR, DONOR_SHA = detail.DONOR, detail.DONOR_SHA
RUN = ROOT/'runs/equipment-component-discriminator-v5'; REPORTS = ROOT/'reports'
PROTOCOL = REPORTS/'equipment-component-discriminator-v5-protocol.json'
EPOCHS, STEPS, PATCHES, SEED = 40, 64, 256, 811
BUDGETS = {'normalCAL10': .10, 'normalCAL05': .05}
SOURCES = ('safelog_ai/component_discriminator.py', 'scripts/study_component_discriminator.py', 'tests/test_component_discriminator.py')


def load(path, device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['architecture'] != ARCH: return reference.load(path, device)
    model = DiscriminatorMain(state['reference_counts'], state['discriminator_active_components']); model.load_state_dict(state['state_dict'], strict=True)
    return model.to(device).eval().requires_grad_(False), state


def freeze():
    assert not RUN.exists() and sha(START) == EXPECTED and sha(DONOR) == DONOR_SHA
    assert read(REPORTS/'equipment-component-detail-v4-verification.json')['status'] == 'passed'
    assert not read(REPORTS/'equipment-component-detail-v4-selection.json')['newly_selected_categories']
    RUN.mkdir(); ledger = dict(read(detail.RUN/'protected-inputs.json'))
    for directory in (REPORTS, detail.RUN):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    new(RUN/'protected-inputs.json', ledger)
    original = read(reference.PROTOCOL)
    new(PROTOCOL, {'schema': 'safelog-component-discriminator-fixed-v5', 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_training_CAL_and_TEST_rescoring': True,
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'tests', 'safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256': EXPECTED, 'donor_sha256': DONOR_SHA, 'manifest_sha256': sha(MANIFEST), 'parent_evaluation_sha256': sha(PARENT_EVALUATION),
        'protected_inputs_sha256': sha(RUN/'protected-inputs.json'), 'categories': list(NEW_CATEGORIES),
        'baseline': original['baseline'], 'split_counts': original['split_counts'], 'budgets': BUDGETS,
        'fit_source': 'Only existing v3 normalFIT527 photos134912 pooled1000-channel patch features; verify source IDs against unchanged manifest, no CAL/TEST in feature banks',
        'architecture': 'ONE accepted frozen backbone+all accepted heads/buffers; add ONE shared category-conditioned MLP discriminator: embedding3x16 scaled.05, Linear1016to256 LeakyReLU.2 Linear256to128 LeakyReLU.2 Linear128to1',
        'synthetic_training': 'Normal label0; F.normalize(unitNormal+GaussianNoiseStd.005) synthetic label1; equal normal/synthetic count; synthetic labels are not real defect annotations',
        'epochs': EPOCHS, 'steps_per_epoch': STEPS, 'normal_patches_per_category_per_step': PATCHES,
        'planned_optimizer_updates': EPOCHS*STEPS, 'normal_patch_draws': EPOCHS*STEPS*PATCHES*3, 'synthetic_patch_draws': EPOCHS*STEPS*PATCHES*3,
        'seed': SEED, 'optimizer': 'AdamW lr.001 weight_decay.00001; BCEWithLogits; clip5; cosine40 epochs eta_min.00001; FP32, TF32off',
        'checkpoint_rule': 'Final epoch40 only. NO TEST epoch selection. Save per-epoch head/optimizer/scheduler/noise-RNG/cursor progress; final-only normalCAL after all training.',
        'inference': 'One common backbone pass, discriminator sigmoid per16x16patch, bilinear256 and top1percent mean photo score; NOT calibrated real defect probability',
        'thresholds': 'Only normalCAL, ranksceil((n+1)*(1-alpha)), alpha.10/.05 strict >; no TEST quantile sweep',
        'guards': original['prespecified_guards'], 'selection': original['selection'],
        'all_original_location_outputs_and_parent_buffers_preserved': True, 'existing_passed_four_categories_preserved': True,
        'MPDD_connector_defect_TEST14_cannot_pass_stage': True, 'known_TEST_reused_for_development_and_candidate_selection': True,
        'new_source_photos_or_real_defect_labels': 0, 'independent_field_validated': False, 'app_default_promoted': False,
        'method_inspiration': 'https://arxiv.org/abs/2303.15140', 'primary_code_reviewed': 'https://github.com/DonaldRR/SimpleNet',
        'exact_SimpleNet_reproduction': False, 'external_code_downloaded_or_executed': False, 'new_unit_tests_passed': 3, 'unit_test_optimizer_updates': 0})
    print(json.dumps({'frozen': True, 'actual_updates_planned': EPOCHS*STEPS, 'normal_FIT_photos_available': 527, 'protected_files': len(ledger)}), flush=True)


def validate():
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path, h in ((START, EXPECTED), (DONOR, DONOR_SHA), (MANIFEST, p['manifest_sha256']), (PARENT_EVALUATION, p['parent_evaluation_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == h, str(path)
    return p


def train():
    p = validate(); repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    commit = subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip(); torch.manual_seed(SEED)
    source = torch.load(START, map_location='cpu', weights_only=True); donor = torch.load(DONOR, map_location='cpu', weights_only=True)
    rows = {r['id']: r for r in read(MANIFEST)['rows']}; bank = {}; source_counts = {}
    for category in NEW_CATEGORIES:
        record = donor['normal_reference_fit'][category]; selected_rows = [rows[n] for n in record['source_ids']]
        assert all(r['phase'] == 'fit' and r['label'] == 0 and r['category'] == category for r in selected_rows)
        tensor = donor['state_dict']['normal_reference_'+category]
        assert len(tensor) == len(selected_rows)*256 and torch.isfinite(tensor).all()
        bank[category] = F.normalize(tensor.cuda().float(), dim=1); source_counts[category] = len(selected_rows)
    model = DiscriminatorMain(source['reference_counts'], list(NEW_CATEGORIES)).cuda().eval().requires_grad_(False)
    missing, extra = model.load_state_dict(source['state_dict'], strict=False)
    assert all(n.startswith('component_discriminator.') for n in missing) and missing and not extra
    model.component_discriminator.requires_grad_(True).train()
    optimizer = torch.optim.AdamW(model.component_discriminator.parameters(), lr=.001, weight_decay=.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=.00001)
    draw_rng = torch.Generator().manual_seed(SEED+1); noise_rng = torch.Generator(device='cuda').manual_seed(SEED+2)
    draws = {c: torch.randint(len(bank[c]), (EPOCHS, STEPS, PATCHES), generator=draw_rng) for c in NEW_CATEGORIES}
    folder = RUN/'train'; folder.mkdir(exist_ok=True); progress = folder/'progress.pt'; journal = folder/'epochs.jsonl'; cursor = 0; updates = 0
    if progress.exists():
        saved = torch.load(progress, map_location='cpu', weights_only=True); assert saved['protocol_sha256'] == sha(PROTOCOL)
        model.component_discriminator.load_state_dict(saved['head']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        noise_rng.set_state(saved['noise_rng']); cursor = saved['epochs_completed']; updates = saved['optimizer_updates']
        assert len(journal.read_text().splitlines()) == cursor and updates == cursor*STEPS
    else:
        assert not journal.exists()
        with (folder/'draws.pt').open('xb') as out: torch.save(draws, out)
    with journal.open('a' if cursor else 'x', encoding='utf-8', newline='\n') as log:
        for epoch in range(cursor, EPOCHS):
            loss_sum = 0.
            for step in range(STEPS):
                normal = torch.cat([bank[c][draws[c][epoch, step].cuda()] for c in NEW_CATEGORIES])
                ids = torch.arange(3, device='cuda').repeat_interleave(PATCHES)
                noise = torch.randn(normal.shape, device='cuda', generator=noise_rng); altered = synthetic_features(normal, noise)
                logits = model.component_discriminator(torch.cat((normal, altered)), torch.cat((ids, ids)))
                truth = torch.cat((torch.zeros(len(normal), device='cuda'), torch.ones(len(altered), device='cuda')))
                loss = F.binary_cross_entropy_with_logits(logits, truth); assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.component_discriminator.parameters(), 5.); assert torch.isfinite(norm)
                optimizer.step(); updates += 1; loss_sum += float(loss.detach())
            scheduler.step()
            record = {'epoch': epoch+1, 'optimizer_updates': updates, 'training_BCE': loss_sum/STEPS, 'lr_next': scheduler.get_last_lr()[0]}
            log.write(json.dumps(record)+'\n'); log.flush()
            temporary = folder/'progress-next.pt'
            torch.save({'head': {n: v.detach().cpu() for n, v in model.component_discriminator.state_dict().items()}, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'noise_rng': noise_rng.get_state(), 'epochs_completed': epoch+1, 'optimizer_updates': updates, 'protocol_sha256': sha(PROTOCOL)}, temporary)
            temporary.replace(progress)
            if (epoch+1) % 5 == 0: print(json.dumps(record), flush=True)
    assert updates == EPOCHS*STEPS
    state = deepcopy(source); state.update(architecture=ARCH, state_dict={n: v.detach().cpu() for n, v in model.state_dict().items()},
        discriminator_active_components=list(NEW_CATEGORIES), discriminator_protocol_sha256=sha(PROTOCOL), discriminator_protocol_commit=commit,
        parent_prototype_sha256=EXPECTED, neural_updates_this_refinement=updates, discriminator_epochs=EPOCHS)
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in source['state_dict'].items())
    assert any(bool(v.abs().sum()) for n, v in state['state_dict'].items() if n.startswith('component_discriminator.'))
    with (folder/'final-trained.pt').open('xb') as out: torch.save(state, out)
    new(folder/'training.json', {'status': 'trained_final_epoch40_only', 'optimizer_updates': updates, 'epochs': EPOCHS, 'normal_FIT_photos_available': source_counts,
        'normal_patch_draws': updates*PATCHES*3, 'synthetic_patch_draws': updates*PATCHES*3, 'new_real_defect_labels': 0,
        'all_parent_tensors_and_reference_buffers_exact': True, 'TEST_or_CAL_used_for_training_or_epoch_selection': False,
        'trained_checkpoint_sha256': sha(folder/'final-trained.pt'), 'progress_checkpoint_sha256': sha(progress)})


def calibrate():
    p = validate(); model, state = load(RUN/'train/final-trained.pt'); points = {}; rows = read(MANIFEST)['rows']
    with torch.inference_mode():
        for category in NEW_CATEGORIES:
            cal = [r for r in rows if r['category'] == category and r['phase'] == 'cal']; scores = []
            for i in range(0, len(cal), 8):
                _, heat = model.forward_equipment_evidence(images(cal[i:i+8]), category); scores.extend(photo_score(heat).cpu().double().tolist())
            points[category] = {n: dict(zip(('threshold', 'rank'), normal_budget_threshold(scores, alpha))) for n, alpha in BUDGETS.items()}; points[category]['scores'] = scores
    state['discriminator_calibrations'] = points; folder = RUN/'candidate'; folder.mkdir()
    with (folder/'final.pt').open('xb') as out: torch.save(state, out)
    cpu, _ = load(folder/'final.pt', 'cpu'); parent, _ = load(START, 'cpu'); proofs = {}
    with torch.inference_mode():
        for category in CATEGORIES:
            row = next(r for r in rows if r['category'] == category and r['phase'] == 'fit'); x = images([row])
            a, b = model.forward_equipment_evidence(x, category); c, d = cpu.forward_equipment_evidence(x.cpu(), category); e, f = parent.forward_equipment_evidence(x.cpu(), category)
            assert torch.equal(c, e) and torch.allclose(a.cpu(), c, atol=2e-4, rtol=.002) and torch.allclose(b.cpu(), d, atol=2e-4, rtol=.002), category
            if category not in NEW_CATEGORIES: assert torch.equal(d, f), category
            proofs[category] = {'CPU_GPU_close': True, 'original_location_CPU_exact': True, 'parent_photo_output_exact': category not in NEW_CATEGORIES}
    new(folder/'calibration.json', {'checkpoint_sha256': sha(folder/'final.pt'), 'normalCAL_only': True, 'new_optimizer_updates_during_CAL': 0,
        'CPU_GPU_and_parent_proofs': proofs, 'fixed_normalCAL_points': {c: {n: v for n, v in cal.items() if n != 'scores'} for c, cal in points.items()}})


def evaluate():
    p = validate(); model, state = load(RUN/'candidate/final.pt'); parent, _ = load(START); rows_all = read(MANIFEST)['rows']; parent_eval = read(PARENT_EVALUATION); results = {}; choices = {}
    for category in NEW_CATEGORIES:
        rows = [r for r in rows_all if r['category'] == category and r['phase'] == 'test']; baseline = p['baseline'][category]; observed = []; pixels = dict(tp=0, fp=0, tn=0, fn=0)
        with torch.inference_mode():
            for i in range(0, len(rows), 8):
                batch = rows[i:i+8]; x = images(batch); location, heat = model.forward_equipment_evidence(x, category); assert torch.equal(location, parent.forward_equipment(x, category))
                for row, local_map, score in zip(batch, location.cpu(), photo_score(heat).cpu().tolist()):
                    mask = np.zeros((SIZE, SIZE), dtype=bool)
                    for path in row['masks']:
                        with Image.open(ROOT/path) as im: mask |= np.array(im.convert('L').resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 0
                    local = counts(mask, local_map.numpy() > state['equipment_operating_points'][category]['pixel_threshold'])
                    for key, value in local.items(): pixels[key] += value
                    observed.append({'id': row['id'], 'category': category, 'label': row['label'], 'defect': row['defect'], 'score': score, 'pixels': local})
        assert pixels == baseline['pixel_counts_model256']; candidates = {}
        for name, alpha in BUDGETS.items():
            point = state['discriminator_calibrations'][category][name]; values = deepcopy(observed)
            for row in values: row['candidate_detected'] = row['score'] > point['threshold']
            rates = classification_rates([r['label'] for r in values], [r['score'] for r in values], point['threshold']); n, d = rates['negative_photos'], rates['positive_photos']; groups = {}
            for group in sorted({r['defect'] for r in values if r['label']}):
                subset = [r for r in values if r['defect'] == group]; fn = sum(not r['candidate_detected'] for r in subset); groups[group] = {'defect_photos': len(subset), 'fn': fn, 'fnr': fn/len(subset)}
            guards = {'photo_FPR_within20': rates['fpr'] <= .20, 'photo_FNR_within20': rates['fnr'] <= .20, 'photo_FNR_better_than_parent': rates['fnr'] < baseline['photo_rates']['fnr'],
                'normal_and_defect_TEST_at_least20': min(n, d) >= 20, 'all_original_location_predictions_exact': True, 'all_parent_tensors_and_reference_buffers_exact': True}
            candidates[name] = {'photo_rates': rates, 'auroc': rank_auroc([r['label'] for r in values], [r['score'] for r in values]), 'normal_test_photos': n, 'defect_test_photos': d,
                'photo_threshold': point['threshold'], 'photo_normalCAL_alpha': alpha, 'photo_normalCAL_rank': point['rank'], 'normal_CAL_photos': p['split_counts'][category]['cal'],
                'pixel_counts_model256': pixels, 'pixel_iou': baseline['pixel_iou'], 'pixel_fnr': baseline['pixel_fnr'], 'by_defect': groups,
                'stage_gate': stage_gate(rates['fpr'], rates['fnr'], n, d), 'guards': guards, 'selected_eligible': all(guards.values()), 'observations': values}
            print(json.dumps({'category': category, 'candidate': name, 'FPR': rates['fpr'], 'FNR': rates['fnr'], 'eligible': all(guards.values())}), flush=True)
        results[category] = candidates; eligible = [n for n, v in candidates.items() if v['selected_eligible']]
        choices[category] = min(eligible, key=lambda n: (max(candidates[n]['photo_rates']['fpr'], candidates[n]['photo_rates']['fnr']), candidates[n]['photo_rates']['fnr'], list(BUDGETS).index(n))) if eligible else 'prior_prototype'
    new(REPORTS/'equipment-component-discriminator-v5-validation.json', {'candidate_checkpoint_sha256': sha(RUN/'candidate/final.pt'), 'categories': results})
    selected = torch.load(START, map_location='cpu', weights_only=True); evaluation = deepcopy(parent_eval); added = [c for c, n in choices.items() if n != 'prior_prototype']
    if added:
        selected['state_dict'].update({n: v for n, v in state['state_dict'].items() if n.startswith('component_discriminator.')})
        selected.update(architecture=ARCH, discriminator_active_components=added)
    for category in added:
        value = results[category][choices[category]]
        selected['equipment_operating_points'][category].update(photo_threshold=value['photo_threshold'], scores=state['discriminator_calibrations'][category]['scores'],
            photo_normalCAL_alpha=value['photo_normalCAL_alpha'], photo_normalCAL_rank=value['photo_normalCAL_rank'], photo_score_method='synthetic_discriminator_top1percent')
        evaluation['categories'][category].update({k: value[k] for k in ('photo_rates', 'auroc', 'photo_threshold', 'stage_gate', 'by_defect', 'photo_normalCAL_alpha', 'photo_normalCAL_rank')})
        evaluation['categories'][category]['photo_score_source'] = 'synthetic_discriminator'; observations = {r['id']: r for r in value['observations']}
        for row in evaluation['rows']:
            if row['category'] == category: row.update(score=observations[row['id']]['score'], candidate_detected=observations[row['id']]['candidate_detected'])
    selected.update(discriminator_selected=choices, discriminator_protocol_sha256=sha(PROTOCOL), discriminator_protocol_commit=state['discriminator_protocol_commit'],
        parent_prototype_sha256=EXPECTED, neural_updates_this_refinement=EPOCHS*STEPS, discriminator_epochs=EPOCHS)
    folder = RUN/'selected'; folder.mkdir()
    with (folder/'final.pt').open('xb') as out: torch.save(selected, out)
    digest = sha(folder/'final.pt'); stage20 = [c for c, v in evaluation['categories'].items() if v['stage_gate']['ready_for_next_stage']]
    evaluation.update(checkpoint_sha256=digest, status='known_component_discriminator_v5', parent_prototype_sha256=EXPECTED, stage20_categories=stage20, known_TEST_reused=True)
    new(folder/'evaluation.json', evaluation)
    new(REPORTS/'equipment-component-discriminator-v5-selection.json', {'selected_per_category': choices, 'newly_selected_categories': added, 'checkpoint_path': (folder/'final.pt').relative_to(ROOT).as_posix(),
        'checkpoint_sha256': digest, 'stage20_categories': stage20, 'actual_new_neural_optimizer_updates': EPOCHS*STEPS, 'final_epoch_only': True,
        'all_parent_tensors_and_reference_buffers_exact': True, 'known_TEST_reused_for_development': True, 'independent_field_validated': False, 'app_default_promoted': False})


def verify():
    p = validate(); ledger = read(RUN/'protected-inputs.json')
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda item: reference.check(*item), ledger.items()))
    for n, h in p['legacy_source_sha256'].items(): assert sha(ROOT/n) == h, n
    source = torch.load(START, map_location='cpu', weights_only=True); trained = torch.load(RUN/'train/final-trained.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(v, trained['state_dict'][n]) for n, v in source['state_dict'].items())
    history = [json.loads(line) for line in (RUN/'train/epochs.jsonl').read_text().splitlines()]
    assert len(history) == EPOCHS and all(r['optimizer_updates'] == (i+1)*STEPS for i, r in enumerate(history))
    new(REPORTS/'equipment-component-discriminator-v5-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved': True, 'all_parent_tensors_and_reference_buffers_exact': True, 'actual_optimizer_updates': EPOCHS*STEPS, 'actual_epochs': EPOCHS,
        'new_real_defect_labels': 0, 'normal_FIT_only_plus_synthetic_features': True, 'TEST_or_CAL_not_used_for_training_or_epoch_choice': True,
        'all_original_candidate_TEST_location_predictions_exact': True, 'selected_per_category': read(REPORTS/'equipment-component-discriminator-v5-selection.json')['selected_per_category']})
    print(json.dumps({'verified': True, 'actual_updates': EPOCHS*STEPS, 'protected_files': len(ledger)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['freeze', 'train', 'calibrate', 'eval', 'verify'], required=True)
    args = parser.parse_args(); {'freeze': freeze, 'train': train, 'calibrate': calibrate, 'eval': evaluate, 'verify': verify}[args.mode]()
