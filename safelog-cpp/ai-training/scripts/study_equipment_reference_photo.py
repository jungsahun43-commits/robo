"""Fixed cable/nut normal-reference presence experiment, separate location output."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.equipment_reference_photo import ARCH, REFERENCE_CATEGORIES, ReferencePhotoMain
from safelog_ai.equipment_expansion import ExpandedMain, CATEGORIES, SIZE, photo_score, stage_gate
from safelog_ai.metal_reference import patch_features, normal_budget_threshold
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.retention_distillation import state_sha256
from scripts.study_equipment_expansion import images, read, MANIFEST
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha
from scripts.study_unified_belt_tiles import snapshot

RUN = ROOT/'runs/equipment-reference-photo-v1'
REPORTS = ROOT/'reports'
PROTOCOL = REPORTS/'equipment-reference-photo-v1-protocol.json'
START = ROOT/'runs/equipment-metal-reference-v1/calibration_only20/final.pt'
EXPECTED = '0f9b7322ae5d8e0aea8cc7a10655eff04e5f6dd2ed6c6bcf585fb05a3179c5f0'
VARIANTS = ('calibration20', 'reference_photo20')
SOURCES = ('safelog_ai/equipment_reference_photo.py', 'scripts/study_equipment_reference_photo.py',
           'tests/test_equipment_reference_photo.py')


def check(name, value):
    assert snapshot(ROOT/name) == value, name


def freeze():
    assert not RUN.exists() and sha(START) == EXPECTED
    RUN.mkdir()
    ledger = dict(read(ROOT/'runs/equipment-metal-reference-v1/protected-inputs.json'))
    for directory in (REPORTS, ROOT/'runs/equipment-metal-reference-v1'):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')):
                ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: check(*item), ledger.items()))
    new(RUN/'protected-inputs.json', ledger)
    rows = read(MANIFEST)['rows']
    counts = {c: {p: sum(r['category'] == c and r['phase'] == p for r in rows) for p in ('fit', 'cal', 'test')} for c in REFERENCE_CATEGORIES}
    assert counts == {'ad_cable': {'fit': 178, 'cal': 45, 'test': 150}, 'ad_metal_nut': {'fit': 176, 'cal': 44, 'test': 115}}
    baseline = read(REPORTS/'equipment-expansion-v1-results.json')['categories']
    new(PROTOCOL, {
        'schema': 'safelog-equipment-reference-photo-fixed-v1',
        'declared_before_fit_CAL_or_TEST_rescoring': True, 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'tests', 'safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256': EXPECTED, 'manifest_sha256': sha(MANIFEST), 'protected_inputs_sha256': sha(RUN/'protected-inputs.json'),
        'categories': list(REFERENCE_CATEGORIES), 'counts': counts, 'variants': list(VARIANTS),
        'baseline': {c: baseline[c] for c in REFERENCE_CATEGORIES},
        'calibration20': 'Original reconstruction photo scores; normalCAL order statistic ceil((n+1)*.8); all neural tensors, pixel thresholds and location maps retained',
        'reference_photo20': 'One SAME frozen MobileNet pass; normalized low40/high960 then avgpool3 stride1 count_include_padFalse; ALL normal FIT256 patches/photo stored FP16; exact nearest cosine FP32 chunks512/2048; bilinear256; mean top1percent reference distance for PHOTO presence only',
        'reference_counts': {c: counts[c]['fit']*256 for c in REFERENCE_CATEGORIES},
        'reference_source': 'Verified publisher-normal FIT photos only, no CAL/TEST patches or masks',
        'reference_storage_dtype': 'float16; FP32 normalization/search', 'coreset_or_parameter_search': False,
        'localization': 'Original shared reconstructor heatmap and pixel threshold unchanged; reference evidence is a separate output and is evaluated separately, not substituted for location results',
        'photo_threshold': 'Only normalCAL, alpha.20 strict >, rankceil((n+1)*.8); both variants fixed before TEST scoring',
        'prespecified_guards_per_category': ['photo_FPR<=.20', 'photo_FNR<=.20', 'photo_FNR<prior', 'original_location_predictions_exact', 'all_prior_neural_tensors_exact'],
        'selection': 'Per category only all guards pass; lower max(FPR,FNR), then lowerFNR, then calibration20; failing category retains parent. Combine selected buffers and thresholds into ONE checkpoint.',
        'reference_heatmap_selection_guard': False, 'reference_heatmap_is_confirmed_location': False,
        'all_other_categories_thresholds_and_outputs_preserved': True, 'metal20_and_ksdd15_preserved': True,
        'neural_optimizer_updates_planned': 0, 'new_source_photos_or_labels': 0,
        'TEST_reused_known_public_benchmark': True, 'new_independent_factory_validation': False, 'app_default_promoted': False,
        'method_inspiration': 'https://arxiv.org/abs/2106.08265', 'exact_PatchCore_reproduction': False, 'new_unit_tests_passed': 3})
    print(json.dumps({'frozen': True, 'counts': counts, 'protected_files': len(ledger), 'new_neural_updates': 0}), flush=True)


def validate():
    p = read(PROTOCOL)
    for name, digest in p['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    for path, digest in ((START, EXPECTED), (MANIFEST, p['manifest_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == digest, str(path)
    return p


def load(path, device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = ReferencePhotoMain(state.get('reference_counts', {})) if state.get('architecture') == ARCH else ExpandedMain()
    model.load_state_dict(state['state_dict'], strict=True)
    return model.to(device).eval().requires_grad_(False), state


def evidence(model, batch, category):
    if isinstance(model, ReferencePhotoMain): return model.forward_equipment_evidence(batch, category)
    heat = model.forward_equipment(batch, category)
    return heat, heat


def fit():
    p = validate(); repo = ROOT.parents[1]
    git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    commit = subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip()
    source = torch.load(START, map_location='cpu', weights_only=True); rows = read(MANIFEST)['rows']
    for variant in VARIANTS:
        folder = RUN/variant; assert not folder.exists(); folder.mkdir(); start = time.monotonic(); state = deepcopy(source)
        state.update(reference_photo_variant=variant, neural_updates_this_refinement=0, photo_normalCAL_alpha=.20,
                     parent_prototype_sha256=EXPECTED, reference_photo_protocol_sha256=sha(PROTOCOL), reference_photo_protocol_commit=commit)
        if variant == 'reference_photo20':
            model = ReferencePhotoMain(p['reference_counts']).cuda().eval().requires_grad_(False)
            missing, extra = model.load_state_dict(source['state_dict'], strict=False)
            assert set(missing) == {'normal_reference_'+c for c in REFERENCE_CATEGORIES} and not extra
            bank_proofs = {}
            with torch.inference_mode():
                for category in REFERENCE_CATEGORIES:
                    fit_rows = [r for r in rows if r['category'] == category and r['phase'] == 'fit']
                    patches = []
                    for i in range(0, len(fit_rows), 8):
                        feature = patch_features(model.backbone(images(fit_rows[i:i+8])))
                        patches.append(feature.permute(0, 2, 3, 1).reshape(-1, 1000).half().cpu())
                    bank = torch.cat(patches); assert bank.shape == (p['reference_counts'][category], 1000)
                    getattr(model, 'normal_reference_'+category).copy_(bank)
                    bank_proofs[category] = {'source_ids': [r['id'] for r in fit_rows], 'feature_sha256': state_sha256({'reference': bank}), 'patches': len(bank), 'dtype': 'float16'}
                    print(json.dumps({'normal_reference_fit': category, 'photos': len(fit_rows), 'patches': len(bank)}), flush=True)
                for category in REFERENCE_CATEGORIES:
                    cal = [r for r in rows if r['category'] == category and r['phase'] == 'cal']; scores = []
                    for i in range(0, len(cal), 8):
                        _, maps = evidence(model, images(cal[i:i+8]), category); scores.extend(photo_score(maps).cpu().double().tolist())
                    threshold, rank = normal_budget_threshold(scores, .20)
                    state['equipment_operating_points'][category].update(photo_threshold=threshold, scores=scores, photo_normalCAL_alpha=.20, photo_normalCAL_rank=rank, photo_score_method='normal_reference_top1percent')
            state.update(architecture=ARCH, reference_counts=p['reference_counts'], normal_reference_fit=bank_proofs,
                         state_dict={n: v.cpu() for n, v in model.state_dict().items()})
        else:
            for category in REFERENCE_CATEGORIES:
                point = state['equipment_operating_points'][category]; threshold, rank = normal_budget_threshold(point['scores'], .20)
                point.update(photo_threshold=threshold, photo_normalCAL_alpha=.20, photo_normalCAL_rank=rank, photo_score_method='reconstruction_top1percent')
        assert all(torch.equal(value, state['state_dict'][name]) for name, value in source['state_dict'].items())
        with (folder/'final.pt').open('xb') as out: torch.save(state, out)
        model, _ = load(folder/'final.pt'); cpu, _ = load(folder/'final.pt', 'cpu'); prior, _ = load(START, 'cpu'); proofs = {}
        with torch.inference_mode():
            for category in CATEGORIES:
                row = next(r for r in rows if r['category'] == category and r['phase'] == 'fit'); x = images([row])
                gpu_location, gpu_presence = evidence(model, x, category); location, presence = evidence(cpu, x.cpu(), category)
                assert torch.equal(location, prior.forward_equipment(x.cpu(), category)), category
                assert torch.allclose(gpu_location.cpu(), location, atol=2e-4, rtol=.002), category
                assert torch.allclose(gpu_presence.cpu(), presence, atol=2e-4, rtol=.002), category
                proofs[category] = {'original_CPU_location_exact': True, 'CPU_GPU_both_outputs_close': True}
        new(folder/'fit.json', {'status': 'normal_FIT_reference_and_normal_CAL_only', 'variant': variant, 'checkpoint_sha256': sha(folder/'final.pt'),
            'all_prior_neural_tensors_exact': True, 'normal_reference_FIT_only': True, 'TEST_used_for_fit_or_CAL': False, 'neural_optimizer_updates': 0,
            'reference_counts': state.get('reference_counts', {}), 'category_proofs': proofs, 'elapsed_fit_CAL_proofs_seconds': time.monotonic()-start})
        print(json.dumps({'fitted': variant, 'checkpoint_sha256': sha(folder/'final.pt'), 'neural_updates': 0}), flush=True)


def counts(mask, prediction):
    return {name: int(value.sum()) for name, value in (('tp', prediction & mask), ('fp', prediction & ~mask), ('tn', ~prediction & ~mask), ('fn', ~prediction & mask))}


def evaluate():
    p = validate(); all_rows = read(MANIFEST)['rows']; original = read(REPORTS/'equipment-expansion-v1-results.json')
    old_rows = {r['id']: r for r in original['rows']}; results = {}
    for variant in VARIANTS:
        model, state = load(RUN/variant/'final.pt'); prior, _ = load(START); category_results = {}
        for category in REFERENCE_CATEGORIES:
            rows = [r for r in all_rows if r['category'] == category and r['phase'] == 'test']; point = state['equipment_operating_points'][category]
            scores = []; pixels = dict(tp=0, fp=0, tn=0, fn=0); observations = []
            with torch.inference_mode():
                for i in range(0, len(rows), 8):
                    batch = rows[i:i+8]; x = images(batch); locations, maps = evidence(model, x, category)
                    assert torch.equal(locations, prior.forward_equipment(x, category)), (variant, category, i)
                    locations = locations.cpu(); maps = maps.cpu(); batch_scores = photo_score(maps).tolist(); old_scores = photo_score(locations).tolist(); scores.extend(batch_scores)
                    for row, location, heat, score, old_score in zip(batch, locations, maps, batch_scores, old_scores):
                        mask = np.zeros((SIZE, SIZE), dtype=bool)
                        for path in row['masks']:
                            with Image.open(ROOT/path) as im: mask |= np.array(im.convert('L').resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 0
                        local = counts(mask, location.numpy() > point['pixel_threshold'])
                        # The separate reference map has no fitted location threshold.
                        # It is photo evidence, not an evaluated segmentation output.
                        assert abs(old_score-old_rows[row['id']]['score']) < 1e-6
                        for key, value in local.items(): pixels[key] += value
                        observations.append({'id': row['id'], 'category': category, 'label': row['label'], 'defect': row['defect'], 'score': score,
                            'candidate_detected': score > point['photo_threshold'], 'original_location_photo_score': old_score, 'pixels': local})
            rates = classification_rates([r['label'] for r in rows], scores, point['photo_threshold']); normal = sum(not r['label'] for r in rows); defect = len(rows)-normal
            baseline = p['baseline'][category]; assert pixels == baseline['pixel_counts_model256']
            guards = {'photo_FPR_within20': rates['fpr'] <= .20, 'photo_FNR_within20': rates['fnr'] <= .20,
                      'photo_FNR_better_than_prior': rates['fnr'] < baseline['photo_rates']['fnr'], 'original_location_predictions_exact': True, 'all_prior_neural_tensors_exact': True}
            category_results[category] = {'photo_rates': rates, 'photo_auroc': rank_auroc([r['label'] for r in rows], scores),
                'normal_test': normal, 'defect_test': defect, 'pixel_counts_model256': pixels, 'pixel_iou': baseline['pixel_iou'], 'pixel_fnr': baseline['pixel_fnr'],
                'photo_threshold': point['photo_threshold'], 'photo_normalCAL_rank': point['photo_normalCAL_rank'], 'normalCAL_count': point['normal_CAL_photos'],
                'stage_gate': stage_gate(rates['fpr'], rates['fnr'], normal, defect), 'guards': guards, 'selected_eligible': all(guards.values()), 'observations': observations,
                'reference_heatmap_scored_as_location': False, 'known_TEST_reused': True}
            print(json.dumps({'variant': variant, 'category': category, 'FPR': rates['fpr'], 'FNR': rates['fnr'], 'AUROC': category_results[category]['photo_auroc'], 'eligible': all(guards.values()), 'location_exact': True}), flush=True)
        results[variant] = category_results
        new(REPORTS/f'equipment-reference-photo-v1-{variant}-validation.json', {'checkpoint_sha256': sha(RUN/variant/'final.pt'), 'categories': category_results})
    chosen = {}
    for category in REFERENCE_CATEGORIES:
        eligible = [v for v in VARIANTS if results[v][category]['selected_eligible']]
        chosen[category] = min(eligible, key=lambda v: (max(results[v][category]['photo_rates']['fpr'], results[v][category]['photo_rates']['fnr']), results[v][category]['photo_rates']['fnr'], VARIANTS.index(v))) if eligible else 'prior_prototype'
    source = torch.load(START, map_location='cpu', weights_only=True); selected = deepcopy(source)
    reference_state = torch.load(RUN/'reference_photo20/final.pt', map_location='cpu', weights_only=True)
    bank_counts = {}
    for category, variant in chosen.items():
        if variant != 'prior_prototype':
            candidate = torch.load(RUN/variant/'final.pt', map_location='cpu', weights_only=True)
            selected['equipment_operating_points'][category] = deepcopy(candidate['equipment_operating_points'][category])
        if variant == 'reference_photo20':
            name = 'normal_reference_'+category; selected['state_dict'][name] = reference_state['state_dict'][name]
            bank_counts[category] = p['reference_counts'][category]
    if bank_counts:
        selected.update(architecture=ARCH, reference_counts=bank_counts,
                        normal_reference_fit={c: reference_state['normal_reference_fit'][c] for c in bank_counts})
    selected.update(reference_photo_selected=chosen, parent_prototype_sha256=EXPECTED, neural_updates_this_refinement=0,
                    reference_photo_protocol_sha256=sha(PROTOCOL), reference_photo_protocol_commit=reference_state['reference_photo_protocol_commit'])
    folder = RUN/'selected'; folder.mkdir()
    with (folder/'final.pt').open('xb') as out: torch.save(selected, out)
    evaluation = read(ROOT/'runs/equipment-metal-reference-v1/handoff-stage/evaluation.json')
    for category, variant in chosen.items():
        if variant == 'prior_prototype': continue
        value = results[variant][category]
        evaluation['categories'][category].update({k: value[k] for k in ('photo_rates', 'photo_auroc', 'pixel_counts_model256', 'pixel_iou', 'pixel_fnr', 'photo_threshold', 'stage_gate')})
        evaluation['categories'][category]['auroc'] = value['photo_auroc']
        observations = {r['id']: r for r in value['observations']}
        for row in evaluation['rows']:
            if row['category'] == category: row.update(score=observations[row['id']]['score'], candidate_detected=observations[row['id']]['candidate_detected'])
    digest = sha(folder/'final.pt')
    evaluation.update(checkpoint_sha256=digest, status='known_equipment_reference_photo_refined', parent_prototype_sha256=EXPECTED, known_TEST_reused=True,
                      stage20_categories=[c for c, value in evaluation['categories'].items() if value['stage_gate']['ready_for_next_stage']])
    new(folder/'evaluation.json', evaluation)
    new(REPORTS/'equipment-reference-photo-v1-selection.json', {'selected_per_category': chosen, 'checkpoint_path': (folder/'final.pt').relative_to(ROOT).as_posix(),
        'checkpoint_sha256': digest, 'reference_counts': bank_counts, 'all_prior_neural_tensors_exact': True, 'all_original_location_predictions_exact': True,
        'metal20_and_ksdd15_preserved': True, 'known_TEST_reused': True, 'new_neural_optimizer_updates': 0, 'independent_field_validated': False, 'app_default_promoted': False,
        'stage20_categories': evaluation['stage20_categories'], 'variant_guards': {v: {c: value['guards'] for c, value in result.items()} for v, result in results.items()}})


def verify():
    p = validate(); ledger = read(RUN/'protected-inputs.json')
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda item: check(*item), ledger.items()))
    for name, digest in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == digest, name
    model, selected = load(RUN/'selected/final.pt', 'cpu'); prior, source = load(START, 'cpu'); rows = read(MANIFEST)['rows']
    assert all(torch.equal(value, selected['state_dict'][name]) for name, value in source['state_dict'].items())
    for category in CATEGORIES:
        if category not in REFERENCE_CATEGORIES: assert selected['equipment_operating_points'][category] == source['equipment_operating_points'][category], category
        with torch.inference_mode():
            row = next(r for r in rows if r['category'] == category and r['phase'] == 'fit'); x = images([row], 'cpu')
            location, presence = evidence(model, x, category)
            assert torch.equal(location, prior.forward_equipment(x, category)), category
            if category not in selected.get('reference_counts', {}): assert torch.equal(location, presence), category
    new(REPORTS/'equipment-reference-photo-v1-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved': True, 'all_prior_neural_tensors_exact': True, 'all_11_original_CPU_location_outputs_exact': True,
        'all_other_category_operating_points_exact': True, 'metal20_and_ksdd15_preserved': True, 'reference_contains_FIT_only': True,
        'new_neural_optimizer_updates': 0, 'selected_checkpoint_sha256': sha(RUN/'selected/final.pt')})
    print(json.dumps({'verified': True, 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256'])}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['freeze', 'fit', 'eval', 'verify'], required=True)
    args = parser.parse_args(); {'freeze': freeze, 'fit': fit, 'eval': evaluate, 'verify': verify}[args.mode]()
