"""Preregistered screw and connector photo study, preserving passed categories."""
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
from safelog_ai.component_reference import ARCH, NEW_CATEGORIES, ComponentReferenceMain
from safelog_ai.equipment_reference_photo import ReferencePhotoMain
from safelog_ai.equipment_expansion import CATEGORIES, SIZE, photo_score, stage_gate
from safelog_ai.metal_reference import patch_features, normal_budget_threshold
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.retention_distillation import state_sha256
from scripts.study_equipment_expansion import images, read, MANIFEST
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_equipment_reference_photo import counts

RUN = ROOT/'runs/equipment-component-reference-v3'
REPORTS = ROOT/'reports'; PROTOCOL = REPORTS/'equipment-component-reference-v3-protocol.json'
START = ROOT/'runs/equipment-reference-budget-v2/selected/final.pt'
EXPECTED = 'bb30750a1a69146c08195e42fb7ea5dada74feaed37e554726216afa4b03b92e'
PARENT_EVALUATION = ROOT/'runs/equipment-reference-budget-v2/selected/evaluation.json'
BUDGETS = {'normalCAL10': .10, 'normalCAL05': .05}
SOURCES = ('safelog_ai/component_reference.py', 'scripts/study_component_reference.py', 'tests/test_component_reference.py')


def check(name, value): assert snapshot(ROOT/name) == value, name


def load(path, device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = ComponentReferenceMain(state['reference_counts']) if state['architecture'] == ARCH else ReferencePhotoMain(state['reference_counts'])
    model.load_state_dict(state['state_dict'], strict=True)
    return model.to(device).eval().requires_grad_(False), state


def freeze():
    assert not RUN.exists() and sha(START) == EXPECTED
    RUN.mkdir(); ledger = dict(read(ROOT/'runs/equipment-reference-budget-v2/protected-inputs.json'))
    for directory in (REPORTS, ROOT/'runs/equipment-reference-budget-v2'):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda item: check(*item), ledger.items()))
    new(RUN/'protected-inputs.json', ledger)
    rows = read(MANIFEST)['rows']; baseline = read(PARENT_EVALUATION)['categories']
    split_counts = {c: {phase: sum(r['category'] == c and r['phase'] == phase for r in rows) for phase in ('fit', 'cal', 'test')} for c in NEW_CATEGORIES}
    assert split_counts == {'ad_screw': {'fit': 96, 'cal': 24, 'test': 160}, 'loco_splicing_connectors': {'fit': 330, 'cal': 54, 'test': 312}, 'mpdd_connector': {'fit': 101, 'cal': 25, 'test': 44}}
    new(PROTOCOL, {'schema': 'safelog-component-reference-fixed-v3', 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_new_fit_CAL_or_TEST_rescoring': True,
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'tests', 'safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256': EXPECTED, 'parent_evaluation_sha256': sha(PARENT_EVALUATION), 'manifest_sha256': sha(MANIFEST), 'protected_inputs_sha256': sha(RUN/'protected-inputs.json'),
        'categories': list(NEW_CATEGORIES), 'split_counts': split_counts, 'baseline': {c: baseline[c] for c in NEW_CATEGORIES},
        'reference_counts_added': {c: v['fit']*256 for c, v in split_counts.items()}, 'budgets': BUDGETS,
        'method': 'Same one frozen accepted MobileNet backbone; low40/high960 branch-normalized features, avgpool3 stride1 count_include_padFalse then unit normalization; all normal FIT256 patches/photo stored FP16, exact nearest cosine FP32 chunks512/2048; bilinear256; photo mean top1percent',
        'reference_source': 'Only verified publisher-normal FIT, no CAL/TEST feature in new memory',
        'normalCAL_thresholds': 'Per category normalCAL only: rankceil((n+1)*(1-alpha)), strict >; ONLY alpha.10 and.05, no TEST threshold quantile or sweep',
        'prespecified_guards': ['photo_FPR<=.20', 'photo_FNR<=.20', 'photo_FNR<parent', 'normal_TEST>=20_and_defect_TEST>=20', 'original_location_predictions_exact', 'all_parent_tensors_and_existing_reference_buffers_exact'],
        'selection': 'Per category eligible lower max(FPR,FNR), then lowerFNR, then normalCAL10. Otherwise retain parent. Save ONE combined selected checkpoint, only selected new buffers.',
        'MPDD_connector_defect_TEST_count': 14, 'MPDD_connector_cannot_pass_stage_due_to_insufficient_evaluation': True,
        'LOCO_defect_group_FNR_required': ['logical_anomalies', 'structural_anomalies'],
        'LOCO_patch_memory_may_miss_relationship_and_count_errors': True,
        'existing_cable_nut_metal_ksdd_passed_results_preserved': True, 'all_11_location_outputs_preserved_not_improved': True,
        'reference_heatmap_is_evaluated_segmentation': False,
        'new_neural_optimizer_updates_planned': 0, 'new_normal_reference_fit_photos': 527, 'new_source_photos_or_labels': 0,
        'known_public_TEST_reused_for_development_and_candidate_selection': True, 'independent_field_validation': False, 'app_default_promoted': False,
        'new_unit_tests_passed': 3, 'method_inspiration': 'https://arxiv.org/abs/2106.08265', 'exact_PatchCore_reproduction': False})
    print(json.dumps({'frozen': True, 'normal_FIT': 527, 'new_reference_patches': 134912, 'budgets': BUDGETS, 'protected_files': len(ledger)}), flush=True)


def validate():
    p = read(PROTOCOL)
    for name, digest in p['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    for path, digest in ((START, EXPECTED), (PARENT_EVALUATION, p['parent_evaluation_sha256']), (MANIFEST, p['manifest_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == digest, str(path)
    return p


def fit():
    p = validate(); repo = ROOT.parents[1]
    git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    commit = subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip(); begin = time.monotonic()
    source = torch.load(START, map_location='cpu', weights_only=True); all_counts = {**source['reference_counts'], **p['reference_counts_added']}
    model = ComponentReferenceMain(all_counts).cuda().eval().requires_grad_(False)
    missing, extra = model.load_state_dict(source['state_dict'], strict=False)
    assert set(missing) == {'normal_reference_'+c for c in NEW_CATEGORIES} and not extra
    rows = read(MANIFEST)['rows']; banks = {}; calibrations = {}
    with torch.inference_mode():
        for category in NEW_CATEGORIES:
            fit_rows = [r for r in rows if r['category'] == category and r['phase'] == 'fit']; patches = []
            for i in range(0, len(fit_rows), 8):
                feature = patch_features(model.backbone(images(fit_rows[i:i+8])))
                patches.append(feature.permute(0, 2, 3, 1).reshape(-1, 1000).half().cpu())
            bank = torch.cat(patches); assert bank.shape == (all_counts[category], 1000)
            getattr(model, 'normal_reference_'+category).copy_(bank)
            banks[category] = {'source_ids': [r['id'] for r in fit_rows], 'feature_sha256': state_sha256({'reference': bank}), 'patches': len(bank), 'dtype': 'float16'}
            print(json.dumps({'normal_reference_fit': category, 'photos': len(fit_rows), 'patches': len(bank)}), flush=True)
            cal = [r for r in rows if r['category'] == category and r['phase'] == 'cal']; scores = []
            for i in range(0, len(cal), 8):
                _, heat = model.forward_equipment_evidence(images(cal[i:i+8]), category); scores.extend(photo_score(heat).cpu().double().tolist())
            calibrations[category] = {name: dict(zip(('threshold', 'rank'), normal_budget_threshold(scores, alpha))) for name, alpha in BUDGETS.items()}
            calibrations[category]['scores'] = scores
    state = deepcopy(source); state.update(architecture=ARCH, reference_counts=all_counts,
        normal_reference_fit={**source['normal_reference_fit'], **banks},
        state_dict={n: v.cpu() for n, v in model.state_dict().items()}, component_reference_calibrations=calibrations,
        component_reference_protocol_sha256=sha(PROTOCOL), component_reference_protocol_commit=commit,
        parent_prototype_sha256=EXPECTED, neural_updates_this_refinement=0)
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in source['state_dict'].items())
    folder = RUN/'candidate'; folder.mkdir()
    with (folder/'final.pt').open('xb') as out: torch.save(state, out)
    cpu, _ = load(folder/'final.pt', 'cpu'); prior, _ = load(START, 'cpu'); proofs = {}
    with torch.inference_mode():
        for category in CATEGORIES:
            row = next(r for r in rows if r['category'] == category and r['phase'] == 'fit'); x = images([row])
            gpu_location, gpu_presence = model.forward_equipment_evidence(x, category)
            location, presence = cpu.forward_equipment_evidence(x.cpu(), category)
            prior_location, prior_presence = prior.forward_equipment_evidence(x.cpu(), category)
            assert torch.equal(location, prior_location), category
            assert torch.allclose(gpu_location.cpu(), location, atol=2e-4, rtol=.002) and torch.allclose(gpu_presence.cpu(), presence, atol=2e-4, rtol=.002), category
            if category not in NEW_CATEGORIES: assert torch.equal(presence, prior_presence), category
            proofs[category] = {'original_location_CPU_exact': True, 'CPU_GPU_both_outputs_close': True, 'prior_photo_presence_exact': category not in NEW_CATEGORIES}
    new(folder/'fit.json', {'status': 'normal_FIT_reference_and_normal_CAL_only', 'checkpoint_sha256': sha(folder/'final.pt'),
        'parent_sha256': EXPECTED, 'normal_reference_FIT_photos_added': 527, 'normal_reference_patches_added': 134912,
        'all_parent_neural_tensors_and_reference_buffers_exact': True, 'new_neural_optimizer_updates': 0,
        'TEST_used_for_fit_or_CAL': False, 'proofs': proofs, 'normalCAL_fixed_points': {c: {n: v for n, v in cal.items() if n != 'scores'} for c, cal in calibrations.items()},
        'elapsed_fit_CAL_and_proofs_seconds': time.monotonic()-begin})
    print(json.dumps({'fitted': True, 'new_neural_updates': 0, 'candidate_sha256': sha(folder/'final.pt')}), flush=True)


def evaluate():
    p = validate(); model, state = load(RUN/'candidate/final.pt'); parent, _ = load(START); all_rows = read(MANIFEST)['rows']
    parent_eval = read(PARENT_EVALUATION); old_rows = {r['id']: r for r in parent_eval['rows']}; results = {}; selected_names = {}
    for category in NEW_CATEGORIES:
        rows = [r for r in all_rows if r['category'] == category and r['phase'] == 'test']; baseline = p['baseline'][category]
        observations = []; pixels = dict(tp=0, fp=0, tn=0, fn=0)
        with torch.inference_mode():
            for i in range(0, len(rows), 8):
                batch = rows[i:i+8]; x = images(batch); location, reference = model.forward_equipment_evidence(x, category)
                assert torch.equal(location, parent.forward_equipment(x, category)), (category, i)
                scores = photo_score(reference).cpu().tolist(); original_scores = photo_score(location).cpu().tolist()
                for row, heat, score, old_score in zip(batch, location.cpu(), scores, original_scores):
                    mask = np.zeros((SIZE, SIZE), dtype=bool)
                    for path in row['masks']:
                        with Image.open(ROOT/path) as im: mask |= np.array(im.convert('L').resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 0
                    local = counts(mask, heat.numpy() > state['equipment_operating_points'][category]['pixel_threshold'])
                    assert abs(old_score-old_rows[row['id']]['score']) < 1e-6
                    for key, value in local.items(): pixels[key] += value
                    observations.append({'id': row['id'], 'category': category, 'label': row['label'], 'defect': row['defect'], 'score': score,
                        'original_localization_photo_score': old_score, 'pixels': local})
        assert pixels == baseline['pixel_counts_model256'], category
        category_results = {}
        for name, alpha in BUDGETS.items():
            point = state['component_reference_calibrations'][category][name]; observed = deepcopy(observations)
            for row in observed: row['candidate_detected'] = row['score'] > point['threshold']
            rates = classification_rates([r['label'] for r in observed], [r['score'] for r in observed], point['threshold'])
            normal, defect = rates['negative_photos'], rates['positive_photos']; stage = stage_gate(rates['fpr'], rates['fnr'], normal, defect)
            by_defect = {}
            for group in sorted({r['defect'] for r in observed if r['label']}):
                subset = [r for r in observed if r['defect'] == group]; fn = sum(not r['candidate_detected'] for r in subset)
                by_defect[group] = {'defect_photos': len(subset), 'fn': fn, 'fnr': fn/len(subset)}
            guards = {'photo_FPR_within20': rates['fpr'] <= .20, 'photo_FNR_within20': rates['fnr'] <= .20,
                'photo_FNR_better_than_parent': rates['fnr'] < baseline['photo_rates']['fnr'], 'normal_and_defect_TEST_at_least20': min(normal, defect) >= 20,
                'original_location_predictions_exact': True, 'all_parent_tensors_and_existing_reference_buffers_exact': True}
            value = {'photo_rates': rates, 'auroc': rank_auroc([r['label'] for r in observed], [r['score'] for r in observed]),
                'normal_test_photos': normal, 'defect_test_photos': defect, 'normal_CAL_photos': p['split_counts'][category]['cal'], 'by_defect': by_defect,
                'photo_threshold': point['threshold'], 'photo_normalCAL_rank': point['rank'], 'photo_normalCAL_alpha': alpha,
                'pixel_counts_model256': pixels, 'pixel_iou': baseline['pixel_iou'], 'pixel_fnr': baseline['pixel_fnr'],
                'stage_gate': stage, 'guards': guards, 'selected_eligible': all(guards.values()), 'observations': observed,
                'reference_heatmap_is_evaluated_segmentation': False, 'known_TEST_reused': True}
            category_results[name] = value
            print(json.dumps({'category': category, 'candidate': name, 'FPR': rates['fpr'], 'FNR': rates['fnr'], 'stage': stage['state'], 'eligible': all(guards.values())}), flush=True)
        results[category] = category_results; eligible = [n for n, v in category_results.items() if v['selected_eligible']]
        selected_names[category] = min(eligible, key=lambda n: (max(category_results[n]['photo_rates']['fpr'], category_results[n]['photo_rates']['fnr']), category_results[n]['photo_rates']['fnr'], list(BUDGETS).index(n))) if eligible else 'prior_prototype'
    new(REPORTS/'equipment-component-reference-v3-validation.json', {'candidate_checkpoint_sha256': sha(RUN/'candidate/final.pt'), 'categories': results})
    selected = torch.load(START, map_location='cpu', weights_only=True); evaluation = deepcopy(parent_eval); added = []
    for category, name in selected_names.items():
        if name == 'prior_prototype': continue
        added.append(category); value = results[category][name]; key = 'normal_reference_'+category
        selected['state_dict'][key] = state['state_dict'][key]; selected['reference_counts'][category] = state['reference_counts'][category]
        selected['normal_reference_fit'][category] = state['normal_reference_fit'][category]
        selected['equipment_operating_points'][category].update(photo_threshold=value['photo_threshold'], scores=state['component_reference_calibrations'][category]['scores'],
            photo_normalCAL_rank=value['photo_normalCAL_rank'], photo_normalCAL_alpha=value['photo_normalCAL_alpha'], photo_score_method='normal_reference_top1percent')
        evaluation['categories'][category].update({k: value[k] for k in ('photo_rates', 'auroc', 'photo_threshold', 'stage_gate', 'pixel_counts_model256', 'pixel_iou', 'pixel_fnr', 'by_defect', 'photo_normalCAL_rank', 'photo_normalCAL_alpha')})
        evaluation['categories'][category]['photo_score_source'] = 'normal_reference'
        observed = {r['id']: r for r in value['observations']}
        for row in evaluation['rows']:
            if row['category'] == category: row.update(score=observed[row['id']]['score'], candidate_detected=observed[row['id']]['candidate_detected'])
    if added: selected['architecture'] = ARCH
    selected.update(component_reference_selected=selected_names, component_reference_protocol_sha256=sha(PROTOCOL),
        component_reference_protocol_commit=state['component_reference_protocol_commit'], parent_prototype_sha256=EXPECTED, neural_updates_this_refinement=0)
    folder = RUN/'selected'; folder.mkdir()
    with (folder/'final.pt').open('xb') as out: torch.save(selected, out)
    digest = sha(folder/'final.pt'); stage20 = [c for c, v in evaluation['categories'].items() if v['stage_gate']['ready_for_next_stage']]
    evaluation.update(checkpoint_sha256=digest, status='known_component_reference_v3', parent_prototype_sha256=EXPECTED,
        known_TEST_reused=True, adaptive_development_on_known_TEST=True, stage20_categories=stage20)
    new(folder/'evaluation.json', evaluation)
    new(REPORTS/'equipment-component-reference-v3-selection.json', {'selected_per_category': selected_names, 'newly_selected_categories': added,
        'checkpoint_path': (folder/'final.pt').relative_to(ROOT).as_posix(), 'checkpoint_sha256': digest, 'reference_counts': selected['reference_counts'],
        'stage20_categories': stage20, 'new_neural_optimizer_updates': 0, 'all_original_location_outputs_preserved': True,
        'known_TEST_reused_for_development': True, 'independent_field_validated': False, 'app_default_promoted': False,
        'all_parent_tensors_and_reference_buffers_exact': True, 'variant_guards': {c: {n: v['guards'] for n, v in group.items()} for c, group in results.items()}})


def verify():
    p = validate(); ledger = read(RUN/'protected-inputs.json')
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda item: check(*item), ledger.items()))
    for name, digest in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == digest, name
    selected, state = load(RUN/'selected/final.pt', 'cpu'); prior, parent = load(START, 'cpu'); rows = read(MANIFEST)['rows']
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in parent['state_dict'].items())
    added = read(REPORTS/'equipment-component-reference-v3-selection.json')['newly_selected_categories']; proofs = {}
    with torch.inference_mode():
        for category in CATEGORIES:
            if category not in added: assert state['equipment_operating_points'][category] == parent['equipment_operating_points'][category], category
            row = next(r for r in rows if r['category'] == category and r['phase'] == 'fit'); x = images([row], 'cpu')
            location, presence = selected.forward_equipment_evidence(x, category); old_location, old_presence = prior.forward_equipment_evidence(x, category)
            assert torch.equal(location, old_location), category
            if category not in added: assert torch.equal(presence, old_presence), category
            proofs[category] = {'original_CPU_location_exact': True, 'prior_photo_presence_exact': category not in added}
    new(REPORTS/'equipment-component-reference-v3-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved': True, 'all_parent_neural_tensors_and_reference_buffers_exact': True, 'CPU_category_proofs': proofs,
        'all_new_reference_sources_normal_FIT_only': True, 'all_original_new_category_TEST_location_predictions_exact': True,
        'prior_cable_nut_metal_ksdd_passed_states_preserved': True, 'new_neural_optimizer_updates': 0, 'selected_checkpoint_sha256': sha(RUN/'selected/final.pt')})
    print(json.dumps({'verified': True, 'protected_files': len(ledger), 'newly_selected': added}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['freeze', 'fit', 'eval', 'verify'], required=True)
    args = parser.parse_args(); {'freeze': freeze, 'fit': fit, 'eval': evaluate, 'verify': verify}[args.mode]()
