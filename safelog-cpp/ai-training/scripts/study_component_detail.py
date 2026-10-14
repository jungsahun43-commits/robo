"""Fixed follow-up: finer screw descriptors and coherent connector templates."""
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
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.component_detail import ARCH, METHODS, DetailLayoutMain, fine_features
from safelog_ai.equipment_expansion import CATEGORIES, SIZE, stage_gate
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from scripts import study_component_reference as prior_study
from scripts.study_equipment_expansion import images, read, MANIFEST
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_equipment_reference_photo import counts
START, EXPECTED, PARENT_EVALUATION = prior_study.START, prior_study.EXPECTED, prior_study.PARENT_EVALUATION
REPORTS = ROOT/'reports'; RUN = ROOT/'runs/equipment-component-detail-v4'
PROTOCOL = REPORTS/'equipment-component-detail-v4-protocol.json'
DONOR = prior_study.RUN/'candidate/final.pt'; DONOR_SHA = '7f757a643dd61c776aceda3497cb90cdbfc0217e81ea9096655930073e44a5ff'
BUDGETS = {'normalCAL10': .10, 'normalCAL05': .05}
SOURCES = ('safelog_ai/component_detail.py', 'scripts/study_component_detail.py', 'tests/test_component_detail.py')


def load(path, device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['architecture'] != ARCH: return prior_study.load(path, device)
    model = DetailLayoutMain(state['reference_counts'], state['reference_methods']); model.load_state_dict(state['state_dict'], strict=True)
    return model.to(device).eval().requires_grad_(False), state


def freeze():
    assert not RUN.exists() and sha(START) == EXPECTED and sha(DONOR) == DONOR_SHA
    assert read(REPORTS/'equipment-component-reference-v3-verification.json')['status'] == 'passed'
    assert not read(REPORTS/'equipment-component-reference-v3-selection.json')['newly_selected_categories']
    RUN.mkdir(); ledger = dict(read(prior_study.RUN/'protected-inputs.json'))
    for directory in (REPORTS, prior_study.RUN):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    new(RUN/'protected-inputs.json', ledger)
    original = read(prior_study.PROTOCOL)
    new(PROTOCOL, {'schema': 'safelog-component-detail-fixed-v4', 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_followup_fit_CAL_and_TEST_scoring': True, 'adaptive_followup_on_known_TEST': True,
        'reason': 'v3 local high/low patch memory failed screw and LOCO20percent joint photo goals; MPDD lacked20defect evaluation photos',
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'tests', 'safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256': EXPECTED, 'donor_sha256': DONOR_SHA, 'manifest_sha256': sha(MANIFEST), 'parent_evaluation_sha256': sha(PARENT_EVALUATION),
        'protected_inputs_sha256': sha(RUN/'protected-inputs.json'), 'methods': METHODS, 'budgets': BUDGETS,
        'screw_method': 'ONE frozen backbone pass; accepted low40 at32x32, avgpool3 stride1 count_include_padFalse, unit normalization, appendconstant.01 and normalize41channels; ALL96normalFIT1024patches/photo stored FP16; exact nearest cosine FP32; bilinear256 then top1percent mean photo score',
        'connector_method': 'Reuse v3 ordered330normalFIT low40/high960 pooled16x16 patch templates, same buffer bytes; for each reference PHOTO compare corresponding positions, mean worst26/256patch cosine distances; choose SINGLE normal reference photo with smallest score; visualization uses that reference, not independent nearest patches',
        'reference_counts_added': {'ad_screw': 98304, 'loco_splicing_connectors': 84480},
        'screw_reference_dimension': 41, 'connector_reference_dimension': 1000,
        'normalCAL_only_thresholds': 'Same alpha.10/.05; rankceil((n+1)*(1-alpha)), strict >; no TEST quantile/sweep',
        'baseline': {c: original['baseline'][c] for c in METHODS}, 'split_counts': {c: original['split_counts'][c] for c in METHODS},
        'guards': original['prespecified_guards'], 'selection': original['selection'],
        'LOCO_logical_and_structural_subgroup_FNR_reported': True, 'all_original_location_outputs_and_parent_buffers_preserved': True,
        'new_normal_FIT_photos_encoded': 96, 'existing_normal_FIT_template_photos_reused': 330,
        'new_source_photos_or_labels': 0, 'new_neural_optimizer_updates_planned': 0,
        'known_TEST_reused_for_development_and_candidate_selection': True, 'independent_field_validated': False,
        'reference_heatmaps_are_not_evaluated_segmentation': True, 'app_default_promoted': False, 'new_unit_tests_passed': 3})
    print(json.dumps({'frozen': True, 'methods': METHODS, 'normal_FIT_encoded': 96, 'normal_FIT_templates_reused': 330, 'protected_files': len(ledger)}), flush=True)


def validate():
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path, h in ((START, EXPECTED), (DONOR, DONOR_SHA), (MANIFEST, p['manifest_sha256']), (PARENT_EVALUATION, p['parent_evaluation_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == h, str(path)
    return p


def fit():
    p = validate(); repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    commit = subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip()
    source = torch.load(START, map_location='cpu', weights_only=True); donor = torch.load(DONOR, map_location='cpu', weights_only=True)
    rows = read(MANIFEST)['rows']; all_counts = {**source['reference_counts'], **p['reference_counts_added']}
    methods = {c: METHODS.get(c, 'patch') for c in all_counts}; model = DetailLayoutMain(all_counts, methods).cuda().eval().requires_grad_(False)
    missing, extra = model.load_state_dict(source['state_dict'], strict=False)
    assert set(missing) == {'normal_reference_'+c for c in METHODS} and not extra
    bank_proofs = deepcopy(source['normal_reference_fit']); calibrations = {}
    with torch.inference_mode():
        screw_rows = [r for r in rows if r['category'] == 'ad_screw' and r['phase'] == 'fit']; features = []
        for i in range(0, len(screw_rows), 8):
            fine = fine_features(model.backbone(images(screw_rows[i:i+8]))); features.append(fine.permute(0, 2, 3, 1).reshape(-1, 41).half().cpu())
        bank = torch.cat(features); assert bank.shape == (98304, 41); model.normal_reference_ad_screw.copy_(bank)
        bank_proofs['ad_screw'] = {'source_ids': [r['id'] for r in screw_rows], 'feature_sha256': state_sha256({'reference': bank}), 'patches': len(bank), 'dtype': 'float16', 'channels': 41}
        layout = donor['state_dict']['normal_reference_loco_splicing_connectors']; model.normal_reference_loco_splicing_connectors.copy_(layout)
        bank_proofs['loco_splicing_connectors'] = deepcopy(donor['normal_reference_fit']['loco_splicing_connectors'])
        assert len(bank_proofs['loco_splicing_connectors']['source_ids']) == 330
        for category in METHODS:
            cal = [r for r in rows if r['category'] == category and r['phase'] == 'cal']; scores = []
            for i in range(0, len(cal), 8):
                _, _, score = model.forward_component_outputs(images(cal[i:i+8]), category); scores.extend(score.cpu().double().tolist())
            calibrations[category] = {n: dict(zip(('threshold', 'rank'), normal_budget_threshold(scores, alpha))) for n, alpha in BUDGETS.items()}
            calibrations[category]['scores'] = scores
            print(json.dumps({'normal_CAL_fixed': category, 'method': METHODS[category], 'count': len(cal)}), flush=True)
    state = deepcopy(source); state.update(architecture=ARCH, reference_counts=all_counts, reference_methods=methods, normal_reference_fit=bank_proofs,
        state_dict={n: v.cpu() for n, v in model.state_dict().items()}, detail_calibrations=calibrations,
        detail_protocol_sha256=sha(PROTOCOL), detail_protocol_commit=commit, parent_prototype_sha256=EXPECTED, neural_updates_this_refinement=0)
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in source['state_dict'].items())
    folder = RUN/'candidate'; folder.mkdir()
    with (folder/'final.pt').open('xb') as out: torch.save(state, out)
    cpu, _ = load(folder/'final.pt', 'cpu'); parent, _ = load(START, 'cpu'); proofs = {}
    with torch.inference_mode():
        for category in CATEGORIES:
            row = next(r for r in rows if r['category'] == category and r['phase'] == 'fit'); x = images([row])
            loc, ref, score = model.forward_component_outputs(x, category); cpu_loc, cpu_ref, cpu_score = cpu.forward_component_outputs(x.cpu(), category)
            old_loc, old_ref = parent.forward_equipment_evidence(x.cpu(), category); assert torch.equal(cpu_loc, old_loc), category
            assert torch.allclose(loc.cpu(), cpu_loc, atol=2e-4, rtol=.002) and torch.allclose(ref.cpu(), cpu_ref, atol=2e-4, rtol=.002) and torch.allclose(score.cpu(), cpu_score, atol=2e-4, rtol=.002), category
            if category not in METHODS: assert torch.equal(cpu_ref, old_ref), category
            proofs[category] = {'original_location_CPU_exact': True, 'CPU_GPU_all_outputs_close': True, 'parent_photo_evidence_exact': category not in METHODS}
    new(folder/'fit.json', {'status': 'normal_FIT_and_CAL_only', 'checkpoint_sha256': sha(folder/'final.pt'), 'new_neural_optimizer_updates': 0,
        'all_parent_tensors_and_buffers_exact': True, 'normal_FIT_encoded': 96, 'normal_FIT_templates_reused': 330, 'TEST_used_for_fit_or_CAL': False,
        'CPU_GPU_and_parent_proofs': proofs, 'normalCAL_points': {c: {n: v for n, v in cal.items() if n != 'scores'} for c, cal in calibrations.items()}})


def evaluate():
    p = validate(); model, state = load(RUN/'candidate/final.pt'); parent, _ = load(START); all_rows = read(MANIFEST)['rows']
    original = read(PARENT_EVALUATION); old_rows = {r['id']: r for r in original['rows']}; results = {}; choices = {}
    for category in METHODS:
        rows = [r for r in all_rows if r['category'] == category and r['phase'] == 'test']; baseline = p['baseline'][category]
        observations = []; pixels = dict(tp=0, fp=0, tn=0, fn=0)
        with torch.inference_mode():
            for i in range(0, len(rows), 8):
                batch = rows[i:i+8]; x = images(batch); location, reference, scores = model.forward_component_outputs(x, category)
                assert torch.equal(location, parent.forward_equipment(x, category)), (category, i)
                for row, heat, score in zip(batch, location.cpu(), scores.cpu().tolist()):
                    mask = np.zeros((SIZE, SIZE), dtype=bool)
                    for path in row['masks']:
                        with Image.open(ROOT/path) as im: mask |= np.array(im.convert('L').resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 0
                    local = counts(mask, heat.numpy() > state['equipment_operating_points'][category]['pixel_threshold'])
                    for key, value in local.items(): pixels[key] += value
                    observations.append({'id': row['id'], 'category': category, 'label': row['label'], 'defect': row['defect'], 'score': score, 'pixels': local})
        assert pixels == baseline['pixel_counts_model256']; candidates = {}
        for name, alpha in BUDGETS.items():
            point = state['detail_calibrations'][category][name]; observed = deepcopy(observations)
            for row in observed: row['candidate_detected'] = row['score'] > point['threshold']
            rates = classification_rates([r['label'] for r in observed], [r['score'] for r in observed], point['threshold']); n, d = rates['negative_photos'], rates['positive_photos']
            groups = {}
            for group in sorted({r['defect'] for r in observed if r['label']}):
                subset = [r for r in observed if r['defect'] == group]; fn = sum(not r['candidate_detected'] for r in subset); groups[group] = {'defect_photos': len(subset), 'fn': fn, 'fnr': fn/len(subset)}
            guards = {'photo_FPR_within20': rates['fpr'] <= .20, 'photo_FNR_within20': rates['fnr'] <= .20,
                'photo_FNR_better_than_parent': rates['fnr'] < baseline['photo_rates']['fnr'], 'normal_and_defect_TEST_at_least20': min(n, d) >= 20,
                'original_location_predictions_exact': True, 'all_parent_tensors_and_buffers_exact': True}
            candidates[name] = {'photo_rates': rates, 'auroc': rank_auroc([r['label'] for r in observed], [r['score'] for r in observed]), 'normal_test_photos': n, 'defect_test_photos': d,
                'normal_CAL_photos': p['split_counts'][category]['cal'], 'photo_threshold': point['threshold'], 'photo_normalCAL_rank': point['rank'], 'photo_normalCAL_alpha': alpha,
                'pixel_counts_model256': pixels, 'pixel_iou': baseline['pixel_iou'], 'pixel_fnr': baseline['pixel_fnr'], 'by_defect': groups,
                'stage_gate': stage_gate(rates['fpr'], rates['fnr'], n, d), 'guards': guards, 'selected_eligible': all(guards.values()), 'observations': observed, 'method': METHODS[category]}
            print(json.dumps({'method': METHODS[category], 'category': category, 'candidate': name, 'FPR': rates['fpr'], 'FNR': rates['fnr'], 'eligible': all(guards.values()), 'by_defect': groups}), flush=True)
        results[category] = candidates; eligible = [n for n, v in candidates.items() if v['selected_eligible']]
        choices[category] = min(eligible, key=lambda n: (max(candidates[n]['photo_rates']['fpr'], candidates[n]['photo_rates']['fnr']), candidates[n]['photo_rates']['fnr'], list(BUDGETS).index(n))) if eligible else 'prior_prototype'
    new(REPORTS/'equipment-component-detail-v4-validation.json', {'candidate_checkpoint_sha256': sha(RUN/'candidate/final.pt'), 'categories': results})
    selected = torch.load(START, map_location='cpu', weights_only=True); evaluation = deepcopy(original); added = []
    methods = {c: 'patch' for c in selected['reference_counts']}
    for category, name in choices.items():
        if name == 'prior_prototype': continue
        added.append(category); value = results[category][name]; key = 'normal_reference_'+category; methods[category] = METHODS[category]
        selected['state_dict'][key] = state['state_dict'][key]; selected['reference_counts'][category] = state['reference_counts'][category]; selected['normal_reference_fit'][category] = state['normal_reference_fit'][category]
        selected['equipment_operating_points'][category].update(photo_threshold=value['photo_threshold'], scores=state['detail_calibrations'][category]['scores'],
            photo_normalCAL_rank=value['photo_normalCAL_rank'], photo_normalCAL_alpha=value['photo_normalCAL_alpha'], photo_score_method=METHODS[category])
        evaluation['categories'][category].update({k: value[k] for k in ('photo_rates', 'auroc', 'photo_threshold', 'stage_gate', 'pixel_counts_model256', 'pixel_iou', 'pixel_fnr', 'by_defect', 'photo_normalCAL_rank', 'photo_normalCAL_alpha')})
        evaluation['categories'][category]['photo_score_source'] = METHODS[category]; observed = {r['id']: r for r in value['observations']}
        for row in evaluation['rows']:
            if row['category'] == category: row.update(score=observed[row['id']]['score'], candidate_detected=observed[row['id']]['candidate_detected'])
    if added: selected.update(architecture=ARCH, reference_methods=methods)
    selected.update(component_detail_selected=choices, detail_protocol_sha256=sha(PROTOCOL), detail_protocol_commit=state['detail_protocol_commit'], parent_prototype_sha256=EXPECTED, neural_updates_this_refinement=0)
    folder = RUN/'selected'; folder.mkdir()
    with (folder/'final.pt').open('xb') as out: torch.save(selected, out)
    digest = sha(folder/'final.pt'); stage20 = [c for c, v in evaluation['categories'].items() if v['stage_gate']['ready_for_next_stage']]
    evaluation.update(checkpoint_sha256=digest, status='known_component_detail_v4', parent_prototype_sha256=EXPECTED, known_TEST_reused=True, adaptive_development_on_known_TEST=True, stage20_categories=stage20)
    new(folder/'evaluation.json', evaluation)
    new(REPORTS/'equipment-component-detail-v4-selection.json', {'selected_per_category': choices, 'newly_selected_categories': added, 'checkpoint_path': (folder/'final.pt').relative_to(ROOT).as_posix(),
        'checkpoint_sha256': digest, 'reference_counts': selected['reference_counts'], 'reference_methods': methods, 'stage20_categories': stage20,
        'new_neural_optimizer_updates': 0, 'all_parent_tensors_and_reference_buffers_exact': True, 'all_original_location_outputs_preserved': True,
        'known_TEST_reused_for_development': True, 'independent_field_validated': False, 'app_default_promoted': False})


def verify():
    p = validate(); ledger = read(RUN/'protected-inputs.json')
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda item: prior_study.check(*item), ledger.items()))
    for name, h in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == h, name
    model, state = load(RUN/'selected/final.pt', 'cpu'); parent, source = load(START, 'cpu'); rows = read(MANIFEST)['rows']; choice = read(REPORTS/'equipment-component-detail-v4-selection.json')
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in source['state_dict'].items()); proofs = {}
    with torch.inference_mode():
        for category in CATEGORIES:
            row = next(r for r in rows if r['category'] == category and r['phase'] == 'fit'); x = images([row], 'cpu')
            location, reference = model.forward_equipment_evidence(x, category); old_loc, old_ref = parent.forward_equipment_evidence(x, category)
            assert torch.equal(location, old_loc), category
            if category not in choice['newly_selected_categories']:
                assert torch.equal(reference, old_ref) and state['equipment_operating_points'][category] == source['equipment_operating_points'][category], category
            proofs[category] = {'original_location_CPU_exact': True, 'parent_photo_output_and_operating_point_exact': category not in choice['newly_selected_categories']}
    new(REPORTS/'equipment-component-detail-v4-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved': True, 'all_parent_tensors_and_buffers_exact': True, 'CPU_category_proofs': proofs,
        'all_original_candidate_TEST_location_predictions_exact': True, 'new_reference_sources_only_normal_FIT': True, 'new_neural_optimizer_updates': 0,
        'selected_checkpoint_sha256': sha(RUN/'selected/final.pt')})
    print(json.dumps({'verified': True, 'newly_selected': choice['newly_selected_categories'], 'protected_files': len(ledger)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['freeze', 'fit', 'eval', 'verify'], required=True)
    args = parser.parse_args(); {'freeze': freeze, 'fit': fit, 'eval': evaluate, 'verify': verify}[args.mode]()
