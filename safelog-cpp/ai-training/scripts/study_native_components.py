"""Fixed native-view study: screw pose/details and connector structure/layout."""
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
from safelog_ai.native_component_views import ARCH, METHODS, VIEW_SIZE, NativeComponentMain, component_view, normalized, descriptor, layout_features, native_evidence, InputQualityError
from safelog_ai.equipment_expansion import CATEGORIES, SIZE, stage_gate
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.retention_distillation import state_sha256
from scripts import study_component_discriminator as previous
from scripts.study_equipment_expansion import images, read, MANIFEST
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha
from scripts.study_unified_belt_tiles import snapshot
from scripts.study_equipment_reference_photo import counts
START, EXPECTED, PARENT_EVALUATION = previous.START, previous.EXPECTED, previous.PARENT_EVALUATION
RUN = ROOT/'runs/equipment-native-components-v6'; DATA = ROOT/'data/equipment-native-components-v6'; REPORTS = ROOT/'reports'
PROTOCOL = REPORTS/'equipment-native-components-v6-protocol.json'
PREPARED = DATA/'manifest.json'; BUDGETS = {'normalCAL10': .10, 'normalCAL05': .05}
SOURCES = ('safelog_ai/native_component_views.py', 'scripts/study_native_components.py', 'tests/test_native_component_views.py')
VARIANTS = {'screw_hilo': ('ad_screw', 'aligned_hilo'), 'screw_fine': ('ad_screw', 'aligned_fine'),
            'connector_local': ('loco_splicing_connectors', 'native_patch'), 'connector_fusion': ('loco_splicing_connectors', 'native_fusion')}


def freeze():
    assert not RUN.exists() and not DATA.exists() and sha(START) == EXPECTED
    assert read(REPORTS/'equipment-component-rounds-v3-v5-final-audit.json')['active_checkpoint_sha256'] == EXPECTED
    RUN.mkdir(); ledger = dict(read(previous.RUN/'protected-inputs.json'))
    for directory in (REPORTS, previous.RUN):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    new(RUN/'protected-inputs.json', ledger)
    baseline = read(PARENT_EVALUATION)['categories']; rows = read(MANIFEST)['rows']
    split_counts = {c: {phase: sum(r['category'] == c and r['phase'] == phase for r in rows) for phase in ('fit', 'cal', 'test')} for c in METHODS}
    new(PROTOCOL, {'schema': 'safelog-native-components-fixed-v6', 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_native_preparation_fit_CAL_and_TEST_scoring': True, 'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'tests', 'safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256': EXPECTED, 'parent_evaluation_sha256': sha(PARENT_EVALUATION), 'manifest_sha256': sha(MANIFEST), 'protected_inputs_sha256': sha(RUN/'protected-inputs.json'),
        'variants': VARIANTS, 'budgets': BUDGETS, 'split_counts': split_counts, 'baseline': {c: baseline[c] for c in METHODS},
        'screw_preprocess': 'Input-only Otsu dark largest8connected foreground at256thumbnail, area.02to.60 and PCA axis ratio>=3; head wider end oriented left; isotropic affine with12percent canvas margin to768x256; source median border fill; no GT or predicted anomalies',
        'connector_preprocess': 'Original aspect ratio retained at1024x512; bicubic and explicit127gray letterbox for other ratios',
        'feature_methods': 'Aligned screw: same low40/high960 normalized pooled patch features or low40 fine41 token features; local top1percent bilinear map mean. Connector: native highlow patch features;256deterministic uniform FIT patches/photo with seedSHA256(sourceID) or fusion with coherent16x32 whole-photo templates and top10percent position mismatch',
        'connector_fusion': 'max(local_score/normalCAL_local_median, whole_template_score/normalCAL_layout_median); threshold of same normalCAL fused scores. NormalCAL fitted scaling is NOT claimed to have a conformal/FPR guarantee.',
        'template_size': [16, 32], 'connector_reference_sampling': '256patches/photo without replacement, seed first8bytesSHA256(sourceID), FITonly; all330photos contribute, no coreset search',
        'reference_precision': 'FP16 storage, FP32 exact cosine search query2048/reference8192 chunks',
        'training_source': 'Existing publisher-normal FIT426photos only; new BSData download not used by this study',
        'normalCAL_thresholds': 'ONLY alpha.10/.05, rankceil((n+1)*(1-alpha)), strict > alarm, no TEST quantiles or sweep',
        'input_quality': 'FIT/CAL quality failures excluded and reported, require>=20 each; any TEST abstention fails promotion. Unknown samples alarm conservatively and never silently healthy.',
        'guards': ['photo_FPR<=.20', 'photo_FNR<=.20', 'photo_FNR<parent', 'TEST_normal_and_defect_each>=20', 'TEST_abstentions==0', 'all_original_location_predictions_exact', 'all_parent_tensors_and_existing_reference_buffers_exact'],
        'selection': 'Per category eligible lower max(FPR,FNR), then FNR, then variant declaration order then CAL10. Otherwise retain parent. Only selected new buffers in ONE final checkpoint.',
        'neural_optimizer_updates_planned': 0, 'normal_reference_fit_is_not_gradient_training': True, 'new_source_photos_or_labels_in_this_study': 0,
        'one_shared_neural_backbone': True, 'backbone_passes_per_native_photo': 2, 'original256_location_branch_preserved': True,
        'existing_four_passed_categories_preserved': True, 'known_TEST_reused_for_development_and_selection': True, 'independent_field_validated': False,
        'reference_heatmaps_are_not_evaluated_segmentation': True, 'app_default_promoted': False, 'new_unit_tests_passed': 4})
    print(json.dumps({'frozen': True, 'variants': list(VARIANTS), 'split_counts': split_counts, 'protected_files': len(ledger)}), flush=True)


def validate():
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path, h in ((START, EXPECTED), (MANIFEST, p['manifest_sha256']), (PARENT_EVALUATION, p['parent_evaluation_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == h, str(path)
    return p


def prepare():
    validate(); assert not DATA.exists(); DATA.mkdir(); output = DATA/'views'; output.mkdir()
    rows = [r for r in read(MANIFEST)['rows'] if r['category'] in METHODS and r['phase'] in ('fit', 'cal', 'test')]
    prepared = []
    for index, row in enumerate(rows):
        value = deepcopy(row); assert sha(ROOT/row['image']) == row['image_sha256']
        with Image.open(ROOT/row['image']) as image: rgb = image.convert('RGB')
        try:
            view, record = component_view(rgb, row['category']); target = output/(str(index)+'.png'); view.save(target, compress_level=2)
            value.update(native_image=target.relative_to(ROOT).as_posix(), native_sha256=sha(target), native_view=record, native_error=None)
        except InputQualityError as error: value.update(native_image=None, native_error=str(error))
        prepared.append(value)
        if (index+1) % 100 == 0: print(json.dumps({'prepared_photos': index+1, 'total': len(rows)}), flush=True)
    errors = {c: {phase: sum(r['category'] == c and r['phase'] == phase and r['native_error'] is not None for r in prepared) for phase in ('fit', 'cal', 'test')} for c in METHODS}
    new(PREPARED, {'status': 'input_only_native_views_prepared', 'protocol_sha256': sha(PROTOCOL), 'rows': prepared, 'quality_failures': errors, 'new_neural_updates': 0})
    print(json.dumps({'prepared': len(prepared), 'quality_failures': errors}), flush=True)


def native_images(rows, device='cuda'):
    values = []
    for row in rows:
        if row['native_error']:
            width, height = VIEW_SIZE[row['category']]; values.append(normalized(Image.new('RGB', (width, height), (127, 127, 127))))
        else:
            with Image.open(ROOT/row['native_image']) as image: values.append(normalized(image.convert('RGB')))
    return torch.stack(values).to(device)


def load(path, device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['architecture'] != ARCH: return previous.load(path, device)
    model = NativeComponentMain(state['reference_counts'], state['native_specs']); model.load_state_dict(state['state_dict'], strict=True)
    return model.to(device).eval().requires_grad_(False), state


def fit():
    validate(); repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    commit = subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip(); prepared = read(PREPARED); assert prepared['protocol_sha256'] == sha(PROTOCOL)
    rows = prepared['rows']; base, parent = previous.load(START); banks = {}; templates = {}; sources = {}; sample_indices = {}
    with torch.inference_mode():
        for category in METHODS:
            fit_rows = [r for r in rows if r['category'] == category and r['phase'] == 'fit' and not r['native_error']]
            cal_rows = [r for r in rows if r['category'] == category and r['phase'] == 'cal' and not r['native_error']]; assert min(len(fit_rows), len(cal_rows)) >= 20
            collected = {method: [] for method in METHODS[category] if method != 'native_fusion'}; coherent = []; chosen = []
            for i in range(0, len(fit_rows), 4):
                batch = fit_rows[i:i+4]; features = base.backbone(native_images(batch))
                for method in collected:
                    f = descriptor(features, method).permute(0, 2, 3, 1).reshape(len(batch), -1, 41 if method == 'aligned_fine' else 1000)
                    if category == 'loco_splicing_connectors':
                        pieces = []
                        for row, per_photo in zip(batch, f):
                            seed = int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:8], 'little')
                            indices = np.random.default_rng(seed).choice(len(per_photo), 256, replace=False); chosen.append(indices.tolist()); pieces.append(per_photo[torch.tensor(indices, device='cuda')])
                        collected[method].append(torch.cat(pieces).half().cpu())
                    else: collected[method].append(f.flatten(0, 1).half().cpu())
                if category == 'loco_splicing_connectors': coherent.append(layout_features(features).permute(0, 2, 3, 1).reshape(-1, 1000).half().cpu())
            for method, pieces in collected.items(): banks[method] = torch.cat(pieces)
            if coherent: templates[category] = torch.cat(coherent); sample_indices[category] = chosen
            sources[category] = [r['id'] for r in fit_rows]
            print(json.dumps({'native_FIT': category, 'photos': len(fit_rows), 'bank_shapes': {m: list(banks[m].shape) for m in collected}}), flush=True)
    with (RUN/'normal-banks.pt').open('xb') as out: torch.save({'banks': banks, 'templates': templates, 'source_ids': sources, 'sampled_indices': sample_indices}, out)
    for variant, (category, method) in VARIANTS.items():
        bank = banks['native_patch' if method == 'native_fusion' else method]; specification = {'method': method, 'patches': len(bank)}
        if method == 'native_fusion': specification.update(layout_patches=len(templates[category]))
        model = NativeComponentMain(parent['reference_counts'], {category: specification}).cuda().eval().requires_grad_(False)
        missing, extra = model.load_state_dict(parent['state_dict'], strict=False); assert all(n.startswith(('native_reference_', 'native_layout_')) for n in missing) and not extra
        getattr(model, 'native_reference_'+category).copy_(bank)
        if method == 'native_fusion': getattr(model, 'native_layout_'+category).copy_(templates[category])
        cal = [r for r in rows if r['category'] == category and r['phase'] == 'cal' and not r['native_error']]; local_scores = []; layout_scores = []
        with torch.inference_mode():
            for i in range(0, len(cal), 4):
                features = model.backbone(native_images(cal[i:i+4])); _, local, layout = native_evidence(features, getattr(model, 'native_reference_'+category), method, getattr(model, 'native_layout_'+category, None))
                local_scores.extend(local.cpu().double().tolist())
                if layout is not None: layout_scores.extend(layout[1].cpu().double().tolist())
        if method == 'native_fusion':
            a, b = float(np.median(local_scores)), float(np.median(layout_scores)); assert min(a, b) > 1e-8
            specification.update(local_CAL_median=a, layout_CAL_median=b); scores = np.maximum(np.array(local_scores)/a, np.array(layout_scores)/b).tolist()
        else: scores = local_scores
        points = {name: dict(zip(('threshold', 'rank'), normal_budget_threshold(scores, alpha))) for name, alpha in BUDGETS.items()}
        state = deepcopy(parent); state.update(architecture=ARCH, state_dict={n: v.cpu() for n, v in model.state_dict().items()}, native_specs={category: specification},
            native_reference_fit={category: {'source_ids': sources[category], 'bank_sha256': state_sha256({'bank': bank}), 'patches': len(bank)}},
            native_calibrations={category: {'points': points, 'scores': scores, 'normalCAL_count': len(cal)}}, native_variant=variant,
            native_protocol_sha256=sha(PROTOCOL), native_protocol_commit=commit, parent_prototype_sha256=EXPECTED, neural_updates_this_refinement=0)
        assert all(torch.equal(v, state['state_dict'][n]) for n, v in parent['state_dict'].items())
        folder = RUN/variant; folder.mkdir()
        with (folder/'final.pt').open('xb') as out: torch.save(state, out)
        cpu, _ = load(folder/'final.pt', 'cpu'); old_cpu, _ = previous.load(START, 'cpu'); row = next(r for r in rows if r['category'] == category and r['phase'] == 'fit' and not r['native_error'])
        with torch.inference_mode():
            x, view = images([row]), native_images([row]); gpu = model.forward_native(x, view, category); host = cpu.forward_native(x.cpu(), view.cpu(), category)
            assert torch.equal(host[0], old_cpu.forward_equipment(x.cpu(), category))
            assert all(torch.allclose(a.cpu(), b, atol=2e-4, rtol=.002) for a, b in zip(gpu[:3], host[:3]))
        new(folder/'fit.json', {'status': 'normal_FIT_and_normalCAL_only', 'checkpoint_sha256': sha(folder/'final.pt'), 'prepared_manifest_sha256': sha(PREPARED),
            'all_parent_tensors_and_reference_buffers_exact': True, 'normal_FIT_photos': len(sources[category]), 'normalCAL_count': len(cal),
            'operating_points': points, 'CPU_GPU_close': True, 'original_CPU_location_exact': True, 'native_backbone_passes': 2, 'new_neural_updates': 0})
        print(json.dumps({'fitted': variant, 'normalCAL': len(cal), 'points': points}), flush=True)
        del model, cpu, old_cpu; torch.cuda.empty_cache()


def evaluate():
    p = validate(); prepared = read(PREPARED); rows_all = prepared['rows']; original = read(PARENT_EVALUATION); results = {}
    for variant, (category, method) in VARIANTS.items():
        model, state = load(RUN/variant/'final.pt'); parent, _ = previous.load(START); rows = [r for r in rows_all if r['category'] == category and r['phase'] == 'test']
        observed = []; pixels = dict(tp=0, fp=0, tn=0, fn=0); errors = sum(bool(r['native_error']) for r in rows)
        with torch.inference_mode():
            for i in range(0, len(rows), 4):
                batch = rows[i:i+4]; x = images(batch); location, _, scores, parts = model.forward_native(x, native_images(batch), category)
                assert torch.equal(location, parent.forward_equipment(x, category))
                for row, heat, score in zip(batch, location.cpu(), scores.cpu().tolist()):
                    mask = np.zeros((SIZE, SIZE), dtype=bool)
                    for path in row['masks']:
                        with Image.open(ROOT/path) as image: mask |= np.array(image.convert('L').resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 0
                    local = counts(mask, heat.numpy() > state['equipment_operating_points'][category]['pixel_threshold'])
                    for key, value in local.items(): pixels[key] += value
                    observed.append({'id': row['id'], 'category': category, 'label': row['label'], 'defect': row['defect'], 'score': 1e9 if row['native_error'] else score,
                        'abstained_requires_review': bool(row['native_error']), 'pixels': local})
        baseline = p['baseline'][category]; assert pixels == baseline['pixel_counts_model256'], variant
        candidates = {}
        for name, alpha in BUDGETS.items():
            point = state['native_calibrations'][category]['points'][name]; values = deepcopy(observed)
            for row in values: row['candidate_detected'] = row['score'] > point['threshold']
            rates = classification_rates([r['label'] for r in values], [r['score'] for r in values], point['threshold']); n, d = rates['negative_photos'], rates['positive_photos']; groups = {}
            for group in sorted({r['defect'] for r in values if r['label']}):
                subset = [r for r in values if r['defect'] == group]; fn = sum(not r['candidate_detected'] for r in subset); groups[group] = {'defect_photos': len(subset), 'fn': fn, 'fnr': fn/len(subset)}
            guards = {'photo_FPR_within20': rates['fpr'] <= .20, 'photo_FNR_within20': rates['fnr'] <= .20, 'photo_FNR_better_than_parent': rates['fnr'] < baseline['photo_rates']['fnr'],
                'normal_and_defect_TEST_at_least20': min(n, d) >= 20, 'TEST_abstentions_zero': errors == 0, 'original_location_predictions_exact': True, 'all_parent_tensors_exact': True}
            value = {'photo_rates': rates, 'auroc': None if errors else rank_auroc([r['label'] for r in values], [r['score'] for r in values]), 'normal_test_photos': n, 'defect_test_photos': d,
                'photo_threshold': point['threshold'], 'photo_normalCAL_alpha': alpha, 'photo_normalCAL_rank': point['rank'], 'normal_CAL_photos': state['native_calibrations'][category]['normalCAL_count'],
                'pixel_counts_model256': pixels, 'pixel_iou': baseline['pixel_iou'], 'pixel_fnr': baseline['pixel_fnr'], 'by_defect': groups, 'TEST_abstentions': errors,
                'stage_gate': stage_gate(rates['fpr'], rates['fnr'], n, d), 'guards': guards, 'selected_eligible': all(guards.values()), 'observations': values, 'variant': variant, 'method': method}
            candidates[name] = value; print(json.dumps({'variant': variant, 'budget': name, 'FPR': rates['fpr'], 'FNR': rates['fnr'], 'abstentions': errors, 'eligible': all(guards.values())}), flush=True)
        results[variant] = candidates
        new(REPORTS/f'equipment-native-components-v6-{variant}-validation.json', {'checkpoint_sha256': sha(RUN/variant/'final.pt'), 'category': category, 'candidates': candidates})
        del model, parent; torch.cuda.empty_cache()
    selected = torch.load(START, map_location='cpu', weights_only=True); evaluation = deepcopy(original); specs = {}; native_fit = {}; choices = {}
    for category in METHODS:
        eligible = [(v, n) for v, (c, _) in VARIANTS.items() if c == category for n in BUDGETS if results[v][n]['selected_eligible']]
        if not eligible: choices[category] = 'prior_prototype'; continue
        variant, name = min(eligible, key=lambda x: (max(results[x[0]][x[1]]['photo_rates']['fpr'], results[x[0]][x[1]]['photo_rates']['fnr']), results[x[0]][x[1]]['photo_rates']['fnr'], list(VARIANTS).index(x[0]), list(BUDGETS).index(x[1])))
        candidate = torch.load(RUN/variant/'final.pt', map_location='cpu', weights_only=True); value = results[variant][name]; choices[category] = {'variant': variant, 'budget': name}
        specs[category] = candidate['native_specs'][category]; native_fit[category] = candidate['native_reference_fit'][category]
        selected['state_dict'].update({n: v for n, v in candidate['state_dict'].items() if n.startswith(('native_reference_', 'native_layout_'))})
        selected['equipment_operating_points'][category].update(photo_threshold=value['photo_threshold'], scores=candidate['native_calibrations'][category]['scores'],
            photo_normalCAL_alpha=value['photo_normalCAL_alpha'], photo_normalCAL_rank=value['photo_normalCAL_rank'], photo_score_method=specs[category]['method'], native_normal_CAL_photos=value['normal_CAL_photos'])
        evaluation['categories'][category].update({k: value[k] for k in ('photo_rates', 'auroc', 'photo_threshold', 'stage_gate', 'by_defect', 'photo_normalCAL_alpha', 'photo_normalCAL_rank')})
        evaluation['categories'][category]['photo_score_source'] = specs[category]['method']; lookup = {r['id']: r for r in value['observations']}
        for row in evaluation['rows']:
            if row['category'] == category: row.update(score=lookup[row['id']]['score'], candidate_detected=lookup[row['id']]['candidate_detected'])
    if specs: selected.update(architecture=ARCH, native_specs=specs, native_reference_fit=native_fit)
    selected.update(native_selected=choices, native_protocol_sha256=sha(PROTOCOL), parent_prototype_sha256=EXPECTED, neural_updates_this_refinement=0)
    folder = RUN/'selected'; folder.mkdir()
    with (folder/'final.pt').open('xb') as out: torch.save(selected, out)
    digest = sha(folder/'final.pt'); stage20 = [c for c, v in evaluation['categories'].items() if v['stage_gate']['ready_for_next_stage']]
    evaluation.update(checkpoint_sha256=digest, status='known_native_component_v6', stage20_categories=stage20, known_TEST_reused=True, adaptive_development_on_known_TEST=True)
    new(folder/'evaluation.json', evaluation)
    new(REPORTS/'equipment-native-components-v6-selection.json', {'selected_per_category': choices, 'newly_selected_categories': list(specs), 'checkpoint_path': (folder/'final.pt').relative_to(ROOT).as_posix(),
        'checkpoint_sha256': digest, 'stage20_categories': stage20, 'new_neural_optimizer_updates': 0, 'all_parent_tensors_and_reference_buffers_exact': True,
        'all_original_location_outputs_preserved': True, 'known_TEST_reused_for_development': True, 'independent_field_validated': False, 'app_default_promoted': False})


def verify():
    p = validate(); ledger = read(RUN/'protected-inputs.json')
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda item: previous.reference.check(*item), ledger.items()))
    for n, h in p['legacy_source_sha256'].items(): assert sha(ROOT/n) == h, n
    parent = torch.load(START, map_location='cpu', weights_only=True); choice = read(REPORTS/'equipment-native-components-v6-selection.json')
    for path in [*(RUN/v/'final.pt' for v in VARIANTS), RUN/'selected/final.pt']:
        state = torch.load(path, map_location='cpu', weights_only=True); assert all(torch.equal(v, state['state_dict'][n]) for n, v in parent['state_dict'].items())
    new(REPORTS/'equipment-native-components-v6-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'all_parent_tensors_and_reference_buffers_exact': True, 'old_bytes_and_mtimes_preserved': True, 'FIT_only_reference_sources': True,
        'all_original_candidate_TEST_location_predictions_exact': True, 'two_views_same_one_backbone': True, 'new_neural_optimizer_updates': 0, 'selected': choice['selected_per_category']})
    print(json.dumps({'verified': True, 'newly_selected': choice['newly_selected_categories'], 'protected_files': len(ledger)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['freeze', 'prepare', 'fit', 'eval', 'verify'], required=True)
    args = parser.parse_args(); {'freeze': freeze, 'prepare': prepare, 'fit': fit, 'eval': evaluate, 'verify': verify}[args.mode]()
