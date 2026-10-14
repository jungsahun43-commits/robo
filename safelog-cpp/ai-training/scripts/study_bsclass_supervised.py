"""Fixed real-labelled wear training; fresh TEST is opened after VAL/CAL decisions."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
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
from safelog_ai.industrial_supervised import ARCH, WEAR_CATEGORY, VARIANTS, wear_descriptor, wear_head, IndustrialSupervisedMain
from safelog_ai.native_component_views import NativeComponentMain
from safelog_ai.equipment_expansion import CATEGORIES, stage_gate
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.retention_distillation import state_sha256
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_unified_belt_tiles import snapshot
from scripts.unified_main_data import normalize_batch

RUN = ROOT/'runs/industrial-bsclass-supervised-v8'; REPORTS = ROOT/'reports'
DATA = ROOT/'data/industrial-bsclass-v1'; MANIFEST = DATA/'manifest.json'
START = ROOT/'runs/equipment-native-screw-budget-v7/selected/final.pt'
EXPECTED = '14a26d0b4c69f4d86c12cfe50daeadad6d79935ae552fc37610a9c465815097a'
PARENT_EVALUATION = START.parent.parent/'evaluation.json'
PROTOCOL = REPORTS/'industrial-bsclass-supervised-v8-protocol.json'
SOURCES = ('safelog_ai/industrial_supervised.py', 'scripts/study_bsclass_supervised.py', 'tests/test_industrial_supervised.py')
EPOCHS, BATCH, SEED = 30, 256, 1289


def role(row):
    if row['phase'] != 'cal': return row['phase']
    coin = int(hashlib.sha256(('bsclass-v8-calroles:'+row['group_id']).encode()).hexdigest()[:8], 16)
    return 'validation' if coin < 2**31 else 'calibration'


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(START) == EXPECTED
    assert read(REPORTS/'equipment-native-screw-v6-v7-final-audit.json')['status'] == 'passed_native_screw_improvement_packaged'
    manifest = read(MANIFEST); rows = manifest['rows']; assert len(rows) == 21835
    assert not manifest['audit']['new_training_performed'] and not manifest['audit']['new_reserved_TEST_model_predictions_run']
    for name, digest in manifest['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    counts = {phase: dict(Counter(str(r['label']) for r in rows if role(r) == phase)) for phase in ('fit', 'validation', 'calibration', 'reserved_test')}
    assert all(min(v.values()) >= 20 and len(v) == 2 for v in counts.values())
    RUN.mkdir(); ledger = dict(read(START.parent.parent/'protected-inputs.json'))
    for directory in (REPORTS, START.parent.parent):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    ledger[MANIFEST.relative_to(ROOT).as_posix()] = snapshot(MANIFEST)
    with ThreadPoolExecutor(max_workers=4) as pool:
        def protect(row):
            value = snapshot(ROOT/row['image']); assert value['sha256'] == row['image_sha256']; return row['image'], value
        ledger.update(pool.map(protect, rows))
    new(RUN/'protected-inputs.json', ledger)
    new(PROTOCOL, {
        'schema': 'safelog-industrial-bsclass-supervised-fixed-v8', 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_feature_cache_training_validation_calibration_and_reserved_TEST_scores': True,
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'tests', 'safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256': EXPECTED, 'manifest_sha256': sha(MANIFEST), 'parent_evaluation_sha256': sha(PARENT_EVALUATION), 'protected_inputs_sha256': sha(RUN/'protected-inputs.json'),
        'primary_source': 'https://publikationen.bibliothek.kit.edu/1000133819', 'license': manifest['audit']['license'],
        'domain': 'Cropped ball-screw visible pitting; NOT fastener screw, whole-factory health or internal mechanical failure',
        'counts': counts, 'grouping': manifest['audit']['split_rule'], 'cal_role_split': 'SHA256(bsclass-v8-calroles:+groupID) first32bits <2^31 validation else calibration',
        'physical_spindle_acquisition_groups_verified': False, 'independent_field_validation': False,
        'one_frozen_accepted_backbone': True, 'old_heads_thresholds_and_all_reference_buffers_frozen': True,
        'input': 'RGB bicubic256 square, ImageNet normalization; supervised dataset is small cropped images, no localization mask claims',
        'descriptor': '3000 channels: channelwise mean, population std, max of low40 then high960; raw accepted frozen backbone features',
        'standardization': 'FIT-only per-channel mean/std clamp1e-4; standardized descriptor clamp[-8,8]',
        'registered_heads': {'linear': 'Linear3000to1', 'mlp': 'Linear3000to256 ReLU Dropout.2 Linear256to64 ReLU Dropout.2 Linear64to1'},
        'head_order': list(VARIANTS), 'epochs_each_head': EPOCHS, 'batch': BATCH, 'seed': SEED,
        'optimizer': 'AdamW lr.001 wd.0001; cosine30 eta_min.00001; BCEWithLogits; gradient clip5; FP32 TF32off',
        'training': 'Every FIT photo each epoch; publisher N=0/P=1 real labels; inverse within-FIT group multiplicity weights normalized mean1; no synthetic labels',
        'planned_optimizer_updates_each_head': EPOCHS*math.ceil(sum(counts['fit'].values())/BATCH),
        'checkpoint_selection': 'VAL-only best epoch/head by lowest max(photoFPR,photoFNR) at sigmoid.5, then VAL_BCE, then epoch/head order; no TEST epoch selection',
        'threshold': 'After winner fixed, labelled calibration-only threshold minimizes max(photoFPR,photoFNR), then their sum, then distance to.5. Strict score>threshold, no TEST tuning.',
        'baseline': 'One non-gradient normal-FIT centroid in same FIT-standardized descriptor, mean squared distance; normal calibration alpha.15 order statistic fixed before TEST',
        'fresh_reserved_TEST': 'Do not cache or forward reserved_TEST until head+epoch+threshold have been saved. Then evaluate selected classifier and fixed baseline once.',
        'selection_guards': ['FPR<=.20', 'FNR<=.20', 'normal_and_defect_each>=20', 'max(FPR,FNR)<normal_centroid_baseline', 'all_parent_tensors_and_buffers_exact'],
        'existing11_category_outputs_unchanged': True, 'existing11_stage20_count': 5, 'new_category_id': WEAR_CATEGORY,
        'no_pitting_localization_masks': True, 'score_is_calibrated_real_failure_probability': False, 'app_default_promoted': False,
        'new_unit_tests_passed': 5
    })
    print(json.dumps({'protocol_frozen': True, 'counts': counts, 'updates_per_head': EPOCHS*math.ceil(sum(counts['fit'].values())/BATCH), 'protected_files': len(ledger)}), flush=True)


def validate(committed=False):
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path, digest in ((START, EXPECTED), (MANIFEST, p['manifest_sha256']), (PARENT_EVALUATION, p['parent_evaluation_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == digest, str(path)
    if committed:
        repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
        for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
            assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    return p


def feature_cache(rows, label):
    path = RUN/(label+'-features.pt'); ids = [r['id'] for r in rows]
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['ids'] == ids and saved['protocol_sha256'] == sha(PROTOCOL)
        return saved['features'].float()
    state = torch.load(START, map_location='cpu', weights_only=True)
    parent = NativeComponentMain(state['reference_counts'], state['native_specs']).eval().requires_grad_(False)
    parent.load_state_dict(state['state_dict'], strict=True); backbone = parent.backbone.cuda(); values = []
    def image(row):
        raw = (ROOT/row['image']).read_bytes(); assert hashlib.sha256(raw).hexdigest() == row['image_sha256'], row['id']
        with Image.open(io.BytesIO(raw)) as im:
            return torch.from_numpy(np.array(im.convert('RGB').resize((256, 256), Image.Resampling.BICUBIC))).permute(2, 0, 1)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for i in range(0, len(rows), 32):
            batch = normalize_batch(torch.stack(list(pool.map(image, rows[i:i+32]))), 'cuda')
            descriptor = wear_descriptor(backbone(batch)); assert torch.isfinite(descriptor).all()
            values.append(descriptor.cpu().half())
            if i % 1024 == 0: print(json.dumps({'feature_cache': label, 'completed': min(i+32, len(rows)), 'total': len(rows), 'seconds': round(time.monotonic()-start, 1)}), flush=True)
    value = torch.cat(values); del parent, backbone; torch.cuda.empty_cache()
    with path.open('xb') as out: torch.save({'ids': ids, 'features': value, 'protocol_sha256': sha(PROTOCOL)}, out)
    return value.float()


def calibrated_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int64); scores = np.asarray(scores, dtype=np.float64)
    assert len(scores) == len(labels) and np.isfinite(scores).all() and set(labels.tolist()) == {0, 1}
    values = np.unique(scores)
    candidates = np.r_[np.nextafter(values[0], -np.inf), values, .5]
    order = np.argsort(scores, kind='stable'); sorted_scores = scores[order]; sorted_labels = labels[order]
    prefix = np.r_[0, np.cumsum(sorted_labels)]; k = np.searchsorted(sorted_scores, candidates, side='right')
    n, d = int((labels == 0).sum()), int(labels.sum()); fn = prefix[k]; fp = n-(k-fn)
    fpr, fnr = fp/n, fn/d
    j = min(range(len(candidates)), key=lambda i: (max(fpr[i], fnr[i]), fpr[i]+fnr[i], abs(candidates[i]-.5), float(candidates[i])))
    return float(candidates[j])


def train():
    p = validate(committed=True); assert not (RUN/'chosen-head.json').exists()
    rows = read(MANIFEST)['rows']; dev = [r for r in rows if role(r) in ('fit', 'validation', 'calibration')]
    features = feature_cache(dev, 'development'); indices = {phase: torch.tensor([i for i, r in enumerate(dev) if role(r) == phase]) for phase in ('fit', 'validation', 'calibration')}
    fit = features[indices['fit']]; mean = fit.mean(0); std = fit.std(0, correction=0).clamp_min(1e-4)
    standard = ((features-mean)/std).clamp(-8, 8).cuda(); labels = torch.tensor([r['label'] for r in dev], dtype=torch.float32, device='cuda')
    fit_idx = indices['fit'].cuda(); val_idx = indices['validation'].cuda(); cal_idx = indices['calibration'].cuda()
    group_count = Counter(dev[int(i)]['group_id'] for i in indices['fit']); weights = torch.tensor([1/group_count[dev[int(i)]['group_id']] for i in indices['fit']], device='cuda'); weights /= weights.mean()
    summaries = []
    for variant_index, variant in enumerate(VARIANTS):
        folder = RUN/variant; folder.mkdir(exist_ok=True); progress = folder/'progress.pt'
        torch.manual_seed(SEED+variant_index); torch.cuda.manual_seed_all(SEED+variant_index)
        head = wear_head(variant).cuda(); initial_sha = state_sha256(head)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=.00001)
        cursor = 0; updates = 0; best = None
        if progress.exists():
            saved = torch.load(progress, map_location='cpu', weights_only=True); assert saved['protocol_sha256'] == sha(PROTOCOL)
            head.load_state_dict(saved['head']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            cursor, updates, best = saved['epochs_completed'], saved['updates'], saved['best']; torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
            assert len((folder/'epochs.jsonl').read_text().splitlines()) == cursor
        with (folder/'epochs.jsonl').open('a' if cursor else 'x', encoding='utf-8', newline='\n') as journal:
            for epoch in range(cursor, EPOCHS):
                head.train(); order = torch.randperm(len(fit_idx), generator=torch.Generator().manual_seed(SEED+variant_index*1000+epoch)).cuda(); loss_sum = 0.
                for start in range(0, len(order), BATCH):
                    draw = order[start:start+BATCH]; picked = fit_idx[draw]
                    logits = head(standard[picked]).flatten(); loss = (F.binary_cross_entropy_with_logits(logits, labels[picked], reduction='none')*weights[draw]).mean()
                    assert torch.isfinite(loss); optimizer.zero_grad(set_to_none=True); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.); assert torch.isfinite(norm); optimizer.step(); updates += 1; loss_sum += float(loss.detach())
                scheduler.step(); head.eval()
                with torch.inference_mode():
                    logits = head(standard[val_idx]).flatten(); loss = float(F.binary_cross_entropy_with_logits(logits, labels[val_idx])); scores = logits.sigmoid().cpu().tolist()
                rates = classification_rates(labels[val_idx].cpu().tolist(), scores, .5)
                key = [max(rates['fpr'], rates['fnr']), loss, epoch+1, variant_index]
                if best is None or key < best['key']:
                    best = {'key': key, 'epoch': epoch+1, 'variant': variant, 'validation_rates_at_half': rates, 'validation_BCE': loss,
                            'head': {n: v.detach().cpu().clone() for n, v in head.state_dict().items()}}
                record = {'epoch': epoch+1, 'optimizer_updates': updates, 'training_weighted_BCE': loss_sum/math.ceil(len(order)/BATCH), 'validation_BCE': loss, 'validation_rates_at_half': rates}
                journal.write(json.dumps(record)+'\n'); journal.flush()
                temporary = folder/'progress-next.pt'
                torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epochs_completed': epoch+1, 'updates': updates,
                    'best': best, 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'protocol_sha256': sha(PROTOCOL)}, temporary); temporary.replace(progress)
                if (epoch+1) % 5 == 0: print(json.dumps({'head': variant, **record}), flush=True)
        assert updates == p['planned_optimizer_updates_each_head'] and initial_sha != state_sha256(head)
        summary = {k: v for k, v in best.items() if k != 'head'}; summary.update(actual_epochs=EPOCHS, actual_optimizer_updates=updates, initial_head_sha256=initial_sha, final_head_sha256=state_sha256(head))
        if not (folder/'training.json').exists():
            new(folder/'training.json', summary)
            with (folder/'best-head.pt').open('xb') as out: torch.save(best['head'], out)
        else: assert read(folder/'training.json') == summary
        summaries.append(summary)
    winner = min(summaries, key=lambda s: s['key']); head = wear_head(winner['variant']).cuda().eval()
    head.load_state_dict(torch.load(RUN/winner['variant']/'best-head.pt', map_location='cpu', weights_only=True))
    with torch.inference_mode(): scores = head(standard[cal_idx]).flatten().sigmoid().cpu().tolist()
    cal_labels = labels[cal_idx].cpu().tolist(); threshold = calibrated_threshold(cal_labels, scores)
    normal_mean = standard[fit_idx][labels[fit_idx] == 0].mean(0)
    baseline_scores = ((standard[cal_idx]-normal_mean)**2).mean(1).cpu().tolist()
    baseline_threshold, rank = normal_budget_threshold([s for y, s in zip(cal_labels, baseline_scores) if not y], .15)
    chosen = {'protocol_sha256': sha(PROTOCOL), 'selected_before_reserved_TEST_forward': True, 'variant': winner['variant'], 'epoch': winner['epoch'],
        'threshold': threshold, 'calibration_rates': classification_rates(cal_labels, scores, threshold), 'validation_chosen': winner, 'all_head_training': summaries,
        'total_actual_optimizer_updates': sum(s['actual_optimizer_updates'] for s in summaries), 'baseline_threshold': baseline_threshold, 'baseline_normal_CAL_rank': rank,
        'baseline_calibration_rates': classification_rates(cal_labels, baseline_scores, baseline_threshold)}
    new(RUN/'chosen-head.json', chosen)
    state = torch.load(START, map_location='cpu', weights_only=True); state['state_dict'] = dict(state['state_dict'])
    state['state_dict'].update({'wear_head.'+n: v for n, v in torch.load(RUN/winner['variant']/'best-head.pt', map_location='cpu', weights_only=True).items()})
    state['state_dict'].update(wear_mean=mean, wear_std=std)
    state.update(architecture=ARCH, wear_variant=winner['variant'], wear_category=WEAR_CATEGORY, wear_protocol_sha256=sha(PROTOCOL), wear_parent_sha256=EXPECTED,
        wear_operating_point={'photo_threshold': threshold, 'threshold_source': 'labelled_calibration_minimax', 'selected_epoch_by_validation': winner['epoch'], 'localization_evaluated': False},
        wear_actual_optimizer_updates=chosen['total_actual_optimizer_updates'], wear_crop_domain_only=True, independent_field_validated=False)
    state['trained_dataset_sources'] = deepcopy(state['trained_dataset_sources'])+['BSClass ball-screw pitting https://publikationen.bibliothek.kit.edu/1000133819']
    folder = RUN/'candidate'; folder.mkdir()
    with (folder/'final.pt').open('xb') as out: torch.save(state, out)
    with (RUN/'baseline-normal-centroid.pt').open('xb') as out: torch.save({'mean': normal_mean.cpu()}, out)
    print(json.dumps({'supervised_training_completed': True, 'chosen_head': winner['variant'], 'chosen_epoch': winner['epoch'], 'total_updates': chosen['total_actual_optimizer_updates'], 'calibration_rates': chosen['calibration_rates'], 'reserved_TEST_forwarded': False}), flush=True)


def load(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = IndustrialSupervisedMain(state['reference_counts'], state['native_specs'], state['wear_variant']).eval().requires_grad_(False)
    model.load_state_dict(state['state_dict'], strict=True); return model.to(device), state


def evaluate():
    validate(); chosen = read(RUN/'chosen-head.json'); assert chosen['selected_before_reserved_TEST_forward'] and chosen['protocol_sha256'] == sha(PROTOCOL)
    assert not (REPORTS/'industrial-bsclass-supervised-v8-validation.json').exists()
    rows = [r for r in read(MANIFEST)['rows'] if role(r) == 'reserved_test']; x = feature_cache(rows, 'reserved_TEST_once')
    model, state = load(RUN/'candidate/final.pt', 'cuda'); baseline = torch.load(RUN/'baseline-normal-centroid.pt', map_location='cuda', weights_only=True)['mean']
    with torch.inference_mode():
        scores = torch.cat([model.forward_wear_descriptor(x[i:i+256].cuda()).sigmoid().cpu() for i in range(0, len(x), 256)]).tolist()
        baseline_scores = torch.cat([((model.standardized_wear(x[i:i+256].cuda())-baseline)**2).mean(1).cpu() for i in range(0, len(x), 256)]).tolist()
    labels = [r['label'] for r in rows]; threshold = chosen['threshold']; rates = classification_rates(labels, scores, threshold)
    base_rates = classification_rates(labels, baseline_scores, chosen['baseline_threshold'])
    guards = {'photo_FPR_within20': rates['fpr'] <= .20, 'photo_FNR_within20': rates['fnr'] <= .20,
        'each_class_at_least20': min(rates['negative_photos'], rates['positive_photos']) >= 20,
        'max_error_better_than_baseline': max(rates['fpr'], rates['fnr']) < max(base_rates['fpr'], base_rates['fnr'])}
    parent = torch.load(START, map_location='cpu', weights_only=True)
    guards['all_parent_tensors_and_buffers_exact'] = all(torch.equal(v, state['state_dict'][n]) for n, v in parent['state_dict'].items())
    assert guards['all_parent_tensors_and_buffers_exact']; eligible = all(guards.values())
    observations = [{'id': r['id'], 'label': r['label'], 'group_id': r['group_id'], 'score': s, 'candidate_detected': s > threshold, 'baseline_score': b} for r, s, b in zip(rows, scores, baseline_scores)]
    value = {'category': WEAR_CATEGORY, 'photo_rates': rates, 'auroc': rank_auroc(labels, scores), 'baseline_photo_rates': base_rates,
        'photo_threshold': threshold, 'stage_gate': stage_gate(rates['fpr'], rates['fnr'], rates['negative_photos'], rates['positive_photos']),
        'guards': guards, 'eligible': eligible, 'observations': observations, 'fresh_reserved_TEST_first_scoring': True, 'physical_spindle_groups_verified': False,
        'localized_damage_masks_available': False, 'independent_field_validated': False}
    new(REPORTS/'industrial-bsclass-supervised-v8-validation.json', value)
    checkpoint = START
    if eligible:
        folder = RUN/'selected'; folder.mkdir(); checkpoint = folder/'final.pt'
        import shutil
        shutil.copy2(RUN/'candidate/final.pt', checkpoint)
    evaluation = read(PARENT_EVALUATION)
    if eligible: evaluation['categories'][WEAR_CATEGORY] = {k: value[k] for k in ('photo_rates', 'auroc', 'photo_threshold', 'stage_gate')}
    evaluation.update(checkpoint_sha256=sha(checkpoint), stage20_categories=[c for c, v in evaluation['categories'].items() if v['stage_gate']['ready_for_next_stage']],
        status='industrial_bsclass_supervised_v8', fresh_bsclass_reserved_TEST_first_scoring=True, physical_spindle_groups_verified=False)
    new(RUN/'evaluation.json', evaluation)
    new(REPORTS/'industrial-bsclass-supervised-v8-selection.json', {'selected': 'supervised_wear' if eligible else 'prior_prototype',
        'checkpoint_path': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(checkpoint), 'newly_selected_categories': [WEAR_CATEGORY] if eligible else [],
        'stage20_categories': evaluation['stage20_categories'], 'actual_optimizer_updates': chosen['total_actual_optimizer_updates'], 'guards': guards,
        'one_model_one_shared_backbone': True, 'old11_category_points_and_tensors_preserved': True, 'app_default_promoted': False, 'independent_field_validated': False})
    print(json.dumps({'fresh_TEST': rates, 'baseline': base_rates, 'eligible': eligible, 'selected_checkpoint_sha256': sha(checkpoint)}), flush=True)


def verify():
    p = validate(); selection = read(REPORTS/'industrial-bsclass-supervised-v8-selection.json')
    for n, h in p['legacy_source_sha256'].items(): assert sha(ROOT/n) == h, n
    ledger = read(RUN/'protected-inputs.json')
    def check(item):
        name, record = item; assert snapshot(ROOT/name) == record, name
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, ledger.items()))
    parent = torch.load(START, map_location='cpu', weights_only=True); model, state = load(RUN/'candidate/final.pt')
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in parent['state_dict'].items())
    assert state['equipment_operating_points'] == parent['equipment_operating_points'] and state['thresholds'] == parent['thresholds']
    old = NativeComponentMain(parent['reference_counts'], parent['native_specs']).eval(); old.load_state_dict(parent['state_dict'], strict=True)
    proofs = {}
    with torch.inference_mode():
        x = torch.zeros(1, 3, 256, 256)
        for category in CATEGORIES:
            a = model.forward_equipment_evidence(x, category); b = old.forward_equipment_evidence(x, category)
            assert all(torch.equal(v, w) for v, w in zip(a, b)), category; proofs[category] = True
        for task in ('facility_surface', 'belt_damage', 'conveyor_foreign'):
            a, b = model.forward_task(x, task), old.forward_task(x, task)
            a, b = (a if isinstance(a, tuple) else (a,)), (b if isinstance(b, tuple) else (b,))
            assert all(torch.equal(v, w) for v, w in zip(a, b)); proofs[task] = True
    new(REPORTS/'industrial-bsclass-supervised-v8-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved': True, 'all_parent_tensors_and_buffers_exact': True, 'all_old_points_and_thresholds_exact': True,
        'CPU_old_equipment_evidence_and_legacy_tasks_exact': proofs, 'selected': selection['selected'], 'actual_optimizer_updates': selection['actual_optimizer_updates'],
        'one_model_one_backbone': True, 'new_data_domain_not_mislabeled_as_fastener_screw': True})
    print(json.dumps({'verified': True, 'protected_files': len(ledger), 'old_equipment_and_legacy_CPU_paths': len(proofs)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'train', 'evaluate', 'verify']); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    else:
        from safelog_ai.process_probe import process_exists
        lock = RUN/'operation.lock'
        if lock.exists():
            owner = int(lock.read_text())
            if process_exists(owner): raise RuntimeError('Owned BSClass operation already active; do not launch a duplicate')
            lock.rename(RUN/('operation-stale-'+str(time.time_ns())+'.txt'))
        with lock.open('x') as out: out.write(str(os.getpid()))
        try: globals()[args.mode]()
        finally:
            if lock.exists() and lock.read_text() == str(os.getpid()): lock.unlink()
