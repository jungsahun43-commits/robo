"""Prespecified belt-only background focus after the accepted native crop main."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.belt_background import VARIANTS, training_loss, select_background
from safelog_ai.belt_detail import configure_belt_only, non_belt_state
from safelog_ai.unified_main import ARCH, TASKS, UnifiedMain, freeze_bn_statistics
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.belt_evaluation import summarize
from safelog_ai.belt_segmentation import pixel_counts
from scripts.unified_main_data import DATA, Cache, read, sha
from scripts.study_unified_belt_detail import batch_for

BASE = ROOT/'runs/unified-belt-hard-background-v1'
CROPS = ROOT/'data/unified-belt-native-detail-v1'
PROTOCOL = ROOT/'reports/unified-belt-background-protocol.json'
LEDGER = BASE/'protected-inputs.json'
START = ROOT/'runs/unified-belt-native-detail-v1/native_mix/final.pt'
START_SHA = '4b6d91c045cf6a33631ed5542c81d8e116166f541e97e6d745e1dfbe3f7fe5e5'
SOURCES = ('safelog_ai/belt_background.py', 'scripts/study_unified_belt_background.py', 'tests/test_belt_background.py')
EPOCHS, BATCH, SEED = 10, 8, 427


def new(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('x', encoding='utf-8', newline='\n') as out: json.dump(value, out, ensure_ascii=False, indent=2)


def snapshot(path):
    stat = path.stat(); return {'sha256': sha(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def check(pair):
    name, expected = pair; assert snapshot(ROOT/name) == expected, name


def freeze():
    assert not BASE.exists() and sha(START) == START_SHA
    manifest = read(DATA/'manifest.json'); crops = read(CROPS/'manifest.json')
    assert len(crops['rows']) == 1397 and [r['id'] for r in crops['rows']] == [r['id'] for r in manifest['fit']['belt_damage']]
    assert all(r['split'] == 'train' for r in crops['rows'])
    train_groups = {r['sequence'] for r in crops['rows']}; val_groups = {r['sequence'] for r in manifest['belt_known_validation']}
    assert not train_groups & val_groups
    files = dict(read(ROOT/'runs/unified-belt-native-detail-v1/protected-inputs.json'))
    for folder in [ROOT/'reports', ROOT/'runs/unified-belt-native-detail-v1']:
        paths = list(folder.iterdir())
        if folder.name == 'unified-belt-native-detail-v1':
            for v in ['full_control', 'native_mix']: paths += list((folder/v).iterdir())
        for path in paths:
            if path.is_file() and not path.name.endswith(('.log', '.lock')): files[path.relative_to(ROOT).as_posix()] = snapshot(path)
    for path in [DATA/'manifest.json', CROPS/'manifest.json']: files[path.relative_to(ROOT).as_posix()] = snapshot(path)
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, files.items()))
    BASE.mkdir(); new(LEDGER, files)
    draws = np.load(ROOT/'runs/unified-belt-native-detail-v1/draws.npz')['belt_damage'].copy()
    assert draws.shape == (10, 1024); np.savez_compressed(BASE/'draws.npz', belt_damage=draws)
    legacy = {p.relative_to(ROOT).as_posix(): sha(p) for folder in ['scripts', 'safelog_ai', 'tests'] for p in (ROOT/folder).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES}
    new(PROTOCOL, {'schema': 'unified-belt-hard-background-v1', 'declared_before_fit': True, 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES}, 'legacy_source_sha256': legacy, 'input_ledger_sha256': sha(LEDGER),
        'start_sha256': START_SHA, 'unified_manifest_sha256': sha(DATA/'manifest.json'), 'crop_manifest_sha256': sha(CROPS/'manifest.json'),
        'draws_sha256': sha(BASE/'draws.npz'), 'variants': list(VARIANTS), 'epochs_each': 10, 'updates_each': 1280, 'total_updates': 2560,
        'batch': BATCH, 'seed': SEED, 'unique_scheduled_TRAIN': len(set(draws.flatten().tolist())), 'draws_each': 10240,
        'crop_sampling': 'Both exactly4of8 native TRAIN crops plus4fullframes using accepted study same ids/order/paired flip',
        'optimizer': 'Fresh AdamW belt-head only LR3e-4 weight_decay.0002 cosine10 eta_min5e-6; clip10',
        'freeze': 'All non-belt parameters and buffers, all BN running stats, frozen accepted-model reference; teacher training only',
        'shared_loss': 'Unchanged BCE posweight20 plus Dice; +2 mean squared positive-probability deficit to frozen reference ONLY where publisher mask&valid&reference>=.5',
        'only_difference': 'hard_background adds .05 times per-image top1percent softplus over valid publisher background outside7x7 dilated damage; anchor_control adds0',
        'teacher_probability_is_new_label': False, 'new_photos': 0, 'new_labels': 0, 'runtime_checkpoint_count': 1,
        'precision': 'FP32 TF32 disabled', 'selection_epoch': 'Final10 only; no per-epoch VAL or early stopping',
        'evaluation': 'Same known reused VAL353 full640 threshold.5; author reserved TEST unused; foreign CAL/TEST not evaluated',
        'selection': 'Background pixel FPR<=.9*accepted AND IoU>=accepted AND photoFNR<=accepted AND pixelFNR<=accepted AND zero-GT-overlap<=accepted; rank FPR then IoU then control; else keep prior_native_mix',
        'baseline': read(ROOT/'reports/unified-belt-detail-native_mix-validation.json')['summary'],
        'baseline_training_lineage': 'Base unified20epoch6400updates + accepted native-belt10epoch1280updates; now each candidate additional10epoch1280updates',
        'known_VAL_reused': True, 'normal_belt_photo_FPR': None, 'independent_factory_verified': False, 'app_default_promoted': False,
        'tests_passed': 5, 'checkpoint_every_updates': 64})
    print(json.dumps({'protocol': 'frozen', 'protected_files': len(files), 'legacy_sources': len(legacy)}), flush=True)


def validate():
    p = read(PROTOCOL)
    for name, digest in p['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    for path, digest in [(START, START_SHA), (LEDGER, p['input_ledger_sha256']), (DATA/'manifest.json', p['unified_manifest_sha256']),
                         (CROPS/'manifest.json', p['crop_manifest_sha256']), (BASE/'draws.npz', p['draws_sha256'])]: assert sha(path) == digest, path
    return p


def save_progress(path, model, opt, scheduler, epoch, cursor, updates, losses, elapsed):
    assert path.resolve().is_relative_to(BASE.resolve()); temp = path.with_suffix('.pending.pt')
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'cursor': cursor, 'updates': updates, 'losses': losses,
        'elapsed_seconds': elapsed, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temp); temp.replace(path)


def train(variant):
    p = validate(); repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    commit = subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip()
    run = BASE/variant; run.mkdir(exist_ok=True); report = ROOT/f'reports/unified-belt-background-{variant}-training.json'
    if (run/'final.pt').exists():
        final = torch.load(run/'final.pt', map_location='cpu', weights_only=True)
        assert final['variant'] == variant and final['optimizer_updates'] == 1280
        if not report.exists(): new(report, {**{k: v for k, v in final.items() if k != 'state_dict'}, 'weights_sha256': sha(run/'final.pt')})
        assert sha(run/'final.pt') == read(report)['weights_sha256']; print(variant+' already complete', flush=True); return
    lock = run/'training.lock'
    if lock.exists():
        pid = int(lock.read_text())
        try: os.kill(pid, 0)
        except OSError: lock.unlink()
        else: raise RuntimeError('Owned study already running '+str(pid))
    with lock.open('x') as out: out.write(str(os.getpid()))
    try:
        torch.manual_seed(SEED); start = torch.load(START, map_location='cpu', weights_only=True)
        model = UnifiedMain().cuda(); model.load_state_dict(start['state_dict'], strict=True)
        reference = UnifiedMain().cuda().eval(); reference.load_state_dict(start['state_dict'], strict=True); reference.requires_grad_(False)
        reference_sha = state_sha256(reference); fixed = state_sha256(non_belt_state(model)); belt_sha = state_sha256(model.belt_head)
        opt = torch.optim.AdamW(configure_belt_only(model), lr=3e-4, weight_decay=.0002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=5e-6)
        manifest = read(DATA/'manifest.json'); draws = np.load(BASE/'draws.npz')['belt_damage']; allowed = set(draws.flatten().tolist())
        full = Cache(manifest['fit']['belt_damage'], allowed); crops = Cache(read(CROPS/'manifest.json')['rows'], allowed)
        progress = run/'progress.pt'; epoch, cursor, updates, losses, elapsed = 1, 0, 0, 0., 0.; torch.manual_seed(SEED)
        if progress.exists():
            s = torch.load(progress, map_location='cpu', weights_only=True); model.load_state_dict(s['state_dict'], strict=True)
            opt.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
            epoch, cursor, updates, losses, elapsed = (s[k] for k in ['epoch', 'cursor', 'updates', 'losses', 'elapsed_seconds'])
            torch.set_rng_state(s['torch_rng']); torch.cuda.set_rng_state_all(s['cuda_rng'])
        started = time.monotonic()
        with (run/'update-journal.jsonl').open('a', encoding='utf-8', buffering=1) as journal:
            while epoch <= EPOCHS:
                model.eval(); model.belt_head.train(); freeze_bn_statistics(model)
                chunks = np.split(draws[epoch-1], 128)
                for index in range(cursor, len(chunks)):
                    images, truth = batch_for(full, crops, chunks[index], 'native_mix')
                    with torch.no_grad(): target_reference = reference.forward_task(images, 'belt_damage')
                    opt.zero_grad(set_to_none=True); logits = model.forward_task(images, 'belt_damage')
                    value = training_loss(variant, logits, target_reference, truth); assert torch.isfinite(value)
                    value.backward(); torch.nn.utils.clip_grad_norm_(model.belt_head.parameters(), 10, error_if_nonfinite=True); opt.step()
                    updates += 1; cursor = index+1; losses += float(value.detach())
                    journal.write(json.dumps({'epoch': epoch, 'cursor': cursor, 'update': updates, 'native_crop_draws': 4})+'\n')
                    if updates%64 == 0: save_progress(progress, model, opt, scheduler, epoch, cursor, updates, losses, elapsed+time.monotonic()-started)
                assert state_sha256(non_belt_state(model)) == fixed and state_sha256(reference) == reference_sha
                scheduler.step(); entry = {'variant': variant, 'epoch': epoch, 'updates': updates, 'mean_loss': losses/128,
                    'elapsed_minutes': (elapsed+time.monotonic()-started)/60, 'non_belt_and_reference_unchanged': True}
                dest = run/f'epoch-{epoch:02d}.json'
                if dest.exists(): assert read(dest)['updates'] == updates
                else: new(dest, entry)
                print(json.dumps(entry), flush=True); epoch += 1; cursor = 0; losses = 0.
                save_progress(progress, model, opt, scheduler, epoch, cursor, updates, losses, elapsed+time.monotonic()-started)
        assert updates == 1280 and state_sha256(model.belt_head) != belt_sha
        assert all(v.grad is None for n, v in model.named_parameters() if not n.startswith('belt_head.')) and all(v.grad is None for v in reference.parameters())
        model.eval(); cpu = UnifiedMain().eval(); cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()}, strict=True)
        original = UnifiedMain().eval(); original.load_state_dict(start['state_dict'], strict=True); probes = {}
        with torch.inference_mode():
            for task in TASKS:
                images, _ = Cache(manifest['fit'][task], {0, 1}).batch([0, 1]); a, b = model.forward_task(images, task), cpu.forward_task(images.cpu(), task)
                if task == 'facility_surface': a, b = a[0], b[0]
                assert torch.allclose(a.cpu(), b, atol=1e-4, rtol=.001)
                if task != 'belt_damage':
                    r = original.forward_task(images.cpu(), task); r = r[0] if task == 'facility_surface' else r; assert torch.equal(r, b)
                probes[task] = {'strict_CPU_GPU_close': True, 'prior_CPU_predictions_exactly_equal': task != 'belt_damage'}
        metadata = {'status': 'trained_fixed10', 'architecture': ARCH, 'tasks': list(TASKS), 'one_shared_backbone': True, 'input_size': 640,
            'thresholds': start['thresholds'], 'variant': variant, 'epochs': 10, 'optimizer_updates': 1280,
            'training_lineage': {'unified_initial_epochs': 20, 'accepted_native_belt_epochs': 10, 'additional_belt_epochs': 10},
            'initial_checkpoint_sha256': START_SHA, 'protocol_sha256': sha(PROTOCOL), 'fit_protocol_commit': commit,
            'non_belt_state_sha256': fixed, 'reference_model_unchanged': True, 'reference_has_no_gradients': True,
            'reference_used_at_runtime': False, 'strict_CPU_GPU_three_tasks': probes,
            'elapsed_training_minutes': (elapsed+time.monotonic()-started)/60,
            'actual_journal_updates_including_replay': sum(1 for _ in (run/'update-journal.jsonl').open()),
            'new_photos': 0, 'new_labels': 0, 'checkpoint_count': 1, 'backbone_count': 1, 'field_validated': False}
        with (run/'final.pt').open('xb') as out: torch.save({**metadata, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}}, out)
        new(report, {**metadata, 'weights_sha256': sha(run/'final.pt')}); print(json.dumps({'completed': variant, 'sha256': sha(run/'final.pt')}), flush=True)
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()): lock.unlink()


def evaluate():
    validate(); rows = read(DATA/'manifest.json')['belt_known_validation']; cache = Cache(rows, range(len(rows)))
    for variant in VARIANTS:
        path = ROOT/f'reports/unified-belt-background-{variant}-validation.json'
        if path.exists(): continue
        trained = read(ROOT/f'reports/unified-belt-background-{variant}-training.json'); checkpoint = BASE/variant/'final.pt'
        assert sha(checkpoint) == trained['weights_sha256']; model = UnifiedMain().cuda().eval()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'], strict=True); observations = []
        with torch.inference_mode():
            for offset in range(0, len(rows), BATCH):
                images, truth = cache.batch(range(offset, min(offset+BATCH, len(rows)))); scores = model.forward_task(images, 'belt_damage').sigmoid()
                for i, r in enumerate(rows[offset:offset+BATCH]):
                    s, m, v = scores[i:i+1], truth['mask'][i:i+1], truth['valid'][i:i+1]
                    observations.append({'id': r['id'], 'sequence': r['sequence'], 'photo_positive': r['photo_positive'],
                        'score': float(s[v.bool()].max()), 'pixels': pixel_counts(s, m, v)})
        summary = summarize(observations); new(path, {'status': 'known_reused_VAL353', 'variant': variant,
            'model_sha256': trained['weights_sha256'], 'threshold': .5, 'rows': observations, 'summary': summary,
            'normal_photo_FPR_available': False, 'reserved_TEST_used': False})
        print(json.dumps({'validation': variant, 'summary': summary}), flush=True)


def select():
    p = validate(); results = {v: read(ROOT/f'reports/unified-belt-background-{v}-validation.json')['summary'] for v in VARIANTS}
    choice = select_background(results, p['baseline']); path = START if choice['selected'] == 'prior_native_mix' else BASE/choice['selected']/'final.pt'
    choice.update(checkpoint_path=path.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(path), protocol_sha256=sha(PROTOCOL), decided_utc=datetime.now(timezone.utc).isoformat())
    new(ROOT/'reports/unified-belt-background-selection.json', choice); print(json.dumps(choice), flush=True)


def verify():
    p = validate()
    for name, digest in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == digest, name
    ledger = read(LEDGER)
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, ledger.items()))
    original = torch.load(START, map_location='cpu', weights_only=True)['state_dict']; proofs = {}
    for variant in VARIANTS:
        run = BASE/variant; trained = read(ROOT/f'reports/unified-belt-background-{variant}-training.json')
        final = torch.load(run/'final.pt', map_location='cpu', weights_only=True); progress = torch.load(run/'progress.pt', map_location='cpu', weights_only=True)
        assert progress['epoch'] == 11 and progress['cursor'] == 0 and progress['updates'] == 1280
        assert all(torch.equal(v, progress['state_dict'][k]) for k, v in final['state_dict'].items())
        assert all(torch.equal(v, original[k]) for k, v in final['state_dict'].items() if not k.startswith('belt_head.'))
        assert sha(run/'final.pt') == trained['weights_sha256'] and trained['reference_model_unchanged'] and trained['reference_has_no_gradients']
        history = [read(run/f'epoch-{e:02d}.json') for e in range(1, 11)]; assert all(r['updates'] == r['epoch']*128 for r in history)
        journal = [json.loads(s) for s in (run/'update-journal.jsonl').read_text().splitlines()]; committed = {r['update']: r for r in journal}
        assert set(committed) == set(range(1, 1281)) and sum(r['native_crop_draws'] for r in committed.values()) == 5120
        proofs[variant] = {'committed_updates': 1280, 'actual_journal_updates': len(journal), 'native_crop_draws': 5120, 'fullframe_draws': 5120,
            'all_non_belt_tensors_exact_prior': True, 'final_equals_last_progress': True, 'reference_unchanged_and_no_gradients': True,
            'strict_CPU_GPU_three_tasks': trained['strict_CPU_GPU_three_tasks'], 'model_sha256': trained['weights_sha256']}
    choice = read(ROOT/'reports/unified-belt-background-selection.json'); assert sha(ROOT/choice['checkpoint_path']) == choice['checkpoint_sha256']
    new(ROOT/'reports/unified-belt-background-verification.json', {'status': 'passed', 'models': proofs, 'total_real_updates': 2560,
        'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']), 'selected': choice['selected'],
        'new_photos': 0, 'new_labels': 0, 'reserved_TEST_used': False, 'normal_photo_FPR_available': False, 'app_default_promoted': False})
    print(json.dumps({'verified': 'passed', 'selected': choice['selected'], 'protected_files': len(ledger)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'train', 'evaluate', 'select', 'verify']); parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.mode == 'train':
        for variant in ([args.variant] if args.variant else VARIANTS): train(variant)
    else: globals()[args.mode]()
