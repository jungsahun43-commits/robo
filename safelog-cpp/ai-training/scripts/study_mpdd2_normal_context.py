"""One preregistered normal-only trial; source-specific photo evidence only."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.mpdd2_normal_context import MPDD2NormalMain, VERSION, white_targets
from safelog_ai.native_component_views import normalized
from safelog_ai.research_delta import pack_delta, restore_delta
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

START = ROOT / 'runs/industrial-magnetic-tiles-v16/released/final.pt'
EXPECTED = '5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e'
DATA = ROOT / 'data/industrial-mpdd2-white-v39'
MANIFEST = DATA / 'manifest.json'
RUN = ROOT / 'runs/industrial-mpdd2-normal-context-v40'
PROTOCOL = ROOT / 'reports/industrial-mpdd2-normal-context-v40-protocol.json'
SOURCES = ('safelog_ai/mpdd2_normal_context.py', 'scripts/study_mpdd2_normal_context.py', 'tests/test_mpdd2_normal_context.py')
EPOCHS, STEPS, BATCH, SEED = 30, 64, 16, 4040


def utc():
    return datetime.now(timezone.utc).isoformat()


def rows_for(phase):
    return [row for row in read(MANIFEST)['rows'] if row['phase'] == phase]


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(START) == EXPECTED
    audit = read(ROOT / 'reports/industrial-mpdd2-white-v39-data-audit.json')
    assert audit['ready'] and audit['manifest_SHA256'] == sha(MANIFEST)
    assert shutil.disk_usage(ROOT).free > 1024 ** 3
    RUN.mkdir()
    # Stat preservation checks are distinct from full content rehashing.
    protected = {}
    for directory in ('data', 'runs', 'reports'):
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and not path.is_relative_to(RUN) and not path.name.endswith(('.lock', '.log')):
                stat = path.stat()
                protected[path.relative_to(ROOT).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    new(RUN / 'protected-stat.json', protected)
    rng = np.random.default_rng(SEED)
    np.savez_compressed(RUN / 'draws.npz', indices=rng.integers(0, len(rows_for('fit')), size=(EPOCHS, STEPS, BATCH)),
                        variants=rng.integers(0, 4, size=(EPOCHS, STEPS, BATCH)))
    policy = {'UTC': utc(), 'schema': VERSION, 'fixed_before_new_features_gradients_model_scores_optimizer': True,
        'source_SHA256': {name: sha(ROOT / name) for name in SOURCES},
        'legacy_Python_SHA256': {path.relative_to(ROOT).as_posix(): sha(path)
            for directory in ('safelog_ai', 'scripts', 'tests') for path in (ROOT / directory).glob('*.py')
            if path.relative_to(ROOT).as_posix() not in SOURCES},
        'parent_SHA256': EXPECTED, 'manifest_SHA256': sha(MANIFEST), 'data_audit_SHA256': sha(ROOT / 'reports/industrial-mpdd2-white-v39-data-audit.json'),
        'protected_stat_SHA256': sha(RUN / 'protected-stat.json'), 'draws_SHA256': sha(RUN / 'draws.npz'),
        'source_scope': 'MPDD2 white-bracket 256x256 aligned close-up only, not old MPDD white identity/general field or mechanical health',
        'split': audit['counts'], 'groups': {key: len(value) for key, value in audit['groups'].items()},
        'FIT_visual_QA': 'fixed first3FIT sourceIDs; narrow light metal bar on dark background, pose/endpoint cropping varies; AI review not human certification of all labels',
        'input': 'original RGB256x256 unchanged geometry, normalized ImageNet; FIT original/H/V/HV RGB flips, not extra natural photos',
        'features': 'same frozen accepted backbone ONCE perRGB, existing low40/high960 unit-normalized merge1000, adaptiveavg4x4 then explicit FP16 roundtrip at cache AND runtime',
        'cache': 'FIT113x4 and VAL26 at first; CAL25 only after epoch selection; TEST140 only after CAL stored; IDs/RGB/rawSHA/order/encoderSHA/geometry/dtype bound in each cache. New total<25MB; state/progress<50MB, reserve1GiB',
        'architecture': 'new white_context_head only: Conv1000to64/GELU/flatten1024to128/GELU/dropout.1/Linear128to64; decoder64to128/GELU/to1024/GELU/unflatten64x4x4/Conv64to1000; no category/source embedding or second encoder',
        'normalization': 'original normalFIT113 only channel mean/popstd across photos+spatial, scale>=.01; standardized values clamp[-8,8]; stored buffers fixed thereafter',
        'initialization': 'fresh head seed4040; all original accepted tensors/buffers/metadata unchanged; no failed research head warmstart',
        'loss': 'normal-only mean squared standardized feature reconstruction; no defect/mask/relation labels used for training',
        'optimizer': 'AdamW lr.001 wd.0001, cosine30 min.00001, clip5, FP32, TF32off, torchthreads4',
        'epochs': EPOCHS, 'steps_per_epoch': STEPS, 'batch': BATCH, 'seed': SEED, 'planned_optimizer': EPOCHS * STEPS,
        'sampling': 'fixed seeded replacement normalFIT photo+variant schedule; 30720 draws are not30720 new natural photos',
        'VAL_selection': 'minimum original-only normalVAL MSE including epoch0, earliest tie. If epoch0 selected, skip freshCAL/TEST and reject',
        'photo_score': 'mean largest4 of16 spatial channel-MSE cells, strict score>cut; not defect probability',
        'CAL': 'normalCAL25 rankceil((25+1)*.85)=23, alpha.15 order statistic saved BEFORE one TEST evaluation; no defectCAL or threshold retries',
        'adoption_guards': ['selectedepoch>0', 'TESTlabel counts each>=20 and conservative groups each>=3', 'source FPR<=.20 AND FNR<=.20',
                            'all parent tensors/buffers/metadata exact', 'fresh package CPU/GPU/realC++/allalias/oldlocation checks before app adoption'],
        'location': 'no source pixelGT;4x4 error is unverified coarse evidence, existing primary maps/pixel thresholds remain',
        'target_transfer': 'not evaluated in this trial; source score not blackbracket/tube performance gain',
        'research_delta': 'parent-bound added white_context_head prefix only; serialized reload restore exact allstate+metadata; not standalone app model',
        'prior_public_TEST_development_exposure': True, 'physical_capture_groups_unverified': True, 'independent_field_validation': False,
        'new_real_defect_FIT_photographs': 0, 'human_reviewed_relationship_labels_used': False, 'auto_app_adoption': False,
        'new_units_passed': 3}
    new(PROTOCOL, policy)
    print({'protocol_SHA256': sha(PROTOCOL), 'protected_files_stat': len(protected), 'planned_optimizer': EPOCHS * STEPS}, flush=True)


def validate(committed=False):
    policy = read(PROTOCOL)
    for name, expected in policy['source_SHA256'].items():
        assert sha(ROOT / name) == expected, name
    for path, expected in ((START, EXPECTED), (MANIFEST, policy['manifest_SHA256']),
                           (RUN / 'draws.npz', policy['draws_SHA256']), (RUN / 'protected-stat.json', policy['protected_stat_SHA256'])):
        assert sha(path) == expected, path
    if committed:
        git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory=' + ROOT.parents[1].as_posix(), '-C', str(ROOT.parents[1])]
        for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
            assert hashlib.sha256(subprocess.check_output(git + ['show', 'HEAD:safelog-cpp/ai-training/' + name])).hexdigest() == sha(ROOT / name), name
    return policy


def load(device='cuda'):
    parent = torch.load(START, map_location='cpu', weights_only=True)
    torch.manual_seed(SEED)
    model = MPDD2NormalMain(parent['reference_counts'], parent['native_specs'], parent['wear_variant'], parent['surface_reference_specs'])
    missing, extra = model.load_state_dict(parent['state_dict'], strict=False)
    assert not extra and missing and all(name.startswith('white_context_head.') for name in missing)
    return model.eval().requires_grad_(False).to(device), parent


def cache(phase, model):
    path = RUN / (phase + '-features.pt')
    rows = rows_for(phase)
    identities = [{key: row[key] for key in ('id', 'image_sha256', 'rgb_sha256')} for row in rows]
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['protocol_SHA256'] == sha(PROTOCOL) and saved['identities'] == identities
        return rows, saved['targets']
    assert phase == 'test' or all(row['label'] == 0 for row in rows)
    targets = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            assert sha(ROOT / row['image']) == row['image_sha256']
            with Image.open(ROOT / row['image']) as image:
                rgb = image.convert('RGB')
                assert rgb.size == (256, 256) and hashlib.sha256(rgb.tobytes()).hexdigest() == row['rgb_sha256']
            variants = [rgb]
            if phase == 'fit':
                variants += [rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT), rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM), rgb.transpose(Image.Transpose.ROTATE_180)]
            x = torch.stack([normalized(image) for image in variants]).cuda()
            targets.append(white_targets(model.backbone(x)).cpu().half())
            if index % 40 == 0:
                print({'cache': phase, 'photos': index + 1, 'total': len(rows)}, flush=True)
    value = torch.stack(targets)
    with path.open('xb') as output:
        torch.save({'targets': value, 'protocol_SHA256': sha(PROTOCOL), 'identities': identities,
                    'parent_SHA256': EXPECTED, 'geometry': [256, 256, 4, 4], 'dtype': 'FP16-roundtrip'}, output)
    return rows, value


def train():
    policy = validate(committed=True)
    assert not (RUN / 'training.json').exists() and not (RUN / 'train/progress.pt').exists(), 'Do not repeat completed/partial study'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    lock = RUN / 'operation.lock'
    if lock.exists():
        raise RuntimeError('Existing operation lock preserved; inspect PID before recovery')
    new(lock, {'pid': os.getpid(), 'start_UTC': utc(), 'operation': 'actual_train_and_fixed_evaluation'})
    actual = 0
    try:
        model, parent = load()
        _, fit_cache = cache('fit', model)
        _, val_cache = cache('validation', model)
        fit, val = fit_cache.cuda().float(), val_cache[:, 0].cuda().float()
        head = model.white_context_head
        with torch.no_grad():
            head.FIT_mean.copy_(fit[:, 0].mean((0, 2, 3))[None, :, None, None])
            head.FIT_scale.copy_(fit[:, 0].std((0, 2, 3), unbiased=False).clamp_min(.01)[None, :, None, None])
        head.requires_grad_(True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=.00001)
        draws = np.load(RUN / 'draws.npz')
        (RUN / 'train').mkdir()

        def val_loss():
            head.eval()
            with torch.no_grad():
                return float((head(val) - head.standardize(val)).square().mean())

        best = val_loss()
        initial = best
        selected_epoch, selected_updates = 0, 0
        selected = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        history = [{'epoch': 0, 'actual_optimizer': 0, 'VAL_MSE': best}]
        journal = RUN / 'train/optimizer-journal.jsonl'
        with journal.open('x', encoding='utf-8') as log:
            for epoch in range(1, EPOCHS + 1):
                head.train()
                for step in range(STEPS):
                    x = fit[draws['indices'][epoch - 1, step], draws['variants'][epoch - 1, step]]
                    optimizer.zero_grad(set_to_none=True)
                    loss = (head(x) - head.standardize(x)).square().mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5)
                    assert torch.isfinite(norm)
                    optimizer.step()
                    actual += 1
                    log.write(json.dumps({'actual_optimizer': actual, 'epoch': epoch, 'step': step + 1, 'loss': float(loss.detach())}) + '\n')
                    log.flush()
                scheduler.step()
                value = val_loss()
                assert np.isfinite(value)
                if value < best:
                    best, selected_epoch, selected_updates = value, epoch, actual
                    selected = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
                history.append({'epoch': epoch, 'actual_optimizer': actual, 'VAL_MSE': value, 'selected_epoch': selected_epoch})
                temp = RUN / 'train/progress.pending.pt'
                torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'actual_optimizer': actual, 'epoch': epoch, 'CPU_RNG': torch.get_rng_state(),
                            'CUDA_RNG': torch.cuda.get_rng_state_all()}, temp)
                temp.replace(RUN / 'train/progress.pt')
                print(history[-1], flush=True)
        assert actual == EPOCHS * STEPS
        torch.save({name: value.detach().cpu() for name, value in head.state_dict().items()}, RUN / 'train/last-head.pt')
        head.load_state_dict(selected)
        head.eval()
        candidate = deepcopy(parent)
        candidate['state_dict'].update({'white_context_head.' + name: value for name, value in selected.items()})
        candidate['mpdd2_white_context_version'] = VERSION
        candidate['mpdd2_white_context_protocol_SHA256'] = sha(PROTOCOL)
        candidate['mpdd2_white_selected_epoch'] = selected_epoch
        payload = pack_delta(parent, candidate, EXPECTED, START.relative_to(ROOT).as_posix(), ('white_context_head.',))
        (RUN / 'trained').mkdir()
        checkpoint = RUN / 'trained/final-delta.pt'
        with checkpoint.open('xb') as output:
            torch.save(payload, output)
        restored = restore_delta(parent, torch.load(checkpoint, map_location='cpu', weights_only=True), EXPECTED)
        assert candidate.keys() == restored.keys()
        assert all(torch.equal(value, restored['state_dict'][name]) for name, value in candidate['state_dict'].items())
        assert all(value == restored[name] for name, value in candidate.items() if name != 'state_dict')
        current_state = model.state_dict()
        for name, value in parent['state_dict'].items():
            assert torch.equal(value, current_state[name].detach().cpu()), name
        training = {'UTC': utc(), 'actual_optimizer': actual, 'selected_epoch': selected_epoch, 'selected_actual_optimizer': selected_updates,
                    'initial_VAL_MSE': initial, 'selected_VAL_MSE': best, 'history': history,
                    'normal_FIT_photos': 113, 'RGB_FIT_extractions': 452, 'normal_VAL_photos': 26,
                    'new_natural_defect_training_photos': 0, 'parent_all_tensors_buffers_metadata_exact': True,
                    'serialized_delta_restore_exact': True, 'checkpoint_SHA256': sha(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
                    'journal_SHA256': sha(journal), 'journal_rows': sum(1 for _ in journal.open(encoding='utf-8'))}
        assert training['journal_rows'] == actual
        new(RUN / 'training.json', training)
        new(ROOT / 'reports/industrial-mpdd2-normal-context-v40-training.json', training)
        if selected_epoch == 0:
            evaluation = {'UTC': utc(), 'eligible_research_candidate': False, 'reason': 'epoch0 selected; no learned VAL improvement',
                          'new_CAL_TEST_scores': 0, 'new_API_CPP_ZIP_checks': 0}
        else:
            cal_rows, cal_features = cache('cal', model)
            with torch.no_grad():
                cal_scores = head.evidence(cal_features[:, 0].cuda().float())[1].cpu().tolist()
            cut, rank = normal_budget_threshold(cal_scores, alpha=.15)
            new(RUN / 'calibration.json', {'UTC': utc(), 'normal_CAL': len(cal_rows), 'scores': cal_scores,
                                          'threshold': cut, 'rank': rank, 'alpha': .15, 'strict_score_greater': True,
                                          'stored_before_TEST_features_or_scores': True})
            test_rows, test_features = cache('test', model)
            with torch.no_grad():
                scores = head.evidence(test_features[:, 0].cuda().float())[1].cpu().tolist()
            labels = [row['label'] for row in test_rows]
            rates = classification_rates(labels, scores, cut)
            ready = bool(rates['fpr'] <= .20 and rates['fnr'] <= .20)
            evaluation = {'UTC': utc(), **rates, 'AUROC': rank_auroc(labels, scores), 'threshold': cut,
                'eligible_research_candidate': ready, 'reason': 'both source photo rates <=20%' if ready else 'source photo rate guard failed',
                'rows': [{'id': row['id'], 'group_id': row['group_id'], 'label': row['label'], 'score': score} for row, score in zip(test_rows, scores)],
                'calibration_SHA256': sha(RUN / 'calibration.json'), 'fresh_CAL_scores': 25, 'fresh_TEST_scores': 140,
                'physical_group_independence_unverified': True, 'independent_field_validation': False,
                'actual_pixel_location_ground_truth': False, 'black_bracket_or_tube_transfer_evaluated': False,
                'app_adopted': False, 'new_API_CPP_ZIP_checks': 0, 'new_ZIPs': 0}
        new(RUN / 'evaluation.json', evaluation)
        new(ROOT / 'reports/industrial-mpdd2-normal-context-v40-evaluation.json', evaluation)
        verify = {'UTC': utc(), 'parent_SHA256_unchanged': sha(START) == EXPECTED,
                  'legacy_Python_files_SHA256_checked': len(policy['legacy_Python_SHA256']),
                  'protected_files_stat_checked': len(read(RUN / 'protected-stat.json')),
                  'full_protected_content_SHA_recalculated': False}
        for name, expected in policy['legacy_Python_SHA256'].items():
            assert sha(ROOT / name) == expected, name
        for name, expected in read(RUN / 'protected-stat.json').items():
            stat = (ROOT / name).stat()
            assert [stat.st_size, stat.st_mtime_ns] == expected, name
        new(RUN / 'verification.json', verify)
        new(ROOT / 'reports/industrial-mpdd2-normal-context-v40-verification.json', verify)
        print({key: value for key, value in evaluation.items() if key != 'rows'}, flush=True)
    except BaseException as error:
        if 'head' in locals() and 'optimizer' in locals():
            torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(), 'actual_optimizer': actual}, RUN / 'failure-partial.pt')
        new(RUN / ('failure-' + datetime.now(timezone.utc).strftime('%H%M%S') + '.json'),
            {'UTC': utc(), 'error': repr(error), 'actual_optimizer_completed': actual})
        raise
    finally:
        if read(lock)['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'train'))
    globals()[parser.parse_args().stage]()
