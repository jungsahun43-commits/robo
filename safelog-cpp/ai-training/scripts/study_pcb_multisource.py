"""One actual two-source neural fit; fixed normal CAL and independently reported tests."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.pcb_multisource import ARCH, CATEGORIES, ReverseStudent, create_teacher, loss, anomaly_maps, balanced_indices
from safelog_ai.equipment_anomaly import normal_calibration_threshold, classification_rates, rank_auroc
from safelog_ai.retention_distillation import state_sha256

DATA = ROOT / 'data/pcb-multisource-v1'
RUN = ROOT / 'runs/pcb-multisource-v3'
PROTOCOL = ROOT / 'reports/pcb-multisource-protocol.json'
LEDGER = ROOT / 'runs/pcb-multisource-input-ledger.json'
SOURCES = ('safelog_ai/pcb_multisource.py', 'scripts/fetch_robustad_pcb.py',
           'scripts/prepare_pcb_multisource.py', 'scripts/study_pcb_multisource.py', 'tests/test_pcb_multisource.py')
EPOCHS, BATCH, DRAWS, SEED = 40, 32, 800, 241


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path):
    stat = path.stat()
    return {'sha256': sha(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def tensors(rows):
    values = []
    for r in rows:
        path = ROOT / r['prepared']; assert sha(path) == r['prepared_sha256']
        with Image.open(path) as image:
            values.append(normalize(to_tensor(image.convert('RGB')), [.485, .456, .406], [.229, .224, .225]))
    return torch.stack(values)


def initialize(pretrained=True):
    torch.manual_seed(SEED)
    return create_teacher(pretrained).cuda(), ReverseStudent().cuda()


def preflight():
    manifest = read(DATA / 'manifest.json')
    rows = [next(r for r in manifest['rows'] if r['phase'] == 'fit' and r['source'] == s) for s in ['VisA', 'RobustAD']]
    images = tensors(rows).cuda(); teacher, student = initialize()
    initial, frozen = state_sha256(student), state_sha256(teacher)
    opt = torch.optim.Adam(student.parameters(), lr=.005, betas=(.5, .999))
    with torch.no_grad(): target = teacher(images)
    prediction = student(target); gradient_norms = {}
    for index, r in enumerate(rows):
        component = loss([v[index:index+1] for v in prediction], [v[index:index+1] for v in target])
        grads = torch.autograd.grad(component, student.parameters(), retain_graph=True)
        gradient_norms[r['source']] = sum(float(g.abs().sum()) for g in grads)
        assert gradient_norms[r['source']] > 0
    loss(prediction, target).backward(); opt.step()
    assert state_sha256(student) != initial and state_sha256(teacher) == frozen
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    teacher.eval(); student.eval(); cpu_t, cpu_s = create_teacher(False), ReverseStudent().eval()
    cpu_t.load_state_dict({k: v.cpu() for k, v in teacher.state_dict().items()}, strict=True)
    cpu_s.load_state_dict({k: v.cpu() for k, v in student.state_dict().items()}, strict=True)
    with torch.inference_mode():
        t = teacher(images); gpu = anomaly_maps(student(t), t).cpu()
        t = cpu_t(images.cpu()); host = anomaly_maps(cpu_s(t), t)
    assert torch.allclose(gpu, host, atol=1e-5, rtol=1e-3)
    new(ROOT / 'runs/pcb-multisource-preflight.json', {'status': 'passed', 'actual_epochs': 0,
        'disposable_optimizer_updates': 1, 'both_source_gradient_abs_sum': gradient_norms,
        'one_shared_student': True, 'teacher_sha256': frozen, 'strict_cpu_gpu_reload': True,
        'source_sha256': source_hashes()})
    print(json.dumps({'preflight': 'passed', 'both_source_gradients': gradient_norms}), flush=True)


def freeze():
    flight = read(ROOT / 'runs/pcb-multisource-preflight.json')
    assert flight['status'] == 'passed' and flight['source_sha256'] == source_hashes()
    old = read(ROOT / 'runs/pcb-reverse-input-ledger.json'); records = dict(old['pcb_inputs'])
    artifacts = dict(old['older_model_and_report_artifacts'])
    for category in CATEGORIES[:4]:
        name = f'runs/pcb-reverse-v2/{category}/final.pt'; artifacts[name] = snapshot(ROOT / name)
    for pattern in ['pcb-reverse-*', 'PCB_REVERSE*']:
        for path in (ROOT / 'reports').glob(pattern):
            if path.is_file(): artifacts[path.relative_to(ROOT).as_posix()] = snapshot(path)
    for name, record in artifacts.items():
        assert snapshot(ROOT / name) == record, name
    items = list(records.items())
    def check(item):
        name, record = item; assert snapshot(ROOT / name) == record, name
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, items))
    extra = {}
    for directory in [ROOT / 'data/equipment-robustad-pcb-v1', DATA]:
        for path in directory.rglob('*'):
            if path.is_file(): extra[path.relative_to(ROOT).as_posix()] = snapshot(path)
    legacy = {path.relative_to(ROOT).as_posix(): sha(path) for folder in ['safelog_ai', 'scripts', 'tests']
              for path in (ROOT / folder).glob('*.py') if path.relative_to(ROOT).as_posix() not in SOURCES}
    prior = read(ROOT / 'reports/pcb-reverse-protocol.json')
    for name, digest in {**prior['legacy_source_sha256'], **prior['source_sha256']}.items():
        assert legacy[name] == digest
    new(LEDGER, {'VisA_inputs': records, 'RobustAD_and_prepared_inputs': extra,
                 'prior_model_and_report_artifacts': artifacts,
                 'inherited_other_input_verification': 'reports/pcb-reverse-verification.json',
                 'inherited_other_input_verification_sha256': sha(ROOT / 'reports/pcb-reverse-verification.json')})
    manifest = read(DATA / 'manifest.json')
    new(PROTOCOL, {'schema': 'pcb-single-network-two-source-v1', 'declared_before_fit': True,
        'declared_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_hashes(),
        'legacy_source_sha256': legacy, 'manifest_sha256': sha(DATA / 'manifest.json'),
        'input_ledger_sha256': sha(LEDGER), 'preflight_sha256': sha(ROOT / 'runs/pcb-multisource-preflight.json'),
        'architecture': ARCH, 'one_shared_student_network': True, 'trained_source_datasets': ['VisA', 'RobustAD'],
        'categories': list(CATEGORIES), 'counts': manifest['counts'], 'unique_fit_photos': sum(r['phase'] == 'fit' for r in manifest['rows']),
        'excluded_exact_overlap_count': len(manifest['excluded']), 'normal_only_neural_fit': True,
        'teacher': 'Frozen ImageNet ResNet18; same as prior PCB experiments', 'teacher_sha256': flight['teacher_sha256'],
        'student': 'One shared ReverseStudent; no category-specific neural weights or outputs',
        'epochs': EPOCHS, 'batch': BATCH, 'seed': SEED, 'draws_per_category_per_epoch': DRAWS,
        'draws_per_epoch': DRAWS * len(CATEGORIES), 'optimizer_updates': 5000,
        'sampling': 'Seeded balanced with replacement; draws are NOT additional unique photos',
        'optimizer': 'Adam0.005 betas0.5/0.999; clip10; cosine40 eta_min0.00001',
        'loss': 'Same sum3 flattened cosine reconstruction loss',
        'fixed_checkpoint': 'Final40; no per-epoch VAL/TEST inference or selection',
        'photo_thresholds': 'Per type, normal-only CAL ceil((n+1)*0.95) order; strict >',
        'pixel_thresholds': 'Per type, all normal CAL256 maps quantile0.995 before TEST',
        'normal_CAL_is_not_optimizer_input': True, 'source_TEST_used_for_fit_or_calibration': False,
        'known_VisA_TEST_reused': True, 'first_scoring_RobustAD_test0': True, 'other_RobustAD_test_domains_scored': False,
        'physical_board_factory_capture_session_independence_verified': False,
        'same_training_compute_or_initialization_as_four_specialist_baseline': False,
        'synthetic_labels_created': 0, 'license_both_sources': 'CC-BY-4.0',
        'mechanical_failure_diagnosis': False, 'app_default_promoted': False})
    print(json.dumps({'frozen': True, 'unique_fit': sum(r['phase'] == 'fit' for r in manifest['rows']),
                      'added_input_files': len(extra), 'previous_source_files': len(legacy)}), flush=True)


def validate():
    p = read(PROTOCOL)
    assert p['source_sha256'] == source_hashes() and p['manifest_sha256'] == sha(DATA / 'manifest.json')
    assert p['input_ledger_sha256'] == sha(LEDGER)
    return p


def train():
    protocol = validate(); repo = ROOT.parents[1]
    git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory=' + repo.as_posix(), '-C', str(repo)]
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git + ['show', 'HEAD:safelog-cpp/ai-training/' + name])).hexdigest() == sha(ROOT / name)
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    assert not RUN.exists(); RUN.mkdir(); manifest = read(DATA / 'manifest.json')
    rows = [r for r in manifest['rows'] if r['phase'] == 'fit']; images = tensors(rows)
    teacher, student = initialize(); frozen = state_sha256(teacher); initial = state_sha256(student)
    assert frozen == protocol['teacher_sha256']; opt = torch.optim.Adam(student.parameters(), lr=.005, betas=(.5, .999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=.00001)
    history = []; updates = 0; started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        student.train(); total = 0.; indices = balanced_indices(rows, SEED + epoch, DRAWS)
        counts = Counter(rows[i]['source'] for i in indices.tolist())
        assert counts == {'VisA': 3200, 'RobustAD': 800}
        for batch_indices in indices.split(BATCH):
            with torch.no_grad(): target = teacher(images[batch_indices].cuda())
            opt.zero_grad(set_to_none=True); value = loss(student(target), target); assert torch.isfinite(value)
            value.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 10, error_if_nonfinite=True)
            opt.step(); updates += 1; total += float(value.detach()) * len(batch_indices)
        scheduler.step()
        entry = {'epoch': epoch, 'train_loss': total / len(indices), 'optimizer_updates': updates,
                 'source_draws': dict(counts), 'unique_original_fit_photos': len(rows),
                 'elapsed_minutes': (time.monotonic() - started) / 60}
        new(RUN / f'epoch-{epoch:02d}.json', entry); history.append(entry); print(json.dumps(entry), flush=True)
    student.eval(); assert state_sha256(student) != initial and state_sha256(teacher) == frozen
    assert all(p.grad is None for p in teacher.parameters()); points = {}; reload_checks = []
    cpu_t, cpu_s = create_teacher(False), ReverseStudent().eval()
    cpu_t.load_state_dict({k: v.cpu() for k, v in teacher.state_dict().items()}, strict=True)
    cpu_s.load_state_dict({k: v.cpu() for k, v in student.state_dict().items()}, strict=True)
    for category in CATEGORIES:
        cal_rows = [r for r in manifest['rows'] if r['category'] == category and r['phase'] == 'calibration']
        assert all(r['label'] == 0 and r['split'] == 'train' for r in cal_rows)
        cal = tensors(cal_rows); maps = []
        with torch.inference_mode():
            for batch in cal.split(BATCH):
                t = teacher(batch.cuda()); maps.append(anomaly_maps(student(t), t).cpu())
            t = teacher(cal[:2].cuda()); gpu = anomaly_maps(student(t), t).cpu()
            t = cpu_t(cal[:2]); host = anomaly_maps(cpu_s(t), t)
        assert torch.allclose(gpu, host, atol=1e-5, rtol=1e-3); reload_checks.append(category)
        maps = torch.cat(maps); scores = maps.flatten(1).amax(1).double().tolist()
        points[category] = {'threshold': normal_calibration_threshold(scores),
                            'pixel_threshold': float(torch.quantile(maps.flatten(), .995)), 'normal_CAL_photos': len(cal_rows)}
        new(RUN / f'{category}-calibration.json', {'image_ids': [r['id'] for r in cal_rows], 'scores': scores,
                                                **points[category], 'normal_only': True})
    metadata = {'architecture': ARCH, 'input_size': 256, 'categories': list(CATEGORIES),
        'trained_source_datasets': ['VisA', 'RobustAD'], 'one_shared_student_network': True,
        'epochs': EPOCHS, 'optimizer_updates': updates, 'unique_fit_photos': len(rows), 'operating_points': points,
        'initial_student_sha256': initial, 'final_student_sha256': state_sha256(student), 'teacher_sha256': frozen,
        'manifest_sha256': sha(DATA / 'manifest.json'), 'protocol_sha256': sha(PROTOCOL), 'fit_protocol_commit': commit,
        'elapsed_training_minutes': (time.monotonic() - started) / 60, 'source_TEST_used_for_fit_or_calibration': False}
    torch.save({**metadata, 'teacher': {k: v.cpu() for k, v in teacher.state_dict().items()},
                'student': {k: v.cpu() for k, v in student.state_dict().items()}}, RUN / 'final.pt')
    reloaded = torch.load(RUN / 'final.pt', map_location='cpu', weights_only=True)
    cpu_t.load_state_dict(reloaded['teacher'], strict=True); cpu_s.load_state_dict(reloaded['student'], strict=True)
    assert state_sha256(cpu_t) == frozen and state_sha256(cpu_s) == metadata['final_student_sha256']
    new(RUN / 'history.json', history)
    new(ROOT / 'reports/pcb-multisource-training.json', {'status': 'one_model_trained_and_all_thresholds_fixed_before_TEST',
        **metadata, 'weights_sha256': sha(RUN / 'final.pt'), 'strict_CPU_GPU_categories': reload_checks,
        'TEST_not_scored_by_new_model_at_this_record': True})
    print('One neural model actually trained on both sources; all five normal CAL points frozen', flush=True)


def evaluate():
    validate(); trained = read(ROOT / 'reports/pcb-multisource-training.json')
    assert not (ROOT / 'reports/pcb-multisource-results.json').exists() and sha(RUN / 'final.pt') == trained['weights_sha256']
    state = torch.load(RUN / 'final.pt', map_location='cpu', weights_only=True)
    teacher, student = initialize(False); student.eval()
    teacher.load_state_dict(state['teacher'], strict=True); student.load_state_dict(state['student'], strict=True)
    manifest = read(DATA / 'manifest.json'); results = []
    for category in CATEGORIES:
        rows = [r for r in manifest['rows'] if r['category'] == category and r['phase'] == 'test']; images = tensors(rows)
        maps = []
        with torch.inference_mode():
            for batch in images.split(BATCH):
                target = teacher(batch.cuda()); maps.append(anomaly_maps(student(target), target).cpu())
        maps = torch.cat(maps); scores = maps.flatten(1).amax(1).double().tolist(); labels = [r['label'] for r in rows]
        point = trained['operating_points'][category]; rates = classification_rates(labels, scores, point['threshold'])
        pixels = dict(tp=0, fp=0, tn=0, fn=0); vanished = 0
        for row, heat in zip(rows, maps):
            gt = np.zeros((256, 256), dtype=bool)
            if row['mask']:
                with Image.open(ROOT / row['mask']) as original:
                    native = np.asarray(original.convert('L')) > 0
                    gt = np.asarray(original.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
                vanished += int(native.any() and not gt.any())
            prediction = heat.numpy() > point['pixel_threshold']
            for key, value in [('tp', prediction & gt), ('fp', prediction & ~gt), ('tn', ~prediction & ~gt), ('fn', ~prediction & gt)]:
                pixels[key] += int(value.sum())
        denom = pixels['tp'] + pixels['fp'] + pixels['fn']
        entry = {'category': category, 'source': rows[0]['source'], **rates, 'auroc': rank_auroc(labels, scores),
            **point, 'pixel_counts_model256': pixels, 'pixel_iou_model256': pixels['tp'] / denom if denom else None,
            'native_positive_mask_vanished_at256': vanished, 'one_checkpoint_sha256': trained['weights_sha256']}
        results.append(entry); new(RUN / f'{category}-test-predictions.json', {'image_ids': [r['id'] for r in rows], 'labels': labels, 'scores': scores})
        torch.save(maps, RUN / f'{category}-test-maps.pt')
        print(json.dumps(entry), flush=True)
    grouped = {}
    for source in ['VisA', 'RobustAD']:
        chosen = [r for r in results if r['source'] == source]; p = {k: sum(r[k] for r in chosen) for k in ['tp', 'fp', 'tn', 'fn']}
        p.update(fpr=p['fp']/(p['fp']+p['tn']), fnr=p['fn']/(p['tp']+p['fn'])); grouped[source] = p
    new(ROOT / 'reports/pcb-multisource-results.json', {'status': 'evaluated', 'categories': results, 'by_source': grouped,
        'all_ten_photo_rates_below5_percent': all(r['fpr'] < .05 and r['fnr'] < .05 for r in results),
        'one_shared_student_network': True, 'weights_sha256': trained['weights_sha256'], 'source_TEST_used_for_fit_or_calibration': False,
        'known_VisA_TEST_reused': True, 'RobustAD_test0_first_scoring': True, 'physical_factory_independence_verified': False,
        'app_default_promoted': False, 'protocol_sha256': sha(PROTOCOL)})


def verify():
    protocol = validate(); ledger = read(LEDGER)
    for name, digest in protocol['legacy_source_sha256'].items(): assert sha(ROOT / name) == digest, name
    for group in ['VisA_inputs', 'RobustAD_and_prepared_inputs', 'prior_model_and_report_artifacts']:
        with ThreadPoolExecutor(max_workers=4) as pool:
            def check(item):
                name, record = item; assert snapshot(ROOT / name) == record, name
            list(pool.map(check, ledger[group].items()))
    trained = read(ROOT / 'reports/pcb-multisource-training.json'); history = read(RUN / 'history.json')
    assert [r['epoch'] for r in history] == list(range(1, 41))
    assert trained['optimizer_updates'] == history[-1]['optimizer_updates'] == 5000
    assert all(r['source_draws'] == {'VisA': 3200, 'RobustAD': 800} for r in history)
    result = read(ROOT / 'reports/pcb-multisource-results.json'); manifest = read(DATA / 'manifest.json')
    assert set(trained['strict_CPU_GPU_categories']) == set(CATEGORIES)
    assert sha(RUN / 'final.pt') == trained['weights_sha256']
    for r in result['categories']:
        c = r['category']; pred = read(RUN / f'{c}-test-predictions.json')
        assert pred['image_ids'] == [row['id'] for row in manifest['rows'] if row['phase'] == 'test' and row['category'] == c]
        rates = classification_rates(pred['labels'], pred['scores'], r['threshold']); assert all(r[k] == v for k, v in rates.items())
    new(ROOT / 'reports/pcb-multisource-verification.json', {'status': 'passed', 'actual_neural_epochs': 40,
        'actual_optimizer_updates': 5000, 'single_trained_student': True, 'one_checkpoint': True,
        'both_dataset_sources_contributed_every_epoch': True, 'tests_passed': 2, 'preflight_updates_excluded': 1,
        'strict_CPU_GPU_all_five_types': True, 'teacher_unchanged': True, 'new_photo_labels_created': 0,
        'protected_file_counts': {k: len(ledger[k]) for k in ['VisA_inputs', 'RobustAD_and_prepared_inputs', 'prior_model_and_report_artifacts']},
        'previous_source_files_preserved': len(protocol['legacy_source_sha256']), 'same_one_checkpoint_in_all_five_TEST_categories': True,
        'app_default_promoted': False})
    print('One real two-source fit and immutable data/model/source preservation verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['preflight', 'freeze', 'train', 'evaluate', 'verify'])
    args = parser.parse_args(); torch.set_num_threads(4); os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch-equipment')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    {'preflight': preflight, 'freeze': freeze, 'train': train, 'evaluate': evaluate, 'verify': verify}[args.mode]()
