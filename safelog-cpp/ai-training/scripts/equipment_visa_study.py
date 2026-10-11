"""Declare a bounded equipment experiment, then verify immutable inputs and report."""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_equipment_visa import DATA, SOURCES, PROTOCOL, read, write_new
from fetch_equipment_visa import sha

LEGACY = ROOT / 'runs/equipment-visa-legacy-ledger.json'
INPUTS = ROOT / 'runs/equipment-visa-input-ledger.json'


def snapshot(path):
    stat = path.stat()
    return {'sha256': sha(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def verify_ledger(ledger, prefix):
    for i, (name, expected) in enumerate(ledger.items()):
        assert snapshot(ROOT / name) == expected, 'Protected input differs: ' + name
        if i and i % 10000 == 0:
            print(prefix + ' verified ' + str(i), flush=True)


def declare():
    assert not PROTOCOL.exists()
    manifest = read(DATA / 'manifest.json')
    preflight = read(ROOT / 'runs/equipment-visa-preflight.json')
    assert preflight['status'] == 'passed' and preflight['source_sha256'] == {n: sha(ROOT / n) for n in SOURCES}
    legacy = read(ROOT / 'runs/facility-primary-preservation-input-ledger.json')['snapshots_before']
    previous_proof = read(ROOT / 'reports/facility-primary-preservation-study-verification.json')
    assert previous_proof['status'] == 'passed'
    for name, digest in previous_proof['source_sha256'].items():
        assert sha(ROOT / name) == digest
    # Inherit the full old ledger already verified at this turn's building closeout.
    extra = list(ROOT.joinpath('runs/facility-presence-target-primary-preservation').rglob('*'))
    extra += list(ROOT.joinpath('reports').glob('*primary-preservation*'))
    for path in extra:
        if path.is_file():
            legacy[path.relative_to(ROOT).as_posix()] = snapshot(path)
    write_new(LEGACY, legacy)
    new_inputs = {}
    for path in DATA.rglob('*'):
        if path.is_file():
            new_inputs[path.relative_to(ROOT).as_posix()] = snapshot(path)
    teacher_source = ROOT / '.cache/torch-equipment/hub/checkpoints/resnet18-f37072fd.pth'
    new_inputs[teacher_source.relative_to(ROOT).as_posix()] = snapshot(teacher_source)
    write_new(INPUTS, new_inputs)
    write_new(PROTOCOL, {'schema': 'equipment-visa-normal-student-teacher-study-v1',
        'declared_utc': datetime.now(timezone.utc).isoformat(), 'declared_before_training': True,
        'dataset': 'VisA official 1cls PCB1-4', 'license': 'CC-BY-4.0',
        'source': 'https://github.com/amazon-science/spot-diff',
        'method_inspiration': 'https://arxiv.org/abs/2103.04257',
        'exact_paper_reproduction': False, 'method': 'Random student / ImageNet frozen teacher feature pyramid matching',
        'category_models': 4, 'epochs_per_category': 20, 'batch_size': 32, 'image_size': 256,
        'optimizer': 'Adam', 'initial_lr': 0.003, 'weight_decay': 0.0001,
        'scheduler': 'Cosine 20, minimum LR 0.00001', 'dtype': 'float32, TF32 disabled',
        'seeds': [81, 82, 83, 84], 'normal_only_training': True, 'calibration_counts': manifest['counts'],
        'manifest_sha256': sha(DATA / 'manifest.json'),
        'source_sha256': {n: sha(ROOT / n) for n in SOURCES},
        'preflight_sha256': sha(ROOT / 'runs/equipment-visa-preflight.json'),
        'disposable_preflight_updates': sum(read(p)['disposable_gpu_updates'] for p in ROOT.joinpath('runs').glob('equipment-visa-preflight*.json')),
        'teacher_state_sha256': preflight['teacher_state_sha256'],
        'teacher_source_weights_sha256': preflight['teacher_source_weights_sha256'],
        'model_selection': 'Final epoch 20 fixed; no test selection or threshold optimization',
        'score': 'Max of fixed Gaussian-smoothed product of three normalized feature-distance maps',
        'threshold': 'Normal TRAIN holdout finite-sample ceil((n+1)*0.95) score, strict >',
        'test_policy': 'Once after all four models and thresholds frozen; no TEST fit/calibration',
        'strict_5_percent_gate': 'All four category FPR and FNR individually <0.05 on official test; not alpha or AUROC',
        'independent_factory_test': False, 'physical_board_group_identity_available': False,
        'mechanical_failure_diagnosis': False, 'app_default_promoted': False,
        'new_ai_or_human_labels': 0, 'building_model_retrained': False,
        'legacy_runtime_sources': previous_proof['source_sha256'],
        'legacy_ledger_sha256': sha(LEGACY), 'legacy_protected_files': len(legacy),
        'equipment_ledger_sha256': sha(INPUTS), 'equipment_protected_files': len(new_inputs),
        'building_closeout_proof_sha256': sha(ROOT / 'reports/facility-primary-preservation-study-verification.json')})
    print('Equipment protocol and inputs frozen', flush=True)


def verify():
    protocol = read(PROTOCOL)
    assert protocol['source_sha256'] == {n: sha(ROOT / n) for n in SOURCES}
    assert protocol['legacy_ledger_sha256'] == sha(LEGACY)
    assert protocol['equipment_ledger_sha256'] == sha(INPUTS)
    for name, digest in protocol['legacy_runtime_sources'].items():
        assert sha(ROOT / name) == digest
    verify_ledger(read(LEGACY), 'Building inputs')
    verify_ledger(read(INPUTS), 'Equipment inputs')
    result = read(ROOT / 'reports/equipment-visa-results.json')
    frozen = read(ROOT / 'reports/equipment-visa-frozen-operating-points.json')
    assert result['thresholds_fixed_before_test'] and frozen['completed_category_model_epochs'] == 80
    assert all(r['strict_cpu_gpu_reload_passed'] for r in frozen['categories'])
    updates = sum(r['optimizer_steps'] for r in frozen['categories'])
    expected = sum(math.ceil(r['normal_fit_photos'] / 32) * 20 for r in frozen['categories'])
    assert updates == expected
    write_new(ROOT / 'reports/equipment-visa-verification.json', {
        'status': 'passed', 'legacy_runtime_sources_preserved': len(protocol['legacy_runtime_sources']),
        'legacy_files_sha_size_mtime_preserved': len(read(LEGACY)),
        'equipment_files_sha_size_mtime_preserved': len(read(INPUTS)),
        'completed_category_model_epochs': 80, 'actual_optimizer_updates': updates,
        'disposable_preflight_updates': protocol['disposable_preflight_updates'], 'new_ai_or_human_labels': 0,
        'strict_cpu_gpu_reload_passed': True, 'default_app_model_changed': False,
        'protocol_sha256': sha(PROTOCOL), 'result_sha256': sha(ROOT / 'reports/equipment-visa-results.json')})
    print('Equipment post-training verification passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('declare', 'verify'))
    args = parser.parse_args()
    declare() if args.mode == 'declare' else verify()
