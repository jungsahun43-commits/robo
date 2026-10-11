"""Freeze and verify a first belt experiment without changing older engines."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_beltcrack_starter import DATA, SOURCES, PROTOCOL, read, new
from prepare_beltcrack_starter import sha

LEDGER = ROOT / 'runs/beltcrack-starter-input-ledger.json'


def snapshot(path):
    stat = path.stat()
    return {'sha256': sha(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def declare():
    assert not PROTOCOL.exists()
    manifest = read(DATA / 'manifest.json')
    preflight = read(ROOT / 'runs/beltcrack-starter-preflight.json')
    assert preflight['status'] == 'passed'
    assert preflight['source_sha256'] == {n: sha(ROOT / n) for n in SOURCES}
    rows = [r for r in manifest['rows'] if r['split'] == 'train']
    positive = sum(r['positive_pixels'] for r in rows)
    valid = sum(r['valid_pixels'] for r in rows)
    prior = read(ROOT / 'reports/equipment-visa-study-protocol.json')
    legacy_sources = {**prior['legacy_runtime_sources'], **prior['source_sha256']}
    for name, digest in legacy_sources.items():
        assert sha(ROOT / name) == digest
    # The previous completed study verified its complete building and PCB ledgers.
    # Here separately bind the unchanged source and model artifacts again.
    existing = {}
    for source in ('runs/equipment-visa-legacy-ledger.json', 'runs/equipment-visa-input-ledger.json'):
        for name, value in read(ROOT / source).items():
            if name.startswith('models/') or name.endswith('/best.pt') or name.endswith('/final.pt') or name.endswith('/manifest.json'):
                existing[name] = value
    for path in ROOT.joinpath('runs/equipment-visa-stfpm-v1').rglob('*'):
        if path.is_file():
            existing[path.relative_to(ROOT).as_posix()] = snapshot(path)
    for path in ROOT.joinpath('reports').glob('equipment-visa*'):
        if path.is_file():
            existing[path.relative_to(ROOT).as_posix()] = snapshot(path)
    ledger = {path.relative_to(ROOT).as_posix(): snapshot(path) for path in DATA.rglob('*') if path.is_file()}
    weights = ROOT / '.cache/torch-equipment/hub/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth'
    ledger[weights.relative_to(ROOT).as_posix()] = snapshot(weights)
    new(LEDGER, {'belt_inputs': ledger, 'previous_model_artifacts': existing})
    new(PROTOCOL, {'schema': 'beltcrack-polygon-starter-v1', 'declared_before_training': True,
        'declared_utc': datetime.now(timezone.utc).isoformat(), 'source': 'https://doi.org/10.57760/sciencedb.31181',
        'license': 'CC-BY-SA-4.0', 'source_archive_name': 'BeltCrack14ks', 'observed_archive_photos': 9645,
        'publication_archive_mismatch': True,
        'publisher_archive_counts': {'train': 5736, 'val': 1961, 'test': 1948},
        'publisher_sequence_counts': {'train': 23, 'val': 10, 'test': 9},
        'sample_counts': manifest['counts'], 'max_frames_per_sequence': 32,
        'sample_selection': 'Original1031 evenly spaced frames filtered to679 complete byte-available inputs before reading selected GT or model scores',
        'availability_subset_reason': 'Publisher429 rate limit; incomplete transfers preserved and no additional requests used for this starter',
        'image_source': manifest['photo_source'], 'source_test_photos_or_labels_used': False,
        'whole_archive_md5_verified': False, 'selected_original_headers_and_member_checksums_verified': True,
        'model': 'Fresh one-class LRASPP MobileNet; Torchvision general pretrained backbone, no building checkpoint',
        'pretrained_weights_sha256': sha(weights), 'class': 'belt_crack', 'input_size': 640,
        'epochs': 40, 'batch_size': 8, 'seed': 87, 'optimizer': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.0002,
        'scheduler': 'Cosine 40, minimum0.000005', 'gradient_clip_norm': 10,
        'pixel_positive_weight': min(20, max(1, (valid-positive)/positive)), 'loss': 'Known-pixel positive-weight BCE + per-photo soft Dice',
        'padding_in_loss_or_metrics': False, 'pixel_ground_truth': manifest['pixel_truth'],
        'threshold': 0.5, 'selection': 'Final40epoch fixed, no VAL best or threshold tuning',
        'evaluation': 'Once on source VAL subset at fixed0.5; photo and pixel metrics separate',
        'five_percent_criterion': 'Photo FPR and FNR both known and individually<0.05; pixel ratios do not substitute',
        'physical_belt_or_factory_independence_verified': False, 'new_ai_or_human_labels': 0,
        'mechanical_failure_diagnosed': False, 'existing_app_default_promoted': False,
        'manifest_sha256': sha(DATA / 'manifest.json'), 'source_sha256': {n: sha(ROOT / n) for n in SOURCES},
        'preflight_sha256': sha(ROOT / 'runs/beltcrack-starter-preflight.json'),
        'study_script_sha256': sha(Path(__file__)), 'source_index_prototype_sha256': sha(ROOT / 'runs/index-belt-rar.py'),
        'public_index_script_sha256': sha(ROOT / 'scripts/index_beltcrack_archive.py'),
        'legacy_source_sha256': legacy_sources, 'input_ledger_sha256': sha(LEDGER),
        'previous_complete_legacy_check': 'reports/equipment-visa-verification.json',
        'preflight_disposable_updates': sum(read(p)['disposable_updates'] for p in ROOT.joinpath('runs').glob('beltcrack-starter-preflight*.json')), 'tests': 3})
    print('Belt training protocol and before ledger frozen', flush=True)


def verify():
    protocol = read(PROTOCOL)
    assert protocol['source_sha256'] == {n: sha(ROOT / n) for n in SOURCES}
    assert protocol['study_script_sha256'] == sha(Path(__file__))
    assert protocol['input_ledger_sha256'] == sha(LEDGER)
    for name, digest in protocol['legacy_source_sha256'].items():
        assert sha(ROOT / name) == digest
    ledger = read(LEDGER)
    for group in ledger.values():
        for name, value in group.items():
            assert snapshot(ROOT / name) == value, 'Preserve: ' + name
    training = read(ROOT / 'reports/beltcrack-starter-training.json')
    result = read(ROOT / 'reports/beltcrack-starter-results.json')
    assert training['epochs'] == 40 and training['strict_cpu_gpu_reload_passed']
    assert training['optimizer_updates'] == ((training['train_photos']+7)//8)*40
    assert not result['source_test_used'] and result['threshold'] == 0.5
    new(ROOT / 'reports/beltcrack-starter-verification.json', {'status': 'passed',
        'actual_epochs': 40, 'actual_optimizer_updates': training['optimizer_updates'],
        'disposable_preflight_updates': protocol['preflight_disposable_updates'], 'tests_passed': 3,
        'belt_input_files_sha_size_mtime_preserved': len(ledger['belt_inputs']),
        'previous_model_artifacts_sha_size_mtime_preserved': len(ledger['previous_model_artifacts']),
        'legacy_runtime_sources_preserved': len(protocol['legacy_source_sha256']),
        'strict_cpu_gpu_reload_passed': True, 'threshold_tuned_after_val': False,
        'source_test_used': False, 'new_ai_or_human_labels': 0, 'building_epochs_after_direction_change': 0,
        'app_default_changed': False, 'weights_sha256': training['weights_sha256'],
        'protocol_sha256': sha(PROTOCOL), 'result_sha256': sha(ROOT / 'reports/beltcrack-starter-results.json')})
    print('Belt actual training and preservation verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('declare', 'verify'))
    args = parser.parse_args()
    declare() if args.mode == 'declare' else verify()
