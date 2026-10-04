"""Verify completed paired pooling checkpoints and their frozen technical record.

This is a CPU shape/provenance check, not another validation or test run. Public
output contains aggregate evidence and source-code hashes, never photo paths.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH as AUX_ARCH
from safelog_ai.context_classifier import ContextClassifier, ARCH as CONTEXT_ARCH
from safelog_ai.presence_classifier import build_model, MEAN, STD
from scripts.train_facility_target import read, save, sha
from scripts.preflight_facility_context import SOURCE_PATHS

PROTOCOL = ROOT / 'reports/facility-pool-context-protocol.json'
PREFLIGHT = ROOT / 'runs/facility-context-preflight.json'
FREEZE = ROOT / 'runs/facility-context-source-freeze.json'
TESTS = ROOT / 'runs/facility-context-test-verification.json'
OUTPUT = ROOT / 'reports/facility-pool-context-technical-verification.json'
REQUIRED_FROZEN_SOURCES = set(SOURCE_PATHS.values()) | {
    'reports/facility-pool-context-protocol.json',
    'scripts/preflight_facility_context.py',
    'scripts/preflight_facility_detail.py',
}
HEX64 = re.compile(r'[0-9a-f]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_snapshot():
    freeze = read(FREEZE)
    commit = freeze.get('pre_training_commit', '')
    require(isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{7,40}', commit),
            'Source freeze needs an actual pre-training Git commit')
    require(freeze.get('comparison_started_before_outcomes') is True,
            'Source snapshot was not recorded before training outcomes')
    sources = freeze.get('sources', {})
    require(isinstance(sources, dict) and REQUIRED_FROZEN_SOURCES <= set(sources),
            'Pre-training freeze omits a required trainer, model, protocol or preflight source')
    repo = ROOT.parents[1]
    public_sources = {}
    for relative, proof in sources.items():
        require(isinstance(relative, str) and '\\' not in relative,
                'Frozen source path is not a canonical relative code path')
        path = PurePosixPath(relative)
        require(not path.is_absolute() and '..' not in path.parts and bool(path.parts)
                and path.parts[0] in ('scripts', 'safelog_ai', 'tests', 'reports')
                and path.suffix in ('.py', '.json'), 'Frozen source path is outside the code/report scope')
        require(isinstance(proof, dict) and proof.get('equal') is True,
                'Frozen Git and executed source bytes were not equal')
        current = sha(ROOT / relative)
        require(proof.get('git_blob_sha256') == current
                and proof.get('executed_file_sha256') == current,
                f'Pre-training source changed: {relative}')
        git_relative = (ROOT.relative_to(repo) / Path(*path.parts)).as_posix()
        blob = subprocess.run(['git', 'show', f'{commit}:{git_relative}'], cwd=repo,
                              check=True, capture_output=True).stdout
        require(hashlib.sha256(blob).hexdigest() == current,
                f'Recorded source is not present unchanged in the frozen Git commit: {relative}')
        public_sources[relative] = {'git_blob_sha256': current,
                                   'executed_file_sha256': current, 'equal': True}
    return {'pre_training_commit': commit, 'sources': public_sources,
            'comparison_started_before_outcomes': True}


def preflight_proof(protocol):
    value = read(PREFLIGHT)
    for flag in ('initial_fp32_outputs_equal', 'constructor_cpu_rng_equal',
                 'augmented_batch_replay_equal', 'finite_loss_and_gradients',
                 'learned_gate_gradient_nonzero', 'source_checkpoint_unchanged', 'frozen_inputs_unchanged'):
        require(value.get(flag) is True, f'Actual GPU preflight is incomplete: {flag}')
    require(value.get('status') == 'passed' and value.get('schema') == 'facility_context_preflight_v1'
            and value.get('checkpoint_saved') is False, 'Disposable preflight did not pass')
    require(value.get('train_photo_count') == 8
            and value.get('train_joint_tag_counts') == {s: 2 for s in ('00', '10', '01', '11')}
            and value.get('backbone_forward_calls') == 1, 'Native TRAIN or one-backbone proof changed')
    require(value.get('public_shape') == [8, 7] and value.get('pixel_shape') == [8, 7, 80, 80]
            and value.get('auxiliary_shape') == [8, 19], 'Preflight output shapes changed')
    require(value.get('replay_num_workers') == 4 and value.get('replay_batches') == 3
            and value.get('loader_randomness') == protocol['loader_randomness'], 'Bounded augmentation replay changed')
    require(value.get('preflight_script_sha256') == sha(ROOT / 'scripts/preflight_facility_context.py')
            and value.get('reused_preflight_helper_sha256') == sha(ROOT / 'scripts/preflight_facility_detail.py'),
            'Executed preflight source changed')
    for field, path in (('study_protocol_sha256', PROTOCOL),
                        ('initial_weights_sha256', ROOT / 'runs' / protocol['reference'] / 'best.pt'),
                        ('core_spatial_manifest_sha256', ROOT / 'data/facility-spatial-training/train.json'),
                        ('auxiliary_manifest_sha256', ROOT / 'data/facility-auxiliary-training/train.json')):
        require(value.get(field) == sha(path), 'Preflight frozen input changed')
    for key, relative in SOURCE_PATHS.items():
        require(value.get('executed_sources_sha256', {}).get(key) == sha(ROOT / relative),
                f'Executed preflight model/trainer source changed: {key}')
    transfer = value.get('initial_state_transfer', {})
    require(transfer.get('control', {}).get('shared_state_tensors_equal') is True
            and transfer.get('control', {}).get('new_state_tensor_count') == 0
            and transfer.get('context', {}).get('shared_state_tensors_equal') is True
            and transfer.get('context', {}).get('new_state_tensor_count') == 1
            and transfer.get('context', {}).get('new_output_projection_zero') is True,
            'Exact shared state and zero gate proof missing')
    steps = value.get('disposable_optimizer_steps', [])
    require(len(steps) == 2 and [v.get('step') for v in steps] == [1, 2]
            and all(v.get('all_gradients_finite') is True and v.get('gate_gradient_nonzero') is True
                    and v.get('learned_gate_nonzero') is True for v in steps), 'Gate learning proof missing')
    keys = ('status', 'schema', 'train_photo_count', 'train_joint_tag_counts',
            'public_shape', 'pixel_shape', 'auxiliary_shape', 'backbone_forward_calls',
            'initial_fp32_outputs_equal', 'initial_state_transfer', 'constructor_cpu_rng_equal', 'loader_randomness',
            'replay_num_workers', 'replay_batches', 'augmented_batch_replay_equal', 'bounded_replay_digests',
            'finite_loss_and_gradients', 'learned_gate_gradient_nonzero', 'disposable_optimizer_steps',
            'amp_initial_scale', 'checkpoint_saved', 'source_checkpoint_unchanged', 'frozen_inputs_unchanged',
            'model_shapes_and_parameters', 'latency_diagnostic', 'amp_two_step_peak_allocated_bytes',
            'python', 'torch', 'device')
    result = {key: value[key] for key in keys}
    result['scope'] = 'Eight original TRAIN DACL photos; disposable GPU integration and bounded augmentation replay only'
    result['full_epoch_sampling_proved_by_this_replay'] = False
    result['accuracy_measured_by_this_check'] = False
    return result


def completed_run(name, variant, protocol):
    require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_-]+', name), 'Invalid declared run name')
    run = ROOT / 'runs' / name
    training = read(run / 'TRAINING.json')
    history = read(run / 'history.json')
    weights = run / 'best.pt'
    architecture = AUX_ARCH if variant == 'control' else CONTEXT_ARCH
    require(training.get('status') == 'complete' and training.get('model_variant') == variant
            and training.get('architecture') == architecture, 'Wrong or incomplete architecture run')
    require(training.get('weights_sha256') == sha(weights)
            and training.get('study_protocol_sha256') == sha(PROTOCOL), 'Run weights or declared protocol changed')
    for key in ('classes', 'seed', 'requested_epochs', 'patience', 'imgsz', 'batch_size',
                'draws_per_epoch', 'backbone_lr', 'head_lr', 'auxiliary_weight',
                'domain_proportions', 'loader_randomness'):
        require(training.get(key) == protocol[key], f'Executed paired condition changed: {key}')
    epochs = protocol['requested_epochs']
    require(training.get('actual_epochs') == epochs and isinstance(history, list)
            and len(history) == epochs and [row.get('epoch') for row in history] == list(range(1, epochs + 1)),
            'Paired fixed epoch budget was not completed')
    require(training.get('initial_weights_sha256') == protocol['initial_weights_sha256']
            and training.get('spatial_manifest_path') == 'data/facility-spatial-training/train.json'
            and training.get('spatial_manifest_sha256') == protocol['core_spatial_manifest_sha256']
            and training.get('core_spatial_manifest_sha256') == protocol['core_spatial_manifest_sha256']
            and training.get('auxiliary_manifest_sha256') == protocol['auxiliary_manifest_sha256'],
            'Run changed the initializer or original core/auxiliary data')
    require(training.get('target_ranking', {}).get('weight') == 0.
            and not any(key in training for key in ('photo_supplement_sha256',
                                                   'supplement_spatial_sha256', 'hard_training_sampling')),
            'An undeclared data, ranking or mining intervention was used')
    for key, relative in SOURCE_PATHS.items():
        require(training.get(key) == sha(ROOT / relative), f'Executed training source changed: {key}')
    transfer = training.get('initial_state_transfer', {})
    require(transfer.get('shared_state_tensors_equal') is True
            and (transfer.get('new_state_tensor_count') == 0 if variant == 'control'
                 else transfer.get('new_state_tensor_count', 0) > 0
                 and transfer.get('new_output_projection_zero') is True),
            'Exact shared initializer transfer was not recorded')
    require(training.get('context_architecture') == (None if variant == 'control' else protocol['context_architecture']),
            'Run architecture recipe differs from its declaration')
    require(training['split_sha256'] == sha(run / 'SPLIT.json'), 'Source group split bytes changed')
    for key, relative in (('additional_validation', 'data/codebrim-training/val.json'),
                          ('additional_test', 'data/codebrim-training/test.json')):
        require(training.get(key) == {'path': relative, 'sha256': sha(ROOT / relative)},
                'Frozen CODEBRIM holdout metadata changed')
    for row in history:
        digest = row.get('sampled_row_indices_sha256', '')
        require(isinstance(digest, str) and HEX64.fullmatch(digest), 'Actual sampled-row digest missing')
        require(sum(row['sampled_domain_counts'].values()) == protocol['draws_per_epoch']
                and sum(row['sampled_row_type_counts'].values()) == protocol['draws_per_epoch'],
                'Executed sampling budget differs from declaration')
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    require(checkpoint.get('architecture') == architecture
            and checkpoint.get('classes') == protocol['classes']
            and checkpoint.get('auxiliary_classes') == list(AUX_CLASSES)
            and checkpoint.get('imgsz') == 640
            and checkpoint.get('mean') == MEAN and checkpoint.get('std') == STD
            and checkpoint.get('selection_split') == 'val'
            and checkpoint.get('split_sha256') == training['split_sha256'],
            'Completed checkpoint public/auxiliary/input/source contract changed')
    selected = checkpoint.get('epoch')
    require(type(selected) is int and 1 <= selected <= epochs, 'Checkpoint selection epoch is invalid')
    best = history[selected - 1]
    require(read(run / 'VALIDATION.json') == best
            and checkpoint.get('worst_target_error') == best['worst_target_error']
            and training.get('best_worst_target_error') == best['worst_target_error'],
            'Selected checkpoint and recorded epoch outcome disagree')
    state = checkpoint['state_dict']
    require(all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                for value in state.values()), 'Checkpoint contains nonfinite or non-tensor states')
    model = build_model(7, pretrained=False, architecture=architecture).cpu().eval()
    require(type(model) is (AuxiliaryClassifier if variant == 'control' else ContextClassifier),
            'Model factory did not return the declared architecture')
    model.load_state_dict(state, strict=True)
    calls = []
    if variant == 'context':
        require('context_gate' in state and state['context_gate'].shape == (7,)
                and bool(torch.count_nonzero(model.context_gate)), 'Treatment has no learned seven-class pooling gates')
    hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    with torch.inference_mode():
        inputs = torch.zeros(1, 3, 640, 640)
        scores, maps, auxiliary = model.forward_training(inputs)
    hook.remove()
    require(len(calls) == 1, 'Completed model did not use a single backbone pass')
    with torch.inference_mode():
        public = model(inputs)
    require(public.shape == (1, 7) and maps.shape == (1, 7, 80, 80)
            and auxiliary.shape == (1, 19) and torch.equal(public, scores)
            and all(bool(torch.isfinite(value).all()) for value in (public, maps, auxiliary)),
            'Strict CPU finite public/map/auxiliary inference contract failed')
    entry = {'run': name, 'architecture': architecture, 'model_variant': variant,
             'weights_sha256': sha(weights), 'history_sha256': sha(run / 'history.json'),
             'training_metadata_sha256': sha(run / 'TRAINING.json'),
             'public_shape': list(public.shape), 'private_spatial_shape': list(maps.shape),
             'auxiliary_shape': list(auxiliary.shape), 'outputs_finite': True,
             'public_training_logits_equal': True, 'strict_state_load': True,
             'parameter_count': sum(p.numel() for p in model.parameters()),
             'study_protocol_sha256': training['study_protocol_sha256'],
             'selected_epoch': selected, 'actual_epochs': epochs,
             'source_split_sha256': training['split_sha256'],
             'learned_context_gate_values': model.context_gate.detach().cpu().tolist() if variant == 'context' else None,
             'learned_context_gate_tanh': model.context_gate.tanh().detach().cpu().tolist() if variant == 'context' else None}
    return entry, training, history


def run():
    torch.set_num_threads(4)
    protocol = read(PROTOCOL)
    require(protocol.get('schema') == 'facility_pool_context_protocol_v1'
            and protocol.get('declared_before_training') is True
            and protocol.get('control_architecture') == AUX_ARCH
            and protocol.get('treatment_architecture') == CONTEXT_ARCH and AUX_ARCH != CONTEXT_ARCH,
            'Architecture comparison declaration changed')
    require(len(protocol['classes']) == 7 and protocol['imgsz'] == 640,
            'Seven-class public input contract changed')
    require(sha(ROOT / 'data/facility-spatial-training/train.json') == protocol['core_spatial_manifest_sha256']
            and sha(ROOT / 'data/facility-auxiliary-training/train.json') == protocol['auxiliary_manifest_sha256'],
            'Frozen original TRAIN metadata changed')
    reference = ROOT / 'runs' / protocol['reference'] / 'best.pt'
    require(sha(reference) == protocol['initial_weights_sha256'], 'Original initializer bytes changed')
    initial = torch.load(reference, map_location='cpu', weights_only=True)
    require(initial['architecture'] == AUX_ARCH and initial['classes'] == protocol['classes']
            and initial['auxiliary_classes'] == list(AUX_CLASSES), 'Original initializer contract changed')
    freeze = source_snapshot()
    preflight = preflight_proof(protocol)
    tests = read(TESTS)
    require(tests.get('status') == 'passed' and tests.get('failures') == 0 and tests.get('errors') == 0
            and type(tests.get('tests_run')) is int and tests['tests_run'] >= 149,
            'Actual recorded full test suite did not pass')
    profile = ROOT / 'reports/facility-inference-profile.json'
    require(profile.read_bytes() == (ROOT / 'reports/facility-inference-profile-round1.json').read_bytes(),
            'Deployed app profile changed during research')
    require(read(PREFLIGHT)['app_profile_sha256'] == sha(profile), 'App profile changed after preflight')
    values = [completed_run(protocol[key], variant, protocol)
              for key, variant in (('control', 'control'), ('treatment', 'context'))]
    entries = [value[0] for value in values]
    control, treatment = values[0][1], values[1][1]
    for key in ('expected_sampling', 'expected_label_sampling', 'photo_positive_weights',
                'pixel_positive_weights', 'auxiliary_positive_weights', 'split_sha256',
                'train_dacl', 'train_damsegment', 'train_codebrim', 'val_dacl', 'val_damsegment',
                'validation_domains', 'spatial_manifest_audit', 'auxiliary_audit'):
        require(key in control and key in treatment and control[key] == treatment[key],
                f'Paired data/sampling/loss condition differs: {key}')
    require(control['split_sha256'] == initial['split_sha256'], 'Initializer source-group split differs from this pair')
    require(sum(control[key] for key in ('train_dacl', 'train_damsegment', 'train_codebrim')) == 14248,
            'Original full-photo row count changed')
    for first, second in zip(values[0][2], values[1][2]):
        for key in ('sampled_row_indices_sha256', 'sampled_domain_counts',
                    'sampled_row_type_counts', 'sampled_full_target_joint_counts'):
            require(first[key] == second[key], f'Actual paired epoch sampling differs: {key}')
    require(entries[1]['parameter_count'] - entries[0]['parameter_count'] == 7,
            'Treatment is not the declared seven-parameter pooling contrast')
    return {'status': 'passed', 'schema': 'facility_pool_context_technical_verification_v1',
            'experiments': entries, 'python': platform.python_version(), 'torch': torch.__version__,
            'device': 'cpu', 'check_input': 'One zero 1x3x640x640 tensor per completed checkpoint; shape check only',
            'actual_train_gpu_preflight': preflight, 'actual_train_gpu_preflight_sha256': sha(PREFLIGHT),
            'pre_training_source_snapshot': freeze, 'pre_training_source_snapshot_sha256': sha(FREEZE),
            'actual_test_verification': {key: tests[key] for key in ('status', 'tests_run', 'failures', 'errors')},
            'actual_test_verification_sha256': sha(TESTS),
            'executed_sources_sha256': {key: sha(ROOT / relative) for key, relative in SOURCE_PATHS.items()},
            'paired_sampled_row_indices_equal': True, 'paired_sampling_epochs': protocol['requested_epochs'],
            'paired_augmentation_replay_equal': True,
            'augmentation_replay_scope': 'Three batches over eight original TRAIN photos with four workers; configured RNG isolation verified. Full six-epoch ordered sample indices are checked separately; no full-epoch augmented tensor hashes asserted.',
            'different_new_architecture_confirmed': True,
            'new_context_parameter_count': 7, 'study_protocol_sha256': sha(PROTOCOL),
            'core_spatial_manifest_sha256': protocol['core_spatial_manifest_sha256'],
            'auxiliary_manifest_sha256': protocol['auxiliary_manifest_sha256'],
            'app_profile_sha256': sha(profile), 'app_profile_unchanged': True,
            'research_models_deployed': False, 'accuracy_measured_by_this_check': False,
            'independent_field_safety_verified': False, 'verification_script_sha256': sha(Path(__file__))}


def main():
    try:
        result = run()
    except Exception as error:
        save(OUTPUT, {'status': 'failed', 'schema': 'facility_pool_context_technical_verification_v1',
                      'error_type': type(error).__name__, 'accuracy_measured_by_this_check': False})
        raise
    save(OUTPUT, result)
    print(json.dumps({'status': 'passed', 'checkpoints': len(result['experiments']),
                      'public_outputs': 7, 'paired_sampling_equal': True, 'accuracy_measured': False}))


if __name__ == '__main__':
    main()
