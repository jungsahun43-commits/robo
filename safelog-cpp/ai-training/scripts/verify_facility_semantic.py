"""Verify completed fixed other-class retention TRAIN runs on CPU; no held-out inference.

Local hashes prove consistency with recorded inputs, not publisher authenticity,
independent truth or accuracy on industrial sites.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AUX_CLASSES, AuxiliaryClassifier
from safelog_ai.frozen_batchnorm import batchnorm_inventory, batchnorm_state_sha256
from safelog_ai.presence_classifier import MEAN, STD
from safelog_ai.spalling_sampler import draw_sha256
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.semantic_residual_classifier import (SemanticResidualClassifier, model_inventory,
    semantic_encoder_state_sha256)
from safelog_ai.research_presence import ResearchPresenceClassifier
from scripts.facility_semantic_study import (PROTOCOL, SOURCE_FILES, MIN_TEST_COUNTS,
    protected_hashes, validate_protocol, write)
from scripts.preflight_facility_semantic import LEDGER, PREVIOUS_LEDGER, REUSED_PREFLIGHT, input_paths, snapshot_inputs
from scripts.verify_facility_head_lr import validate_run_metadata as validate_old_control_metadata
from scripts.preflight_facility_detail import select_train_replay
from scripts.verify_facility_resolution import (git_blob, hash_map, local_path, read,
    require, sha, verify_cpu_reload)

RUNTIME_SOURCES = tuple(SOURCE_FILES)
TEST_SOURCES = tuple(MIN_TEST_COUNTS)
PROTOCOL_PATH = PROTOCOL
PREFLIGHT_PATH = 'runs/facility-semantic-preflight.json'
SOURCE_RECORD_PATH = 'runs/facility-semantic-source-before-training.json'
OUTPUT_PATH = 'reports/facility-semantic-study-verification.json'
_SHA = re.compile(r'[a-f0-9]{64}\Z')


def equal_fields(record, expected, message):
    for key, value in expected.items():
        require(type(record.get(key)) is type(value) and record[key] == value, f'{message}: {key}')


def verify_frozen_sources(snapshot, protocol, preflight, root, protocol_sha, preflight_sha, blob_reader=None):
    equal_fields(snapshot, {'schema': 'facility_semantic_source_before_training_v1',
        'declared_before_training': True, 'new_completed_training_epochs_at_snapshot': 0},
        'Required before-training source snapshot differs')
    commit = snapshot.get('source_git_commit')
    require(isinstance(commit, str) and re.fullmatch(r'[a-f0-9]{40}', commit), 'A full source Git commit is required')
    for record in (snapshot, protocol, preflight): hash_map(record.get('source_sha256'), RUNTIME_SOURCES)
    sources = protocol['source_sha256']
    require(snapshot['source_sha256'] == preflight['source_sha256'] == sources,
            'Source snapshot/runtime inventories differ')
    require(snapshot.get('protocol_sha256') == preflight.get('protocol_sha256') == protocol_sha
            and snapshot.get('preflight_sha256') == preflight_sha,
            'Source snapshot protocol/preflight binding differs')
    reader = blob_reader or (lambda c, p: git_blob(root, c, p))
    for relative, digest in sources.items():
        require(sha(local_path(root, relative)) == digest, 'Runtime bytes changed after declaration')
        blob = reader(commit, relative)
        require(isinstance(blob, bytes) and hashlib.sha256(blob).hexdigest() == digest,
                'Committed runtime source bytes differ')
    return {'source_git_commit': commit, 'runtime_source_count': len(sources), 'source_sha256': dict(sources),
            'git_blob_bytes_verified': True, 'working_runtime_sources_unchanged': True}


def validate_preflight(record, protocol, protocol_sha, protected):
    equal_fields(record, {'schema': 'facility_semantic_preflight_v1', 'status': 'passed',
        'protocol_sha256': protocol_sha, 'unique_paired_draw_positions': 8,
        'changed_replay_positions': 0, 'candidate_teacher_unchanged_eval_frozen_no_grad': True,
        'candidate_original_batchnorm_buffers_preserved_affine_learnable': True,
        'actual_disposable_optimizer_steps': 1, 'new_candidate_count': 1, 'control_training_repeated': False,
        'reused_control_preflight_path': REUSED_PREFLIGHT, 'reused_control_preflight_status': 'passed',
        'same_original_replay_positions_as_reused_control': True,
        'previous_inputs_preserved': True,
        'all_original_sha_size_mtime_preserved': True, 'new_training_epochs': 0,
        'source_test_inference_executed': False, 'app_model_promoted': False,
        'accuracy_measured': False, 'label_changes': 0, 'new_masks': 0}, 'Actual preflight differs')
    require(record.get('source_sha256') == protocol['source_sha256']
            and record.get('protected_file_sha256') == protected, 'Preflight source/protected bytes changed')
    require(type(record.get('protected_input_file_count')) is int and record['protected_input_file_count'] > 0
            and type(record.get('previous_input_file_count')) is int and record['previous_input_file_count'] == 58995,
            'Actual original input preservation proof is absent')
    require(record.get('protected_input_ledger_path') == LEDGER
            and isinstance(record.get('protected_input_ledger_sha256'), str)
            and _SHA.fullmatch(record['protected_input_ledger_sha256']), 'Protected private input ledger is unbound')
    require(isinstance(record.get('reused_control_preflight_sha256'), str)
            and _SHA.fullmatch(record['reused_control_preflight_sha256']), 'Reused control preflight is unbound')
    require(set(record.get('candidates', {})) == {'semantic'}, 'Only one new candidate preflight is allowed')
    for variant, arm in record['candidates'].items():
        equal_fields(arm, {'actual_batch_size': 8, 'actual_optimizer_steps': 1, 'amp_skipped_steps': 0,
            'public_shape': [8, 7], 'pixel_shape': [8, 7, 80, 80], 'auxiliary_shape': [8, 19],
            'all_outputs_finite': True, 'cpu_reload_verified': True,
            'strict_factory_reload_verified': True, 'public_output_equals_training_photo_output': True},
            'Actual CUDA update/shape/strict CPU reload differs')
        inventory = validate_semantic_inventory(arm.get('semantic_model_inventory', {}), initial_heads_zero=True)
        require(arm.get('parameter_count') == inventory['parameter_count'], 'Measured semantic parameter count differs')
        validate_original_transfer(arm.get('original_state_transfer', {}), inventory)
        validate_pretrained_transfer(arm.get('semantic_pretrained_transfer', {}), protocol, inventory)
        validate_frozen_semantic(arm.get('frozen_semantic_state_preservation', {}),
            protocol['semantic_encoder_state_sha256'], 1)
        equal_fields(arm, {'zero_initial_outputs_identical_to_original': True,
            'original_state_tensors_exactly_equal': 324, 'semantic_head_gradient_tensors': 6,
            'semantic_head_gradient_nonzero_tensors': 6, 'semantic_head_gradients_finite': True,
            'actual_semantic_encoder_forward_eval_verified': True}, 'Actual semantic initial identity/gradient proof differs')
        validate_forward_latency(arm.get('actual_cuda_forward_latency', {}))
        equal_fields(arm, {'teacher_forward_batches': 1, 'actual_distillation_gradient_mask_verified': True,
            'teacher_and_student_same_augmented_image_tensor': True,
            'distillation_weight': protocol['distillation_weight_by_variant'][variant], 'temperature': 2.},
            'Actual disposable masked-loss/teacher compute proof differs')
        validate_teacher_preservation(arm.get('teacher_state_preservation', {}),
            protocol['teacher_state_sha256'], protocol['initial_weights_sha256'], inventory['parameter_count'])
        validate_batchnorm_preservation(arm.get('batchnorm_state_preservation', {}),
            protocol['initial_batchnorm_buffers_sha256'])
        equal_fields(arm, {'actual_bn_forward_eval_layers': 47, 'bn_affine_gradient_tensors': 94,
            'bn_affine_gradients_finite': True, 'bn_buffer_cpu_reload_verified': True},
            'Actual BN forward/gradient/CPU proof differs')
        equal_fields(arm, {'optimizer_learning_rates': {'backbone': .00004, 'head': .0001},
            'optimizer_weight_decay': {'backbone': .0002, 'head': .0002}},
            'Actual one-batch lower head-LR optimizer differs')
        require(type(arm.get('bn_affine_gradient_nonzero_tensors')) is int
                and 0 < arm['bn_affine_gradient_nonzero_tensors'] <= 94,
                'Actual learnable BN affine gradient proof is absent')
        require(type(arm.get('known_other_class_entries')) is int and arm['known_other_class_entries'] > 0
                and type(arm.get('unweighted_distillation_loss')) in (int, float)
                and math.isfinite(arm['unweighted_distillation_loss']) and arm['unweighted_distillation_loss'] >= 0,
                'Actual masked distillation measurement is absent')
        require(type(arm.get('peak_cuda_allocated_bytes')) is int and arm['peak_cuda_allocated_bytes'] > 0
                and isinstance(arm.get('disposable_checkpoint_sha256'), str)
                and _SHA.fullmatch(arm['disposable_checkpoint_sha256']), 'Measured disposable evidence is absent')
    return {'passed': True, 'actual_candidate_train_positions': 8, 'parameter_count': inventory['parameter_count'],
            'semantic_model_inventory': inventory, 'zero_initial_outputs_identical_to_original': True,
            'actual_cuda_forward_latency': dict(arm['actual_cuda_forward_latency']),
            'disposable_amp_optimizer_steps': 1, 'reused_control_preflight': True,
            'candidate_teacher_unchanged_eval_frozen_no_grad': True,
            'actual_distillation_gradient_mask_verified': True}


def validate_test_counts(record):
    require(type(record.get('tests_run')) is int and record['tests_run'] >= sum(MIN_TEST_COUNTS.values())
            and all(type(record.get(k)) is int and record[k] == 0 for k in ('failures', 'errors', 'skipped')),
            'All focused tests must actually pass without skips')
    counts = record.get('tests_by_module', {})
    require(isinstance(counts, dict) and set(counts) == set(MIN_TEST_COUNTS)
            and all(type(counts[k]) is int and counts[k] >= n for k, n in MIN_TEST_COUNTS.items())
            and sum(counts.values()) == record['tests_run'] == record.get('expected_tests_collected'),
            'Full focused-suite collection differs from actual executed tests')
    return {k: record[k] for k in ('tests_run', 'failures', 'errors', 'skipped', 'tests_by_module', 'expected_tests_collected')}


def validate_test_results(record, root):
    equal_fields(record, {'schema': 'facility_semantic_test_results_v1', 'status': 'passed',
                         'before_after_sources_equal': True}, 'Actual focused test record differs')
    counts = validate_test_counts(record)
    hash_map(record.get('test_source_sha256'), TEST_SOURCES)
    hash_map(record.get('source_sha256'), RUNTIME_SOURCES)
    for relative, expected in {**record['test_source_sha256'], **record['source_sha256']}.items():
        require(sha(local_path(root, relative)) == expected, 'Executed test/runtime bytes changed')
    return {**counts, 'test_source_sha256': dict(record['test_source_sha256']), 'source_sha256': dict(record['source_sha256'])}


def expected_counts(indices, items, full):
    domains = {d: 0 for d in ('dacl', 'damsegment', 'codebrim')}
    row_types = {'full': 0, 'crop': 0}
    joint = {d: {s: 0 for s in ('00', '10', '01', '11', 'unknown')} for d in domains}
    classes = ['concrete_crack', 'concrete_spalling', 'rust_stain', 'exposed_rebar',
               'wet_surface', 'efflorescence', 'surface_cavity']
    counts = {c: {'positive': 0, 'negative': 0, 'unknown': 0} for c in classes}
    for i in indices:
        i = int(i); row = items[i]; domain = row['domain']; domains[domain] += 1
        kind = 'full' if i < full else 'crop'; row_types[kind] += 1
        for label, value in zip(classes, row['targets']):
            counts[label]['unknown' if value < 0 else 'positive' if value == 1 else 'negative'] += 1
        if kind == 'full':
            crack, spall = row['targets'][:2]
            state = f'{crack}{spall}' if crack >= 0 and spall >= 0 else 'unknown'
            joint[domain][state] += 1
    return {'sampled_domain_counts': domains, 'sampled_row_type_counts': row_types,
            'sampled_full_target_joint_counts': joint, 'sampled_row_indices_sha256': draw_sha256(indices),
            'sampled_photo_target_counts': counts}


def validate_teacher_preservation(record, initial_hash, initial_weights_sha, student_parameters=3244151):
    equal_fields(record, {'initial_state_sha256': initial_hash, 'final_state_sha256': initial_hash,
        'initial_weights_sha256': initial_weights_sha, 'unchanged': True, 'eval_mode': True,
        'all_parameters_frozen': True, 'no_parameter_gradients': True, 'student_parameter_count': student_parameters,
        'teacher_parameter_count': 3244151, 'teacher_state_tensor_count': 324,
        'teacher_forward_both_arms': True}, 'Actual frozen teacher preservation differs')
    return dict(record)


def validate_semantic_inventory(record, initial_heads_zero):
    equal_fields(record, {'original_state_tensor_count': 324, 'new_head_state_tensor_count': 6,
        'new_semantic_head_parameter_count': 21345, 'all_encoder_parameters_frozen': True,
        'encoder_eval': True, 'all_encoder_modules_eval': True, 'new_semantic_heads_trainable': True,
        'new_semantic_heads_zero': initial_heads_zero, 'low_channels': 192, 'pooled_channels': 768,
        'low_feature_grid_for_640': [80, 80]}, 'Semantic model inventory differs')
    for key in ('parameter_count', 'trainable_parameter_count', 'frozen_encoder_parameter_count',
                'state_tensor_count', 'encoder_state_tensor_count'):
        require(type(record.get(key)) is int and record[key] > 0, 'Actual semantic state/parameter inventory is absent')
    require(record['parameter_count'] == 3244151 + record['frozen_encoder_parameter_count'] + 21345
            and record['trainable_parameter_count'] == 3244151 + 21345
            and record['state_tensor_count'] == 324 + record['encoder_state_tensor_count'] + 6,
            'Original, frozen encoder and new head inventory totals differ')
    return dict(record)


def validate_original_transfer(record, inventory):
    equal_fields(record, {'shared_state_tensors_equal': True, 'shared_state_tensor_count': 324,
        'original_state_tensor_count': 324, 'strict_state_load': True,
        'new_state_tensor_count': inventory['state_tensor_count'] - 324,
        'new_semantic_head_state_tensor_count': 6, 'new_semantic_heads_zero': True,
        'additional_trainable_parameters': 21345, 'full_original_model_state_preserved': True},
        'Original 324-state transfer or zero semantic initializer differs')


def validate_pretrained_transfer(record, protocol, inventory):
    equal_fields(record, {'weights_sha256': protocol['semantic_pretrained_weights_sha256'],
        'encoder_state_sha256': protocol['semantic_encoder_state_sha256'],
        'encoder_state_tensor_count': inventory['encoder_state_tensor_count'],
        'encoder_parameter_count': inventory['frozen_encoder_parameter_count'], 'strict_encoder_load': True,
        'official_pooling_norm_transferred': True,
        'discarded_image_net_classifier_tensors': ['classifier.2.weight', 'classifier.2.bias'],
        'all_encoder_parameters_frozen': True, 'encoder_eval': True, 'new_semantic_heads_zero': True},
        'Declared local official encoder transfer differs')


def validate_frozen_semantic(record, encoder_sha, forward_batches):
    equal_fields(record, {'initial_encoder_state_sha256': encoder_sha, 'final_encoder_state_sha256': encoder_sha,
        'encoder_state_unchanged': True, 'encoder_all_eval': True, 'encoder_parameters_frozen': True,
        'encoder_gradients_absent': True, 'head_parameters_trainable': True,
        'semantic_heads_changed_from_zero': {'map': True, 'photo': True, 'aux': True},
        'encoder_forward_batches': forward_batches}, 'Frozen semantic encoder or learned head proof differs')
    return dict(record)


def validate_semantic_epoch(record, encoder_sha, batches):
    equal_fields(record, {'encoder_state_sha256': encoder_sha, 'encoder_state_unchanged': True,
        'encoder_all_eval': True, 'encoder_parameters_frozen': True, 'encoder_gradients_absent': True,
        'semantic_head_gradient_tensors': 6, 'semantic_head_gradients_finite': True,
        'semantic_heads_changed_from_zero': {'map': True, 'photo': True, 'aux': True},
        'encoder_forward_batches': batches}, 'Actual semantic TRAIN epoch proof differs')
    require(type(record.get('semantic_head_gradient_nonzero_tensors')) is int
            and 0 < record['semantic_head_gradient_nonzero_tensors'] <= 6,
            'Actual semantic head gradient proof lacks a measured nonzero tensor')


def validate_forward_latency(record):
    equal_fields(record, {'device': 'cuda', 'dtype': 'torch.float32', 'batch_size': 1,
        'input_shape': [1, 3, 640, 640], 'warmup_forwards_each': 3, 'timed_forwards_each': 10,
        'same_input_tensor_both_models': True, 'transfer_excluded': True, 'model_forward_only': True,
        'accuracy_measured': False, 'android_latency_measured': False}, 'Actual forward timing scope differs')
    require(isinstance(record.get('device_name'), str) and record['device_name']
            and isinstance(record.get('torch_version'), str) and record['torch_version']
            and isinstance(record.get('input_tensor_sha256'), str) and _SHA.fullmatch(record['input_tensor_sha256']),
            'Actual CUDA timing device/version/input digest is absent')
    for name in ('original', 'semantic'):
        row = record.get(name, {})
        require(set(row) == {'mean_ms', 'median_ms', 'min_ms', 'max_ms'}
                and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in row.values())
                and row['min_ms'] <= row['median_ms'] <= row['max_ms']
                and row['min_ms'] <= row['mean_ms'] <= row['max_ms'],
                'Actual positive finite CUDA timing statistics are inconsistent')
    return dict(record)


def validate_retention_audit(record, weight, expected_batches, known_entries, contributing_batches):
    equal_fields(record, {'weight': weight, 'temperature': 2., 'teacher_forward_batches': expected_batches,
        'contributing_batches': contributing_batches, 'known_other_class_entries': known_entries,
        'excluded_primary_classes': ['concrete_crack', 'concrete_spalling'],
        'teacher_eval': True, 'teacher_frozen': True, 'teacher_state_unchanged': True,
        'teacher_gradients_absent': True}, 'Actual retention loss/teacher audit differs')
    loss = record.get('unweighted_mean_batch_loss'); weighted = record.get('weighted_mean_batch_loss')
    require(type(loss) in (float, int) and math.isfinite(loss) and loss >= 0
            and type(weighted) in (float, int) and math.isfinite(weighted) and weighted == weight * loss,
            'Actual measured teacher loss or weighted mean differs')


def validate_batchnorm_preservation(record, initial_buffers_sha):
    equal_fields(record, {'initial_buffers_sha256': initial_buffers_sha,
        'final_buffers_sha256': initial_buffers_sha, 'buffers_unchanged': True,
        'layer_count': 47, 'channel_count': 12328, 'buffer_tensor_count': 141,
        'affine_parameter_tensor_count': 94, 'affine_parameters_trainable': True,
        'backbone_parameters_trainable': True, 'head_parameters_trainable': True,
        'eps_and_momentum_unchanged': True, 'all_batchnorm_eval': True,
        'policy_applied_after_each_model_train': True}, 'Frozen-statistics/learnable-affine BN proof differs')
    return dict(record)


def validate_batchnorm_epoch(record, initial_buffers_sha):
    equal_fields(record, {'buffer_sha256': initial_buffers_sha, 'buffers_unchanged': True,
        'actual_eval_layers': 47, 'affine_trainable_tensors': 94, 'affine_gradient_tensors': 94,
        'affine_gradients_finite': True, 'policy_applied_after_model_train': True},
        'Actual TRAIN epoch BN buffer/mode/gradient proof differs')
    require(type(record.get('affine_gradient_nonzero_tensors')) is int
            and 0 < record['affine_gradient_nonzero_tensors'] <= 94,
            'Actual TRAIN BN affine gradients need a measured nonzero tensor')


def cosine_learning_rate(initial, step, t_max=6, eta_min=.000005):
    return eta_min + (initial - eta_min) * (1 + math.cos(math.pi * step / t_max)) / 2


def validate_observed_learning_rate_groups(values, step, initial_head=.0001):
    require(isinstance(values, dict) and set(values) == {'backbone', 'head'},
            'Observed optimizer learning-rate groups differ')
    expected = {'backbone': cosine_learning_rate(.00004, step), 'head': cosine_learning_rate(initial_head, step)}
    for key, value in values.items():
        require(type(value) in (int, float) and math.isfinite(value) and value > 0
                and math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-15),
                'Observed lower head-LR cosine rate differs from the fixed condition')


def validate_learning_rate_history(history, training, protocol):
    equal_fields(training, {'head_lr': .0001, 'backbone_lr': .00004,
        'optimizer_schedule': {'name': 'CosineAnnealingLR', 'T_max': 6, 'eta_min': .000005, 'weight_decay': .0002},
        }, 'Actual matched semantic/control scheduler metadata differs')
    require(len(history) == 6, 'Six actual lower head-LR observations are required')
    observations = []
    for index, row in enumerate(history):
        equal_fields(row, {'epoch': index + 1}, 'Actual LR observation epoch differs')
        rates = row.get('optimizer_learning_rates')
        validate_observed_learning_rate_groups(rates, index)
        observations.append({'epoch': index + 1, **rates})
        if 'scheduler_next_learning_rates' in row:
            validate_observed_learning_rate_groups(row['scheduler_next_learning_rates'], index + 1)
    final = training.get('optimizer_final_learning_rates')
    validate_observed_learning_rate_groups(final, 6)
    curves = {'head': [cosine_learning_rate(.0001, step) for step in range(7)],
              'backbone': [cosine_learning_rate(.00004, step) for step in range(7)]}
    return {'candidate_actual_rates_verified': True, 'candidate_observed_epochs': 6,
        'candidate_before_epoch_observations': observations, 'candidate_final_after_step6_observation': dict(final),
        'candidate_initial_head_lr': .0001, 'reused_control_initial_head_lr': .0001,
        'backbone_initial_lr': .00004, 'eta_min': .000005, 'T_max': 6,
        'control_curve_provenance': 'Observed original head-LR run epoch optimizer groups and final step6; compared to new observations',
        'declared_step0_to6_curves': dict(curves), 'weight_decay': .0002}


def verify_matched_actual_learning_rates(control_history, control_training, candidate_history, candidate_training, protocol):
    candidate = validate_learning_rate_history(candidate_history, candidate_training, protocol)
    control = validate_learning_rate_history(control_history, control_training, protocol)
    require(len(control_history) == len(candidate_history) == 6, 'Both observed LR histories are required')
    for step, (left, right) in enumerate(zip(control_history, candidate_history)):
        for group in ('backbone', 'head'):
            require(math.isclose(left['optimizer_learning_rates'][group], right['optimizer_learning_rates'][group],
                rel_tol=1e-12, abs_tol=1e-15), 'Actual semantic/control learning rates differ')
    for group in ('backbone', 'head'):
        require(math.isclose(control_training['optimizer_final_learning_rates'][group],
            candidate_training['optimizer_final_learning_rates'][group], rel_tol=1e-12, abs_tol=1e-15),
            'Actual final semantic/control learning rates differ')
    candidate.update(reused_control_actual_rates_verified=True, reused_control_rates_are_observations=True,
        both_actual_observed_curves_match=True, reused_control_before_epoch_observations=control['candidate_before_epoch_observations'],
        reused_control_final_after_step6_observation=control['candidate_final_after_step6_observation'])
    return candidate


def verify_semantic_cpu_reload(weights, item, root, inventory):
    provider = ResearchPresenceClassifier(weights, 'cpu')
    require(provider.device.type == 'cpu' and isinstance(provider.model, SemanticResidualClassifier),
            'Completed semantic checkpoint needs the strict offline research factory')
    with Image.open(local_path(root, item['image'])) as image:
        inputs = provider.transform(image.convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        photo, maps, auxiliary = provider.model.forward_training(inputs); public = provider.model(inputs)
    require(list(public.shape) == [1, 7] and list(photo.shape) == [1, 7] and list(maps.shape) == [1, 7, 80, 80]
            and list(auxiliary.shape) == [1, 19] and torch.equal(public, photo)
            and all(bool(torch.isfinite(t).all()) for t in (public, photo, maps, auxiliary)),
            'Actual reloaded semantic CPU output contract differs')
    actual = validate_semantic_inventory(model_inventory(provider.model), initial_heads_zero=False)
    require(actual['parameter_count'] == inventory['parameter_count'] and actual['state_tensor_count'] == inventory['state_tensor_count'],
            'Actual research CPU factory state/parameter inventory differs')
    return {'device': 'cpu', 'strict_factory_reload_verified': True, 'train_photos_forwarded': 1,
        'input_shape': list(inputs.shape), 'photo_shape': list(photo.shape), 'loss_and_pool_map_shape': list(maps.shape),
        'auxiliary_shape': list(auxiliary.shape), 'public_shape': list(public.shape), 'all_outputs_finite': True,
        'public_output_equals_training_photo_output': True, 'parameter_count': actual['parameter_count'],
        'state_tensor_count': actual['state_tensor_count'], 'offline_construction_verified': True,
        'frozen_encoder_state_sha256': semantic_encoder_state_sha256(provider.model)}


def validate_actual_sampling(histories, arrays, items, full, epochs=6, draws=14248, batch_size=8):
    require(set(histories) == set(arrays) == {'control', 'semantic'}, 'Both paired histories are required')
    rows = len(items)
    for variant, array in arrays.items():
        require(isinstance(array, np.ndarray) and array.shape == (epochs, draws) and array.dtype == np.dtype('int64')
                and np.all(array >= 0) and np.all(array < rows), 'Prepared study draw shape/dtype/range differs')
        require(len(histories[variant]) == epochs, 'The fixed six epochs per arm must be complete')
    require(np.array_equal(arrays['control'], arrays['semantic']), 'Both arms must keep identical original row draws')
    totals = {v: {'attempted_batches': 0, 'actual_optimizer_steps': 0, 'amp_skipped_steps': 0} for v in histories}
    teacher_batches = {v: 0 for v in histories}; orders = {v: [] for v in histories}
    known_total = 0
    labels = np.asarray([row['targets'] for row in items], dtype=np.int64)
    for e in range(epochs):
        indices = arrays['control'][e]; expected = expected_counts(indices, items, full)
        known = labels[indices, 2:] >= 0
        known_entries = int(known.sum()); known_total += known_entries
        contributing = sum(int(known[b:b + batch_size].any()) for b in range(0, draws, batch_size))
        for variant, history in histories.items():
            row = history[e]; equal_fields(row, {'epoch': e + 1, **expected}, 'Actual epoch/order/all-seven exposure differs')
            actual = row.get('optimizer_step_diagnostics', {})
            require(set(actual) == set(totals[variant]) and all(type(v) is int for v in actual.values())
                    and actual['attempted_batches'] == math.ceil(draws / batch_size)
                    and 0 < actual['actual_optimizer_steps'] <= actual['attempted_batches']
                    and actual['amp_skipped_steps'] == actual['attempted_batches'] - actual['actual_optimizer_steps'],
                    'Actual optimizer attempts/updates/AMP skips differ')
            for key in actual: totals[variant][key] += actual[key]
            equal_fields(row.get('fixed_sampling', {}), {'declared_draw_sha256': expected['sampled_row_indices_sha256'],
                'draw_hash_matches_prepared': True, 'changed_positions_from_control': 0}, 'Actual unchanged sampler audit differs')
            weight = 4.
            validate_retention_audit(row.get('retention_distillation', {}), weight,
                math.ceil(draws / batch_size), known_entries, contributing)
            teacher_batches[variant] += row['retention_distillation']['teacher_forward_batches']
            orders[variant].append(expected['sampled_row_indices_sha256'])
    return {'epochs_compared': epochs, 'draws_per_epoch_each_arm': draws,
        'both_actual_draw_orders_match_fixed_original_control_array': True,
        'actual_ordered_row_index_hashes_identical_each_epoch': True,
        'actual_domain_full_crop_joint_counts_identical_each_epoch': True,
        'actual_all_seven_photo_target_exposure_identical_each_epoch': True,
        'changed_positions': 0, 'ordered_row_hashes_by_variant': orders,
        'optimizer_step_diagnostics_by_variant': totals, 'teacher_forward_batches_by_variant': teacher_batches,
        'known_other_class_entries_each_arm': known_total,
        'teacher_eval_frozen_no_grad_verified_both_arms': True}


def validate_run_metadata(training, checkpoint, history, variant, protocol, protocol_sha,
                          initial_state, weights_sha, split_sha):
    expected = {'status': 'complete', 'actual_epochs': protocol['requested_epochs'],
        'architecture': protocol['architecture_by_variant'][variant], 'imgsz': 640, 'model_variant': variant,
        'classes': protocol['classes'], 'auxiliary_classes': protocol['auxiliary_classes'],
        'auxiliary_source_class_count': 19, 'validation_batch_size': 16,
        'validation_domains': ['dacl', 'damsegment', 'codebrim'],
        'train_dacl': 6225, 'train_damsegment': 1585, 'train_codebrim': 6438, 'val_dacl': 710, 'val_damsegment': 424,
        'initial_weights_sha256': protocol['initial_weights_sha256'],
        'core_spatial_manifest_sha256': protocol['core_spatial_manifest_sha256'],
        'spatial_manifest_sha256': protocol['core_spatial_manifest_sha256'],
        'spatial_manifest_path': 'data/facility-spatial-training/train.json',
        'auxiliary_manifest_sha256': protocol['auxiliary_manifest_sha256'],
        'source_sha256': protocol['source_sha256'], 'study_protocol_sha256': protocol_sha,
        'study_protocol_path': PROTOCOL_PATH, 'distillation_recipe': protocol['distillation_recipe'],
        'distillation_weight': protocol['distillation_weight_by_variant'][variant],
        'sampling_intervention_applied': False, 'paired_draws_key': 'control',
        'sampler_plan_sha256': protocol['sampler_plan_sha256'],
        'historical_control_reused': True, 'reused_control_retrained': False,
        'batchnorm_policy': protocol['batchnorm_policy'],
        'semantic_features_policy': protocol['semantic_features_policy'],
        'optimizer_schedule': {'name': 'CosineAnnealingLR', 'T_max': 6, 'eta_min': .000005, 'weight_decay': .0002},
        'private_draw_archive_sha256': protocol['paired_draws_sha256'],
        'photo_pooling_grid': [80, 80], 'source_pixel_target_grid': [80, 80], 'raw_spatial_grid': [80, 80],
        'new_photo_targets': 0, 'new_pixel_targets': 0, 'app_model_promoted': False,
        'target_ranking_weight': 0., 'weights_sha256': weights_sha, 'split_sha256': split_sha}
    for key in ('seed', 'requested_epochs', 'patience', 'batch_size', 'draws_per_epoch',
                'backbone_lr', 'head_lr', 'auxiliary_weight', 'loader_randomness', 'domain_proportions'):
        expected[key] = protocol[key]
    equal_fields(training, expected, 'Actual fixed retention training condition differs')
    inventory = validate_semantic_inventory(training.get('semantic_model_inventory', {}), initial_heads_zero=True)
    validate_original_transfer(training.get('initial_state_transfer', {}), inventory)
    validate_pretrained_transfer(training.get('semantic_pretrained_transfer', {}), protocol, inventory)
    semantic_proof = validate_frozen_semantic(training.get('frozen_semantic_state_preservation', {}),
        protocol['semantic_encoder_state_sha256'], protocol['requested_epochs'] * math.ceil(protocol['draws_per_epoch'] / protocol['batch_size']))
    for row in history:
        validate_semantic_epoch(row.get('semantic_features_training', {}), protocol['semantic_encoder_state_sha256'],
            math.ceil(protocol['draws_per_epoch'] / protocol['batch_size']))
    require(len(history) == protocol['requested_epochs'], 'Declared training epochs were not completed')
    for key in ('attempted_batches', 'actual_optimizer_steps', 'amp_skipped_steps'):
        actual = sum(row['optimizer_step_diagnostics'][key] for row in history)
        require(type(training.get('optimizer_step_diagnostics', {}).get(key)) is int
                and training['optimizer_step_diagnostics'][key] == actual, 'Optimizer summary differs from actual epochs')
    require(type(training.get('elapsed_training_minutes')) in (int, float)
            and math.isfinite(training['elapsed_training_minutes']) and training['elapsed_training_minutes'] > 0
            and type(training.get('peak_cuda_allocated_bytes')) is int and training['peak_cuda_allocated_bytes'] > 0,
            'Measured training resources are absent')
    equal_fields(checkpoint, {'architecture': expected['architecture'], 'classes': protocol['classes'],
        'auxiliary_classes': list(AUX_CLASSES), 'imgsz': 640, 'mean': MEAN, 'std': STD,
        'selection_split': 'val', 'split_sha256': split_sha}, 'Completed checkpoint metadata differs')
    epoch = checkpoint.get('epoch'); error = checkpoint.get('worst_target_error')
    require(type(epoch) is int and 1 <= epoch <= len(history) and type(error) in (int, float)
            and math.isfinite(error) and 0 <= error <= 1
            and error == history[epoch - 1]['worst_target_error'] == training.get('best_worst_target_error'),
            'Selected checkpoint error/epoch differs from history')
    state = checkpoint.get('state_dict')
    restored_model = SemanticResidualClassifier(7, pretrained=False)
    expected_state = restored_model.state_dict()
    require(isinstance(state, dict) and set(state) == set(expected_state), 'Full semantic checkpoint state inventory changed')
    for key, tensor in state.items():
        original = expected_state[key]
        require(isinstance(tensor, torch.Tensor) and tensor.shape == original.shape and tensor.dtype == original.dtype
                and bool(torch.isfinite(tensor).all()), 'Semantic checkpoint tensor shape/dtype/finite contract differs')
    restored_model.load_state_dict(state, strict=True)
    require(semantic_encoder_state_sha256(restored_model) == protocol['semantic_encoder_state_sha256'],
            'Actual saved semantic encoder state changed')
    actual_inventory = validate_semantic_inventory(model_inventory(restored_model), initial_heads_zero=False)
    require(all(actual_inventory[k] == inventory[k] for k in ('parameter_count', 'trainable_parameter_count',
        'frozen_encoder_parameter_count', 'new_semantic_head_parameter_count', 'state_tensor_count',
        'original_state_tensor_count', 'encoder_state_tensor_count', 'new_head_state_tensor_count')),
        'Measured saved semantic inventory differs from recorded initializer')
    bn_proof = validate_batchnorm_preservation(training.get('batchnorm_state_preservation', {}),
        protocol['initial_batchnorm_buffers_sha256'])
    for row in history:
        validate_batchnorm_epoch(row.get('batchnorm_training', {}), protocol['initial_batchnorm_buffers_sha256'])
    reference_model = AuxiliaryClassifier(7, pretrained=False)
    reference_model.load_state_dict(initial_state, strict=True)
    require(batchnorm_state_sha256(reference_model) == batchnorm_state_sha256(restored_model)
            == protocol['initial_batchnorm_buffers_sha256']
            and batchnorm_inventory(reference_model)['configuration'] == batchnorm_inventory(restored_model)['configuration'],
            'Actual checkpoint BN buffers/configuration differ from original initializer')
    return {'imgsz': 640, 'architecture': expected['architecture'], 'actual_epochs': training['actual_epochs'],
        'selected_epoch': epoch, 'weights_sha256': weights_sha, 'state_tensor_count': len(state),
        'new_state_tensor_count': len(state) - 324, 'original_state_tensor_count': 324,
        'strict_state_inventory_verified': True, 'semantic_model_inventory': actual_inventory,
        'initial_state_transfer': dict(training['initial_state_transfer']),
        'semantic_pretrained_transfer': dict(training['semantic_pretrained_transfer']),
        'frozen_semantic_state_preservation': semantic_proof,
        'optimizer_step_diagnostics': dict(training['optimizer_step_diagnostics']),
        'batchnorm_state_preservation': bn_proof, 'actual_checkpoint_bn_buffers_and_configuration_verified': True,
        'teacher_state_preservation': validate_teacher_preservation(training.get('teacher_state_preservation', {}),
            protocol['teacher_state_sha256'], protocol['initial_weights_sha256'], inventory['parameter_count'])}


def verify_input_ledger(root, preflight, manifest, auxiliary, protected):
    path = local_path(root, LEDGER)
    require(sha(path) == preflight['protected_input_ledger_sha256'], 'Private original input ledger bytes changed')
    ledger = read(path)
    equal_fields(ledger, {'schema': 'facility_semantic_private_input_ledger_v1', 'local_only': True,
        'split': 'train', 'all_inputs_preserved': True, 'core_manifest_sha256': sha(root / 'data/facility-spatial-training/train.json')},
        'Original TRAIN input ledger differs')
    before = ledger.get('snapshots_before')
    require(isinstance(before, dict) and before and before == ledger.get('snapshots_after'),
            'Recorded original SHA/size/mtime preservation is absent')
    data = {'items': manifest['items'], 'auxiliary_manifest': auxiliary}
    expected = input_paths(data, protected)
    require(set(expected) == set(before) and len(before) == preflight['protected_input_file_count'],
            'Private original image/mask/annotation inventory is incomplete')
    require(snapshot_inputs(root, expected) == before, 'Original TRAIN SHA/size/mtime changed')
    previous = read(local_path(root, PREVIOUS_LEDGER))
    equal_fields(ledger, {'previous_ledger_sha256': sha(root / PREVIOUS_LEDGER),
        'previous_input_file_count': len(previous['snapshots_before']), 'previous_inputs_preserved': True},
        'Established original input ledger binding differs')
    require(previous['snapshots_before'] == previous['snapshots_after']
            and len(previous['snapshots_before']) == preflight['previous_input_file_count'] == 58995
            and all(before.get(name) == value for name, value in previous['snapshots_before'].items()),
            'Established original TRAIN SHA/size/mtime preservation differs')
    return {'status': 'passed', 'input_files_checked': len(before),
        'all_original_input_sha_size_mtime_preserved': True, 'original_train_images_masks_annotations_verified': True,
        'ledger_sha256': sha(path), 'individual_input_paths_published': False,
        'previous_input_files_checked': len(previous['snapshots_before']), 'previous_inputs_preserved': True}


def validate_single_candidate_budget(record):
    equal_fields(record, {'actual_completed_training_epochs': 6, 'reused_control_epochs': 6,
        'new_candidate_count': 1, 'control_training_repeated': False,
        'existing_control_preserved': True}, 'Single-candidate/reused-control budget differs')
    return {'new_training_epochs': 6, 'reused_control_epochs': 6, 'new_candidate_count': 1,
            'control_training_repeated': False}


def verify_reused_control(root, protocol, training, checkpoint, history, initial_state, weights_sha, split_sha):
    old_protocol_path = root / 'reports/facility-head-lr-study-protocol.json'
    old_verification_path = root / 'reports/facility-head-lr-study-verification.json'
    old_protocol = read(old_protocol_path); previous = read(old_verification_path)
    require(previous.get('schema') == 'facility_head_lr_study_verification_v1'
            and previous.get('status') == 'passed' and previous.get('actual_completed_training_epochs') == 6
            and previous.get('runtime_source_count') == 97 and previous.get('protected_files_unchanged') is True,
            'Completed frozen-BN weight-four control evidence must remain passed')
    require(previous['protocol_sha256'] == sha(old_protocol_path)
            and previous['source_sha256'] == old_protocol['source_sha256']
            and protocol['control'] == old_protocol['treatment'], 'Reused control belongs to another fixed study')
    entry = validate_old_control_metadata(training, checkpoint, history, 'low_lr', old_protocol,
        sha(old_protocol_path), initial_state, weights_sha, split_sha)
    equal_fields(training, {'head_lr': .0001, 'backbone_lr': .00004}, 'Reused control initial LR metadata differs')
    prior_entry = next(e for e in previous['experiments'] if e['variant'] == 'low_lr')
    for key in ('weights_sha256', 'actual_epochs', 'selected_epoch', 'state_tensor_count', 'optimizer_step_diagnostics'):
        require(entry[key] == prior_entry[key], 'Actual reused control checkpoint/history differs from completed evidence')
    validate_teacher_preservation(training.get('teacher_state_preservation', {}),
        protocol['teacher_state_sha256'], protocol['initial_weights_sha256'])
    control_model = AuxiliaryClassifier(7, pretrained=False)
    control_model.load_state_dict(checkpoint['state_dict'], strict=True)
    control_bn_sha = batchnorm_state_sha256(control_model)
    counters = [int(tensor) - int(initial_state[name]) for name, tensor in checkpoint['state_dict'].items()
                if name.endswith('.num_batches_tracked')]
    require(len(counters) == 47, 'Reused frozen-BN control counter inventory differs')
    counter_deltas = {str(delta): counters.count(delta) for delta in sorted(set(counters))}
    entry.update(variant='control', reused=True, training_repeated=False,
        previous_protocol_sha256=sha(old_protocol_path), previous_verification_sha256=sha(old_verification_path),
        teacher_state_preservation=dict(training['teacher_state_preservation']),
        batchnorm_control_measurement={'running_buffers_sha256': control_bn_sha,
            'running_buffers_differ_from_initializer': control_bn_sha != protocol['initial_batchnorm_buffers_sha256'],
            'counter_delta_layer_counts': counter_deltas, 'buffer_tensor_count': 141, 'layer_count': 47,
            'scope': 'Measured saved frozen-BN control buffers/counters; original frozen policy verified against prior source97 proof'},
        batchnorm_state_preservation=dict(training['batchnorm_state_preservation']),
        actual_checkpoint_bn_buffers_and_configuration_verified=True)
    return entry


def snapshot_before_training(root=ROOT):
    """Bind actual committed source bytes after preflight and before study runs."""
    root = Path(root).resolve()
    require(not (root / SOURCE_RECORD_PATH).exists(), 'Preserve existing before-training snapshot')
    protocol = validate_protocol(read(root / PROTOCOL_PATH), root)
    require(not (root / 'runs' / protocol['treatment']).exists(),
            'Source snapshot must precede creation of the one new candidate run')
    require((root / 'runs' / protocol['control'] / 'TRAINING.json').is_file(),
            'Completed fixed control evidence must exist before the one new candidate')
    protocol_sha = sha(root / PROTOCOL_PATH); preflight = read(root / PREFLIGHT_PATH)
    validate_preflight(preflight, protocol, protocol_sha, protected_hashes(root))
    repo = root.parents[1]
    git = subprocess.run(['git', '-c', f'safe.directory={repo}', 'rev-parse', 'HEAD'],
                         cwd=repo, capture_output=True, check=False)
    require(git.returncode == 0, 'Source Git HEAD is unavailable')
    commit = git.stdout.decode('ascii').strip()
    snapshot = {'schema': 'facility_semantic_source_before_training_v1',
        'declared_before_training': True, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'source_git_commit': commit, 'source_sha256': dict(protocol['source_sha256']),
        'protocol_sha256': protocol_sha, 'preflight_sha256': sha(root / PREFLIGHT_PATH),
        'initial_weights_sha256': protocol['initial_weights_sha256'],
        'new_completed_training_epochs_at_snapshot': 0, 'git_blob_bytes_match_protocol_sources': True,
        'control_training_repeated': False}
    verify_frozen_sources(snapshot, protocol, preflight, root, protocol_sha, snapshot['preflight_sha256'])
    write(root / SOURCE_RECORD_PATH, snapshot)
    print({'status': 'snapshotted', 'source_git_commit': commit,
           'runtime_sources': len(RUNTIME_SOURCES), 'study_epochs_started': 0}, flush=True)
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-results', type=Path)
    parser.add_argument('--snapshot-before-training', action='store_true')
    args = parser.parse_args(argv)
    if args.snapshot_before_training:
        require(args.test_results is None, 'Before-training snapshot does not use post-training test evidence')
        return snapshot_before_training()
    require(args.test_results is not None, 'Actual completion verification requires --test-results')
    tests_path = args.test_results.resolve()
    require(tests_path.is_relative_to((ROOT / 'runs').resolve()) and tests_path.is_file(), 'Test evidence must be local runs file')
    require(not (ROOT / OUTPUT_PATH).exists(), 'Preserve existing completion evidence')
    torch.set_num_threads(1)
    protocol = validate_protocol(read(ROOT / PROTOCOL_PATH), ROOT); protocol_sha = sha(ROOT / PROTOCOL_PATH)
    preflight = read(ROOT / PREFLIGHT_PATH); snapshot = read(ROOT / SOURCE_RECORD_PATH)
    proof_paths = [ROOT / p for p in (PROTOCOL_PATH, PREFLIGHT_PATH, SOURCE_RECORD_PATH, LEDGER)] + [tests_path]
    before_proof = {p: sha(p) for p in proof_paths}; protected = protected_hashes(ROOT)
    sources = verify_frozen_sources(snapshot, protocol, preflight, ROOT, protocol_sha, before_proof[ROOT / PREFLIGHT_PATH])
    preflight_proof = validate_preflight(preflight, protocol, protocol_sha, protected)
    tests = validate_test_results(read(tests_path), ROOT)
    manifest = read(ROOT / 'data/facility-spatial-training/train.json')
    auxiliary = read(ROOT / 'data/facility-auxiliary-training/train.json')
    print({'phase': 'verify-original-train-inputs', 'files': preflight['protected_input_file_count']}, flush=True)
    data_proof = verify_input_ledger(ROOT, preflight, manifest, auxiliary, protected)
    names = {'control': protocol['control'], 'semantic': protocol['treatment']}
    require(not any((ROOT / 'reports' / f'{name}-target-test.json').exists() for name in [protocol['reference'], *names.values()]),
            'This fixed study refuses held-out TEST inference records')
    histories = {v: read(ROOT / 'runs' / n / 'history.json') for v, n in names.items()}
    with np.load(local_path(ROOT, protocol['paired_draws_path']), allow_pickle=False) as archive:
        require(set(archive.files) == {'control', 'treatment'}, 'Prepared draw archive keys differ')
        arrays = {'control': archive['control'].copy(), 'semantic': archive['control'].copy()}
    require(sha(ROOT / protocol['paired_draws_path']) == protocol['paired_draws_sha256'], 'Prepared fixed draw archive changed')
    sampling = validate_actual_sampling(histories, arrays, manifest['items'], manifest['full_count'],
        protocol['requested_epochs'], protocol['draws_per_epoch'], protocol['batch_size'])
    selected, _ = select_train_replay(manifest, auxiliary)
    initial = torch.load(ROOT / 'runs' / protocol['reference'] / 'best.pt', map_location='cpu', weights_only=True)
    require(len(initial['state_dict']) == 324, 'Original initializer state count differs')
    require(state_sha256(initial['state_dict']) == protocol['teacher_state_sha256'],
            'Original teacher canonical tensor state hash differs')
    entries = []; trainings = []; before_runs = {}; reused_control = None; learning_rates = None
    for variant, name in names.items():
        run = ROOT / 'runs' / name; weights = run / 'best.pt'
        before_runs.update({run / f: sha(run / f) for f in ('history.json', 'TRAINING.json', 'best.pt', 'SPLIT.json')})
        require(sha(run / 'SPLIT.json') == initial['split_sha256'], 'Original split bytes changed')
        training = read(run / 'TRAINING.json'); trainings.append(training)
        require(type(training.get('actual_teacher_forward_batches')) is int
                and training['actual_teacher_forward_batches'] == sampling['teacher_forward_batches_by_variant'][variant],
                'Actual teacher forward summary differs')
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        if variant == 'control':
            entry = verify_reused_control(ROOT, protocol, training, checkpoint, histories[variant],
                initial['state_dict'], sha(weights), sha(run / 'SPLIT.json'))
            reused_control = entry
        else:
            entry = validate_run_metadata(training, checkpoint, histories[variant], variant, protocol, protocol_sha,
                initial['state_dict'], sha(weights), sha(run / 'SPLIT.json'))
            entry['variant'] = variant
            entry['actual_matched_learning_rate_curve_verified'] = True
            entries.append(entry)
        if variant == 'control':
            entry['cpu_reload'] = verify_cpu_reload(weights, selected[0], ROOT, 3244151, 'control')
        else:
            entry['cpu_reload'] = verify_semantic_cpu_reload(weights, selected[0], ROOT, entry['semantic_model_inventory'])
    learning_rates = verify_matched_actual_learning_rates(histories['control'], trainings[0], histories['semantic'], trainings[1], protocol)
    for key in ('classes', 'split_sha256', 'core_spatial_manifest_sha256', 'auxiliary_manifest_sha256',
        'expected_sampling', 'expected_label_sampling', 'photo_positive_weights', 'pixel_positive_weights',
        'auxiliary_positive_weights', 'additional_validation', 'additional_test', 'total_loss_formula', 'loss', 'paired_label_preservation'):
        require(key in trainings[0] and trainings[0][key] == trainings[1].get(key), 'Paired original targets/loss/base weights differ')
    require(protected_hashes(ROOT) == protected, 'Completion verification changed protected bytes')
    require(preflight['reused_control_preflight_sha256'] == sha(ROOT / REUSED_PREFLIGHT), 'Reused control preflight bytes changed')
    verify_frozen_sources(snapshot, protocol, preflight, ROOT, protocol_sha, before_proof[ROOT / PREFLIGHT_PATH])
    validate_test_results(read(tests_path), ROOT)
    require(all(sha(p) == digest for p, digest in {**before_proof, **before_runs}.items()), 'Proof/run evidence changed during verification')
    result = {'schema': 'facility_semantic_study_verification_v1', 'status': 'passed',
        'verified_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': protocol_sha,
        'preflight_sha256': before_proof[ROOT / PREFLIGHT_PATH],
        'source_before_training_sha256': before_proof[ROOT / SOURCE_RECORD_PATH], **sources,
        'preflight': preflight_proof, 'test_results_sha256': before_proof[tests_path], 'tests': tests,
        'prepared_data_integrity': data_proof, 'actual_sampling_verification': sampling, 'experiments': entries,
        'reused_control': reused_control, 'reused_control_epochs': reused_control['actual_epochs'],
        'new_candidate_count': len(entries), 'control_training_repeated': False, 'existing_control_preserved': True,
        'actual_completed_training_epochs': sum(e['actual_epochs'] for e in entries),
        'both_teachers_unchanged_eval_frozen_no_grad': True,
        'candidate_original_batchnorm_buffers_preserved_affine_learnable': True,
        'learning_rate_verification': learning_rates,
        'frozen_semantic_encoder_and_learned_heads_verified': True,
        'original324_and_zero_initial_outputs_verified': True,
        'parameter_and_compute_budget_identical': False,
        'protected_file_sha256': protected, 'protected_files_unchanged': True,
        'verification_training_epochs': 0, 'source_test_inference_executed': False,
        'app_model_promoted': False, 'deployed': False, 'accuracy_measured_by_verifier': False,
        'additional_expert_confirmed_labels': 0, 'label_changes': 0, 'new_photo_targets': 0, 'new_pixel_targets': 0,
        'scope': 'Local original TRAIN bytes/source/sampler/checkpoint/CPU proof; not field accuracy'}
    result['single_candidate_budget_verification'] = validate_single_candidate_budget(result)
    write(ROOT / OUTPUT_PATH, result)
    print({'status': 'passed', 'runtime_sources': len(RUNTIME_SOURCES), 'tests_run': tests['tests_run'],
        'actual_completed_training_epochs': result['actual_completed_training_epochs'], 'reused_control_epochs': reused_control['actual_epochs'],
        'original_input_files_verified': data_proof['input_files_checked'], 'cpu_train_photo_forwards': 2}, flush=True)
    return result


if __name__ == '__main__': main()
