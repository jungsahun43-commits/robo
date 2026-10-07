"""Zero residual initial identity and frozen semantic state remain distinct from original324."""
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from scripts import verify_facility_semantic as verify
from safelog_ai.spalling_sampler import draw_sha256


class SemanticVerificationTests(TestCase):
    def sampling_fixture(self):
        targets = ([0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0], [0, 0, -1, -1, -1, -1, -1],
                   [1, 0, 0, 0, 0, 0, 0])
        items = [{'image': f'data/full{i}.png', 'domain': d, 'targets': list(t), 'split': 'train'}
                 for i, (d, t) in enumerate(zip(('dacl', 'dacl', 'dacl', 'damsegment', 'codebrim'), targets))]
        items.append({'image': 'data/crop.png', 'parent_image': 'data/full0.png',
                      'parent_split': 'train', 'domain': 'dacl', 'targets': [-1] * 7})
        array = np.asarray([[0, 1, 2, 3, 4, 5, 0, 2]], dtype=np.int64)
        arrays = {'control': array.copy(), 'semantic': array.copy()}
        history = {}
        for variant, values in arrays.items():
            counts = verify.expected_counts(values[0], items, 5); weight = 4.
            # Different AMP skip counts are permitted while the attempted
            # budget and both teacher-forward counts remain exactly matched.
            updates = 1 if variant == 'control' else 2
            row = {'epoch': 1, **counts, 'optimizer_step_diagnostics':
                   {'attempted_batches': 2, 'actual_optimizer_steps': updates, 'amp_skipped_steps': 2 - updates},
                   'fixed_sampling': {'changed_positions_from_control': 0,
                       'declared_draw_sha256': draw_sha256(values[0]), 'draw_hash_matches_prepared': True},
                   'retention_distillation': {'weight': weight, 'temperature': 2.,
                       'unweighted_mean_batch_loss': .125, 'weighted_mean_batch_loss': weight * .125,
                       'teacher_forward_batches': 2, 'contributing_batches': 2, 'known_other_class_entries': 30,
                       'excluded_primary_classes': ['concrete_crack', 'concrete_spalling'],
                       'teacher_eval': True, 'teacher_frozen': True, 'teacher_gradients_absent': True,
                       'teacher_state_unchanged': True}}
            history[variant] = [row]
        return history, arrays, items, 5

    def validate_fixture(self, values):
        return verify.validate_actual_sampling(*values, epochs=1, draws=8, batch_size=4)

    def test_identical_original_draws_and_teacher_forwards_allow_independent_amp_skips(self):
        proof = self.validate_fixture(self.sampling_fixture())
        self.assertEqual(proof['changed_positions'], 0)
        self.assertEqual(proof['teacher_forward_batches_by_variant'], {'control': 2, 'semantic': 2})
        self.assertTrue(proof['actual_all_seven_photo_target_exposure_identical_each_epoch'])
        self.assertNotEqual(proof['optimizer_step_diagnostics_by_variant']['control'],
                            proof['optimizer_step_diagnostics_by_variant']['semantic'])

    def test_all_seven_label_exposure_and_private_array_equality_are_required(self):
        values = self.sampling_fixture()
        values[0]['semantic'][0]['sampled_photo_target_counts']['rust_stain']['positive'] += 1
        with self.assertRaisesRegex(ValueError, 'all-seven'): self.validate_fixture(values)
        values = self.sampling_fixture(); values[1]['semantic'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'identical original'): self.validate_fixture(values)

    def test_teacher_requires_actual_frozen_state_and_correct_masked_loss_accounting(self):
        for key, value in (('teacher_eval', False), ('teacher_frozen', False), ('teacher_gradients_absent', False),
                           ('teacher_state_unchanged', False), ('teacher_forward_batches', 1),
                           ('known_other_class_entries', 31), ('weighted_mean_batch_loss', .25),
                           ('excluded_primary_classes', []), ('weight', 1.)):
            values = self.sampling_fixture(); values[0]['semantic'][0]['retention_distillation'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate_fixture(values)

    def test_optimizer_attempts_updates_and_skips_require_consistent_integers(self):
        for key, value in (('attempted_batches', 3), ('actual_optimizer_steps', 0),
                           ('amp_skipped_steps', -1), ('actual_optimizer_steps', True)):
            values = self.sampling_fixture(); values[0]['control'][0]['optimizer_step_diagnostics'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'optimizer'): self.validate_fixture(values)

    def test_resource_latency_has_measured_scope_and_both_actual_lr_curves_match(self):
        timing = {'device': 'cuda', 'device_name': 'test GPU', 'torch_version': 'test version',
                  'dtype': 'torch.float32', 'batch_size': 1, 'input_shape': [1, 3, 640, 640],
                  'warmup_forwards_each': 3, 'timed_forwards_each': 10,
                  'same_input_tensor_both_models': True, 'transfer_excluded': True, 'model_forward_only': True,
                  'accuracy_measured': False, 'android_latency_measured': False, 'input_tensor_sha256': 'a' * 64,
                  'original': {'mean_ms': 2.4, 'median_ms': 2.2, 'min_ms': 2., 'max_ms': 3.},
                  'semantic': {'mean_ms': 12., 'median_ms': 11., 'min_ms': 10., 'max_ms': 20.}}
        verify.validate_forward_latency(timing)
        for key, value in (('transfer_excluded', False), ('android_latency_measured', True), ('timed_forwards_each', 0)):
            modified = deepcopy(timing); modified[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): verify.validate_forward_latency(modified)
        modified = deepcopy(timing); modified['semantic']['mean_ms'] = 0.
        with self.assertRaises(ValueError): verify.validate_forward_latency(modified)
        history = [{'epoch': t + 1, 'optimizer_learning_rates':
                   {'backbone': verify.cosine_learning_rate(.00004, t),
                    'head': verify.cosine_learning_rate(.0001, t)}} for t in range(6)]
        training = {'head_lr': .0001, 'backbone_lr': .00004,
                    'optimizer_schedule': {'name': 'CosineAnnealingLR', 'T_max': 6,
                                           'eta_min': .000005, 'weight_decay': .0002},
                    'optimizer_final_learning_rates': {'backbone': .000005, 'head': .000005}}
        proof = verify.verify_matched_actual_learning_rates(history, training, deepcopy(history), dict(training), {})
        self.assertTrue(proof['both_actual_observed_curves_match'])
        self.assertTrue(proof['reused_control_rates_are_observations'])
        modified = deepcopy(history); modified[0]['optimizer_learning_rates']['head'] = .00025
        with self.assertRaises(ValueError):
            verify.verify_matched_actual_learning_rates(history, training, modified, dict(training), {})

    def semantic_inventory(self, initial=True):
        return {'parameter_count': 31085624, 'trainable_parameter_count': 3265496,
                'frozen_encoder_parameter_count': 27820128, 'new_semantic_head_parameter_count': 21345,
                'state_tensor_count': 510, 'original_state_tensor_count': 324,
                'encoder_state_tensor_count': 180, 'new_head_state_tensor_count': 6,
                'all_encoder_parameters_frozen': True, 'encoder_eval': True, 'all_encoder_modules_eval': True,
                'new_semantic_heads_trainable': True, 'new_semantic_heads_zero': initial,
                'low_channels': 192, 'pooled_channels': 768, 'low_feature_grid_for_640': [80, 80]}

    def test_original324_and_official_frozen_encoder_have_separate_exact_inventories(self):
        inventory = self.semantic_inventory()
        verify.validate_semantic_inventory(inventory, initial_heads_zero=True)
        transfer = {'shared_state_tensors_equal': True, 'shared_state_tensor_count': 324,
                    'original_state_tensor_count': 324, 'strict_state_load': True, 'new_state_tensor_count': 186,
                    'new_semantic_head_state_tensor_count': 6, 'new_semantic_heads_zero': True,
                    'additional_trainable_parameters': 21345, 'full_original_model_state_preserved': True}
        verify.validate_original_transfer(transfer, inventory)
        for key, value in (('parameter_count', 3244151), ('state_tensor_count', 324),
                           ('original_state_tensor_count', 323), ('new_semantic_heads_zero', False),
                           ('all_encoder_parameters_frozen', False)):
            changed = dict(inventory); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.validate_semantic_inventory(changed, initial_heads_zero=True)
        changed = dict(transfer); changed['new_semantic_heads_zero'] = False
        with self.assertRaises(ValueError): verify.validate_original_transfer(changed, inventory)

    def test_encoder_cannot_train_and_all_three_heads_must_change_from_zero(self):
        proof = {'initial_encoder_state_sha256': 'a' * 64, 'final_encoder_state_sha256': 'a' * 64,
                 'encoder_state_unchanged': True, 'encoder_all_eval': True, 'encoder_parameters_frozen': True,
                 'encoder_gradients_absent': True, 'head_parameters_trainable': True,
                 'semantic_heads_changed_from_zero': {'map': True, 'photo': True, 'aux': True},
                 'encoder_forward_batches': 10686}
        verify.validate_frozen_semantic(proof, 'a' * 64, 10686)
        for key, value in (('final_encoder_state_sha256', 'b' * 64), ('encoder_all_eval', False),
                           ('encoder_parameters_frozen', False), ('encoder_gradients_absent', False),
                           ('encoder_forward_batches', 1781),
                           ('semantic_heads_changed_from_zero', {'map': True, 'photo': True, 'aux': False})):
            changed = dict(proof); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.validate_frozen_semantic(changed, 'a' * 64, 10686)
        epoch = {'encoder_state_sha256': 'a' * 64, 'encoder_state_unchanged': True,
                 'encoder_all_eval': True, 'encoder_parameters_frozen': True, 'encoder_gradients_absent': True,
                 'semantic_head_gradient_tensors': 6, 'semantic_head_gradient_nonzero_tensors': 3,
                 'semantic_head_gradients_finite': True,
                 'semantic_heads_changed_from_zero': {'map': True, 'photo': True, 'aux': True},
                 'encoder_forward_batches': 1781}
        verify.validate_semantic_epoch(epoch, 'a' * 64, 1781)
        epoch['semantic_head_gradient_nonzero_tensors'] = 0
        with self.assertRaises(ValueError): verify.validate_semantic_epoch(epoch, 'a' * 64, 1781)

    def test_single_new_candidate_budget_excludes_reused_control_training(self):
        record = {'actual_completed_training_epochs': 6, 'reused_control_epochs': 6,
                  'new_candidate_count': 1, 'control_training_repeated': False,
                  'existing_control_preserved': True}
        proof = verify.validate_single_candidate_budget(record)
        self.assertEqual(proof['new_training_epochs'], 6)
        self.assertEqual(proof['reused_control_epochs'], 6)
        for key, value in (('actual_completed_training_epochs', 12), ('reused_control_epochs', 0),
                           ('new_candidate_count', 2), ('control_training_repeated', True),
                           ('existing_control_preserved', False)):
            changed = dict(record); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): verify.validate_single_candidate_budget(changed)
