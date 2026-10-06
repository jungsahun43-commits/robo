"""Single lower head-LR candidate preserves original draws, teacher and BN statistics."""
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from scripts import verify_facility_head_lr as verify
from safelog_ai.spalling_sampler import draw_sha256


class HeadLrVerificationTests(TestCase):
    def sampling_fixture(self):
        targets = ([0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0], [0, 0, -1, -1, -1, -1, -1],
                   [1, 0, 0, 0, 0, 0, 0])
        items = [{'image': f'data/full{i}.png', 'domain': d, 'targets': list(t), 'split': 'train'}
                 for i, (d, t) in enumerate(zip(('dacl', 'dacl', 'dacl', 'damsegment', 'codebrim'), targets))]
        items.append({'image': 'data/crop.png', 'parent_image': 'data/full0.png',
                      'parent_split': 'train', 'domain': 'dacl', 'targets': [-1] * 7})
        array = np.asarray([[0, 1, 2, 3, 4, 5, 0, 2]], dtype=np.int64)
        arrays = {'control': array.copy(), 'low_lr': array.copy()}
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
        self.assertEqual(proof['teacher_forward_batches_by_variant'], {'control': 2, 'low_lr': 2})
        self.assertTrue(proof['actual_all_seven_photo_target_exposure_identical_each_epoch'])
        self.assertNotEqual(proof['optimizer_step_diagnostics_by_variant']['control'],
                            proof['optimizer_step_diagnostics_by_variant']['low_lr'])

    def test_all_seven_label_exposure_and_private_array_equality_are_required(self):
        values = self.sampling_fixture()
        values[0]['low_lr'][0]['sampled_photo_target_counts']['rust_stain']['positive'] += 1
        with self.assertRaisesRegex(ValueError, 'all-seven'): self.validate_fixture(values)
        values = self.sampling_fixture(); values[1]['low_lr'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'identical original'): self.validate_fixture(values)

    def test_teacher_requires_actual_frozen_state_and_correct_masked_loss_accounting(self):
        for key, value in (('teacher_eval', False), ('teacher_frozen', False), ('teacher_gradients_absent', False),
                           ('teacher_state_unchanged', False), ('teacher_forward_batches', 1),
                           ('known_other_class_entries', 31), ('weighted_mean_batch_loss', .25),
                           ('excluded_primary_classes', []), ('weight', 1.)):
            values = self.sampling_fixture(); values[0]['low_lr'][0]['retention_distillation'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate_fixture(values)

    def test_optimizer_attempts_updates_and_skips_require_consistent_integers(self):
        for key, value in (('attempted_batches', 3), ('actual_optimizer_steps', 0),
                           ('amp_skipped_steps', -1), ('actual_optimizer_steps', True)):
            values = self.sampling_fixture(); values[0]['control'][0]['optimizer_step_diagnostics'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'optimizer'): self.validate_fixture(values)

    def test_bn_statistics_stay_fixed_while_affine_and_backbone_remain_trainable(self):
        proof = {'initial_buffers_sha256': 'a' * 64, 'final_buffers_sha256': 'a' * 64,
                 'buffers_unchanged': True, 'layer_count': 47, 'channel_count': 12328,
                 'buffer_tensor_count': 141, 'affine_parameter_tensor_count': 94,
                 'affine_parameters_trainable': True, 'backbone_parameters_trainable': True,
                 'head_parameters_trainable': True, 'eps_and_momentum_unchanged': True,
                 'all_batchnorm_eval': True, 'policy_applied_after_each_model_train': True}
        self.assertEqual(verify.validate_batchnorm_preservation(proof, 'a' * 64), proof)
        for key, value in (('final_buffers_sha256', 'b' * 64), ('affine_parameters_trainable', False),
                           ('backbone_parameters_trainable', False), ('head_parameters_trainable', False),
                           ('eps_and_momentum_unchanged', False), ('all_batchnorm_eval', False)):
            changed = dict(proof); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.validate_batchnorm_preservation(changed, 'a' * 64)
        epoch = {'buffer_sha256': 'a' * 64, 'buffers_unchanged': True, 'actual_eval_layers': 47,
                 'affine_trainable_tensors': 94, 'affine_gradient_tensors': 94,
                 'affine_gradient_nonzero_tensors': 73, 'affine_gradients_finite': True,
                 'policy_applied_after_model_train': True}
        verify.validate_batchnorm_epoch(epoch, 'a' * 64)
        for key, value in (('buffer_sha256', 'c' * 64), ('affine_gradient_tensors', 93),
                           ('affine_gradient_nonzero_tensors', 0), ('affine_gradients_finite', False),
                           ('policy_applied_after_model_train', False)):
            changed = dict(epoch); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.validate_batchnorm_epoch(changed, 'a' * 64)

    def lr_fixture(self):
        curves = {key: [verify.cosine_learning_rate(initial, t) for t in range(7)]
                  for key, initial in (('control_head', .00025), ('candidate_head', .0001), ('backbone', .00004))}
        policy = {'step0_to6_curves': curves}
        training = {'head_lr': .0001, 'backbone_lr': .00004, 'head_lr_policy': policy,
                    'optimizer_schedule': {'name': 'CosineAnnealingLR', 'T_max': 6,
                                           'eta_min': .000005, 'weight_decay': .0002},
                    'optimizer_final_learning_rates': {'backbone': .000005, 'head': .000005}}
        history = [{'epoch': t + 1, 'optimizer_learning_rates':
                    {'backbone': curves['backbone'][t], 'head': curves['candidate_head'][t]},
                    'scheduler_next_learning_rates': {'backbone': curves['backbone'][t + 1],
                                                      'head': curves['candidate_head'][t + 1]}}
                   for t in range(6)]
        return history, training, {'head_lr_policy': policy}

    def test_candidate_learning_rates_are_observed_and_control_curve_is_only_derived(self):
        proof = verify.validate_learning_rate_history(*self.lr_fixture())
        self.assertTrue(proof['candidate_actual_rates_verified'])
        self.assertFalse(proof['reused_control_rates_are_observations'])
        self.assertEqual(proof['candidate_observed_epochs'], 6)
        self.assertEqual(proof['candidate_final_after_step6_observation'], {'backbone': .000005, 'head': .000005})
        curves = proof['declared_step0_to6_curves']
        self.assertAlmostEqual(curves['candidate_head'][0] / curves['control_head'][0], .4)
        self.assertEqual(curves['candidate_head'][6] / curves['control_head'][6], 1.)

    def test_wrong_head_rate_scheduler_or_observed_final_rate_cannot_pass(self):
        for actual in (.00025, True, float('nan')):
            history, training, protocol = self.lr_fixture()
            history[0]['optimizer_learning_rates']['head'] = actual
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                verify.validate_learning_rate_history(history, training, protocol)
        history, training, protocol = self.lr_fixture()
        training['optimizer_final_learning_rates']['head'] = .0001
        with self.assertRaises(ValueError): verify.validate_learning_rate_history(history, training, protocol)
        history, training, protocol = self.lr_fixture()
        training['optimizer_schedule']['eta_min'] = 0.
        with self.assertRaises(ValueError): verify.validate_learning_rate_history(history, training, protocol)
        history, training, protocol = self.lr_fixture()
        with self.assertRaises(ValueError): verify.validate_learning_rate_history(history[:-1], training, protocol)

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
