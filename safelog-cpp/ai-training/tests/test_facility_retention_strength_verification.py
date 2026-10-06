"""Focused retention proof mutations for exact draws and a frozen teacher."""
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from scripts import verify_facility_retention_strength as verify
from safelog_ai.spalling_sampler import draw_sha256


class RetentionStrengthVerificationTests(TestCase):
    def sampling_fixture(self):
        targets = ([0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0], [0, 0, -1, -1, -1, -1, -1],
                   [1, 0, 0, 0, 0, 0, 0])
        items = [{'image': f'data/full{i}.png', 'domain': d, 'targets': list(t), 'split': 'train'}
                 for i, (d, t) in enumerate(zip(('dacl', 'dacl', 'dacl', 'damsegment', 'codebrim'), targets))]
        items.append({'image': 'data/crop.png', 'parent_image': 'data/full0.png',
                      'parent_split': 'train', 'domain': 'dacl', 'targets': [-1] * 7})
        array = np.asarray([[0, 1, 2, 3, 4, 5, 0, 2]], dtype=np.int64)
        arrays = {'control': array.copy(), 'strong': array.copy()}
        history = {}
        for variant, values in arrays.items():
            counts = verify.expected_counts(values[0], items, 5); weight = 0. if variant == 'control' else 4.
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
        self.assertEqual(proof['teacher_forward_batches_by_variant'], {'control': 2, 'strong': 2})
        self.assertTrue(proof['actual_all_seven_photo_target_exposure_identical_each_epoch'])
        self.assertNotEqual(proof['optimizer_step_diagnostics_by_variant']['control'],
                            proof['optimizer_step_diagnostics_by_variant']['strong'])

    def test_actual_draw_order_and_source_exposure_mutation_are_rejected(self):
        for key in ('sampled_row_indices_sha256', 'sampled_domain_counts', 'sampled_row_type_counts',
                    'sampled_full_target_joint_counts'):
            values = self.sampling_fixture(); values[0]['strong'][0][key] = '0' * 64 if 'sha256' in key else {}
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate_fixture(values)

    def test_all_seven_label_exposure_and_private_array_equality_are_required(self):
        values = self.sampling_fixture()
        values[0]['strong'][0]['sampled_photo_target_counts']['rust_stain']['positive'] += 1
        with self.assertRaisesRegex(ValueError, 'all-seven'): self.validate_fixture(values)
        values = self.sampling_fixture(); values[1]['strong'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'identical original'): self.validate_fixture(values)

    def test_teacher_requires_actual_frozen_state_and_correct_masked_loss_accounting(self):
        for key, value in (('teacher_eval', False), ('teacher_frozen', False), ('teacher_gradients_absent', False),
                           ('teacher_state_unchanged', False), ('teacher_forward_batches', 1),
                           ('known_other_class_entries', 31), ('weighted_mean_batch_loss', .25),
                           ('excluded_primary_classes', []), ('weight', 1.)):
            values = self.sampling_fixture(); values[0]['strong'][0]['retention_distillation'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate_fixture(values)

    def test_optimizer_attempts_updates_and_skips_require_consistent_integers(self):
        for key, value in (('attempted_batches', 3), ('actual_optimizer_steps', 0),
                           ('amp_skipped_steps', -1), ('actual_optimizer_steps', True)):
            values = self.sampling_fixture(); values[0]['control'][0]['optimizer_step_diagnostics'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'optimizer'): self.validate_fixture(values)

    def test_teacher_parameter_and_buffer_digest_cannot_change_or_allow_gradients(self):
        proof = {'initial_state_sha256': 'a' * 64, 'final_state_sha256': 'a' * 64,
                 'initial_weights_sha256': 'b' * 64, 'unchanged': True, 'eval_mode': True,
                 'all_parameters_frozen': True, 'no_parameter_gradients': True,
                 'student_parameter_count': 3244151, 'teacher_parameter_count': 3244151,
                 'teacher_state_tensor_count': 324, 'teacher_forward_both_arms': True}
        self.assertEqual(verify.validate_teacher_preservation(proof, 'a' * 64, 'b' * 64), proof)
        for key, value in (('final_state_sha256', 'c' * 64), ('no_parameter_gradients', False),
                           ('teacher_state_tensor_count', 323), ('eval_mode', False)):
            modified = dict(proof); modified[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.validate_teacher_preservation(modified, 'a' * 64, 'b' * 64)

    def test_private_snapshot_binds_file_bytes_size_and_mtime(self):
        with tempfile.TemporaryDirectory(dir=verify.ROOT / 'runs') as directory:
            root = Path(directory); (root / 'data').mkdir(); path = root / 'data' / 'x.png'; path.write_bytes(b'original')
            first = verify.snapshot_inputs(root, ['data/x.png'])
            self.assertEqual(first, verify.snapshot_inputs(root, ['data/x.png']))
            path.write_bytes(b'modified')
            self.assertNotEqual(first, verify.snapshot_inputs(root, ['data/x.png']))
            with self.assertRaises(ValueError): verify.snapshot_inputs(root, ['../escape.png'])

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
