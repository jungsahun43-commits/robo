"""Focused proof mutations for actual sampler/update/source preservation."""
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from scripts import verify_facility_subtype as verify
from safelog_ai.spalling_sampler import draw_sha256


class SubtypeVerificationTests(TestCase):
    def sampling_fixture(self):
        # Rows 0 and 1 have the same two targets but intentionally different
        # other-five targets; only row1 belongs to the eligible subtype.
        targets = ([0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0])
        items = [{'image': f'data/full{i}.png', 'domain': d, 'targets': list(t), 'split': 'train'}
                 for i, (d, t) in enumerate(zip(('dacl', 'dacl', 'dacl', 'damsegment', 'codebrim'), targets))]
        items.append({'image': 'data/crop.png', 'parent_image': 'data/full0.png',
                      'parent_split': 'train', 'domain': 'dacl', 'targets': [-1] * 7})
        left = np.asarray([[0, 1, 2, 3, 4, 5, 0, 2]], dtype=np.int64)
        right = np.asarray([[1, 1, 2, 3, 4, 5, 0, 2]], dtype=np.int64)
        arrays = {'control': left, 'negative': right}; eligible = np.asarray([False, True, False, False, False, False])
        history = {}
        for variant, values in arrays.items():
            counts = verify.expected_counts(values[0], items, 5)
            row = {'epoch': 1, **counts, 'optimizer_step_diagnostics':
                   {'attempted_batches': 2, 'actual_optimizer_steps': 1, 'amp_skipped_steps': 1},
                   'subtype_sampling': {'eligible_draws': int(eligible[values[0]].sum()),
                       'changed_positions_from_control': 1 if variant == 'negative' else 0,
                       'declared_draw_sha256': draw_sha256(values[0]), 'draw_hash_matches_prepared': True}}
            history[variant] = [row]
        return history, arrays, items, 5, eligible

    def validate_fixture(self, values):
        return verify.validate_actual_sampling(*values, epochs=1, draws=8, batch_size=4)

    def test_changed_other_five_targets_are_allowed_with_fixed_exact_two_target_strata(self):
        proof = self.validate_fixture(self.sampling_fixture())
        self.assertEqual(proof['changed_positions'], 1)
        self.assertEqual(proof['eligible_draws_by_variant'], {'control': 1, 'negative': 2})
        self.assertTrue(proof['both_actual_draw_orders_match_fixed_prepared_arrays'])

    def test_actual_history_order_hash_mutation_is_rejected(self):
        values = self.sampling_fixture(); values[0]['negative'][0]['sampled_row_indices_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'epoch/order'): self.validate_fixture(values)

    def test_actual_source_or_crop_exposure_mutation_is_rejected(self):
        for key in ('sampled_domain_counts', 'sampled_row_type_counts', 'sampled_full_target_joint_counts'):
            values = self.sampling_fixture(); values[0]['negative'][0][key] = {}
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate_fixture(values)

    def test_replacement_cannot_change_a_target_or_existing_eligible_draw(self):
        for index, value in ((0, 2), (1, 0)):
            values = self.sampling_fixture(); values[1]['negative'][0, index] = value
            with self.subTest(position=index), self.assertRaises(ValueError): self.validate_fixture(values)

    def test_optimizer_attempts_updates_skips_need_actual_consistent_integers(self):
        for key, value in (('attempted_batches', 3), ('actual_optimizer_steps', 0),
                           ('amp_skipped_steps', -1), ('actual_optimizer_steps', True)):
            values = self.sampling_fixture(); values[0]['control'][0]['optimizer_step_diagnostics'][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'optimizer'): self.validate_fixture(values)

    def test_missing_epoch_or_changed_subtype_exposure_is_rejected(self):
        values = self.sampling_fixture(); values[0]['negative'] = []
        with self.assertRaises(ValueError): self.validate_fixture(values)
        values = self.sampling_fixture(); values[0]['negative'][0]['subtype_sampling']['eligible_draws'] = 1
        with self.assertRaisesRegex(ValueError, 'subtype sampling'): self.validate_fixture(values)

    def test_private_snapshot_binds_file_bytes_size_and_mtime(self):
        with tempfile.TemporaryDirectory(dir=verify.ROOT / 'runs') as directory:
            root = Path(directory); (root / 'data').mkdir(); path = root / 'data' / 'x.png'; path.write_bytes(b'original')
            first = verify.snapshot_inputs(root, ['data/x.png'])
            self.assertEqual(first, verify.snapshot_inputs(root, ['data/x.png']))
            path.write_bytes(b'modified')
            self.assertNotEqual(first, verify.snapshot_inputs(root, ['data/x.png']))
            with self.assertRaises(ValueError): verify.snapshot_inputs(root, ['../escape.png'])

    def test_actual_tests_require_complete_collection_and_no_skips(self):
        counts = {'tests/a.py': 2, 'tests/b.py': 3}
        record = {'tests_run': 5, 'expected_tests_collected': 5, 'tests_by_module': counts,
                  'failures': 0, 'errors': 0, 'skipped': 0}
        with patch.object(verify, 'MIN_TEST_COUNTS', counts):
            self.assertEqual(verify.validate_test_counts(record)['tests_run'], 5)
            for key, value in (('skipped', 1), ('tests_run', 4), ('expected_tests_collected', 6), ('failures', True)):
                modified = deepcopy(record); modified[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError): verify.validate_test_counts(modified)
