from copy import deepcopy
import unittest
from scripts.report_facility_retention_results import bind_verified_plan


class RetentionReportAdapterTests(unittest.TestCase):
    def fixture(self):
        source = {'frozen.py': 'a'*64}
        p = {'source_sha256': source, 'paired_draws_sha256': 'b'*64, 'sampler_plan_sha256': 'c'*64}
        t = [{'reference': True}] + [{'study_protocol_sha256': 'd'*64,
            'private_draw_archive_sha256': 'b'*64, 'source_sha256': source} for _ in range(2)]
        v = {'status': 'passed', 'protocol_sha256': 'd'*64, 'source_sha256': source,
             'actual_completed_training_epochs': 12, 'protected_files_unchanged': True,
             'working_runtime_sources_unchanged': True}
        return t, p, v

    def test_verified_missing_field_is_derived_only_in_copy(self):
        t, p, v = self.fixture(); before = deepcopy(t)
        bound = bind_verified_plan(t, p, 'd'*64, v)
        self.assertEqual(t, before)
        self.assertEqual(bound[0], t[0])
        for arm in bound[1:]: self.assertEqual(arm['sampler_plan_sha256'], p['sampler_plan_sha256'])

    def test_explicit_conflicting_field_or_unverified_binding_is_rejected(self):
        for key, value in (('status', 'failed'), ('protocol_sha256', 'e'*64),
                           ('actual_completed_training_epochs', 6), ('protected_files_unchanged', False)):
            t, p, v = self.fixture(); v[key] = value
            with self.assertRaises(ValueError): bind_verified_plan(t, p, 'd'*64, v)
        t, p, v = self.fixture(); t[1]['sampler_plan_sha256'] = 'e'*64
        with self.assertRaises(ValueError): bind_verified_plan(t, p, 'd'*64, v)
        t, p, v = self.fixture(); t[1]['private_draw_archive_sha256'] = 'e'*64
        with self.assertRaises(ValueError): bind_verified_plan(t, p, 'd'*64, v)
