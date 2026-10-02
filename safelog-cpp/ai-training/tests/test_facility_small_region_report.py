import copy
import hashlib
import json
import unittest

from scripts.report_facility_small_region import (
    CLASSES, DOMAINS, COUNTS, PAIR_KEYS, TARGETS,
    validate_pair, validate_frozen, measured_metrics,
    train_review_evidence,
)


class SmallRegionReportTests(unittest.TestCase):
    def pair(self):
        control = {key: 'fixed' for key in PAIR_KEYS}
        control.update(architecture='lraspp_mobilenet_facility_auxiliary_v1', classes=CLASSES,
                       seed=51, requested_epochs=6, actual_epochs=6, patience=6, imgsz=640,
                       draws_per_epoch=14248, backbone_lr=.00004, head_lr=.00025,
                       auxiliary_weight=.5, domain_proportions=[.7, .1, .2],
                       validation_domains=DOMAINS, initial_weights_sha256='initializer',
                       spatial_manifest_sha256='core', core_spatial_manifest_sha256='core')
        treatment = copy.deepcopy(control)
        treatment.update(spatial_manifest_sha256='derived', spatial_manifest_audit={
            'status': 'prepared', 'recipe': 'small_source_region_context_v1',
            'source_manifest_sha256': 'core', 'unchanged_full_count': 14248, 'replaced_rows': 12})
        return control, treatment

    def selection(self):
        points = {}
        for label in TARGETS:
            domains = {d: {'tp': 90, 'fn': 10, 'fp': 10, 'tn': n - 110,
                           'fnr': .1, 'fpr': 10 / (n - 100)} for d, n in COUNTS.items()}
            points[label] = {'threshold': .5, 'domains': domains}
        return {'weights_sha256': 'checkpoint', 'selection_split': 'val', 'classes': CLASSES,
                'configured_grids': [1], 'validation_counts': COUNTS,
                'selected': {'grid': 1, 'views': 1, 'worst_error': .1, 'target_passed': False, 'per_class': points}}

    def test_completed_matched_pair_accepts_crop_recipe_difference(self):
        control, treatment = self.pair()
        treatment['expected_label_sampling'] = {'a_crop_presence_can_change': .1}
        conditions = validate_pair(control, treatment, 'initializer')
        self.assertEqual(conditions['seed'], 51)
        self.assertEqual(conditions['actual_epochs'], 6)

    def test_missing_or_changed_pair_conditions_rejected(self):
        for key, value in (('seed', 52), ('actual_epochs', 5), ('head_lr', .001),
                           ('split_sha256', 'different'), ('training_script_sha256', 'different')):
            control, treatment = self.pair()
            treatment[key] = value
            with self.assertRaises(ValueError): validate_pair(control, treatment, 'initializer')
        control, treatment = self.pair()
        treatment.pop('auxiliary_manifest_sha256')
        with self.assertRaises(ValueError): validate_pair(control, treatment, 'initializer')
        with self.assertRaises(ValueError): validate_pair(*self.pair(), 'different-initializer')

    def test_running_wrong_weights_and_different_view_rejected(self):
        training = {'status': 'complete', 'weights_sha256': 'checkpoint', 'classes': CLASSES}
        selection = self.selection()
        self.assertEqual(validate_frozen(training, selection, 'checkpoint')['grid'], 1)
        with self.assertRaises(ValueError): validate_frozen({**training, 'status': 'running'}, selection, 'checkpoint')
        with self.assertRaises(ValueError): validate_frozen(training, selection, 'changed')
        selection['configured_grids'] = [1, 2, 3]
        with self.assertRaises(ValueError): validate_frozen(training, selection, 'checkpoint')
        self.assertEqual(validate_frozen(training, selection, 'checkpoint', full_only=False)['grid'], 1)
        selection['selected'].update(grid=2, views=5)
        with self.assertRaises(ValueError): validate_frozen(training, selection, 'checkpoint', full_only=False)

    def test_metrics_require_actual_counts_and_consistent_rates(self):
        selected = self.selection()['selected']
        result = measured_metrics(selected)
        self.assertEqual(result['per_class'][TARGETS[0]]['domains']['dacl']['positive_photos'], 100)
        self.assertEqual(result['worst_error'], .1)
        for mutation in ('missing', 'rate', 'zero_support', 'missing_source'):
            point = copy.deepcopy(selected)
            row = point['per_class'][TARGETS[0]]['domains']['dacl']
            if mutation == 'missing': row.pop('fn')
            if mutation == 'rate': row['fnr'] = .01
            if mutation == 'zero_support': row.update(tp=0, fn=0, tn=700)
            if mutation == 'missing_source': point['per_class'][TARGETS[0]]['domains'].pop('codebrim')
            with self.assertRaises(ValueError): measured_metrics(point)

    def test_review_proof_requires_real_matching_hashes_and_export_rows(self):
        package = {'schema': 'facility_train_review_v1', 'split': 'train', 'classes': CLASSES,
                   'weights_sha256': 'checkpoint', 'original_full_train_count': 14248,
                   'policy': 'Opinions only; no automatic target edits',
                   'provenance': {'cache_sha256': 'cache', 'manifest_sha256': 'manifest'},
                   'cases': [{'case_id': 'case1'}, {'case_id': 'case2'}]}
        content_hash = hashlib.sha256(json.dumps(package, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        package['package_content_sha256'] = content_hash
        verification = {'browser_photo_annotation_display_verified': True,
                        'browser_export_saved_json_verified': True, 'user_proposals_automatically_applied': False,
                        'weights_sha256': 'checkpoint', 'local_package_sha256': 'package',
                        'local_html_sha256': 'html', 'cache_sha256': 'cache',
                        'selected_photos': 2, 'export_rows': 2, 'original_full_train_count': 14248}
        proposals = {'schema': 'facility_train_review_proposals_v1', 'weights_sha256': 'checkpoint',
                     'source_package_sha256': content_hash, 'policy': package['policy'],
                     'cases': [{'case_id': 'case1', 'reason': 'unreviewed', 'note': ''},
                               {'case_id': 'case2', 'reason': 'unreviewed', 'note': ''}]}
        hashes = {'verification_sha256': 'verification', 'package_sha256': 'package', 'html_sha256': 'html',
                  'cache_sha256': 'cache', 'manifest_sha256': 'manifest', 'proposals_sha256': 'proposals'}
        result = train_review_evidence(verification, package, proposals, hashes, 'checkpoint', 'manifest')
        self.assertEqual(result['selected_photos'], 2)
        self.assertEqual(result['proposals_with_reason_or_note'], 0)
        self.assertEqual(result['automatic_original_label_edits'], 0)
        self.assertFalse(result['manual_review_completion_verified'])
        for mutation in ('missing_browser_proof', 'modified_package', 'different_export_package',
                         'missing_export_row', 'rewritten_target', 'different_file_hash', 'changed_manifest', 'missing_saved_hash'):
            v, p, e, h = map(copy.deepcopy, (verification, package, proposals, hashes))
            if mutation == 'missing_browser_proof': v.pop('browser_export_saved_json_verified')
            if mutation == 'modified_package': p['policy'] = 'Changed'
            if mutation == 'different_export_package': e['source_package_sha256'] = 'different'
            if mutation == 'missing_export_row': e['cases'].pop()
            if mutation == 'rewritten_target': e['cases'][0]['targets'] = [0] * 7
            if mutation == 'different_file_hash': h['html_sha256'] = 'different'
            if mutation == 'changed_manifest': h['manifest_sha256'] = 'changed'
            if mutation == 'missing_saved_hash': h.pop('proposals_sha256')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                train_review_evidence(v, p, e, h, 'checkpoint', 'manifest')


if __name__ == '__main__':
    unittest.main()
