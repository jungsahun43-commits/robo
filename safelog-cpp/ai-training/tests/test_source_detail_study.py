import unittest
from safelog_ai.source_detail_study import source_only_rows, photo_bucket_means


class SourceDetailTests(unittest.TestCase):
    def test_subset_keeps_actual_source_roles_and_excludes_target_normal(self):
        def row(key, phase, source, neu=True, active=True):
            return {'photo_id': key, 'phase': phase, 'source_folder': source, 'is_NEU': neu,
                    'active_support': active, 'training_active': phase == 'fit',
                    'validation_active': phase == 'validation', 'bucket': 'real_positive', 'positive_pixels': 32}
        rows = [row('sdi', 'fit', 'NEU_SDI'), row('oil', 'validation', 'NEU_Oil'),
                row('target', 'fit', None, False), row('small', 'fit', 'NEU_SPDI', active=False)]
        result = source_only_rows(rows)
        self.assertEqual([r['photo_id'] for r in result['fit']], ['sdi'])
        self.assertEqual([r['photo_id'] for r in result['validation']], ['oil'])

    def test_validation_weights_photos_before_crops(self):
        rows = [{'photo_id': p, 'bucket': b} for p, b in [('a', 'real_positive'), ('a', 'real_positive'),
                ('b', 'real_positive'), ('a', 'annotation_negative')]]
        means = photo_bucket_means(rows, [1, 3, 10, 4])
        self.assertEqual(means, {'real_positive': 6, 'annotation_negative': 4})


if __name__ == '__main__':
    unittest.main()
