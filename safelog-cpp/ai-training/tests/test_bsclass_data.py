import unittest
from scripts.prepare_bsclass import publisher_label, split_groups


class BSClassDataTests(unittest.TestCase):
    def test_only_author_N_P_names_receive_labels(self):
        self.assertEqual(publisher_label('training_data/N (10)_.png'), (0, 'N:10', True))
        self.assertEqual(publisher_label('P (1).png'), (1, 'P:1', False))
        with self.assertRaises(ValueError): publisher_label('unknown.png')

    def test_rotated_exact_and_near_images_cannot_cross_split(self):
        rows = [{'publisher_original_id': 'N:1', 'canonical_rgb_sha256': 'a', 'canonical_phash': '0000000000000000', 'label': 0},
                {'publisher_original_id': 'N:1', 'canonical_rgb_sha256': 'b', 'canonical_phash': 'ffffffffffffffff', 'label': 0},
                {'publisher_original_id': 'P:2', 'canonical_rgb_sha256': 'c', 'canonical_phash': '0000000000000001', 'label': 1}]
        audit = split_groups(rows)
        self.assertEqual(audit['groups'], 1); self.assertEqual(len({r['phase'] for r in rows}), 1)


if __name__ == '__main__': unittest.main()
