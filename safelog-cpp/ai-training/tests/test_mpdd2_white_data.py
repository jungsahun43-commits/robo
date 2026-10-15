import unittest
from scripts.prepare_mpdd2_white import source_role, split_groups


class MPDD2ReservationTests(unittest.TestCase):
    def test_only_reserved_source_roles_and_labels(self):
        self.assertEqual(source_role('bracket_white/test/anomalous/000.png'), ('test', 1))
        self.assertEqual(source_role('bracket_white/train/normal/194.png'), ('train', 0))
        for name in ('../bracket_white/test/normal/000.png', 'bracket_grey/test/normal/000.png',
                     'bracket_white/train/anomalous/000.png', 'bracket_white/test/normal/0.png'):
            with self.assertRaises(ValueError):
                source_role(name)

    def test_groups_allocated_once_order_independent(self):
        allocation = split_groups(range(100))
        self.assertEqual(allocation, split_groups(reversed(range(100))))
        self.assertEqual(len(allocation), 100)
        self.assertEqual([sum(value == phase for value in allocation.values())
                          for phase in ('fit', 'validation', 'cal')], [70, 15, 15])


if __name__ == '__main__':
    unittest.main()
