from safelog_ai.gearbox_groups import stratified_allocation
import unittest


class GearboxGroupsTests(unittest.TestCase):
    def test_groups_stay_intact_and_allocation_is_order_invariant(self):
        rows = [{'group_id': str(g), 'label': (g + i) % 2, 'phase': 'fit'} for g in range(20) for i in range(10 + g)]
        split = stratified_allocation(rows)
        self.assertEqual(split, stratified_allocation(list(reversed(rows))))
        self.assertEqual({p: list(split.values()).count(p) for p in ('fit', 'validation', 'cal')}, {'fit': 14, 'validation': 3, 'cal': 3})
        self.assertEqual(set(split), {r['group_id'] for r in rows})


    def test_reserved_test_cannot_be_reallocated(self):
        with self.assertRaisesRegex(ValueError, 'nonTEST'):
            stratified_allocation([{'group_id': 'a', 'label': 0, 'phase': 'test'}])


    def test_insufficient_groups_stop_before_learning(self):
        with self.assertRaisesRegex(ValueError, '14'):
            stratified_allocation([{'group_id': 'a', 'label': 0, 'phase': 'fit'}])


if __name__ == '__main__':
    unittest.main()
