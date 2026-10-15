import unittest
from safelog_ai.ostrava_split import assign


class SplitTests(unittest.TestCase):
    def test_test_and_validation_priority_never_moves_original_test(self):
        rows=[{'id':i,'publisher_role':role,'group_id':g} for i,role,g in (
            ('test','test','a'),('near_train','train','a'),('near_val','validation','a'),
            ('val','validation','b'),('near_val_train','train','b'),('external_test','test','c'),
            ('external_train','train','c'))]
        phases={r['id']:r['phase'] for r in assign(rows,{'c'})}
        self.assertEqual(phases,{'test':'test','near_train':'excluded_TEST_connected','near_val':'excluded_TEST_connected',
            'val':'validation','near_val_train':'excluded_publisher_VAL_connected','external_test':'test','external_train':'excluded_TEST_connected'})

    def test_deterministic_whole_group_allocation(self):
        rows=[{'id':str(i),'publisher_role':'train','group_id':str(i//2)} for i in range(22)]
        a={r['id']:r['phase'] for r in assign(rows,set())}
        b={r['id']:r['phase'] for r in assign(list(reversed(rows)),set())}
        self.assertEqual(a,b)
        self.assertEqual(sum(v=='cal' for v in a.values()),6)
        self.assertTrue(all(a[str(i)]==a[str(i+1)] for i in range(0,22,2)))
        self.assertFalse({'test','validation'}&set(a.values()))


if __name__=='__main__':unittest.main()
