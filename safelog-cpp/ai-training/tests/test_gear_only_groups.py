import unittest
from safelog_ai.gear_only_groups import allocate_gear_groups


class GearOnlyGroupsTests(unittest.TestCase):
    def test_viewed_groups_forced_fit_with_intact_capacity(self):
        rows=[{'group_id':str(g),'label':i%2,'phase':'fit','functional_component_category':'gear'} for g in range(14) for i in range(20)]
        result=allocate_gear_groups(rows,{'0','1'})
        self.assertEqual(result,allocate_gear_groups(list(reversed(rows)),{'1','0'}))
        self.assertEqual(result['0'],'fit');self.assertEqual(result['1'],'fit')
        self.assertEqual({p:list(result.values()).count(p) for p in ('fit','validation','cal')},{'fit':8,'validation':3,'cal':3})

    def test_original_test_and_other_categories_cannot_be_allocated(self):
        for phase,category in [('test','gear'),('fit','spline')]:
            with self.assertRaisesRegex(ValueError,'eligible TRAIN'):
                allocate_gear_groups([{'group_id':'0','label':0,'phase':phase,'functional_component_category':category}],set())


if __name__=='__main__':unittest.main()
