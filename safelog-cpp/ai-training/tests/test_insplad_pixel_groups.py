import unittest
from safelog_ai.insplad_pixel_groups import Groups,allocate


class PixelGroupTests(unittest.TestCase):
    def test_transitive_bridge_to_TEST_excludes_normal_candidate(self):
        g=Groups(4);g.union(0,1);g.union(2,3);g.union(1,2)
        candidate=str(g.find(0));held={str(g.find(3))}
        self.assertEqual(allocate([candidate],held,{}),{candidate:'excluded_TEST_connected'})

    def test_preserve_prior_roles_and_exclude_conflicts(self):
        roles={'a':{'fit'},'b':{'cal'},'c':{'fit','validation'}}
        self.assertEqual(allocate(['a','b','c'],set(),roles),{'a':'fit','b':'cal','c':'excluded_prior_role_conflict'})

    def test_allocation_order_independent_and_never_assigns_a_held_group(self):
        keys=[f'normal-{i}' for i in range(10)]+['held']
        a=allocate(keys,{'held'},{});b=allocate(list(reversed(keys)),{'held'}, {})
        self.assertEqual(a,b);self.assertEqual(a['held'],'excluded_TEST_connected')
        self.assertEqual(sorted(a.values()).count('fit'),6)
        self.assertEqual(sorted(a.values()).count('validation'),2)
        self.assertEqual(sorted(a.values()).count('cal'),2)
