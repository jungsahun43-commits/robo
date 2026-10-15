import unittest
import torch
from safelog_ai.gear_only_cache import select_descriptors


class GearOnlyCacheTests(unittest.TestCase):
    def test_joins_reordered_subset_by_identity(self):
        values=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
        self.assertTrue(torch.equal(select_descriptors(['c','a'],['a','b','c'],values),values[[2,0]]))

    def test_missing_or_duplicated_ids_are_rejected(self):
        for ids,source in [(['heldout'],['fit']),(['fit'],['fit','fit'])]:
            with self.assertRaises(ValueError):select_descriptors(ids,source,torch.ones(len(source),2))


if __name__=='__main__':unittest.main()
