import unittest
import numpy as np
from safelog_ai.group_rank_metrics import label_group_weights,weighted_auc


class GroupRankMetricsTests(unittest.TestCase):
    def test_group_weights_balance_each_label_and_group(self):
        y=[0,0,0,1];w=label_group_weights(y,['a','a','b','c'])
        self.assertTrue(np.allclose(w,[.25,.25,.5,1.]))

    def test_weighted_ordering_and_ties(self):
        self.assertAlmostEqual(weighted_auc([0,1,0,1],[0,.5,.75,1],[1,2,3,1]),.5)
        self.assertAlmostEqual(weighted_auc([0,1],[.2,.2],[1,3]),.5)
        self.assertAlmostEqual(weighted_auc([0,1],[.1,.9],[2,1]),1.)

    def test_nonfinite_or_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):weighted_auc([0,1],[0,1],[0,1])


if __name__=='__main__':unittest.main()
