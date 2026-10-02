import unittest
import torch
from scripts.facility_hard_sampling import hard_multipliers


class HardSamplingTests(unittest.TestCase):
    def test_unknown_labels_do_not_boost_exposure(self):
        weights=hard_multipliers([[-1,-1,1],[1,0,-1]],[[1.,1.,0.],[.1,.9,1.]],[0,1])
        self.assertEqual(weights[0].item(),1.)
        self.assertAlmostEqual(weights[1].item(),2.62)

    def test_confident_wrong_training_examples_are_bounded(self):
        weights=hard_multipliers([[1,0],[1,0],[0,1]],[[1.,0.],[0.,1.],[.5,.5]],[0,1])
        self.assertTrue(torch.allclose(weights,torch.tensor([1.,3.,1.5],dtype=torch.float64)))

    def test_invalid_scores_and_unasserted_targets_rejected(self):
        for labels,scores,strength in (([[2]],[[.5]],2),([[1]],[[float('nan')]],2),
                                      ([[0]],[[1.1]],2),([[0]],[[.5]],float('inf'))):
            with self.assertRaises(ValueError):hard_multipliers(labels,scores,[0],strength)
