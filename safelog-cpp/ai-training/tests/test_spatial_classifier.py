import unittest
import torch
import numpy as np
from safelog_ai.spatial_classifier import SpatialClassifier,spatial_loss
from scripts.prepare_facility_spatial import masks_for


class SpatialTests(unittest.TestCase):
    def test_codebrim_positives_do_not_invent_pixel_regions(self):
        targets=[1,0,0,1,-1,0,-1]
        mask,known,conflicts=masks_for({'targets':targets},{'domain':'codebrim'},None,list(range(7)))
        self.assertEqual(int(mask.sum()),0)
        self.assertTrue(np.array_equal(known,[0,1,1,0,0,1,0]))
        self.assertEqual(conflicts,0)

    def test_unknown_pixel_categories_have_zero_gradient(self):
        logits=torch.zeros(1,2,8,8,requires_grad=True);target=torch.zeros_like(logits);target[0,0,:2,:2]=1
        loss=spatial_loss(logits,target,torch.tensor([[1.,0.]]),torch.ones(2));loss.backward()
        self.assertEqual(logits.grad[0,1].abs().sum().item(),0.)
        self.assertGreater(logits.grad[0,0].abs().sum().item(),0.)
        unknown=spatial_loss(logits,target,torch.zeros(1,2),torch.ones(2))
        self.assertEqual(unknown.item(),0.)

    def test_photo_contract_keeps_seven_logits_and_separate_coarse_maps(self):
        torch.set_num_threads(2);model=SpatialClassifier().eval()
        with torch.inference_mode():photo,maps=model.forward_details(torch.zeros(2,3,128,128))
        self.assertEqual(tuple(photo.shape),(2,7));self.assertEqual(tuple(maps.shape),(2,7,16,16))
        self.assertTrue(torch.isfinite(photo).all())
