"""Catch image/mask/unknown-grid misalignment before any new fit."""
import unittest
from unittest.mock import patch
import torch
from scripts.study_unified_preservation_aligned import AlignedCache
from scripts.unified_main_data import normalize_batch


class AlignmentTests(unittest.TestCase):
    def make_cache(self):
        image=torch.arange(3*4*6,dtype=torch.uint8).reshape(3,4,6)
        truth={'mask':torch.arange(4*6).reshape(1,4,6)%2,
               'pixel_known':torch.tensor([[[0,0,1,1,1,1]]*4]),
               'valid':torch.tensor([[[0,1,1,1,1,1]]*4]),
               'labels':torch.tensor([1.,0.]),'known':torch.tensor([1.,0.])}
        cache=AlignedCache([],[]);cache.row=lambda index:(image,truth)
        return cache,image,truth

    def test_flip_keeps_unknown_and_padding_at_same_coordinates_as_image_mask(self):
        cache,image,truth=self.make_cache()
        with patch('torch.rand',return_value=torch.zeros(1)):
            inputs,actual=cache.batch([0],'cpu',training=True)
        self.assertTrue(torch.equal(inputs,normalize_batch(image.unsqueeze(0).flip(-1),'cpu')))
        for key in ('mask','pixel_known','valid'):
            self.assertTrue(torch.equal(actual[key],truth[key].float().unsqueeze(0).flip(-1)))
            self.assertEqual(float(actual[key].sum()),float(truth[key].sum()))
        self.assertTrue(torch.equal(actual['known'][0],truth['known']))

    def test_validation_never_changes_images_masks_or_known_scope(self):
        cache,image,truth=self.make_cache();inputs,actual=cache.batch([0],'cpu',training=False)
        self.assertTrue(torch.equal(inputs,normalize_batch(image.unsqueeze(0),'cpu')))
        for key in truth:self.assertTrue(torch.equal(actual[key][0],truth[key].float()))


if __name__=='__main__':unittest.main()
