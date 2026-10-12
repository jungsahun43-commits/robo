"""Check real author data shapes and preserve unknown class identity under a flip."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.study_unified_preservation_typed import TypedCache,ROOT
from scripts.unified_main_data import DATA,read,normalize_batch


class TypedKnownTests(unittest.TestCase):
    def test_real_three_sources_have_class_vector_not_spatial_known_mask(self):
        manifest=read(DATA/'manifest.json')
        for domain in ['dacl','damsegment','codebrim']:
            row=next(r for r in manifest['fit']['facility_surface'] if r['domain']==domain)
            with np.load(ROOT/row['pixel_target'],allow_pickle=False) as target:
                self.assertEqual(target['known'].shape,(7,));self.assertEqual(target['mask'].shape,(7,80,80))

    def test_forced_flip_never_reorders_known_classes_but_flips_pixel_coordinates(self):
        image=torch.arange(3*4*6,dtype=torch.uint8).reshape(3,4,6)
        truth={'mask':torch.arange(7*4*6).reshape(7,4,6)%2,
               'pixel_known':torch.tensor([1.,1.,0.,0.,0.,0.,0.])}
        cache=TypedCache([],[]);cache.row=lambda index:(image,truth)
        with patch('torch.rand',return_value=torch.zeros(1)):
            x,result=cache.batch([0],'cpu',training=True)
        self.assertTrue(torch.equal(x,normalize_batch(image.unsqueeze(0).flip(-1),'cpu')))
        self.assertTrue(torch.equal(result['mask'][0],truth['mask'].flip(-1).float()))
        self.assertTrue(torch.equal(result['pixel_known'][0],truth['pixel_known']))
        self.assertFalse(torch.equal(result['pixel_known'][0],truth['pixel_known'].flip(-1)))


if __name__=='__main__':unittest.main()
