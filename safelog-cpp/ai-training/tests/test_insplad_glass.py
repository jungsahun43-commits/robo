from copy import deepcopy
import unittest
from PIL import Image
import torch
from safelog_ai.insplad_glass import SIDE,glass_view,glass_descriptor,assert_preserved,InspladGlassMain
from safelog_ai.insplad_classifier import InspladMain
from safelog_ai.gearbox_supervised import gearbox_descriptor


class GlassTests(unittest.TestCase):
    def test_unpadded_rectangle_geometry(self):
        view,box=glass_view(Image.new('RGB',(17,51),(255,0,0)))
        self.assertEqual(view.size,(384,384));self.assertEqual(box,[128,0,256,384])
        self.assertEqual(view.getpixel((0,0)),(127,127,127))
        self.assertEqual(view.getpixel((128,0)),(255,0,0))

    def test_padding_exclusion_and_full_descriptor(self):
        features={k:torch.ones(2,c,4,4) for k,c in (('fine',24),('low',40),('high',960))}
        for v in features.values():v[:,:,:,2:]=99
        desc=glass_descriptor(features,[[0,0,SIDE//2,SIDE]]*2)
        expected=[]
        for c in (24,40,960):expected.extend((torch.ones(2,c),torch.zeros(2,c),torch.ones(2,c)))
        self.assertTrue(torch.equal(desc,torch.cat(expected,1)))
        self.assertTrue(torch.equal(glass_descriptor(features,[[0,0,SIDE,SIDE]]*2),gearbox_descriptor(features)))
        with self.assertRaises(ValueError):glass_descriptor(features,[[0,0,1,1]]*2)

    def test_prior_research_state_and_methods_guard(self):
        parent={'state_dict':{'base.weight':torch.ones(2),'insplad_classifier.weight':torch.ones(3)},'insplad_scope':'rust'}
        candidate=deepcopy(parent);candidate['state_dict']['insplad_glass_classifier.weight']=torch.zeros(4)
        assert_preserved(parent,candidate)
        candidate['state_dict']['insplad_classifier.weight'][0]=0
        with self.assertRaises(ValueError):assert_preserved(parent,candidate)
        self.assertIs(InspladGlassMain.forward_surface,InspladMain.forward_surface)
        self.assertIs(InspladGlassMain.forward_insplad_upper_shackle,InspladMain.forward_insplad_upper_shackle)
