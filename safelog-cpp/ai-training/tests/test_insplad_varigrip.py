import unittest
import torch
from safelog_ai.insplad_varigrip import CONDITIONS,condition_decisions,InspladVariGripMain
from safelog_ai.insplad_suspension import InspladSuspensionMain
from safelog_ai.insplad_glass import assert_preserved


class VariGripTests(unittest.TestCase):
    def test_original_good_mapping_and_typed_threshold(self):
        self.assertEqual(CONDITIONS,('good','bird-nest','rust'))
        probabilities=torch.tensor([[.8,.1,.1],[.3,.6,.1],[.3,.1,.6],[.5,.2,.3]])
        scores,classes=condition_decisions(probabilities,.5)
        self.assertEqual(classes.tolist(),[0,1,2,0])
        self.assertTrue(torch.allclose(scores,torch.tensor([.2,.7,.7,.5])))

    def test_previous_forward_contracts_preserved(self):
        for name in ('forward_surface','forward_real_mask','forward_insplad_upper_shackle','forward_insplad_glass','forward_insplad_suspension'):
            self.assertIs(getattr(InspladVariGripMain,name),getattr(InspladSuspensionMain,name))

    def test_old_three_heads_and_metadata_protected(self):
        old={'state_dict':{n:torch.ones(1) for n in ('base','insplad_classifier.a','insplad_glass_classifier.b','insplad_suspension_classifier.c')},'prior_metadata':{'selected':2}}
        new={'state_dict':{**old['state_dict'],'insplad_varigrip_classifier.d':torch.zeros(1)},'prior_metadata':{'selected':2},'extra_metadata':True}
        assert_preserved(old,new)
        for key in old['state_dict']:
            changed={**new,'state_dict':{**new['state_dict'],key:torch.zeros(1)}}
            with self.assertRaises(ValueError):assert_preserved(old,changed)
        with self.assertRaises(ValueError):assert_preserved(old,{**new,'prior_metadata':{'selected':1}})
