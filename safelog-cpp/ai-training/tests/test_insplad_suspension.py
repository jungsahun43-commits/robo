from copy import deepcopy
import unittest
import torch
from safelog_ai.insplad_suspension import InspladSuspensionMain
from safelog_ai.insplad_glass import InspladGlassMain,assert_preserved


class SuspensionTests(unittest.TestCase):
    def test_all_previous_research_namespaces_protected(self):
        prior={'state_dict':{'backbone.a':torch.ones(1),'insplad_classifier.a':torch.ones(2),'insplad_glass_classifier.a':torch.ones(3)},
               'insplad_scope':'rust','insplad_glass_scope':'missingcap'}
        candidate=deepcopy(prior);candidate['state_dict']['insplad_suspension_classifier.a']=torch.zeros(4)
        assert_preserved(prior,candidate)
        for name in prior['state_dict']:
            changed=deepcopy(candidate);changed['state_dict'][name][0]=0
            with self.assertRaises(ValueError):assert_preserved(prior,changed)
        changed=deepcopy(candidate);changed.pop('insplad_glass_scope')
        with self.assertRaises(ValueError):assert_preserved(prior,changed)

    def test_distinct_routing_and_inherited_contracts(self):
        self.assertIs(InspladSuspensionMain.forward_surface,InspladGlassMain.forward_surface)
        self.assertIs(InspladSuspensionMain.forward_real_mask,InspladGlassMain.forward_real_mask)
        self.assertIs(InspladSuspensionMain.forward_insplad_upper_shackle,InspladGlassMain.forward_insplad_upper_shackle)
        self.assertIs(InspladSuspensionMain.forward_insplad_glass,InspladGlassMain.forward_insplad_glass)
        stub=InspladSuspensionMain.__new__(InspladSuspensionMain);torch.nn.Module.__init__(stub)
        stub.insplad_suspension_classifier=torch.nn.Identity()
        value=torch.randn(2,3072)
        self.assertIs(stub.forward_insplad_suspension_descriptor(value),value)
