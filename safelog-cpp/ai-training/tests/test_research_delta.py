import unittest
from copy import deepcopy
import torch
from safelog_ai.research_delta import pack_delta,restore_delta


class ResearchDeltaTests(unittest.TestCase):
    def setUp(self):
        self.parent={'state_dict':{'base.weight':torch.tensor([1.,2.])},'version':'same'}
        self.candidate=deepcopy(self.parent);self.candidate['state_dict']['gear_head.weight']=torch.tensor([3.]);self.candidate['new_protocol']='new'

    def test_restore_exact_and_no_input_mutation(self):
        payload=pack_delta(self.parent,self.candidate,'a'*64,'runs/base/model.pt',('gear_',))
        rebuilt=restore_delta(self.parent,payload,'a'*64)
        for n,v in self.candidate['state_dict'].items():self.assertTrue(torch.equal(v,rebuilt['state_dict'][n]))
        self.assertEqual(rebuilt['new_protocol'],'new');self.assertNotIn('gear_head.weight',self.parent['state_dict'])

    def test_parent_mutation_rejected(self):
        self.candidate['state_dict']['base.weight'][0]=9.
        with self.assertRaisesRegex(ValueError,'parent tensor'):pack_delta(self.parent,self.candidate,'a'*64,'runs/base.pt',('gear_',))

    def test_wrong_parent_and_modified_overlay_rejected(self):
        payload=pack_delta(self.parent,self.candidate,'a'*64,'runs/base.pt',('gear_',))
        with self.assertRaisesRegex(ValueError,'identity'):restore_delta(self.parent,payload,'b'*64)
        payload['added_state_dict']['gear_head.weight'][0]=8.
        with self.assertRaisesRegex(ValueError,'Overlay'):restore_delta(self.parent,payload,'a'*64)


if __name__=='__main__':unittest.main()
