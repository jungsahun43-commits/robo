import unittest
import torch
from safelog_ai.equipment_expansion import EquipmentHead, reconstruction_loss, stage_gate, photo_score


class EquipmentExpansionTests(unittest.TestCase):
    def test_normal_reconstruction_does_actual_neural_update_without_updating_features(self):
        torch.manual_seed(321); head=EquipmentHead(); target=torch.randn(2,1000,4,4,requires_grad=True)
        ids=torch.tensor([0,3]); before=head.decoder[-1].weight.detach().clone()
        opt=torch.optim.AdamW(head.parameters(),lr=.001); value=reconstruction_loss(head(target,ids),target)
        value.backward();opt.step()
        self.assertIsNone(target.grad);self.assertFalse(torch.equal(before,head.decoder[-1].weight))

    def test_missing_category_is_rejected(self):
        with self.assertRaises(ValueError):EquipmentHead()(torch.zeros(1,1000,4,4),torch.tensor([-1]))

    def test_unknown_or_tiny_evaluation_never_passes20percent(self):
        for args in [(None,.01,0,353),(.01,.01,3,353),(.01,.01,50,14)]:
            self.assertFalse(stage_gate(*args)['ready_for_next_stage'])

    def test_both_rates_must_pass_for_each_category(self):
        self.assertEqual(stage_gate(.1,.2,100,100)['state'],'target20_met')
        self.assertEqual(stage_gate(.15,.15,100,100)['state'],'target15_met')
        self.assertFalse(stage_gate(.01,.3,100,100)['ready_for_next_stage'])

    def test_photo_score_uses_top1percent_not_max(self):
        maps=torch.zeros(1,100,100);maps[0,0,0]=1
        self.assertAlmostEqual(float(photo_score(maps)),.01,places=6)


if __name__=='__main__':
    torch.set_num_threads(4);unittest.main()
