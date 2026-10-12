"""Prove freeze integrity, identical facility exposure and honest selection guards."""
import unittest
import numpy as np
import torch
from scripts.study_unified_preservation import configure,schedule,research_selection
from safelog_ai.unified_main import UnifiedMain
from safelog_ai.retention_distillation import state_sha256


class PreservationTests(unittest.TestCase):
    def test_frozen_backbone_has_no_gradient_or_optimizer_state_but_foreign_head_learns(self):
        torch.manual_seed(1);model=UnifiedMain();opt=torch.optim.AdamW(configure(model,'frozen'),lr=.001)
        before=state_sha256(model.backbone);head=state_sha256(model.foreign_head)
        value=model.forward_task(torch.randn(2,3,128,128),'conveyor_foreign').square().mean()
        value.backward();opt.step()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(all(p not in opt.state for p in model.backbone.parameters()))
        self.assertEqual(before,state_sha256(model.backbone));self.assertNotEqual(head,state_sha256(model.foreign_head))

    def test_more_facility_per_epoch_keeps_exact_old_total_facility_draws(self):
        old={'facility_surface':np.arange(20*512).reshape(20,512),
             'belt_damage':np.arange(20*1024).reshape(20,1024),
             'conveyor_foreign':np.arange(20*1024).reshape(20,1024)}
        draws=schedule(old)
        self.assertEqual(draws['facility_surface'].shape,(10,1024))
        np.testing.assert_array_equal(np.sort(draws['facility_surface'].flatten()),old['facility_surface'].flatten())
        np.testing.assert_array_equal(np.sort(draws['belt_damage'].flatten()),old['belt_damage'][:10].flatten())

    def test_photo_improvement_cannot_hide_location_regression_or_invent_belt_FPR(self):
        baseline={'facility_primary_max':.25,'belt':{'photo_rates':{'fnr':.02},'pixel_iou':.4}}
        rows={v:{'facility_primary_max':.20,'belt':{'photo_rates':{'fnr':.01},'pixel_iou':.3}} for v in ['joint','frozen']}
        result=research_selection(rows,baseline)
        self.assertEqual(result['selected'],'prior_v1')
        self.assertFalse(result['normal_belt_photo_FPR_available']);self.assertFalse(result['foreign_TEST_used_for_selection'])


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
