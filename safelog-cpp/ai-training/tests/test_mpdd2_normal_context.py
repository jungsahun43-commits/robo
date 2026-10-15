import inspect
import unittest
import torch
from safelog_ai.mpdd2_normal_context import WhiteContextHead, MPDD2NormalMain
from safelog_ai.real_mask_server import RuntimeRealMaskMain


class WhiteContextTests(unittest.TestCase):
    def test_original_location_contract_inherited(self):
        self.assertIs(MPDD2NormalMain.forward_surface, RuntimeRealMaskMain.forward_surface)
        self.assertIs(MPDD2NormalMain.forward_real_mask, RuntimeRealMaskMain.forward_real_mask)
        self.assertEqual(list(inspect.signature(MPDD2NormalMain.forward_surface).parameters),
                         ['self', 'original', 'view', 'category'])

    def test_invalid_feature_geometry_rejected(self):
        with self.assertRaises(ValueError):
            WhiteContextHead().standardize(torch.zeros(1, 1000, 4, 8))

    def test_loss_has_finite_decoder_gradients(self):
        torch.manual_seed(4040)
        head = WhiteContextHead()
        x = torch.randn(2, 1000, 4, 4) * .01
        loss = (head(x) - head.standardize(x)).square().mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in head.parameters()))


if __name__ == '__main__':
    unittest.main()
