import unittest
import torch
from safelog_ai.mpdd2_local_reference import white_local_features, reference_evidence, MPDD2ReferenceMain
from safelog_ai.real_mask_server import RuntimeRealMaskMain


class LocalReferenceTests(unittest.TestCase):
    def test_old_contracts_unchanged(self):
        self.assertIs(MPDD2ReferenceMain.forward_surface, RuntimeRealMaskMain.forward_surface)
        self.assertIs(MPDD2ReferenceMain.forward_real_mask, RuntimeRealMaskMain.forward_real_mask)

    def test_blank_background_has_valid_descriptor(self):
        x = white_local_features({'fine': torch.zeros(1, 24, 64, 64), 'low': torch.zeros(1, 40, 32, 32)})
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue(torch.all(x.norm(dim=1) > .99))

    def test_known_reference_and_orthogonal_query(self):
        reference = torch.zeros(1, 65)
        reference[:, 0] = 1
        query = torch.zeros(1, 65, 64, 64)
        query[:, 0] = 1
        _, same = reference_evidence(query, reference)
        self.assertLess(float(same[0]), 1e-6)
        query.zero_()
        query[:, 1] = 1
        _, different = reference_evidence(query, reference)
        self.assertGreater(float(different[0]), .99)


if __name__ == '__main__':
    unittest.main()
