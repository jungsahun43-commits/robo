import unittest
import torch
from safelog_ai.ksdd_photo_only import KSDDPhotoMain
from tests.test_equipment_reference_photo import FakeBackbone


class KSDDPhotoOnlyTests(unittest.TestCase):
    def test_same_backbone_two_passes_and_original_location_is_exact(self):
        model = KSDDPhotoMain({}, {}, 'linear', {}).eval(); model.backbone = FakeBackbone()
        original = torch.zeros(1, 3, 256, 256); native = torch.zeros(1, 3, 704, 256); valid = torch.ones(1, 704, 256, dtype=torch.bool)
        with torch.inference_mode():
            location, auxiliary, score = model.forward_ksdd_photo(original, native, valid)
            self.assertEqual(model.backbone.calls, 2)
            reference = model.forward_equipment(original, 'ksdd2_surface')
        self.assertTrue(torch.equal(location, reference)); self.assertEqual(tuple(auxiliary.shape), (1, 704, 256)); self.assertTrue(torch.isfinite(score).all())


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
