import unittest
import torch
from safelog_ai.gearbox_supervised import gearbox_descriptor, DIM


class GearboxDescriptorTests(unittest.TestCase):
    def test_descriptor_matches_explicit_statistics_and_roundtrip(self):
        torch.manual_seed(26)
        features = {k: torch.randn(2, c, 4, 4) for k,c in (('fine',24),('low',40),('high',960))}
        expected = []
        for value in features.values():
            v = value.half().float().flatten(2)
            expected += [v.mean(2), v.std(2, correction=0), v.amax(2)]
        result = gearbox_descriptor(features)
        self.assertEqual(tuple(result.shape), (2, DIM))
        self.assertTrue(torch.equal(result, torch.cat(expected, 1).half().float()))

    def test_population_std_has_no_nan_for_single_spatial_value(self):
        features = {k: torch.ones(1,c,1,1) for k,c in (('fine',24),('low',40),('high',960))}
        self.assertTrue(torch.isfinite(gearbox_descriptor(features)).all())

    def test_rejects_different_encoder(self):
        with self.assertRaises(ValueError):
            gearbox_descriptor({'fine':torch.zeros(1,25,4,4)})


if __name__ == '__main__': unittest.main()
