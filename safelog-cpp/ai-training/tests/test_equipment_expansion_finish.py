import unittest
import torch
from scripts.finish_equipment_expansion import percentile995


class PercentileRepairTests(unittest.TestCase):
    def test_same_linear_quantile_definition_as_torch(self):
        for values in [torch.linspace(0,2,5000),torch.tensor([.3,.2,.9,.1])]:
            self.assertAlmostEqual(percentile995(values),float(torch.quantile(values,.995)),places=6)

    def test_large_calibration_array_has_no_torch_size_limit(self):
        values=torch.ones(2**24+1,dtype=torch.float32)*.25
        self.assertEqual(percentile995(values),.25)


if __name__=='__main__':torch.set_num_threads(4);unittest.main()
