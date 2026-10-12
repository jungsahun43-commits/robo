import unittest
import torch
from safelog_ai.pcb_reverse import ReverseStudent,create_teacher,loss,anomaly_maps
from safelog_ai.equipment_anomaly import normal_calibration_threshold,classification_rates


class ReverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_decoder_preserves_scales_and_teacher_has_no_gradients(self):
        teacher = create_teacher(False)
        student = ReverseStudent().train()
        target = teacher(torch.randn(2,3,256,256))
        predicted = student(target)
        self.assertEqual([tuple(v.shape) for v in predicted],[(2,64,64,64),(2,128,32,32),(2,256,16,16)])
        value = loss(predicted,target);value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in student.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))

    def test_identical_features_have_zero_maps_and_mismatch_is_detected(self):
        targets = [torch.ones(1,c,s,s) for c,s in [(64,64),(128,32),(256,16)]]
        self.assertLess(float(anomaly_maps(targets,targets).abs().max()),1e-5)
        altered = [v.clone() for v in targets];altered[0][:,:,20:40,20:40] *= -1
        maps = anomaly_maps(altered,targets)
        self.assertEqual(tuple(maps.shape),(1,256,256))
        self.assertTrue(torch.isfinite(maps).all());self.assertGreater(float(maps.max()),.1)

    def test_teacher_response_does_not_skip_the_trainable_bottleneck(self):
        student = ReverseStudent()
        self.assertEqual(student.bottleneck[0][0].in_channels,448)
        self.assertEqual(student.bottleneck[0][0].out_channels,128)
        with self.assertRaises(ValueError):
            loss([torch.zeros(1,1,2,2)],[torch.zeros(1,1,2,2)])

    def test_normal_only_calibration_and_missing_denominator(self):
        self.assertEqual(normal_calibration_threshold(list(range(91))),87)
        row = classification_rates([1],[.2],.5)
        self.assertIsNone(row['fpr']);self.assertEqual(row['fnr'],1)


if __name__=='__main__':
    unittest.main()
