import unittest
import numpy as np
from PIL import Image, ImageDraw
import torch
from safelog_ai.native_component_views import component_view, InputQualityError, NativeComponentMain


class NativeViewTests(unittest.TestCase):
    def example(self):
        image = Image.new('RGB', (512, 512), 'white'); draw = ImageDraw.Draw(image)
        draw.rectangle((80, 205, 440, 305), fill=(55, 55, 55)); draw.rectangle((60, 170, 110, 340), fill=(35, 35, 35))
        return image

    def test_pose_is_determined_from_input_and_rotation_keeps_head_left(self):
        image = self.example(); a, pa = component_view(image, 'ad_screw')
        b, pb = component_view(image.transpose(Image.Transpose.ROTATE_90), 'ad_screw')
        self.assertEqual(a.size, (768, 256)); self.assertFalse(pa['GT_or_anomaly_predictions_used'])
        self.assertLess(np.abs(np.array(a).astype(float)-np.array(b).astype(float)).mean(), 2.)
        self.assertGreater(pa['axis_ratio'], 3); self.assertGreater(pb['axis_ratio'], 3)

    def test_uniform_or_ambiguous_input_requires_review(self):
        with self.assertRaises(InputQualityError): component_view(Image.new('RGB', (512, 512), 'white'), 'ad_screw')
        with self.assertRaises(ValueError): component_view(self.example(), 'internal_motor_fault')

    def test_connector_aspect_ratio_is_preserved_and_letterbox_is_explicit(self):
        view, record = component_view(Image.new('RGB', (1700, 850), 'red'), 'loco_splicing_connectors')
        self.assertEqual(view.size, (1024, 512)); self.assertEqual(record['offset'], [0, 0])
        square, detail = component_view(Image.new('RGB', (500, 500), 'red'), 'loco_splicing_connectors')
        self.assertEqual(detail['offset'], [256, 0]); self.assertEqual(square.getpixel((0, 0)), (127, 127, 127))

    def test_unknown_native_method_is_rejected(self):
        with self.assertRaises(ValueError): NativeComponentMain({}, {'ad_screw': {'method': 'unknown', 'patches': 2}})


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
