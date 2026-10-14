import unittest
from PIL import Image
import torch
from safelog_ai.loco_context import ContextHead, ContextMain, context_view, context_targets
from tests.test_equipment_reference_photo import FakeBackbone


class ContextTests(unittest.TestCase):
    def test_global_bottleneck_can_affect_distant_output_cells(self):
        head = ContextHead().eval()
        with torch.no_grad():
            for name, parameter in head.named_parameters(): parameter.fill_(.001 if name.endswith('weight') else .01)
            a = torch.zeros(1, 1000, 4, 8); b = a.clone(); b[:, :, 0, 0] = 1
            changed = head(b)-head(a)
        self.assertGreater(float(changed[:, :, -1, -1].abs().max()), 0)

    def test_letterbox_and_small_input_review(self):
        view, meta = context_view(Image.new('RGB', (1700, 850), 'red'))
        self.assertEqual(view.size, (512, 256)); self.assertEqual(meta['offset'], [0, 0])
        with self.assertRaises(ValueError): context_view(Image.new('RGB', (32, 64)))

    def test_FP16_cached_target_and_original_location_are_preserved(self):
        model = ContextMain({}, {}, 'linear', {}).eval(); model.backbone = FakeBackbone()
        x = torch.zeros(1, 3, 256, 256); native = torch.zeros(1, 3, 256, 512)
        features = {'low': model.backbone.low, 'high': model.backbone.high}
        target = context_targets(features)
        self.assertTrue(torch.equal(target, target.half().float()))
        with torch.inference_mode():
            location, _, score = model.forward_context_and_location(x, native)
            self.assertEqual(model.backbone.calls, 2)
            self.assertTrue(torch.equal(location, model.forward_equipment(x, 'loco_splicing_connectors')))
        self.assertTrue(torch.isfinite(score).all())


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
