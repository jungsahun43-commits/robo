import unittest
import numpy as np
from PIL import Image
from safelog_ai.corrosion_condition_data import native_classes, polygon_classes, allocate


class CorrosionConditionDataTests(unittest.TestCase):
    def test_voc_palette_and_bgr_confusion(self):
        im = Image.fromarray(np.array([[0, 1, 2, 3]], dtype=np.uint8), 'P')
        palette = [0]*768
        palette[3:12] = [128, 0, 0, 0, 128, 0, 128, 128, 0]
        im.putpalette(palette)
        np.testing.assert_array_equal(native_classes(im), [[0, 1, 2, 3]])
        palette[3:6] = [0, 0, 128]
        im.putpalette(palette)
        with self.assertRaises(ValueError): native_classes(im)

    def test_polygon_order_ignore_and_coordinates(self):
        a = {'imageWidth': 8, 'imageHeight': 8, 'shapes': [
            {'label': '2_Fair_Steel_Corrosion', 'shape_type': 'polygon', 'points': [[1,1],[6,1],[6,6],[1,6]]},
            {'label': '__ignore__', 'shape_type': 'polygon', 'points': [[2,2],[3,2],[3,3],[2,3]]}]}
        m = polygon_classes(a, (8,8))
        self.assertEqual(int(m[0,0]), 0); self.assertEqual(int(m[1,1]), 1); self.assertEqual(int(m[2,2]), 255)
        a['imageWidth'] = 9
        with self.assertRaises(ValueError): polygon_classes(a, (8,8))

    def test_held_transitive_priority_and_prior_roles(self):
        keys = ['held', 'conflict', 'fit', 'validation', 'cal']+[str(i) for i in range(20)]
        roles = {'held': {'fit'}, 'conflict': {'fit','cal'}, 'fit': {'fit'}, 'validation': {'validation'}, 'cal': {'cal'}}
        a = allocate(keys, {'held'}, roles)
        self.assertEqual(a, allocate(reversed(keys), {'held'}, roles))
        self.assertEqual(a['held'], 'excluded_TEST_connected'); self.assertEqual(a['conflict'], 'excluded_prior_role_conflict')
        for k in ('fit','validation','cal'): self.assertEqual(a[k], k)
        self.assertEqual(sum(a[str(i)] == 'fit' for i in range(20)), 14)


if __name__ == '__main__': unittest.main()
