import unittest
from safelog_ai.insplad_metadata import parse_member, annotation_binary_condition, filename_base_proxy


class SupervisedMetadataTests(unittest.TestCase):
    def test_original_multiclass_good_index_one_is_not_defect(self):
        self.assertEqual(annotation_binary_condition('vari-grip', 'good/a b.jpg 1')[2], 0)
        self.assertEqual(annotation_binary_condition('vari-grip', 'bird-nest/c.jpg 0')[2], 1)

    def test_related_crops_and_augmentations_keep_frame_group(self):
        a = parse_member('defect_supervised/yoke-suspension/train/rust/01-06-2021_DJI_0183_55_0.jpg')
        b = parse_member('defect_supervised/yoke-suspension/val/rust/01-06-2021_DJI_0183_56.jpg')
        self.assertEqual(a['frame_proxy'], b['frame_proxy'])
        self.assertTrue(b['reserved_evaluation'])
        self.assertEqual(filename_base_proxy('x/crop_55_0.jpg', {'crop_55', 'crop_55_0'}), 'crop_55')
        with self.assertRaises(ValueError):
            parse_member('../defect_supervised/yoke-suspension/train/rust/a.jpg')


if __name__ == '__main__':
    unittest.main()
