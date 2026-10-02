import copy
import unittest

from scripts.facility_spatial_manifest import validate_replacement, RECIPE


class ReplacementManifestTests(unittest.TestCase):
    def fixtures(self):
        full = {'image': 'train/photo.jpg', 'targets': [1, 0, -1, -1, -1, -1, -1], 'domain': 'damsegment', 'pixel_target': 'mask/full.npz'}
        crop = {'image': 'old/crop.jpg', 'targets': [1, 0, -1, -1, -1, -1, -1], 'domain': 'damsegment',
                'parent_image': full['image'], 'parent_split': 'train', 'pixel_target': 'mask/crop.npz'}
        core = {'split': 'train', 'classes': ['concrete_crack', 'concrete_spalling', 'rust_stain', 'exposed_rebar', 'wet_surface', 'efflorescence', 'surface_cavity'],
                'full_count': 1, 'items': [full, crop]}
        candidate = copy.deepcopy(core)
        candidate['items'][1].update(image='data/facility-small-region-training/images/crop.jpg',
                                     pixel_target='data/facility-small-region-training/masks/crop.npz', replacement_of_image=crop['image'])
        candidate['audit'] = {'status': 'prepared', 'recipe': RECIPE, 'source_manifest_sha256': 'core-hash', 'replaced_rows': 1}
        return core, candidate

    def test_verified_train_crop_keeps_full_parent(self):
        core, candidate = self.fixtures()
        self.assertIs(validate_replacement(core, candidate, 'core-hash'), candidate)

    def test_full_label_or_split_edit_is_rejected(self):
        for field in ('labels', 'split', 'source'):
            core, candidate = self.fixtures()
            if field == 'labels': candidate['items'][0]['targets'][0] = 0
            if field == 'split': candidate['split'] = 'val'
            if field == 'source': candidate['audit']['source_manifest_sha256'] = 'changed'
            with self.assertRaises(ValueError): validate_replacement(core, candidate, 'core-hash')

    def test_parent_unknown_or_absent_is_not_promoted(self):
        for index in (1, 2):
            core, candidate = self.fixtures()
            candidate['items'][1]['targets'][index] = 1
            with self.assertRaises(ValueError): validate_replacement(core, candidate, 'core-hash')

    def test_validation_parent_and_escaped_data_path_are_rejected(self):
        for field, value in (('parent_image', 'val/photo.jpg'), ('parent_split', 'val'),
                             ('image', 'data/facility-small-region-training/../val.jpg')):
            core, candidate = self.fixtures()
            candidate['items'][1][field] = value
            with self.assertRaises(ValueError): validate_replacement(core, candidate, 'core-hash')

    def test_replacement_count_must_match_rows(self):
        core, candidate = self.fixtures()
        candidate['audit']['replaced_rows'] = 2
        with self.assertRaises(ValueError): validate_replacement(core, candidate, 'core-hash')

    def test_windows_backslash_traversal_and_non_string_paths_are_rejected(self):
        for field in ('image', 'pixel_target'):
            for value in (r'data/facility-small-region-training/sub/..\..\..\..\outside.jpg', 42):
                core, candidate = self.fixtures()
                candidate['items'][1][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_replacement(core, candidate, 'core-hash')


if __name__ == '__main__':
    unittest.main()
