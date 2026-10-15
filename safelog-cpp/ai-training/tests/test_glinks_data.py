import unittest
from collections import Counter
from safelog_ai.glinks_data import publisher_record, normal_group_split


class GLinksDataTests(unittest.TestCase):
    def test_original_role_label_proxy_and_reject_nonpublisher_paths(self):
        key = '05079651-a658-48ac-9586-a0c4b998cb7e-0022.jpg'
        for directory, role, label in (('train/good', 'train', 0), ('test/good', 'test', 0), ('test/defect', 'test', 1)):
            row = publisher_record(directory + '/' + key)
            self.assertEqual((row['publisher_role'], row['label'], row['frame_number']), (role, label, 22))
        for value in ('../test/good/' + key, '/test/good/' + key, 'train/defect/' + key,
                      'test/good/not-an-object.jpg', 'test/good/../good/' + key):
            with self.assertRaises(ValueError): publisher_record(value)

    def test_normal_split_cannot_depend_on_input_order_or_photo_multiplicity(self):
        groups = [str(i) for i in range(25)]
        first = normal_group_split(groups)
        self.assertEqual(first, normal_group_split(list(reversed(groups)) + groups[:8]))
        self.assertEqual(Counter(first.values()), {'fit': 17, 'validation': 4, 'cal': 4})
        self.assertEqual(set(first), set(groups))


if __name__ == '__main__': unittest.main()
