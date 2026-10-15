import unittest
from scripts.inspect_msdd_public_inventory import flatten_tree


class PublicArchiveTreeTests(unittest.TestCase):
    def test_metadata_only_nested_files(self):
        node = {'children': [{'name': 'x.xml', 'path': 'Annotations/x.xml', 'size': 100,
                              'id': 'a' * 32, 'type': 'file'}]}
        self.assertEqual(flatten_tree(node)[0]['path'], 'Annotations/x.xml')

    def test_traversal_and_duplicate_paths_rejected(self):
        row = {'name': 'x.xml', 'path': '../x.xml', 'size': 1, 'id': 'a' * 32, 'type': 'file'}
        with self.assertRaises(ValueError):
            flatten_tree({'children': [row]})
        row['path'] = 'Annotations/x.xml'
        with self.assertRaises(ValueError):
            flatten_tree({'children': [row, row]})


if __name__ == '__main__':
    unittest.main()
