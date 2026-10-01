import unittest
from scripts.prepare_damsegment import targets_from_categories, grouped_splits
from scripts.refine_damsegment import acceptable

CLASSES = ["concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar",
           "wet_surface", "efflorescence", "surface_cavity"]


class DamSegmentTests(unittest.TestCase):
    def test_normals_do_not_label_unannotated_defects_negative(self):
        self.assertEqual(targets_from_categories(set(), CLASSES), [0, 0, -1, -1, -1, -1, -1])
        self.assertEqual(targets_from_categories({1}, CLASSES), [0, 1, -1, -1, -1, -1, -1])
        with self.assertRaises(ValueError): targets_from_categories({2}, CLASSES)

    def test_similarity_components_and_extension_aliases_share_split(self):
        # a~b, b~c within six bits; a~c is twelve bits. Transitive grouping is required.
        items = [{"patch_id": str(i), "pixel_sha256": str(i), "dhash": f"{h:016x}"}
                 for i, h in enumerate((0, 63, 4095, 2**63))]
        items[3]["patch_id"] = items[0]["patch_id"]
        sizes = grouped_splits(items)
        self.assertEqual(sizes, [4])
        self.assertEqual(len({item["split"] for item in items}), 1)
        self.assertEqual(len({item["group_id"] for item in items}), 1)

    def test_per_label_release_rejects_a_false_positive_regression(self):
        self.assertFalse(acceptable([
            {"round1": {"fp": 10, "fn": 10}, "round4": {"fp": 9, "fn": 9}},
            {"round1": {"fp": 0, "fn": 5}, "round4": {"fp": 1, "fn": 1}}]))


if __name__ == "__main__": unittest.main()
