from copy import deepcopy
import unittest

from scripts.evaluate_rc_positive_complete import augmented_metadata


class AdapterTests(unittest.TestCase):
    def fixture(self):
        training = {"status": "complete", "classes": ["a", "b"], "weights_sha256": "original"}
        source = {"path": "data/codebrim-training/val.json", "sha256": "a" * 64}
        validation = {"split": "val", "classes": ["a", "b"], "items": [{}] * 611}
        return training, source, validation

    def test_virtual_link_does_not_modify_actual_metadata(self):
        training, source, validation = self.fixture(); before = deepcopy(training)
        new = augmented_metadata(training, source, validation)
        self.assertEqual(training, before)
        self.assertEqual(new["additional_validation"], source)
        self.assertEqual(new["weights_sha256"], "original")

    def test_rejects_wrong_split_size_or_existing_override(self):
        training, source, validation = self.fixture()
        for corrupted in ({**validation, "split": "test"}, {**validation, "items": [{}] * 610}, {**validation, "classes": ["b", "a"]}):
            with self.assertRaises(ValueError): augmented_metadata(training, source, corrupted)
        with self.assertRaises(ValueError): augmented_metadata({**training, "additional_validation": source}, source, validation)


if __name__ == "__main__": unittest.main()
