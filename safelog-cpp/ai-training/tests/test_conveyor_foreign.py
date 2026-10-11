import json
import sys
import unittest
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_conveyor_foreign import canonical_group, choose, PREVIEW_GROUP
from safelog_ai.conveyor_foreign import letterbox, summarize


class ConveyorForeignTests(unittest.TestCase):
    def test_group_selection_separates_recordings_and_source_test(self):
        inventory = json.loads((ROOT/'data/equipment-conveyor-foreign/curated-label-inventory.json').read_text(encoding='utf-8'))['rows']
        selected, groups, _ = choose(inventory)
        self.assertEqual(len(groups), 19)
        self.assertEqual(groups[PREVIEW_GROUP], 'train')
        phases = {s: {canonical_group(r['recording_name_hint']) for r in selected if r['split']==s} for s in ('train', 'val', 'test')}
        self.assertFalse(phases['train'] & phases['val'] | phases['train'] & phases['test'] | phases['val'] & phases['test'])
        self.assertTrue(all(r['source_split']!='test' for r in selected if r['split']!='test'))
        self.assertTrue(all(r['mechanical_health_label']=='unknown' and r['belt_crack_photo_label']=='unknown' for r in selected))

    def test_no_negative_denominator_cannot_pass(self):
        result = summarize([{'label': 1, 'score': .9}])
        self.assertEqual(result['fnr'], 0)
        self.assertIsNone(result['fpr'])
        self.assertFalse(result['photo_fpr_fnr_below_5_percent'])

    def test_boundary_is_fixed_strict_greater_than(self):
        result = summarize([{'label': 0, 'score': .5}, {'label': 1, 'score': .5}, {'label': 1, 'score': .8}])
        self.assertEqual((result['tn'], result['fn'], result['tp']), (1, 1, 1))
        with self.assertRaises(ValueError):
            summarize([{'label': 0, 'score': float('nan')}])

    def test_prepared_image_matches_serving_transform(self):
        manifest = json.loads((ROOT/'data/equipment-conveyor-foreign/grouped-photo-v1/manifest.json').read_text(encoding='utf-8'))
        row = next(r for r in manifest['rows'] if r['split']=='train')
        base = ROOT/'data/equipment-conveyor-foreign/grouped-photo-v1'
        with Image.open(base/row['native']) as photo, Image.open(base/row['image']) as prepared:
            import numpy as np
            expected = torch.from_numpy(np.array(prepared)).permute(2, 0, 1)
            self.assertTrue(torch.equal(letterbox(photo), expected))


if __name__=='__main__':
    unittest.main()
