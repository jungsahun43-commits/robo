import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from expand_beltcrack_cached import selection
from safelog_ai.belt_evaluation import summarize


class ExpansionTests(unittest.TestCase):
    def test_test_members_and_incomplete_split_members_never_selected(self):
        members = [
            {'name': 'BeltCrack14ks/json/train/data1/0.json', 'volume': 0},
            {'name': 'BeltCrack14ks/json/train/data1/1.json', 'volume': 0},
            {'name': 'BeltCrack14ks/json/train/data1/1.json', 'volume': 1},
            {'name': 'BeltCrack14ks/json/test/data34/0.json', 'volume': 0}]
        with patch('expand_beltcrack_cached.byte_source', side_effect=lambda r, e: ('x',0) if r['volume']==0 else None):
            chosen, _, counts = selection(members, [{},{}], [])
        self.assertEqual([r['name'] for r in chosen], [members[0]['name']])
        self.assertEqual(counts, {'train/data1': 1})

    def test_prior_frames_preserved_outside_new_even_sample(self):
        members = [{'name': f'BeltCrack14ks/json/train/data1/{i}.json', 'volume': 0} for i in range(1000)]
        prior = [{'name': members[7]['name'], 'split':'train', 'sequence':'data1'}]
        with patch('expand_beltcrack_cached.byte_source', return_value=('x',0)):
            chosen, _, _ = selection(members, [{}], prior)
        self.assertIn(prior[0], chosen)
        self.assertEqual(len({r['name'] for r in chosen}), len(chosen))
        self.assertEqual(len(chosen), 97)

    def test_normal_denominator_missing_cannot_pass_five_percent(self):
        result = summarize([{'id':'a', 'sequence':'s', 'photo_positive':True, 'score':0.9,
                             'pixels':{'tp':5,'fp':0,'tn':95,'fn':0}}])
        self.assertIsNone(result['photo_rates']['fpr'])
        self.assertFalse(result['photo_fpr_fnr_below_5_percent'])

    def test_photo_presence_does_not_imply_localization(self):
        result = summarize([{'id':'a', 'sequence':'s', 'photo_positive':True, 'score':0.9,
                             'pixels':{'tp':0,'fp':5,'tn':90,'fn':5}}])
        self.assertEqual(result['photo_rates']['fnr'], 0)
        self.assertEqual(result['pixel_fnr'], 1)
        self.assertEqual(result['positive_photos_without_any_gt_overlap'], 1)


if __name__ == '__main__':
    unittest.main()
