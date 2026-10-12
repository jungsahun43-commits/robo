import sys
import unittest
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_conveyor_expansion import choose, DATA, OLD, SOURCE, read, sha


class ConveyorExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = read(OLD/'manifest.json')
        cls.selected = choose(read(SOURCE/'curated-label-inventory.json')['rows'],cls.prior)

    def test_training_never_takes_source_test_or_heldout_groups(self):
        for row in self.selected:
            if row['split']=='train':
                self.assertNotEqual(row['source_split'],'test')
                self.assertEqual(self.prior['group_splits'][row['recording_group']],'train')

    def test_complete_original_val_and_test_unchanged(self):
        for phase in ('val','test'):
            self.assertEqual({r['id'] for r in self.selected if r['split']==phase},
                             {r['id'] for r in self.prior['rows'] if r['split']==phase})
        old = {r['id']:r for r in self.prior['rows']}
        for row in read(DATA/'manifest.json')['rows']:
            if row['from_completed_v1']:
                self.assertEqual(sha(DATA/row['native']),old[row['id']]['native_sha256'])
                self.assertEqual(sha(DATA/row['image']),old[row['id']]['image_sha256'])

    def test_additional_audit_contains_only_unused_val_frames(self):
        prior_ids = {r['id'] for r in self.prior['rows']}
        additional = [r for r in self.selected if r['split']=='audit_val']
        self.assertTrue(additional)
        self.assertTrue(all(r['id'] not in prior_ids and r['source_split']!='test'
                            and self.prior['group_splits'][r['recording_group']]=='val' for r in additional))

    def test_no_exact_rgb_or_group_leak_and_health_stays_unknown(self):
        rows = read(DATA/'manifest.json')['rows']
        self.assertEqual(len({r['decoded_rgb_sha256'] for r in rows}),len(rows))
        phases = defaultdict(set)
        for row in rows:
            phase = 'val' if row['split']=='audit_val' else row['split']
            phases[phase].add(row['recording_group'])
            self.assertEqual(row['mechanical_health_label'],'unknown')
            self.assertEqual(row['belt_crack_photo_label'],'unknown')
        self.assertFalse(phases['train'] & phases['val'] | phases['train'] & phases['test'] | phases['val'] & phases['test'])


if __name__=='__main__':
    unittest.main()
