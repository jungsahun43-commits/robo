import unittest
from scripts.prepare_equipment_expansion import relative_member


class ExpansionDataTests(unittest.TestCase):
    def test_only_original_category_is_selected(self):
        config={'categories':['bracket_black']}
        self.assertEqual(str(relative_member('MPDD/bracket_black/train/good/001.png','mpdd',config)).replace('\\','/'),'bracket_black/train/good/001.png')
        for name in ['MPDD/annotated_image/bracket_black/test/hole/001.png','MPDD/prompt/bracket_black/test/hole/001.txt','MPDD/cropped/images/001.png']:
            self.assertIsNone(relative_member(name,'mpdd',config))

    def test_path_escape_is_rejected(self):
        for name in ['../cable/train/good/a.png','C:/cable/train/good/a.png','/cable/train/good/a.png']:
            with self.assertRaises(ValueError):relative_member(name,'mvtec_ad',{'categories':['cable']})

    def test_kolektor_ignores_scripts_overlays_and_example_files(self):
        for name in ['KolektorSDD2/split_weakly_0.pyb','KolektorSDD2/annotated_image/test/20000.png','KolektorSDD2/train/example.png']:
            self.assertIsNone(relative_member(name,'kolektor_sdd2',{}))
        self.assertIsNotNone(relative_member('KolektorSDD2/train/10000_GT.png','kolektor_sdd2',{}))


if __name__=='__main__':unittest.main()
