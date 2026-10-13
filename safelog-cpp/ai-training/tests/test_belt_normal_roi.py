import unittest
from PIL import Image,ImageDraw
from safelog_ai.belt_normal_roi import normal_core,provisional_split,fingerprints,distance,grouped


class NormalRoiTests(unittest.TestCase):
    def test_only_explicit_core_without_other_labels_is_known(self):
        self.assertEqual(normal_core([10,10,100,100],[],128,128,min_side=16),(30,30,90,90))
        for label in ['longitudinal tear','slight scratch','worn surface','objects','belt splice']:
            self.assertIsNone(normal_core([0,0,200,200],[(label,[95,95,10,10])],256,256))
        self.assertIsNone(normal_core([0,0,20,20],[],256,256))
        with self.assertRaises(ValueError): normal_core([0,0,float('nan'),100],[],256,256)

    def test_frame_gaps_and_publisher_holdout_are_never_fit(self):
        self.assertEqual(provisional_split('frame_00100_jpg.rf.abcd.jpg','train'),'fit')
        self.assertEqual(provisional_split('frame_00100.jpg','test'),'excluded_publisher_holdout')
        self.assertEqual(provisional_split('frame_00160.jpg','train'),'excluded_temporal_gap')
        self.assertEqual(provisional_split('frame_00200.jpg','train'),'cal')
        self.assertEqual(provisional_split('frame_00280.jpg','train'),'reserved')
        self.assertEqual(provisional_split('other.jpg','train'),'excluded_unknown_group')

    def test_hashes_catch_rotated_augmented_siblings(self):
        image=Image.new('RGB',(100,80),'gray'); d=ImageDraw.Draw(image); d.rectangle((8,15,34,58),fill='white'); d.ellipse((60,5,85,30),fill='black')
        hashes=fingerprints(image)
        self.assertEqual(distance(hashes,fingerprints(image.transpose(Image.Transpose.ROTATE_90))),0)
        rows=[{'parent':'a','rgb_sha256':'x','phash':hashes},{'parent':'a','rgb_sha256':'y','phash':['aaaaaaaaaaaaaaaa']},
            {'parent':'b','rgb_sha256':'z','phash':hashes}]
        self.assertEqual(grouped(rows),[0,0,0])


if __name__=='__main__': unittest.main()
