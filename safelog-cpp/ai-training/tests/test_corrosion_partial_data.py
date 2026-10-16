import unittest
import numpy as np
from safelog_ai.corrosion_partial_data import partial_target,mask_view,pixel_counts


class PartialDataTest(unittest.TestCase):
    def test_vt_good_and_ignore_never_become_binary_background(self):
        raw=np.array([[0,1,2,3,255]],dtype=np.uint8);before=raw.copy()
        np.testing.assert_array_equal(partial_target('vt',raw),[[255,1,1,1,255]])
        np.testing.assert_array_equal(raw,before)

    def test_actual_fullmask_background_and_169_positive_preserved(self):
        raw=np.array([[0,169],[169,0]],dtype=np.uint8)
        self.assertEqual(pixel_counts(partial_target('ostrava',raw)),{'positive':2,'known_negative':2,'unknown':0,'padding':0})
        with self.assertRaises(ValueError):partial_target('ostrava',np.array([[255]],dtype=np.uint8))

    def test_letterbox_coordinates_and_unknown_padding(self):
        raw=np.array([[255,1,1,255],[255,255,1,255]],dtype=np.uint8)
        view,padding,g=mask_view(raw)
        self.assertEqual(g,{'native_size':[4,2],'resized_size':[512,256],'left':0,'top':128})
        self.assertTrue(np.all(view[:128]==255));self.assertTrue(np.all(view[384:]==255))
        self.assertEqual(pixel_counts(view,padding)['known_negative'],0)
        self.assertEqual(pixel_counts(view,padding)['padding'],512*256)
        self.assertEqual(view[200,200],1)


if __name__=='__main__':unittest.main()
