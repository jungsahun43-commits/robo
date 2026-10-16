import unittest,numpy as np
from safelog_ai.native_corrosion_grid import grid,annotation_tile


class NativeGridTest(unittest.TestCase):
    def test_every_pixel_covered_including_last_edges_and_thin_axes(self):
        for w,h in ((1348,132),(513,1100),(200,1904)):
            coverage=np.zeros((h,w),dtype=np.uint8)
            for b in grid((w,h)):coverage[b['top']:b['top']+b['height'],b['left']:b['left']+b['width']]+=1
            self.assertTrue(np.all(coverage>0))

    def test_known_labels_roundtrip_over_overlap_without_unknown_to_negative(self):
        mask=np.full((617,1033),255,dtype=np.uint8);mask[10:30,1000:1020]=1;mask[590:610,3:23]=0
        reconstructed=np.full_like(mask,255)
        for b in grid((1033,617)):
            t=annotation_tile(mask,b);sub=t[:b['height'],:b['width']];reconstructed[b['top']:b['top']+b['height'],b['left']:b['left']+b['width']]=sub
        np.testing.assert_array_equal(reconstructed,mask)

    def test_padding_not_known_background_and_input_grid_independent_of_gt(self):
        m=np.full((100,200),255,dtype=np.uint8);m[1:3,1:3]=1;b=grid((200,100))[0];t=annotation_tile(m,b)
        self.assertTrue(np.all(t[100:]==255));self.assertTrue(np.all(t[:,200:]==255));self.assertEqual(int((t==0).sum()),0)


if __name__=='__main__':unittest.main()
