import unittest
from PIL import Image
import numpy as np
import torch
from safelog_ai.magnetic_surface import surface_view,source_probability,photo_score
from scripts.prepare_magnetic_tiles import Bands


class RealMaskSurfaceTests(unittest.TestCase):
    def test_mask_threshold_alignment_and_original_restoration(self):
        torch.set_num_threads(4);a=np.zeros((100,200),dtype=np.uint8);a[20:40,30:50]=255;a[60:70,80:90]=100
        v,m,k,g=surface_view(Image.new('RGB',(200,100)),Image.fromarray(a))
        self.assertEqual(v.size,(512,512));self.assertEqual(g['offset'],[0,128])
        self.assertEqual(int(m[128+155:128+178,205:229].sum()),0)
        original=source_probability(m.float()[None],g)[0]
        self.assertGreater(float(original[22:38,32:48].mean()),.95);self.assertEqual(tuple(original.shape),(100,200))

    def test_padding_cannot_dominate_photo_score(self):
        p=torch.ones(1,512,512);_,_,valid,_=surface_view(Image.new('RGB',(200,100)))
        p[:,valid]=.2
        self.assertAlmostEqual(float(photo_score(p,valid[None])[0]),.2,places=6)

    def test_hamming_candidate_index_has_no_distance2_false_negatives(self):
        idx=Bands();h=0x125688fab3419017;idx.add([f'{h:016x}'],'source')
        for bits in ((0,23),(21,43),(22,63),(0,1),(32,33)):
            q=h^(1<<bits[0])^(1<<bits[1]);self.assertIn('source',idx.candidates([f'{q:016x}']))


if __name__=='__main__':unittest.main()
