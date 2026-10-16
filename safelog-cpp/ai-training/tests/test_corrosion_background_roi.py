import struct,unittest
import numpy as np
from safelog_ai.corrosion_background_roi import rectangle,sparse_target,target512


def encoded(left,top,right,bottom,**unused):
    raw=bytearray(64);raw[:4]=b'Iout';struct.pack_into('>H',raw,4,228);raw[6]=1
    for i,value in zip((8,10,12,14),(top,left,bottom,right)):struct.pack_into('>h',raw,i,value)
    return raw


class BackgroundRoiTest(unittest.TestCase):
    def test_half_open_native_bounds_without_coordinate_shift(self):
        box=rectangle(encoded(10,20,15,25));self.assertEqual((box['width'],box['height']),(5,5))
        positives=[rectangle(encoded(i*6,0,i*6+5,5)) for i in range(5)]
        negatives=[rectangle(encoded(i*6,10,i*6+5,15)) for i in range(5)]
        t=sparse_target((40,20),positives,negatives)
        self.assertEqual(int((t==1).sum()),125);self.assertEqual(int((t==0).sum()),125)
        self.assertEqual(t[0,4],1);self.assertEqual(t[0,5],255);self.assertEqual(t[5,0],255)

    def test_unknown_geometry_and_cross_label_overlap_rejected(self):
        raw=encoded(1,2,6,7);struct.pack_into('>H',raw,50,128)
        with self.assertRaises(ValueError):rectangle(raw)
        boxes=[rectangle(encoded(i*6,0,i*6+5,5)) for i in range(5)]
        with self.assertRaises(ValueError):sparse_target((40,20),boxes,boxes)

    def test_unknown_and_padding_never_become_negative_on_resize(self):
        t=np.full((20,40),255,dtype=np.uint8);t[2:7,3:8]=1
        small,g=target512(t);self.assertEqual(g['top'],128)
        self.assertEqual(int((small==0).sum()),0);self.assertTrue(np.all(small[:128]==255))


if __name__=='__main__':unittest.main()
