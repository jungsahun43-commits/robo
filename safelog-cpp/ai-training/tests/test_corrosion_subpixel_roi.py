import struct,unittest
import numpy as np
from safelog_ai.corrosion_subpixel_roi import subpixel_rectangle,provided_target


def roi(x,y,w,h):
    a=bytearray(64);a[:4]=b'Iout';struct.pack_into('>H',a,4,227);a[6]=1;struct.pack_into('>H',a,50,128);struct.pack_into('>ffff',a,18,x,y,w,h)
    return a


class SubpixelRoiTest(unittest.TestCase):
    def test_official_constructor_cast_and_ceil_are_not_size_snapping(self):
        b=subpixel_rectangle(roi(10.75,20.25,20.0000038,19.999992))
        self.assertEqual((b['left'],b['top'],b['right'],b['bottom']),(10,20,31,40))

    def test_all_provided_patches_union_and_unknown_scope(self):
        p=[subpixel_rectangle(roi(i*21,0,20,20)) for i in range(5)]
        n=[subpixel_rectangle(roi(i*21,30,20,20)) for i in range(5)]
        t,overlap=provided_target((120,60),p+p[:1],n)
        self.assertEqual(overlap,400);self.assertEqual(int((t==1).sum()),2000);self.assertEqual(int((t==0).sum()),2000)
        self.assertEqual(t[20,0],255)

    def test_conflicting_classes_or_outside_coordinates_are_not_repaired(self):
        p=[subpixel_rectangle(roi(i*21,0,20,20)) for i in range(5)]
        with self.assertRaises(ValueError):provided_target((120,60),p,p)
        with self.assertRaises(ValueError):provided_target((50,60),p,p)
        a=roi(0,0,float('nan'),20)
        with self.assertRaises(ValueError):subpixel_rectangle(a)


if __name__=='__main__':unittest.main()
