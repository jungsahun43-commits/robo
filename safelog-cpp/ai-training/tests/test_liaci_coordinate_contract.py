import io,unittest,zipfile
from PIL import Image
from safelog_ai.liaci_coordinate_contract import checked_polygons,numeric_pair,read_binary_member
from safelog_ai.public_zip_metadata import end_directory,parse_directory


class LiaciCoordinateContractTest(unittest.TestCase):
    def test_binary_zip_crc_and_budget(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w',8) as z:z.writestr('mask.bmp',b'\x00\xff\x01')
        raw=b.getvalue();m=end_directory(raw,len(raw));row=parse_directory(raw[m['offset']:m['offset']+m['bytes']],1,len(raw))[0]
        get=lambda offset,count:raw[offset:offset+count]
        self.assertEqual(read_binary_member(row,get,100),b'\x00\xff\x01')
        with self.assertRaises(ValueError):read_binary_member({**row,'CRC32_declared':'00000000'},get,100)
        with self.assertRaises(ValueError):read_binary_member(row,get,1)

    def test_coordinates_preserve_floats_and_reject_unknown(self):
        a={'iscrowd':0,'segmentation':[[.5,.5,2.5,.5,2.5,2.5]]}
        self.assertEqual(checked_polygons([a],4,4)[0][0],(.5,.5))
        for bad in ({**a,'iscrowd':1},{**a,'segmentation':[[0,0,9,0,2,2]]},{**a,'segmentation':[[float('nan'),0,1,1,2,2]]}):
            with self.assertRaises(ValueError):checked_polygons([bad],4,4)

    def test_native_binary_pair_no_resize_or_GT_edit(self):
        rgb=io.BytesIO();Image.new('RGB',(4,3)).save(rgb,format='JPEG')
        bmp=io.BytesIO();Image.new('1',(4,3)).save(bmp,format='BMP')
        _,mask,result=numeric_pair(rgb.getvalue(),bmp.getvalue(),[4,3],[])
        self.assertFalse(mask.any());self.assertEqual(result['native_positive_pixels'],0)
        with self.assertRaises(ValueError):numeric_pair(rgb.getvalue(),bmp.getvalue(),[3,4],[])


if __name__=='__main__':unittest.main()
