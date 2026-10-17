import io
import unittest
import zipfile
from safelog_ai.public_zip_metadata import end_directory, parse_directory, read_text_member


class PublicZipTextMemberTest(unittest.TestCase):
    def fixture(self,method=8):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w',method) as z:
            z.writestr('split.csv','image,split\n1,test\n')
        raw=b.getvalue();meta=end_directory(raw,len(raw))
        row=parse_directory(raw[meta['offset']:meta['offset']+meta['bytes']],1,len(raw))[0]
        return raw,row,lambda start,count:raw[start:start+count]

    def test_verified_text_for_both_supported_methods(self):
        for method in (0,8):
            raw,row,getter=self.fixture(method)
            self.assertEqual(read_text_member(row,getter,1024),b'image,split\n1,test\n')

    def test_changed_name_or_crc_fails(self):
        raw,row,getter=self.fixture()
        with self.assertRaises(ValueError):
            read_text_member({**row,'member':'other.csv'},getter,1024)
        with self.assertRaises(ValueError):
            read_text_member({**row,'CRC32_declared':'00000000'},getter,1024)

    def test_small_budget_never_requests_payload(self):
        raw,row,getter=self.fixture()
        calls=[]
        with self.assertRaises(ValueError):
            read_text_member(row,lambda *args:calls.append(args),4)
        self.assertEqual(calls,[])


if __name__ == '__main__':unittest.main()
