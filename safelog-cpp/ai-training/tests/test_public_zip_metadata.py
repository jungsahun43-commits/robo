import io
import struct
import unittest
import zipfile
from safelog_ai.public_zip_metadata import check_range, end_directory, parse_directory


def sample(name='train/a.txt'):
    out = io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(name,b'not an image')
    raw = out.getvalue()
    meta = end_directory(raw,len(raw))
    return raw,meta,raw[meta['offset']:meta['offset']+meta['bytes']]


class PublicZipMetadataTest(unittest.TestCase):
    def test_valid_directory_is_metadata_only(self):
        raw,meta,cd=sample()
        rows=parse_directory(cd,meta['entries'],len(raw))
        self.assertEqual(rows[0]['member'],'train/a.txt')
        self.assertEqual(rows[0]['bytes'],12)
        self.assertEqual(rows[0]['local_header_offset'],0)
        self.assertFalse('payload' in rows[0])

    def test_reject_path_and_incomplete_directory(self):
        for name in ('../escape','/abs','C:/abs','back\\slash','a//b','a/./b'):
            raw,meta,cd=sample()
            # Windows ZipInfo normalizes backslashes before writing. Build raw
            # directory names to exercise the consumer, not that producer.
            old_length=struct.unpack_from('<H',cd,28)[0]
            changed=bytearray(cd[:46]+name.encode()+cd[46+old_length:])
            struct.pack_into('<H',changed,28,len(name))
            with self.assertRaises(ValueError):
                parse_directory(bytes(changed),meta['entries'],len(raw)+len(name))
        raw,meta,cd=sample()
        with self.assertRaises(ValueError):
            parse_directory(cd[:-1],meta['entries'],len(raw))
        bad=bytearray(raw);struct.pack_into('<H',bad,len(raw)-18,1)
        with self.assertRaises(ValueError):
            end_directory(bytes(bad),len(bad))

    def test_no_full_fallback_or_changed_range_identity(self):
        headers={'Content-Range':'bytes 5-9/20','Content-Length':'5','ETag':'"same"'}
        check_range(206,headers,5,5,20,'"same"')
        for status,changed in [(200,headers),(206,{**headers,'ETag':'"other"'}),(206,{**headers,'Content-Range':'bytes 0-4/20'})]:
            with self.assertRaises(ValueError):
                check_range(status,changed,5,5,20,'"same"')


if __name__ == '__main__':
    unittest.main()
