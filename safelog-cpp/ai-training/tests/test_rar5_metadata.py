import struct
import unittest
import zlib
from safelog_ai.rar5_metadata import vint, encode_vint, header


class RAR5HeaderTests(unittest.TestCase):
    def test_large_locator_integer_and_truncation(self):
        value = 9773237356
        raw = encode_vint(value)
        self.assertEqual(vint(raw, 0), (value, len(raw)))
        with self.assertRaises(ValueError):
            vint(raw[:-1], 0)

    def test_header_crc_required(self):
        raw = b'\x03\x01\x00\x00'
        block = struct.pack('<I', zlib.crc32(raw)) + raw
        self.assertEqual(header(block)[0]['archive_flags'], 0)
        with self.assertRaises(ValueError):
            header(block[:-1] + b'\x01')


if __name__ == '__main__':
    unittest.main()
