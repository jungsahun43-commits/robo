from pathlib import Path
import tempfile
import unittest
import zipfile
from scripts.extract_codebrim import correct_legacy_offsets, extract


class ArchiveTests(unittest.TestCase):
    def test_legacy_offset_repair_preserves_member_crc_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'data.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('one.txt', b'one'); bundle.writestr('two.txt', b'two')
            before = archive.read_bytes()
            with zipfile.ZipFile(archive) as bundle, archive.open('rb') as raw:
                for info in bundle.infolist(): info.header_offset += 2**32
                report = correct_legacy_offsets(bundle, raw)
                self.assertEqual(report, {str(-2**32): 2})
                self.assertEqual(bundle.read('one.txt'), b'one')
                self.assertEqual(bundle.read('two.txt'), b'two')
            self.assertEqual(before, archive.read_bytes())

    def test_path_escape_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'data.zip'
            with zipfile.ZipFile(archive, 'w') as bundle: bundle.writestr('../escape.txt', 'bad')
            with self.assertRaises(ValueError): extract(archive, Path(directory)/'out')
            self.assertFalse((Path(directory)/'escape.txt').exists())

    def test_bad_crc_is_not_ignored_by_repaired_extractor(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'data.zip'
            with zipfile.ZipFile(archive, 'w') as bundle: bundle.writestr('one.txt', b'one')
            data = archive.read_bytes(); archive.write_bytes(data.replace(b'onePK', b'badPK', 1))
            with self.assertRaises(zipfile.BadZipFile): extract(archive, Path(directory)/'out')
