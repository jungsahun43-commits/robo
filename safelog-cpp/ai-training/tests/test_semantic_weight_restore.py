"""Restore missing ignored weights without rewriting tracked metadata."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

from scripts import restore_facility_semantic_weights as restore


class OfficialResponse(io.BytesIO):
    def geturl(self): return restore.frozen.OFFICIAL_URL


class SemanticWeightRestoreTests(TestCase):
    def fixture(self, root):
        payload = b'unit-test-placeholder-official-weight-bytes'
        digest = hashlib.sha256(payload).hexdigest()
        record = json.loads((restore.ROOT / restore.frozen.METADATA_PATH).read_text(encoding='utf-8'))
        record.update(weights_sha256=digest, expected_filename_sha256_prefix=digest[:8], file_size_bytes=len(payload))
        path = root / restore.frozen.METADATA_PATH; path.parent.mkdir(parents=True)
        path.write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))
        return payload, digest, record, path

    def test_missing_binary_restores_exact_bytes_and_existing_binary_only_verifies(self):
        with tempfile.TemporaryDirectory(dir=restore.ROOT / 'runs') as directory:
            root = Path(directory); payload, digest, record, metadata = self.fixture(root)
            before = restore.fingerprint(metadata); calls = []
            def opener(url, timeout):
                calls.append((url, timeout)); return OfficialResponse(payload)
            with patch.object(restore, 'OFFICIAL_FULL_SHA', digest), patch.object(restore, 'SHA_PREFIX', digest[:8]):
                result = restore.restore(root, opener, lambda path: dict(record))
                target = root / restore.frozen.WEIGHTS_PATH
                self.assertEqual(target.read_bytes(), payload)
                self.assertEqual(result['status'], 'restored_verified_binary')
                self.assertEqual(calls, [(restore.frozen.OFFICIAL_URL, 60)])
                self.assertEqual(restore.fingerprint(metadata), before)
                def forbidden_network(*args, **kwargs): raise AssertionError('Existing binary must not download')
                again = restore.restore(root, forbidden_network, lambda path: dict(record))
                self.assertEqual(again['status'], 'existing_verified')
                self.assertEqual(restore.fingerprint(metadata), before)

    def test_wrong_full_hash_or_prior_partial_or_existing_binary_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory(dir=restore.ROOT / 'runs') as directory:
            root = Path(directory); payload, digest, record, metadata = self.fixture(root)
            before = restore.fingerprint(metadata)
            target = root / restore.frozen.WEIGHTS_PATH
            partial = target.with_suffix('.download.part')
            with patch.object(restore, 'OFFICIAL_FULL_SHA', digest), patch.object(restore, 'SHA_PREFIX', digest[:8]):
                with self.assertRaisesRegex(ValueError, 'full SHA256'):
                    restore.restore(root, lambda *args, **kwargs: OfficialResponse(b'wrong actual bytes'), lambda path: dict(record))
                self.assertFalse(target.exists()); self.assertEqual(partial.read_bytes(), b'wrong actual bytes')
                partial_before = restore.fingerprint(partial)
                with self.assertRaisesRegex(ValueError, 'incomplete earlier download'):
                    restore.restore(root, lambda *args, **kwargs: OfficialResponse(payload), lambda path: dict(record))
                self.assertEqual(restore.fingerprint(partial), partial_before)
                target.write_bytes(b'prior-invalid-target')
                target_before = restore.fingerprint(target)
                with self.assertRaisesRegex(ValueError, 'full SHA256'):
                    restore.restore(root, lambda *args, **kwargs: OfficialResponse(payload), lambda path: dict(record))
                self.assertEqual(restore.fingerprint(target), target_before)
                self.assertEqual(restore.fingerprint(partial), partial_before)
                self.assertEqual(restore.fingerprint(metadata), before)
