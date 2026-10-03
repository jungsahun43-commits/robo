import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from scripts.facility_photo_supplement import ARCHIVE_SHA, CONVID_URL, ORIGINAL_MANIFESTS, validate_photo_supplement
from scripts.report_facility_small_region import CLASSES


class PhotoSupplementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        prefix = self.root / 'data/peccd-training'
        prefix.mkdir(parents=True)
        (prefix / 'photo.jpg').write_bytes(b'path fixture only')
        np.savez_compressed(prefix / 'unknown.npz', mask=np.zeros((7, 80, 80), np.uint8), known=np.zeros(7, np.uint8))
        self.data = {'split': 'train', 'classes': CLASSES, 'audit': {'status': 'prepared', 'source_archive_sha256': ARCHIVE_SHA},
                     'items': [{'image': 'data/peccd-training/photo.jpg', 'pixel_target': 'data/peccd-training/unknown.npz',
                                'domain': 'peccd', 'source_split': 'train_only_unsplit', 'targets': [1, 0] + [-1] * 5}]}

    def check(self, data=None):
        return validate_photo_supplement(data or self.data, CLASSES, self.root, {'original/photo.jpg'})

    def convid(self):
        prefix = self.root / 'data/convid-training'
        prefix.mkdir(parents=True)
        photo = prefix / 'photo.jpg'
        photo.write_bytes(b'path and checksum fixture only')
        spall = prefix / 'spall.jpg'
        spall.write_bytes(b'spall fixture only')
        np.savez_compressed(prefix / 'unknown.npz', mask=np.zeros((7, 80, 80), np.uint8), known=np.zeros(7, np.uint8))
        source = prefix / 'original.jpg'
        source.write_bytes(b'original source fixture')
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        records = [{'id': 'official-id', 'folder': 'crack', 'sha256': source_sha},
                   {'id': 'spall-id', 'folder': 'Spalling', 'sha256': source_sha}]
        index = prefix / 'SOURCE-INDEX.json'
        index.write_text(json.dumps({'dataset_url': CONVID_URL, 'version': 4,
                                    'records': records}), encoding='utf-8')
        picks = prefix / 'SOURCE-PICKS.json'
        picks.write_text(json.dumps({'seed': 52, 'per_folder_limit': 100, 'picks': records}), encoding='utf-8')
        index_sha, picks_sha = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (index, picks)]
        for name, value in (('CONVID_INDEX_SHA', index_sha), ('CONVID_PICKS_SHA', picks_sha)):
            fixture = patch('scripts.facility_photo_supplement.' + name, value)
            fixture.start()
            self.addCleanup(fixture.stop)
        snapshots = {}
        for name in ORIGINAL_MANIFESTS:
            original = self.root / name
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(b'original split/label manifest fixture')
            snapshots[name] = hashlib.sha256(original.read_bytes()).hexdigest()
        audit = {'status': 'prepared', 'source_index_path': 'data/convid-training/SOURCE-INDEX.json',
                 'source_index_sha256': index_sha, 'source_pick_sha256': picks_sha,
                 'original_manifests_unchanged_sha256': snapshots}
        public = self.root / 'reports/facility-convid-data-audit.json'
        public.parent.mkdir()
        public.write_text(json.dumps({k: v for k, v in audit.items() if k != 'source_index_path'}), encoding='utf-8')
        audit['audit_file_sha256'] = hashlib.sha256(public.read_bytes()).hexdigest()
        return {'split': 'train', 'classes': CLASSES, 'audit': audit,
                'items': [{'domain': 'convid', 'source_split': 'train_only_unsplit', 'source_id': 'official-id',
                           'source_folder': 'crack', 'source_sha256': source_sha, 'source_original': 'data/convid-training/original.jpg',
                           'image': 'data/convid-training/photo.jpg', 'pixel_target': 'data/convid-training/unknown.npz',
                           'prepared_image_sha256': hashlib.sha256(photo.read_bytes()).hexdigest(),
                           'targets': [1] + [-1] * 6},
                          {'domain': 'convid', 'source_split': 'train_only_unsplit', 'source_id': 'spall-id',
                           'source_folder': 'Spalling', 'source_sha256': source_sha, 'source_original': 'data/convid-training/original.jpg',
                           'image': 'data/convid-training/spall.jpg', 'pixel_target': 'data/convid-training/unknown.npz',
                           'prepared_image_sha256': hashlib.sha256(spall.read_bytes()).hexdigest(),
                           'targets': [-1, 1] + [-1] * 5}]}

    def test_asserted_photo_labels_with_no_pixel_truth_are_accepted(self):
        self.assertEqual(len(self.check()), 1)

    def test_unannotated_class_is_not_negative_and_boxes_are_not_pixels(self):
        data = copy.deepcopy(self.data)
        data['items'][0]['targets'][2] = 0
        with self.assertRaises(ValueError): self.check(data)
        target = np.zeros((7, 80, 80), np.uint8)
        target[0, 2:10, 3:11] = 1
        np.savez_compressed(self.root / 'data/peccd-training/unknown.npz', mask=target, known=np.ones(7, np.uint8))
        with self.assertRaises(ValueError): self.check()

    def test_wrong_source_holdout_repeated_path_and_changed_archive_are_rejected(self):
        for kind in ('heldout', 'duplicate', 'archive', 'classes'):
            data = copy.deepcopy(self.data)
            if kind == 'heldout': data['items'][0]['source_split'] = 'test'
            if kind == 'duplicate': data['items'] *= 2
            if kind == 'archive': data['audit']['source_archive_sha256'] = 'changed'
            if kind == 'classes': data['classes'] = CLASSES[::-1]
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.check(data)

    def test_windows_escape_and_wrong_pixel_shape_are_rejected(self):
        for path in (r'data/peccd-training/a/..\..\..\outside.jpg', 'data/peccd-training/../outside.jpg'):
            data = copy.deepcopy(self.data)
            data['items'][0]['image'] = path
            with self.subTest(path=path), self.assertRaises(ValueError): self.check(data)
        np.savez_compressed(self.root / 'data/peccd-training/unknown.npz', mask=np.zeros((7, 4, 4)), known=np.zeros(7))
        with self.assertRaises(ValueError): self.check()

    def test_convid_positive_only_folder_is_not_exclusive_class_truth(self):
        data = self.convid()
        self.assertEqual(len(self.check(data)), 2)
        for target in ([1, 0] + [-1] * 5, [1, 1] + [-1] * 5, [-1, 1] + [-1] * 5):
            changed = copy.deepcopy(data)
            changed['items'][0]['targets'] = target
            with self.subTest(target=target), self.assertRaises(ValueError): self.check(changed)

    def test_convid_changed_index_photo_and_source_hash_are_rejected(self):
        data = self.convid()
        for kind in ('index', 'photo', 'source', 'duplicate'):
            changed = copy.deepcopy(data)
            if kind == 'index': changed['audit']['source_index_sha256'] = 'changed'
            if kind == 'photo': changed['items'][0]['prepared_image_sha256'] = 'changed'
            if kind == 'source': changed['items'][0]['source_sha256'] = 'b' * 64
            if kind == 'duplicate': changed['items'] *= 2
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.check(changed)

    def test_convid_preflight_rejects_changed_picks_split_audit_and_original(self):
        data = self.convid()
        for name in ('data/convid-training/SOURCE-PICKS.json', ORIGINAL_MANIFESTS[0],
                     'reports/facility-convid-data-audit.json', 'data/convid-training/original.jpg'):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b' changed')
            with self.subTest(name=name), self.assertRaises(ValueError): self.check(data)
            path.write_bytes(original)
        changed = copy.deepcopy(data)
        changed['items'] = changed['items'][:1]
        with self.assertRaises(ValueError): self.check(changed)


if __name__ == '__main__':
    unittest.main()
