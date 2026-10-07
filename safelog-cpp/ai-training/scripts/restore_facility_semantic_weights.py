"""Restore the ignored official binary while preserving tracked verified metadata.

This maintenance wrapper is outside the frozen training source inventory. A
fresh clone can contain verified metadata without its ignored weight binary.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import fetch_facility_semantic_weights as frozen

OFFICIAL_FULL_SHA = '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'
OFFICIAL_ENCODER_SHA = '712dc4bd02d35026a00afa56e842b33f7100fcb22020bcec74190b4631641d43'
SHA_PREFIX = '983f1562'
SHA_PATTERN = re.compile(r'[a-f0-9]{64}\Z')


def checked_path(root, relative, directory):
    root = Path(root).resolve(); path = (root / relative).resolve()
    frozen.require(path.is_relative_to(root) and path.is_relative_to((root / directory).resolve()),
                   'Restoration path must remain in its checked repository directory')
    return path


def fingerprint(path):
    path = Path(path); stat = path.stat()
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def metadata_record(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = {'schema': 'facility_semantic_pretrained_weights_v1', 'status': 'verified',
        'weights_path': frozen.WEIGHTS_PATH, 'official_url': frozen.OFFICIAL_URL, 'weight_enum': frozen.WEIGHT_ENUM,
        'expected_filename_sha256_prefix': SHA_PREFIX, 'weights_sha256': OFFICIAL_FULL_SHA,
        'encoder_state_sha256': OFFICIAL_ENCODER_SHA, 'encoder_state_tensor_count': 180,
        'encoder_parameter_count': 27820128, 'strict_encoder_load': True,
        'official_pooling_norm_transferred': True, 'all_encoder_parameters_frozen': True, 'encoder_eval': True}
    for key, actual in expected.items():
        frozen.require(type(value.get(key)) is type(actual) and value[key] == actual,
                       f'Preserved official metadata differs: {key}')
    frozen.require(SHA_PATTERN.fullmatch(value['weights_sha256']) and value['weights_sha256'].startswith(SHA_PREFIX)
                   and type(value.get('file_size_bytes')) is int and value['file_size_bytes'] > 0,
                   'Official complete hash and size are required')
    return value


def verify_bound_binary(path, metadata, verifier):
    frozen.require(frozen.file_sha256(path) == metadata['weights_sha256']
                   and Path(path).stat().st_size == metadata['file_size_bytes'],
                   'Binary full SHA256/size differs from preserved official metadata')
    actual = verifier(path)
    for key in ('schema', 'status', 'weights_path', 'official_url', 'weight_enum', 'weights_sha256',
                'file_size_bytes', 'encoder_state_sha256', 'encoder_state_tensor_count', 'encoder_parameter_count',
                'strict_encoder_load', 'official_pooling_norm_transferred', 'all_encoder_parameters_frozen', 'encoder_eval'):
        frozen.require(type(actual.get(key)) is type(metadata[key]) and actual[key] == metadata[key],
                       f'Official strict encoder/metadata verification differs: {key}')
    return actual


def publish_without_overwrite(temporary, target):
    frozen.require(not target.exists(), 'Preserve the existing target binary')
    if os.name == 'nt':
        # Windows rename fails if another process creates the target first.
        temporary.rename(target)
    else:
        # POSIX rename replaces targets. A no-clobber hard link publishes the
        # fully verified file atomically, then removes this wrapper's temp link.
        os.link(temporary, target); temporary.unlink()


def restore(root=ROOT, opener=None, verifier=None):
    root = Path(root).resolve()
    target = checked_path(root, frozen.WEIGHTS_PATH, 'data/pretrained')
    metadata_path = checked_path(root, frozen.METADATA_PATH, 'reports')
    if not metadata_path.exists():
        frozen.require(root == ROOT.resolve(), 'Initial metadata creation must use the canonical fetch command')
        frozen.main()
        return {'status': 'initial_fetch_delegated'}
    frozen.require(metadata_path.is_file() and (not target.exists() or target.is_file()),
                   'Expected verified metadata and a regular optional binary')
    before = fingerprint(metadata_path); metadata = metadata_record(metadata_path)
    verifier = verifier or frozen.verify_existing; opener = opener or urlopen
    try:
        if target.exists():
            actual = verify_bound_binary(target, metadata, verifier)
            status = 'existing_verified'
        else:
            temporary = checked_path(root, str(Path(frozen.WEIGHTS_PATH).with_suffix('.download.part')).replace('\\', '/'), 'data/pretrained')
            frozen.require(not temporary.exists(), 'Preserve the incomplete earlier download for inspection')
            target.parent.mkdir(parents=True, exist_ok=True)
            with opener(frozen.OFFICIAL_URL, timeout=60) as response, temporary.open('xb') as stream:
                final = urlparse(response.geturl())
                frozen.require(final.scheme == 'https' and final.hostname == 'download.pytorch.org'
                               and response.geturl() == frozen.OFFICIAL_URL,
                               'Only the exact official HTTPS PyTorch model URL is permitted')
                while chunk := response.read(1024 * 1024): stream.write(chunk)
            actual = verify_bound_binary(temporary, metadata, verifier)
            frozen.require(fingerprint(metadata_path) == before, 'Verified metadata changed during restoration')
            publish_without_overwrite(temporary, target)
            status = 'restored_verified_binary'
        return {'status': status, 'weights_sha256': actual['weights_sha256'],
                'encoder_state_sha256': actual['encoder_state_sha256'], 'metadata_preserved': True,
                'new_training_epochs': 0, 'source_val_or_test_inference': False}
    finally:
        frozen.require(fingerprint(metadata_path) == before, 'Restoration must preserve metadata raw bytes/size/mtime')


def main():
    frozen.torch.set_num_threads(4)
    print(json.dumps(restore(), ensure_ascii=False))


if __name__ == '__main__': main()
