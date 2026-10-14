"""Pinned author data, checksum every selected Git blob; never execute source code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import requests
ROOT = Path(__file__).resolve().parents[1]
REVISION = '7204b28a621733c01cd75899ed9ea27f6ea0b9e3'
DATA = ROOT/'data/industrial-bsdata-v1'
MAXIMUM_BYTES = 512*1024*1024


def new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f: json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    assert not DATA.exists(); DATA.mkdir(parents=True)
    response = requests.get('https://api.github.com/repos/2Obe/BSData/git/trees/'+REVISION+'?recursive=1', timeout=30)
    response.raise_for_status(); tree = response.json(); assert not tree.get('truncated')
    selected = {item['path']: item for item in tree['tree'] if item['type'] == 'blob' and
        (item['path'] in ('LICENSE', 'README.md', 'image_data.json') or
         (item['path'].split('/')[0] in ('data', 'label') and Path(item['path']).suffix.lower() in ('.jpg', '.jpeg', '.png', '.json')))}
    assert selected and sum(v['size'] for v in selected.values()) < MAXIMUM_BYTES
    new(DATA/'git-tree.json', tree)
    url = 'https://codeload.github.com/2Obe/BSData/zip/'+REVISION; archive = DATA/'author-source.zip'
    with requests.get(url, stream=True, timeout=(20, 60)) as reply:
        reply.raise_for_status(); count = 0
        with archive.open('xb') as out:
            for chunk in reply.iter_content(1024*1024):
                count += len(chunk)
                if count > MAXIMUM_BYTES: raise ValueError('Prespecified512MiB budget exceeded')
                out.write(chunk)
    verified = {}; output = DATA/'originals'; output.mkdir()
    with zipfile.ZipFile(archive) as source:
        assert source.testzip() is None
        for info in source.infolist():
            parts = Path(info.filename.replace('\\', '/')).parts
            if len(parts) < 2: continue
            name = Path(*parts[1:]).as_posix()
            if name not in selected: continue
            path = output/name; assert path.resolve().is_relative_to(output.resolve()) and not info.is_dir()
            blob = source.read(info); expected = selected[name]
            assert len(blob) == expected['size'] and hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest() == expected['sha'], name
            if blob.startswith(b'version https://git-lfs'): raise ValueError('LFS pointer is not dataset contents')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as out: out.write(blob)
            verified[name] = {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob), 'author_Git_blob_sha1': expected['sha']}
    assert set(verified) == set(selected)
    counts = {suffix: sum(Path(n).suffix.lower() == suffix for n in verified) for suffix in ('.jpg', '.jpeg', '.png', '.json')}
    new(DATA/'download.json', {'status': 'downloaded_author_originals_verified_not_yet_in_training', 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'author_repository': 'https://github.com/2Obe/BSData', 'revision': REVISION, 'archive_url': url, 'archive_bytes': archive.stat().st_size,
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'publisher_RADAR_archive_MD5_not_compared_different_archive': True,
        'license': 'CC BY-SA4.0 author repository LICENSE', 'source': 'https://publikationen.bibliothek.kit.edu/1000129520',
        'selected_counts': counts, 'selected_file_checksums': verified, 'foreign_code_or_saved_models_extracted_or_executed': False,
        'is_fastener_screw_dataset': False, 'domain': 'ball_screw_drive_visible_surface_pitting', 'new_training_or_field_performance_claimed': False})
    print(json.dumps({'downloaded': 'BSData_author_originals', 'revision': REVISION, 'counts': counts, 'bytes': archive.stat().st_size, 'not_trained_yet': True}), flush=True)


if __name__ == '__main__': main()
