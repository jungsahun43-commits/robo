"""v87: preregister, then inspect ONLY bounded public ZIP directory ranges."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.public_zip_metadata import check_range, end_directory, parse_directory

META = ROOT/'data/industrial-new-surface-v86-metadata'
OUT = ROOT/'data/industrial-liaci-v87-source'
POLICY = ROOT/'reports/industrial-liaci-v87-range-policy.json'
CODE = [Path(__file__), ROOT/'safelog_ai/public_zip_metadata.py', ROOT/'tests/test_public_zip_metadata.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def freeze():
    assert not POLICY.exists() and not OUT.exists()
    policy = {'UTC': datetime.now(timezone.utc).isoformat(), 'stage': 'public ZIP metadata only, no member contents',
              'URL': 'https://liaci.sintef.cloud/download_data/data.zip', 'archive_bytes': 1044750953,
              'ETag': '"1726140227.0-1044750953-416943642"', 'expected_Last_Modified': 'Thu, 12 Sep 2024 11:23:47 GMT',
              'source': 'SINTEF LIACi official project -> public Dash layout Download data link',
              'data_license': 'CC BY-NC-SA4.0; noncommercial research/author/source/distribution/changes/SA, preserve third-party rights; not paper/code license',
              'max_total_range_bytes': 16*1024**2, 'max_directory_bytes': 8*1024**2, 'reserve_bytes': 1024**3,
              'HTTP': 'Exact206 ContentRange/ContentLength/ETag/identity encoding; 200/redirect identity changes stop before body; no full fallback',
              'ZIP': 'Single disk, nonZIP64, stored/deflate only, <=50000 entries; safe unique paths/no symlink/encryption; directory metadata CRC declarations are not member CRC verification',
              'TEST': 'No image or annotation member payload in v87; identify original split before a separate acquisition/reservation/split/minimum/duplicate policy. All original evaluation images/copies remain held. No automatic TRAIN assignment.',
              'new_photos_pixels_features_scores_gradients_optimizer': 0, 'model_protocol_fixed': False,
              'code_SHA256': {str(p.relative_to(ROOT)):sha(p) for p in CODE},
              'evidence_SHA256': {p.name:sha(p) for p in [META/'liaci-layout.txt', META/'liaci-download-HEAD.json', META/'linked-metadata-receipts.json']}}
    save(POLICY, policy)
    print(json.dumps({'policy_SHA256':sha(POLICY)}, ensure_ascii=False))


def run():
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+ROOT.parents[1].as_posix(), '-C', str(ROOT.parents[1])]
    for p in [*CODE, POLICY]:
        assert subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+p.relative_to(ROOT).as_posix()]) == p.read_bytes()
    for name, digest in policy['code_SHA256'].items():
        assert sha(ROOT/name) == digest
    for name, digest in policy['evidence_SHA256'].items():
        assert sha(META/name) == digest
    assert not OUT.exists()
    assert shutil.disk_usage(ROOT).free >= policy['reserve_bytes']+policy['max_total_range_bytes']+8*1024**2
    OUT.mkdir()
    received, receipts = 0, []
    def get_range(start, count):
        nonlocal received
        assert 0 <= start < start+count <= policy['archive_bytes']
        assert received+count <= policy['max_total_range_bytes']
        with requests.get(policy['URL'], headers={'Range':f'bytes={start}-{start+count-1}', 'If-Range':policy['ETag'], 'Accept-Encoding':'identity'}, stream=True, timeout=(15,45)) as r:
            check_range(r.status_code,r.headers,start,count,policy['archive_bytes'],policy['ETag'])
            assert r.url == policy['URL'] and r.headers.get('Last-Modified') == policy['expected_Last_Modified']
            blocks, length = [], 0
            for b in r.iter_content(65536):
                length += len(b)
                assert length <= count
                blocks.append(b)
            assert length == count
            raw = b''.join(blocks)
        received += length
        path = OUT/f'ZIP-directory-range-{start}-{count}.bin'
        with path.open('xb') as f:
            f.write(raw)
        receipts.append({'start':start,'bytes':count,'local_SHA256':sha(path),'Content_Range':f'bytes {start}-{start+count-1}/{policy["archive_bytes"]}'})
        return raw
    try:
        tail = get_range(policy['archive_bytes']-65557,65557)
        cd = end_directory(tail,policy['archive_bytes'],policy['max_directory_bytes'])
        directory = get_range(cd['offset'],cd['bytes'])
        rows = parse_directory(directory,cd['entries'],policy['archive_bytes'])
        result = {'UTC':datetime.now(timezone.utc).isoformat(),'policy_SHA256':sha(POLICY),'rows':rows,
                  'member_count':len(rows),'declared_expanded_bytes':sum(r['bytes'] for r in rows),
                  'top_level_counts':dict(Counter(r['member'].split('/')[0] for r in rows)),
                  'archive_bytes_NOT_downloaded':policy['archive_bytes'],'actual_range_bytes':received,'ranges':receipts,
                  'full_archive_checksum_verified':False,'member_CRCS_verified':False,
                  'source_photo_annotation_payloads_pixel_decodes_features_scores_gradients_optimizer':0,'ready_for_training':False}
        save(OUT/'public-zip-directory.json',result)
        save(ROOT/'reports/industrial-liaci-v87-directory-summary.json',{k:v for k,v in result.items() if k!='rows'})
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)
    except BaseException as exc:
        save(OUT/'metadata-failure.json',{'UTC':datetime.now(timezone.utc).isoformat(),'error':repr(exc),'actual_range_bytes':received,'ranges':receipts,'full_fallback':False,'optimizer':0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze','run'))
    globals()[parser.parse_args().stage]()
