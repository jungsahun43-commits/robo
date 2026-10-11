"""Fetch an openly licensed conveyor foreign-object source; not healthy-belt GT."""
import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/equipment-conveyor-foreign'
SOURCE = 'https://data.mendeley.com/datasets/s25x2bnshz/1'
API = 'https://data.mendeley.com/public-api/datasets/s25x2bnshz/files'
EXPECTED_SHA = 'f4ced924f274bcd729fd77ab0aca8fa72a48b5419077e2862b87ab175fd12480'
EXPECTED_BYTES = 6228464374


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    metadata = DATA/'publisher-files.json'
    if not metadata.exists():
        response = requests.get(API,params={'folder_id':'root','version':'1','$start':0,'$limit':1000},
            headers={'Accept':'application/vnd.mendeley-public-dataset.1+json'},timeout=(20,60))
        response.raise_for_status()
        with metadata.open('x',encoding='utf-8') as handle:
            json.dump(response.json(),handle,indent=2)
    rows = json.loads(metadata.read_text(encoding='utf-8'))
    selected = [r for r in rows if r['filename']=='image-data.zip']
    assert len(selected)==1
    entry = selected[0]['content_details']
    assert entry['size']==EXPECTED_BYTES and entry['sha256_hash']==EXPECTED_SHA
    target,partial = DATA/'image-data.zip',DATA/'image-data.zip.partial'
    if not target.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        assert offset<=EXPECTED_BYTES
        if offset<EXPECTED_BYTES:
            headers = {'Range':f'bytes={offset}-'} if offset else {}
            with requests.get(entry['download_url'],headers=headers,stream=True,timeout=(30,60)) as response:
                # Stop on429/403; no account or endpoint switching to evade limits.
                response.raise_for_status()
                assert response.status_code==(206 if offset else 200)
                if offset:
                    assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                assert int(response.headers['Content-Length'])==EXPECTED_BYTES-offset
                previous_print,time_start = offset,time.monotonic()
                with partial.open('ab' if offset else 'xb') as handle:
                    for block in response.iter_content(chunk_size=8*1024*1024):
                        handle.write(block)
                        offset += len(block)
                        assert offset<=EXPECTED_BYTES
                        if offset-previous_print>=128*1024*1024:
                            print(json.dumps({'downloaded_bytes':offset,'total_bytes':EXPECTED_BYTES,
                                'percent':round(100*offset/EXPECTED_BYTES,1),'elapsed_minutes':(time.monotonic()-time_start)/60}),flush=True)
                            previous_print = offset
        assert partial.stat().st_size==EXPECTED_BYTES and sha(partial)==EXPECTED_SHA
        partial.rename(target)
    assert target.stat().st_size==EXPECTED_BYTES and sha(target)==EXPECTED_SHA
    index = DATA/'archive-index.json'
    assert not index.exists(),'Preserve completed acquisition'
    with zipfile.ZipFile(target) as archive:
        members = [{'name':r.filename,'bytes':r.file_size,'compressed':r.compress_size,'crc32':r.CRC}
                   for r in archive.infolist() if not r.is_dir()]
    counts = Counter('/'.join(r['name'].replace('\\','/').split('/')[:2]) for r in members)
    with index.open('x',encoding='utf-8') as handle:
        json.dump({'source':SOURCE,'doi':'10.17632/s25x2bnshz.1','license':'CC-BY-4.0',
            'archive_sha256':EXPECTED_SHA,'archive_bytes':EXPECTED_BYTES,'publisher_sha256_verified':True,
            'members':members,'top_level_counts':dict(counts),
            'normal_definition':'Only iron ore; anomalous means introduced foreign objects',
            'healthy_belt_or_mechanical_fault_ground_truth':False,
            'images_or_labels_used_for_belt_crack_training':0,'new_training_performed':False,
            'acquisition_script_sha256':sha(Path(__file__))},handle,indent=2)
    print(json.dumps({'status':'downloaded_sha256_verified','members':len(members),
        'top_level_counts':dict(counts),'normal_is_healthy_belt':False}),flush=True)


if __name__=='__main__':
    main()
