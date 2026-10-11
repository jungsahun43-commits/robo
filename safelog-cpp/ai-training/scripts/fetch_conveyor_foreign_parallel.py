"""Resume the same public ZIP in four disjoint ranges, preserving the prefix."""
import json
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import requests

from fetch_conveyor_foreign import DATA, EXPECTED_BYTES, EXPECTED_SHA, main as index_complete_archive, sha


def main():
    rows = json.loads((DATA/'publisher-files.json').read_text(encoding='utf-8'))
    entry = next(r['content_details'] for r in rows if r['filename']=='image-data.zip')
    assert entry['size']==EXPECTED_BYTES and entry['sha256_hash']==EXPECTED_SHA
    prefix = DATA/'image-data.zip.partial'
    prefix_size = prefix.stat().st_size
    assert 0<=prefix_size<=EXPECTED_BYTES
    prefix_sha = sha(prefix)
    parts_dir = DATA/'download-ranges-v1'
    parts_dir.mkdir(exist_ok=True)
    step = 256*1024*1024
    planned = [(start,min(EXPECTED_BYTES-1,start+step-1)) for start in range(prefix_size,EXPECTED_BYTES,step)]
    plan = parts_dir/'plan.json'
    value = {'prefix_bytes':prefix_size,'prefix_sha256':prefix_sha,'ranges':planned,'workers':4,
             'whole_archive_expected_sha256':EXPECTED_SHA,'sequential_prefix_preserved':True}
    if plan.exists():
        assert json.loads(plan.read_text(encoding='utf-8'))==json.loads(json.dumps(value))
    else:
        with plan.open('x',encoding='utf-8') as handle:
            json.dump(value,handle,indent=2)

    def transfer(bounds):
        start,end = bounds
        target = parts_dir/f'{start}-{end}.bin'
        if target.exists():
            assert target.stat().st_size==end-start+1
            return target
        partial = parts_dir/f'{start}-{end}.partial'
        offset = partial.stat().st_size if partial.exists() else 0
        assert 0<=offset<=end-start+1
        if offset<end-start+1:
            with requests.get(entry['download_url'],headers={'Range':f'bytes={start+offset}-{end}'},
                              stream=True,timeout=(30,60)) as response:
                response.raise_for_status()
                assert response.status_code==206
                assert response.headers['Content-Range']==f'bytes {start+offset}-{end}/{EXPECTED_BYTES}'
                assert int(response.headers['Content-Length'])==end-start+1-offset
                with partial.open('ab' if offset else 'xb') as handle:
                    for block in response.iter_content(8*1024*1024):
                        handle.write(block)
        assert partial.stat().st_size==end-start+1
        partial.rename(target)
        print(json.dumps({'completed_range_start':start,'range_bytes':target.stat().st_size}),flush=True)
        return target

    pending_ranges = iter(planned)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(transfer,bounds) for bounds in [next(pending_ranges,None) for _ in range(4)] if bounds}
        while futures:
            completed,futures = wait(futures,return_when=FIRST_COMPLETED)
            for future in completed:
                future.result()  # Stop submitting on429/403/network failure.
                bounds = next(pending_ranges,None)
                if bounds:
                    futures.add(pool.submit(transfer,bounds))
    assert prefix.stat().st_size==prefix_size and sha(prefix)==prefix_sha
    assembled = DATA/'image-data.zip.assembled'
    assert not assembled.exists()
    with assembled.open('xb') as out:
        for path in [prefix]+[parts_dir/f'{start}-{end}.bin' for start,end in planned]:
            with path.open('rb') as source:
                for block in iter(lambda:source.read(8*1024*1024),b''):
                    out.write(block)
    assert assembled.stat().st_size==EXPECTED_BYTES and sha(assembled)==EXPECTED_SHA
    assembled.rename(DATA/'image-data.zip')
    index_complete_archive()


if __name__=='__main__':
    main()
