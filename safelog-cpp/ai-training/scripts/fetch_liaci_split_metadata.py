"""v88: verify ONLY the original split CSV, not images or annotation contents."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import shutil,subprocess,sys
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from safelog_ai.public_zip_metadata import check_range,read_text_member
SOURCE=ROOT/'data/industrial-liaci-v87-source'
POLICY=ROOT/'reports/industrial-liaci-v88-split-policy.json'
OUT=ROOT/'data/industrial-liaci-v88-split'
CODE=[Path(__file__),ROOT/'safelog_ai/public_zip_metadata.py',ROOT/'tests/test_public_zip_text_member.py']
INDEX=SOURCE/'public-zip-directory.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def freeze():
    assert not POLICY.exists() and not OUT.exists()
    prior=read(ROOT/'reports/industrial-liaci-v87-range-policy.json')
    row=next(r for r in read(INDEX)['rows'] if r['member']=='LIACi_dataset_pretty/train_test_split.csv')
    assert row['bytes']==50824 and row['packed']==9955 and row['CRC32_declared']=='8de40ea5'
    save(POLICY,{'UTC':datetime.now(timezone.utc).isoformat(),'stage':'original split CSV metadata only',
        'URL':prior['URL'],'ETag':prior['ETag'],'archive_bytes':prior['archive_bytes'],'member':row,
        'index_SHA256':sha(INDEX),'max_total_range_bytes':1024**2,'max_text_bytes':128*1024,'reserve_bytes':1024**3,
        'code_SHA256':{str(p.relative_to(ROOT)):sha(p) for p in CODE},
        'checks':'Exact206/ContentRange/ETag/length/identity; local header/name/size+CRC and decoded UTF8 CSV; no full fallback/external code.',
        'scope':'No photo/annotation JSON/mask/member contents beyond this split CSV. Original evaluation reservation will be made from verified schema before any annotation/pixel read. No split changes or TRAIN assignment yet.',
        'new_photos_pixels_features_scores_gradients_optimizer':0,'data_license':'CC BY-NC-SA4.0, attribution/noncommercial/SA/third-party rights preserved; not standalone model.'})
    print({'policy_SHA256':sha(POLICY)})

def run():
    p=read(POLICY)
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for path in [*CODE,POLICY]:
        assert subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+path.relative_to(ROOT).as_posix()])==path.read_bytes()
    assert all(sha(ROOT/name)==digest for name,digest in p['code_SHA256'].items()) and sha(INDEX)==p['index_SHA256']
    assert not OUT.exists() and shutil.disk_usage(ROOT).free>p['reserve_bytes']+4*1024**2
    OUT.mkdir();receipts=[];received=0
    def getter(start,count):
        nonlocal received
        assert count>0 and 0<=start<start+count<=p['archive_bytes'] and received+count<=p['max_total_range_bytes']
        with requests.get(p['URL'],headers={'Range':f'bytes={start}-{start+count-1}','If-Range':p['ETag'],'Accept-Encoding':'identity'},stream=True,timeout=(15,45)) as r:
            check_range(r.status_code,r.headers,start,count,p['archive_bytes'],p['ETag'])
            assert r.url==p['URL']
            chunks=[];size=0
            for chunk in r.iter_content(65536):
                size+=len(chunk);assert size<=count;chunks.append(chunk)
            assert size==count
            data=b''.join(chunks)
        received+=size
        with (OUT/f'range-{start}-{count}.bin').open('xb') as f:f.write(data)
        receipts.append({'start':start,'bytes':count,'local_SHA256':hashlib.sha256(data).hexdigest()})
        return data
    try:
        data=read_text_member(p['member'],getter,p['max_text_bytes'])
        with (OUT/'publisher-train_test_split.csv').open('xb') as f:f.write(data)
        result={'UTC':datetime.now(timezone.utc).isoformat(),'policy_SHA256':sha(POLICY),'member':p['member'],
            'split_CSV_SHA256':hashlib.sha256(data).hexdigest(),'actual_range_bytes':received,'ranges':receipts,
            'member_size_and_CRC_verified':True,'whole_archive_checksum_verified':False,
            'schema_and_original_evaluation_ID_reservation':'pending explicit inspection, not readyFIT',
            'new_photos_annotations_pixels_features_scores_gradients_optimizer':0,'ready_for_training':False}
        save(OUT/'split-receipt.json',result);save(ROOT/'reports/industrial-liaci-v88-split-receipt.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        save(OUT/'failure.json',{'UTC':datetime.now(timezone.utc).isoformat(),'error':repr(exc),'actual_range_bytes':received,'ranges':receipts,'full_fallback':False,'optimizer':0})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'))
    globals()[parser.parse_args().stage]()
