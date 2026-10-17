"""v89: reserve original evaluation IDs, then read only original COCO JSON metadata."""
import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import re,shutil,subprocess,sys
import requests

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.public_zip_metadata import check_range,read_text_member
INDEX=ROOT/'data/industrial-liaci-v87-source/public-zip-directory.json'
SPLIT=ROOT/'data/industrial-liaci-v88-split/publisher-train_test_split.csv'
OUT=ROOT/'data/industrial-liaci-v89-metadata'
POLICY=ROOT/'reports/industrial-liaci-v89-annotation-metadata-policy.json'
CODE=[Path(__file__),ROOT/'safelog_ai/public_zip_metadata.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def freeze():
    assert not OUT.exists() and not POLICY.exists()
    assert sha(SPLIT)=='a53e08d4a666f0b0a7b1e4c38d258b82d0511f2fe7c900e789ef17ce72477ffb'
    with SPLIT.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['','file_name','split'];rows=list(reader)
    assert len(rows)==1893 and Counter(r['split'] for r in rows)=={'train':1370,'test':191,'not_used':332}
    assert len({r['file_name'] for r in rows})==1893
    assert all(int(r[''])==i and re.fullmatch(r'image_\d{4}\.jpg',r['file_name']) for i,r in enumerate(rows))
    index=read(INDEX);members=index['rows'];by_name={r['member']:r for r in members}
    by_stem=defaultdict(list)
    for r in members:
        if not r['directory'] and re.fullmatch(r'image_\d{4}\.(jpg|png|bmp)',PurePosixPath(r['member']).name):
            by_stem[PurePosixPath(r['member']).stem].append(r)
    reserved=[]
    for row in rows:
        source=by_name['LIACi_dataset_pretty/images/'+row['file_name']]
        copies=by_stem[PurePosixPath(row['file_name']).stem]
        assert len(copies)==14
        reserved.append({'file_name':row['file_name'],'publisher_split':row['split'],
                         'role':'reserved_TEST' if row['split']=='test' else 'excluded_publisher_not_used' if row['split']=='not_used' else 'unassigned_TRAIN_candidate_only',
                         'original':source,'all_copies':copies})
    OUT.mkdir()
    reservation=OUT/'reserved-publisher-evaluation-before-annotation-content.json'
    save(reservation,{'UTC':datetime.now(timezone.utc).isoformat(),'source_split_SHA256':sha(SPLIT),'index_SHA256':sha(INDEX),'rows':reserved,
                      'original_TEST_count':191,'publisher_not_used_excluded':332,'TRAIN_candidates_NOT_readyFIT':1370,'pixels_features_optimizer':0})
    prior=read(ROOT/'reports/industrial-liaci-v88-split-policy.json')
    member=by_name['LIACi_dataset_pretty/coco-labels.json']
    assert member['bytes']==15612447 and member['packed']==1791261 and member['CRC32_declared']=='d9eba479'
    save(POLICY,{'UTC':datetime.now(timezone.utc).isoformat(),'stage':'COCO JSON metadata only after immutable original split/copy reservation',
        'URL':prior['URL'],'ETag':prior['ETag'],'archive_bytes':prior['archive_bytes'],'member':member,
        'reservation_SHA256':sha(reservation),'index_SHA256':sha(INDEX),'split_SHA256':sha(SPLIT),
        'max_total_range_bytes':4*1024**2,'max_text_bytes':20*1024**2,'reserve_bytes':1024**3,
        'code_SHA256':{str(p.relative_to(ROOT)):sha(p) for p in CODE},
        'checks':'Exact public206/name/header/size/CRC/UTF8, bounded deflate, strict JSON duplicate-key/nonfinite rejection; preserve raw JSON. No imageData/base64 or code execution.',
        'roles':'All original TEST191 and14copies each reserved; publisher not_used332 excluded, not replenishment. Only original1370TRAIN remain candidates, no finalFIT_VAL_CAL or sample minimum relaxation.',
        'semantics':'Read raw category names/image metadata/annotation counts only. No polygon raster/pixel decode/GT edits or wholephoto normal assignment from missing classes. Capture/physical groups and label completeness remain unverified.',
        'future':'Separate data pixel/duplicate/group/minimum/split policy must be preregistered before source RGB or mask decode. Existing held graph/roles preserved; currentTEST never TRAIN.',
        'data_license':'Official LIACi CC BY-NC-SA4.0, noncommercial/attribution/SA/third-party rights retained; other sources unchanged',
        'new_photos_pixel_features_scores_gradients_optimizer':0,'model_protocol_fixed':False})
    print(json.dumps({'policy_SHA256':sha(POLICY),'reservation_SHA256':sha(reservation),'reserved_TEST':191,'excluded_not_used':332,'TRAIN_candidates':1370}))

def no_duplicates(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('Duplicate JSON key')
        result[k]=v
    return result

def run():
    p=read(POLICY);reservation=OUT/'reserved-publisher-evaluation-before-annotation-content.json'
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for path in [*CODE,POLICY]:assert subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+path.relative_to(ROOT).as_posix()])==path.read_bytes()
    assert sha(reservation)==p['reservation_SHA256'] and sha(INDEX)==p['index_SHA256'] and sha(SPLIT)==p['split_SHA256']
    assert all(sha(ROOT/name)==digest for name,digest in p['code_SHA256'].items())
    assert shutil.disk_usage(ROOT).free>p['reserve_bytes']+48*1024**2
    received=0;ranges=[]
    def getter(start,count):
        nonlocal received
        assert count>0 and 0<=start<start+count<=p['archive_bytes'] and received+count<=p['max_total_range_bytes']
        with requests.get(p['URL'],headers={'Range':f'bytes={start}-{start+count-1}','If-Range':p['ETag'],'Accept-Encoding':'identity'},stream=True,timeout=(15,45)) as r:
            check_range(r.status_code,r.headers,start,count,p['archive_bytes'],p['ETag']);assert r.url==p['URL']
            chunks=[];size=0
            for chunk in r.iter_content(65536):size+=len(chunk);assert size<=count;chunks.append(chunk)
            assert size==count;data=b''.join(chunks)
        received+=size
        with (OUT/f'range-{start}-{count}.bin').open('xb') as f:f.write(data)
        ranges.append({'start':start,'bytes':count,'local_SHA256':hashlib.sha256(data).hexdigest()})
        return data
    try:
        raw=read_text_member(p['member'],getter,p['max_text_bytes'])
        with (OUT/'publisher-coco-labels.json').open('xb') as f:f.write(raw)
        obj=json.loads(raw.decode('utf-8-sig'),object_pairs_hook=no_duplicates,parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
        assert isinstance(obj,dict) and all(isinstance(obj[k],list) for k in ('images','annotations','categories'))
        images=obj['images'];assert len(images)==1893
        ids={r['id'] for r in images};assert len(ids)==1893
        lookup={r['file_name']:r['publisher_split'] for r in read(reservation)['rows']}
        names=[PurePosixPath(r['file_name']).name for r in images];assert len(set(names))==1893 and set(names)==set(lookup)
        cat={r['id']:r['name'] for r in obj['categories']};assert len(cat)==len(obj['categories'])
        phase={r['id']:lookup[PurePosixPath(r['file_name']).name] for r in images}
        support=defaultdict(set)
        for r in obj['annotations']:
            assert r['image_id'] in ids and r['category_id'] in cat
            support[(phase[r['image_id']],cat[r['category_id']])].add(r['image_id'])
        result={'UTC':datetime.now(timezone.utc).isoformat(),'policy_SHA256':sha(POLICY),'reservation_SHA256':sha(reservation),'original_COCO_JSON_SHA256':hashlib.sha256(raw).hexdigest(),
            'original_member_size_CRC_verified':True,'whole_archive_checksum_verified':False,'actual_range_bytes':received,'ranges':ranges,
            'top_level_keys':list(obj),'images':len(images),'annotations':len(obj['annotations']),'categories':obj['categories'],
            'image_field_names':sorted({k for row in images for k in row}),'annotation_field_names':sorted({k for row in obj['annotations'] for k in row}),
            'category_photo_support':{phase:{name:len(support[(phase,name)]) for name in cat.values()} for phase in ('train','test','not_used')},
            'publisher_split_counts':dict(Counter(phase.values())),'capture_groups_label_completeness_and_pixel_contract_verified':False,
            'new_source_images_pixels_features_scores_gradient_optimizer':0,'ready_for_training':False}
        save(OUT/'annotation-metadata-audit.json',result);save(ROOT/'reports/industrial-liaci-v89-annotation-metadata.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        save(OUT/'failure.json',{'UTC':datetime.now(timezone.utc).isoformat(),'error':repr(exc),'ranges':ranges,'actual_range_bytes':received,'pixel_decode_optimizer':0})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));globals()[parser.parse_args().stage]()
