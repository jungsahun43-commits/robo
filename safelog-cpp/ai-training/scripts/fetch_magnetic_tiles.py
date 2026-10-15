"""Fetch author-published magnetic tile images/masks for noncommercial research.

The author expressly permits academic research with citation. No SPDX license
is invented, publisher code is not executed, and unlisted ZIP files are ignored.
"""
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/industrial-magnetic-tiles-v16'
REPO='abin24/Magnetic-tile-defect-datasets.'
COLLECTION='abin24/Surface-Inspection-defect-detection-dataset'
FOLDERS={'MT_Blowhole','MT_Break','MT_Crack','MT_Fray','MT_Free','MT_Uneven'}


def main():
    assert not DATA.exists(),'Keep existing source receipts; do not download twice.'
    DATA.mkdir()
    session=requests.Session();session.headers['User-Agent']='SafeLog-academic-source-audit/1.0'
    receipts=[]
    def get(url,name):
        r=session.get(url,timeout=45);r.raise_for_status();b=r.content
        (DATA/name).write_bytes(b)
        receipts.append({'url':url,'file':name,'status':r.status_code,'sha256':hashlib.sha256(b).hexdigest(),'UTC':datetime.now(timezone.utc).isoformat()})
        return r
    meta=get('https://api.github.com/repos/'+REPO,'repo.json').json()
    commit=get('https://api.github.com/repos/'+REPO+'/commits/'+meta['default_branch'],'commit.json').json()['sha']
    tree=get('https://api.github.com/repos/'+REPO+'/git/trees/'+commit+'?recursive=1','tree.json').json()
    assert not tree['truncated']
    readme=get('https://raw.githubusercontent.com/'+REPO+'/'+commit+'/README.md','README_author.md').text
    cm=get('https://api.github.com/repos/'+COLLECTION+'/commits/master','collection_commit.json').json()['sha']
    conditions=get('https://raw.githubusercontent.com/'+COLLECTION+'/'+cm+'/README.md','README_author_collection.md').text
    assert 'It is welcomed to use our dataset' in readme
    assert 'only for academic research, no commercial purposes are allowed' in conditions
    assert 'Magnetic tile dataset by ourselves' in conditions and REPO in conditions
    selected=[r for r in tree['tree'] if r['type']=='blob' and PurePosixPath(r['path']).parts[0] in FOLDERS and PurePosixPath(r['path']).suffix.lower() in ('.jpg','.png','.bmp')]
    assert selected and sum(r['size'] for r in selected)<250*1024**2
    url='https://codeload.github.com/'+REPO+'/zip/'+commit
    h=hashlib.sha256();size=0;archive=DATA/'author-pinned.zip'
    with session.get(url,timeout=60,stream=True) as response:
        response.raise_for_status()
        with archive.open('xb') as f:
            for chunk in response.iter_content(1024*1024):
                size+=len(chunk);assert size<250*1024**2;h.update(chunk);f.write(chunk)
    files={}
    with zipfile.ZipFile(archive) as z:
        prefixes={PurePosixPath(n).parts[0] for n in z.namelist()};assert len(prefixes)==1
        prefix=next(iter(prefixes))
        for entry in selected:
            name=entry['path'];parts=PurePosixPath(name).parts
            assert '..' not in parts and '\\' not in name and not PurePosixPath(name).is_absolute()
            info=z.getinfo(prefix+'/'+name);assert info.file_size==entry['size'] and info.file_size<8*1024**2
            raw=z.read(info)
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==entry['sha']
            target=DATA/'originals'/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(raw)
            files[name]=hashlib.sha256(raw).hexdigest()
    result={'dataset':'author_magnetic_tiles','author_repository':REPO,'revision':commit,'collection_revision':cm,
        'source_readme_research_permission_checked':True,'research_permission':'academic research only, no commercial use, cite corresponding provider',
        'standardized_dataset_SPDX_license_not_assumed':True,'GitHub_API_license':meta.get('license'),
        'archive_url':url,'archive_sha256':h.hexdigest(),'archive_bytes':size,'files':files,
        'per_file_publisher_Git_blob_SHA_matched':True,'publisher_code_executed':False,
        'training_or_TEST_split_declared_yet':False,'new_optimizer_updates':0,'receipts':receipts}
    with (DATA/'download.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print({'source_revision':commit,'files':len(files),'bytes':size,'research_permission_checked':True},flush=True)


if __name__=='__main__':main()
