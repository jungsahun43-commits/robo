"""Download original fine masks and publisher split files; never execute archive code."""
from datetime import datetime,timezone
import json
from pathlib import Path,PurePosixPath
import shutil
import sys
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
DATA=ROOT/'data/industrial-ksdd1-v18'
PAGE='https://www.vicos.si/resources/kolektorsdd/'
URLS={'fine-originals.zip':'https://data.vicos.si/datasets/KSDD/KolektorSDD.zip',
      'publisher-training-splits.zip':'https://data.vicos.si/datasets/KSDD/KolektorSDD-training-splits.zip',
      'author-page.html':PAGE}


def fetch(name,url):
    path=DATA/name;partial=path.with_suffix(path.suffix+'.partial')
    if path.exists():raise RuntimeError('Existing unsealed download must be reviewed: '+str(path))
    req=urllib.request.Request(url,headers={'User-Agent':'SafeLogAcademicDatasetAudit/1.0'})
    with urllib.request.urlopen(req,timeout=60) as response,partial.open('xb') as out:
        declared=int(response.headers.get('Content-Length',0));assert declared<=160_000_000
        total=0
        while block:=response.read(1024*1024):
            total+=len(block);assert total<=160_000_000;out.write(block)
        assert not declared or total==declared
        metadata={'requested_url':url,'final_url':response.url,'bytes':total,'headers':dict(response.headers)}
    partial.rename(path);metadata['local_sha256']=sha(path)
    return metadata


def extract(archive,destination):
    destination.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())<3000 and sum(i.file_size for i in z.infolist())<750_000_000
        bad=z.testzip();assert bad is None,bad
        for info in z.infolist():
            rel=PurePosixPath(info.filename)
            assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in info.filename and ':' not in info.filename
            target=destination.joinpath(*rel.parts)
            assert target.resolve().is_relative_to(destination.resolve())
            if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,target.open('xb') as out:shutil.copyfileobj(source,out)


def main():
    assert not DATA.exists();DATA.mkdir(parents=True)
    downloaded={name:fetch(name,url) for name,url in URLS.items()}
    extract(DATA/'fine-originals.zip',DATA/'originals')
    extract(DATA/'publisher-training-splits.zip',DATA/'publisher-splits')
    files={p.relative_to(DATA).as_posix():sha(p) for d in ('originals','publisher-splits') for p in (DATA/d).rglob('*') if p.is_file()}
    result={'UTC':datetime.now(timezone.utc).isoformat(),'author_page':PAGE,'downloads':downloaded,'files':files,
        'license':'CC BY-NC-SA 4.0, stated on author page; noncommercial academic research, attribution/share-alike',
        'citation':'Tabernik et al., Segmentation-Based Deep-Learning Approach for Surface-Defect Detection, DOI10.1007/s10845-019-01476-x',
        'fine_annotation_download':True,'box_annotation_archive_not_used':True,
        'publisher_supplied_archive_checksum_available':False,'local_SHA_and_ZIP_CRC_checked':True,
        'archive_code_or_pickle_executed':False,'model_training_or_scoring_performed':False}
    new(DATA/'download.json',result)
    print({'downloaded':True,'source_files':len(files),'archives':downloaded},flush=True)


if __name__=='__main__':main()
