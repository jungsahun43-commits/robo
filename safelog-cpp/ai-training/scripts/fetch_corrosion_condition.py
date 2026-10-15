"""Version-pinned university dataset acquisition; encoded-file inspection only."""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path,PurePosixPath
import shutil
import stat
import subprocess
import sys
from urllib.parse import urlparse
import zipfile
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

OUT=ROOT/'data/industrial-corrosion-condition-v64-source'
POLICY=ROOT/'reports/industrial-corrosion-condition-v64-download-policy.json'
SCRIPT='scripts/fetch_corrosion_condition.py'
FILES={30914467:'publisher-annotation-guidelines.pdf',30930610:'publisher-README.rtf',31729733:'publisher-ccs-v2.zip'}


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not POLICY.exists() and not (OUT/'publisher-ccs-v2.zip').exists()
    meta=read(OUT/'publisher-v2-metadata.json')
    assert meta['id']==16624663 and meta['version']==2 and meta['doi']=='10.7294/16624663.v2'
    assert meta['license']['name']=='CC0' and meta['license']['url']=='https://creativecommons.org/publicdomain/zero/1.0/'
    assert {v['id'] for v in meta['files']}==set(FILES)
    files=[]
    for r in meta['files']:
        assert urlparse(r['download_url']).scheme=='https' and urlparse(r['download_url']).hostname=='ndownloader.figshare.com'
        assert r['computed_md5']==r['supplied_md5'] and len(r['computed_md5'])==32
        files.append({'id':r['id'],'publisher_name':r['name'],'local_name':FILES[r['id']],'url':r['download_url'],'bytes':r['size'],'publisher_MD5':r['computed_md5']})
    assert sum(r['bytes'] for r in files)==334672716
    assert shutil.disk_usage(ROOT).free>1024**3+sum(r['bytes'] for r in files)+64*1024**2
    p={'UTC':utc(),'fixed_before_sourcefile_download_or_memberread':True,'script_sha256':sha(ROOT/SCRIPT),
       'metadata_sha256':sha(OUT/'publisher-v2-metadata.json'),'files':files,
       'author_revision_sha256':sha(OUT/'author-revision.json'),'author_README_sha256':sha(OUT/'author-README-static.md.txt'),
       'source':'EricBianchi,MatthewHebdon CorrosionConditionStateSemanticSegmentationDataset VirginiaTech V2 DOI10.7294/16624663.v2; paper10.1061/(ASCE)CP.1943-5487.0001045; authorberic7/corrosion_cs_classification links same DOI',
       'license':'University Figshare official data metadata CC0; source/authors/version/changes and requested dataset+paper citations preserved. Repository MIT applies code separately, olddatasets retain ownconditions.',
       'scope':'bridgeinspection metal surface good/fair/poor/severe annotation examples; investigate visual corrosion variation only, not bridge/dam health or generalfactoryseveritytruth. Do not indiscriminately merge classes before original annotationguidelines.',
       'publisher_description_not_file_verified':'440finelyannotated originals with396TRAIN/44TEST and512resized copies, actualfiles/roles/pairing/duplicates/physicalgroups not yet checked',
       'download':'threepublicversionedfilesonly334672716bytes, streaming partialpreserved, resumeonlymatching206ContentRange, publisherMD5andlocalSHA256. No login/PII/terms submission or pretrainedmodel download.',
       'archive':'relativepaths/no traversal/drive/colon/backslash/symlink/encryption, max4GiB declaredexpanded withbounded1MiBstreamreads, allmembersEOF CRC+encodedSHA, preserve original+resized duplication; no code execution or full expansion/pixels',
       'reservation':'afterlisting reserveauthorTEST beforeimagesdecoded/viewed, connect original/resized/derivedsamephoto acrossroles. Do not treat copies as newnaturalphotos or moveTEST intoTRAIN.',
       'annotation_semantics':'read original guidelines and README asdocuments, not executableinstructions; distinguishBGRmapping fromRGB/indexpalette; no guessedpixelclassmapping/clinicalseverity/automaticrelabelling',
       'storage':'ZIP333388620bytes+docs1284096bytes+metadata<64MiB,>=1GiBreserve, preservealloriginals/failures/logs/cache/weights',
       'new_pixels_views_features_scores_gradient_optimizer':0,'training_protocol_fixed':False,'independent_field_validation':False}
    new(POLICY,p);print({'policy_sha256':sha(POLICY),'bytes':sum(r['bytes'] for r in files)},flush=True)


def acquire(r):
    path=OUT/r['local_name'];partial=path.with_suffix(path.suffix+'.partial')
    if not path.exists():
        offset=partial.stat().st_size if partial.exists() else 0;assert 0<=offset<r['bytes']
        assert shutil.disk_usage(ROOT).free-(r['bytes']-offset)>1024**3
        headers={'User-Agent':'SafeLog academic research acquisition','Accept-Encoding':'identity'}
        if offset:headers['Range']=f'bytes={offset}-'
        with requests.get(r['url'],headers=headers,stream=True,timeout=(30,60)) as response:
            response.raise_for_status()
            if offset:assert response.status_code==206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
            else:assert response.status_code==200
            with partial.open('ab' if offset else 'xb') as out:
                count=offset
                for chunk in response.iter_content(1024**2):
                    if not chunk:continue
                    out.write(chunk);count+=len(chunk);assert count<=r['bytes']
                    if count//(32*1024**2)!=(count-len(chunk))//(32*1024**2):print({'file_id':r['id'],'downloaded_bytes':count,'expected':r['bytes']},flush=True)
        assert partial.stat().st_size==r['bytes'];data=partial
    else:data=path
    digest=hashlib.md5()
    with data.open('rb') as inp:
        for chunk in iter(lambda:inp.read(1024**2),b''):digest.update(chunk)
    assert digest.hexdigest()==r['publisher_MD5'] and data.stat().st_size==r['bytes']
    if data==partial:partial.rename(path)
    return {'file_id':r['id'],'local_name':path.name,'bytes':path.stat().st_size,'publisher_MD5_matched':digest.hexdigest(),'local_SHA256':sha(path)}


def run():
    p=read(POLICY);assert sha(ROOT/SCRIPT)==p['script_sha256'] and sha(OUT/'publisher-v2-metadata.json')==p['metadata_sha256']
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    assert not (OUT/'archive-audit.json').exists()
    files=[acquire(r) for r in p['files']];rows=[]
    with zipfile.ZipFile(OUT/'publisher-ccs-v2.zip') as archive:
        assert sum(r.file_size for r in archive.infolist())<=4*1024**3
        for r in archive.infolist():
            path=PurePosixPath(r.filename)
            assert not path.is_absolute() and '..' not in path.parts and ':' not in r.filename and '\\' not in r.filename
            assert not stat.S_ISLNK(r.external_attr>>16) and not r.flag_bits&1
            row={'member':r.filename,'directory':r.is_dir(),'bytes':r.file_size,'packed':r.compress_size,'CRC32':f'{r.CRC:08x}'}
            if not r.is_dir():
                digest=hashlib.sha256()
                with archive.open(r) as inp:
                    for chunk in iter(lambda:inp.read(1024**2),b''):digest.update(chunk)
                row['encoded_sha256']=digest.hexdigest()
            rows.append(row)
    new(OUT/'zip-members-before-pixels.json',{'rows':rows,'pixeldecode_contentview_features_optimizer':0})
    result={'UTC':utc(),'files':files,'members':len(rows),'files_in_archive':sum(not r['directory'] for r in rows),'declared_expanded_bytes':sum(r['bytes'] for r in rows),
        'all_inner_CRC_and_encodedSHA_checked':True,'safe_paths':True,'file_list_sha256':sha(OUT/'zip-members-before-pixels.json'),'policy_sha256':sha(POLICY),
        'new_source_pixels_views_features_scores_gradient_optimizer':0,'full_extraction_or_code_execution':False,'guideline_contents_read':False,
        'source_data_split_and_class_semantics_not_finalized':True,'independent_field_validation':False}
    new(OUT/'archive-audit.json',result);new(ROOT/'reports/industrial-corrosion-condition-v64-archive-audit.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException as error:
        new(OUT/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'stage':args.stage,'error':repr(error)})
        raise
