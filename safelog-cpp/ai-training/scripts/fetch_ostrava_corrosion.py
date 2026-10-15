"""Acquire one publisher-approved archive and inspect encoded files only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
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

SOURCE=ROOT/'data/industrial-new-rust-belt-v60-source'
POLICY=ROOT/'reports/industrial-ostrava-v60-download-policy.json'
SCRIPT='scripts/fetch_ostrava_corrosion.py'
ARCHIVE=SOURCE/'publisher-ostrava-v2.zip'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not POLICY.exists() and not ARCHIVE.exists()
    value=read(SOURCE/'ostrava-zenodo-metadata.json');item=value['files'][0]
    assert value['id']==11235637 and value['metadata']['license']['id']=='cc-by-4.0' and value['metadata']['access_right']=='open'
    assert item['size']==384363293 and item['checksum']=='md5:e30f1f62eb85018c2faee44909e69a06'
    url=item['links']['self'];assert urlparse(url).scheme=='https' and urlparse(url).hostname=='zenodo.org'
    assert shutil.disk_usage(ROOT).free>1024**3+item['size']+64*1024**2
    p={'UTC':utc(),'fixed_before_archive_download_or_member_read':True,'script_sha256':sha(ROOT/SCRIPT),
       'metadata_sha256':sha(SOURCE/'ostrava-zenodo-metadata.json'),'url':url,'expected_bytes':item['size'],'publisher_checksum':item['checksum'],
       'source':'Matej Fric, Corrosion in Industrial Complexes in Ostrava Zenodo record11235637 versionv2 DOI10.5281/zenodo.11235637; author repository https://github.com/matejfric/corrosion-segmentation links same DOI',
       'license':'data CC BY4.0 from Zenodo official metadata; cite author/source/version/changes. Other data/code licenses unchanged.',
       'scope':'publisher describes105 actual industrial corrosion photographs with segmentation annotations at2048square. Verify files/counts/polarity/splits, not yet checked. Do not assume allhealthybackground photos or machinehealthGT.',
       'download':'single public384363293byte file, no login/consent/PII, streaming partial preserved; resume only matching206ContentRange, officialMD5 pluslocalSHA256',
       'archive':'ZIP safe relative paths/no drive/colon/backslash/traversal/symlink/encryption, max2GiB declaredexpanded; read each member to EOF forCRC andencodedSHA, no originalcodeexecution or imagepixeldecode',
       'reservation':'record originalmemberpaths/authorTRAIN_VAL_TEST roles before any imageview/pixels; keep publisherTEST unchanged. No modeltraining or finalsplit atthisstep.',
       'storage':'download384MiB plusmetadata<64MiB,>=1GiBreserve; no fullextraction or existingfile deletion',
       'new_source_pixels_features_scores_optimizer':0,'training_protocol_fixed':False,'independent_field_validation':False,
       'other_candidates':'BeltCrack officialdata CC BY-SA4.0 DataCiteV1 42,032,778,597bytes21files; noarchive download atthisstep. ProjectApachelicense notdatasetlicense. CableInspect website license is not datalicense; downloadpage403, notacquired.'}
    new(POLICY,p);print({'policy_sha256':sha(POLICY)},flush=True)


def run():
    p=read(POLICY);assert sha(ROOT/SCRIPT)==p['script_sha256'] and sha(SOURCE/'ostrava-zenodo-metadata.json')==p['metadata_sha256']
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    assert not ARCHIVE.exists() and not (SOURCE/'ostrava-archive-audit.json').exists()
    partial=SOURCE/'publisher-ostrava-v2.partial';offset=partial.stat().st_size if partial.exists() else 0
    assert 0<=offset<p['expected_bytes'] and shutil.disk_usage(ROOT).free-(p['expected_bytes']-offset)>1024**3
    headers={'User-Agent':'SafeLog academic research dataset acquisition','Accept-Encoding':'identity'}
    if offset:headers['Range']=f'bytes={offset}-'
    with requests.get(p['url'],headers=headers,stream=True,timeout=(30,60)) as response:
        response.raise_for_status()
        if offset:assert response.status_code==206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
        else:assert response.status_code==200
        with partial.open('ab' if offset else 'xb') as out:
            count=offset
            for chunk in response.iter_content(1024**2):
                if not chunk:continue
                out.write(chunk);count+=len(chunk);assert count<=p['expected_bytes']
                if count//(32*1024**2)!=(count-len(chunk))//(32*1024**2):print({'downloaded_bytes':count,'expected':p['expected_bytes']},flush=True)
    assert partial.stat().st_size==p['expected_bytes']
    md5=hashlib.md5();digest=hashlib.sha256()
    with partial.open('rb') as inp:
        for chunk in iter(lambda:inp.read(1024**2),b''):md5.update(chunk);digest.update(chunk)
    assert 'md5:'+md5.hexdigest()==p['publisher_checksum']
    partial.rename(ARCHIVE);members=[]
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert sum(v.file_size for v in archive.infolist())<=2*1024**3
        for item in archive.infolist():
            name=item.filename;path=PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and ':' not in name and '\\' not in name
            assert not stat.S_ISLNK(item.external_attr>>16) and not (item.flag_bits&1)
            value={'member':name,'directory':item.is_dir(),'bytes':item.file_size,'packed':item.compress_size,'CRC32':f'{item.CRC:08x}'}
            if not item.is_dir():
                checksum=hashlib.sha256()
                with archive.open(item) as inp:
                    for chunk in iter(lambda:inp.read(1024**2),b''):checksum.update(chunk)
                value['encoded_sha256']=checksum.hexdigest()
            members.append(value)
    new(SOURCE/'ostrava-zip-members.json',{'rows':members,'originalpublisherpaths_before_pixeldecode_or_view':True})
    result={'UTC':utc(),'policy_sha256':sha(POLICY),'archive_bytes':ARCHIVE.stat().st_size,'local_archive_sha256':digest.hexdigest(),
            'publisher_MD5_matched':md5.hexdigest(),'all_inner_files_CRC_and_encodedSHA_checked':True,'members':len(members),
            'files':sum(not r['directory'] for r in members),'declared_expanded_bytes':sum(r['bytes'] for r in members),
            'member_list_sha256':sha(SOURCE/'ostrava-zip-members.json'),'new_pixel_decode_contentview_features_scores_optimizer':0,
            'full_extraction_or_code_execution':False,'training_protocol_fixed':False,'independent_field_validation':False}
    new(SOURCE/'ostrava-archive-audit.json',result);new(ROOT/'reports/industrial-ostrava-v60-archive-audit.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException as error:
        new(SOURCE/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'stage':args.stage,'error':repr(error)})
        raise
