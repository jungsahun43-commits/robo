"""Acquire only the publisher's small labeled archive, with encoded-only audit."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import stat
import subprocess
import sys
import zipfile
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

SOURCE=ROOT/'data/industrial-corrosion-background-v75-source'
POLICY=ROOT/'reports/industrial-corrosion-background-v75-acquisition-policy.json'
SCRIPT='scripts/fetch_corrosion_background.py'
ARCHIVE=SOURCE/'publisher-labeled-images-v1.zip'
SIZE=262967506
EXPECTED='6fbd29524a81f4d8250a3cf978da09ad09d48f65522677792156d0bdf454fcce'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not POLICY.exists() and not ARCHIVE.exists()
    snapshot=read(SOURCE/'publisher-initial-state.json')['dataset']['snapshot']
    assert snapshot['id']=='kcyn4nhv2c' and snapshot['version']==1 and snapshot['licence']['short_name']=='CC BY 4.0'
    item=[r for r in read(SOURCE/'publisher-root-files.json') if r['filename']=='Labeled images.zip'];assert len(item)==1;item=item[0]
    assert item['size']==SIZE and item['content_details']['sha256_hash']==EXPECTED and item['status']=='COMPLETED'
    assert item['content_details']['download_url']=='https://data.mendeley.com/public-files/datasets/kcyn4nhv2c/files/ee33e15a-c7ba-46f8-a06d-619c69d0f950/file_downloaded'
    assert shutil.disk_usage(ROOT).free>SIZE+1024**3+64*1024**2
    p={'UTC':utc(),'fixed_before_labeled_archive_download_member_reads':True,'source_sha256':{SCRIPT:sha(ROOT/SCRIPT)},
       'inputs_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (SOURCE/'publisher-page.html',SOURCE/'publisher-initial-state.json',SOURCE/'publisher-root-files.json',SOURCE/'publisher-folders.json')},
       'URL':item['content_details']['download_url'],'expected_bytes':SIZE,'publisher_SHA256':EXPECTED,
       'dataset':'Rios, Marina; Netto, PauloIvson; Roehl, Deane (2023), Images andpixelannotations forcorrosionsegmentation, MendeleyDataV1 DOI10.17632/kcyn4nhv2c.1',
       'license':'PublisherdatasetCC BY4.0, attribution/source/version/changes; identifiedthirdpartycontent mayneedfurtherpermission. No changingparentVTCC0/OstravaCC BY4.0/InsPLADCC BY-NC3.0 orotherterms.',
       'scope':'Authorpage describes1024completephotos, only35selectedphotos withfive5x5positivepatches andfive5x5backgroundpatches each. These are sparseannotations notfullbinarymasks/wholehealthyphotos. Counts/formats/coordinates/backgroundmeaning/splits/physicalgroups stillunverified. No assumedreadyFIT oradditionalindependentphotos.',
       'selection':'OnlyLabeled images.zip262967506bytes. Do notdownloadComplete database.zip2211263435bytes orsourcecode. No login/contact/consent required bypublicGET; no bypass ifblocked.',
       'download':'Publicstream toexclusivepartial, no retry/resumeoroverwrite inthisscript. Errorpreservespartial/failure. Fullbytes+publisherSHA256 mustmatch before renametoarchive. No guessedchecksum.',
       'archive':'Encodedonly; safe relativepaths/no drive-colon-backslash-traversal/Windowsreservedcomponents/symlinks/encryption/duplicatecasefoldedpaths; <=10000members/2GiB declaredexpanded/128MiB singlemember. ReadtoEOF CRC+encodedSHA, no sourceimages/masksdecode/annotationJSONparse orauthorcodeexecution.',
       'next':'Before pixels/contentview, inspectonlymembernames/metadata andreserveanyauthorsplits/allcopies. Then separatecodeJSONmustfixannotationcoordinate-semanticchecks, group/heldexactRGB+8orientationpHash2audit usingcompletedoldfingerprints, roles/oneallocation/minima/errors/storage. ExcludecurrentTESTlinkedTRAIN; no relabel/GTguessing/thresholdtuning. No newstudy features/gradient untilqualifieddataandseparatepreregisteredmodel.',
       'storage':'Archive263MB plusmetadata<=64MiB; no fullZIPexpansion/densefeaturecache/fullparentcopy/deletion; >=1GiBreserve.',
       'new_source_pixels_features_scores_gradient_optimizer':0,'training_protocol_fixed':False,'independent_field_validation':False}
    new(POLICY,p);print({'policy_sha256':sha(POLICY)},flush=True)


def run():
    p=read(POLICY)
    for n,h in {**p['source_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    partial=SOURCE/'publisher-labeled-images-v1.partial';assert not ARCHIVE.exists() and not partial.exists()
    assert shutil.disk_usage(ROOT).free>SIZE+1024**3+64*1024**2
    digest=hashlib.sha256();count=0
    with requests.get(p['URL'],headers={'User-Agent':'SafeLog academic research dataset acquisition','Accept-Encoding':'identity'},stream=True,timeout=(30,60)) as response:
        response.raise_for_status();assert response.status_code==200
        with partial.open('xb') as out:
            for chunk in response.iter_content(1024**2):
                if not chunk:continue
                count+=len(chunk);assert count<=SIZE;out.write(chunk);digest.update(chunk)
                if count//(32*1024**2)!=(count-len(chunk))//(32*1024**2):print({'downloaded_bytes':count,'expected':SIZE},flush=True)
    assert count==SIZE and digest.hexdigest()==EXPECTED;partial.rename(ARCHIVE)
    rows=[];seen=set()
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert len(archive.infolist())<=10000 and sum(v.file_size for v in archive.infolist())<=2*1024**3
        for item in archive.infolist():
            name=item.filename;path=PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and ':' not in name and '\\' not in name and '\x00' not in name
            assert all(part.split('.')[0].upper() not in {'CON','PRN','AUX','NUL',*['COM'+str(i) for i in range(1,10)],*['LPT'+str(i) for i in range(1,10)]} for part in path.parts)
            assert name.casefold() not in seen;seen.add(name.casefold())
            assert not stat.S_ISLNK(item.external_attr>>16) and not item.flag_bits&1 and item.file_size<=128*1024**2
            value={'member':name,'directory':item.is_dir(),'bytes':item.file_size,'packed':item.compress_size,'CRC32':f'{item.CRC:08x}'}
            if not item.is_dir():
                h=hashlib.sha256()
                with archive.open(item) as inp:
                    for chunk in iter(lambda:inp.read(1024**2),b''):h.update(chunk)
                value['encoded_SHA256']=h.hexdigest()
            rows.append(value)
    new(SOURCE/'publisher-labeled-members.json',{'rows':rows,'no_pixel_or_annotation_content_decode':True})
    audit={'UTC':utc(),'archive_bytes':count,'archive_local_SHA256':digest.hexdigest(),'publisher_SHA256_matched':True,'all_inner_CRC_encodedSHA_safe_paths_checked':True,'members':len(rows),'declared_expanded_bytes':sum(r['bytes'] for r in rows),'member_list_SHA256':sha(SOURCE/'publisher-labeled-members.json'),'source_pixel_or_annotation_content_decodes':0,'new_features_scores_gradient_optimizer':0,'whole_expansion_or_author_code_execution':False,'training_protocol_fixed':False,'independent_field_validation':False}
    new(SOURCE/'archive-audit.json',audit);new(ROOT/'reports/industrial-corrosion-background-v75-archive-audit.json',audit);print(audit,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException as e:
        new(SOURCE/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'stage':args.stage,'error':repr(e)});raise
