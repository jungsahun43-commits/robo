"""Fetch the paper-linked Re-GBC release, preserving metadata and archive integrity."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
from pathlib import Path,PurePosixPath
import shutil
import sys
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha
from scripts.study_equipment_expansion import read
DATA=ROOT/'data/industrial-gearbox-source-v25'
REF='jhnnsbr/gearbox-components-remanufacturing-inspection'
URL='https://www.kaggle.com/api/v1/datasets/download/'+REF+'?datasetVersionNumber=1'


def main():
    metadata=read(DATA/'publisher-metadata.json')
    assert metadata['ref']==REF and not metadata['isPrivate'] and metadata['currentVersionNumber']==1
    assert metadata['licenseName']=='Attribution 4.0 International (CC BY 4.0)'
    assert not (DATA/'download-receipt.json').exists() and shutil.disk_usage(ROOT).free>2*1024**3
    partial=DATA/'publisher-v1.partial.zip';target=DATA/'publisher-v1.zip'
    assert not partial.exists() and not target.exists(),'Preserve/inspect a previous download before any retry'
    request=urllib.request.Request(URL,headers={'User-Agent':'SafeLog-academic-dataset-audit'})
    digest=hashlib.sha256();count=0;mark=64*1024**2
    with urllib.request.urlopen(request,timeout=60) as response,partial.open('xb') as out:
        assert response.status==200
        for block in iter(lambda:response.read(1024**2),b''):
            out.write(block);digest.update(block);count+=len(block)
            assert count<=1024**3,'Unexpected oversized download'
            if count>=mark:
                print({'downloaded_MiB':round(count/1024**2,1)},flush=True);mark+=64*1024**2
    members=[]
    with zipfile.ZipFile(partial) as archive:
        info=archive.infolist();assert len(info)<=25000 and sum(r.file_size for r in info)<=1024**3
        assert archive.testzip() is None
        for item in info:
            path=PurePosixPath(item.filename.replace('\\','/'))
            assert not path.is_absolute() and '..' not in path.parts and not any(':' in p for p in path.parts)
            assert ((item.external_attr>>16)&0o170000)!=0o120000
            members.append({'name':item.filename,'bytes':item.file_size,'CRC':item.CRC})
    partial.rename(target)
    result={'UTC':datetime.now(timezone.utc).isoformat(),'status':'downloaded_integrity_checked_not_trained',
        'publisher_ref':REF,'publisher_version':1,'publisher_page':'https://www.kaggle.com/datasets/'+REF,
        'paper_source':'https://arxiv.org/pdf/2511.15440','paper_DOI':'10.1109/ETFA65518.2025.11205777',
        'publisher_license':metadata['licenseName'],'publisher_metadata_sha256':sha(DATA/'publisher-metadata.json'),
        'download_endpoint':URL,'downloaded_archive_bytes':count,'archive_sha256_local':digest.hexdigest(),
        'publisher_supplied_SHA256':False,'ZIP_CRC_all_members_passed':True,'archive_member_count':len(members),
        'uncompressed_bytes':sum(r['bytes'] for r in members),'suffix_counts':dict(Counter(PurePosixPath(r['name']).suffix.lower() for r in members)),
        'members':members,'archive_code_executed':False,'image_content_viewed_by_agent':False,
        'source_full_photo_vs_patch_scope':'publisher dataset is pre-extracted128x128 visible-surface patches, not whole gearbox functional health',
        'split_and_duplicates_checked':False,'new_model_features_scores_optimizer_updates':0,
        'next':'Read publisher annotation/part/acquisition split files as data; preserve original grouping, choose fixed part-disjoint TEST and FIT/VAL/CAL before model scores, cross-check exact/near against all retained TEST before training.'}
    new(DATA/'download-receipt.json',result)
    print({k:v for k,v in result.items() if k!='members'},flush=True)
    print({'first_members':members[:12],'metadata_members':[r for r in members if PurePosixPath(r['name']).suffix.lower() in ('.csv','.json','.txt','.md')][:30]},flush=True)


if __name__=='__main__':main()
