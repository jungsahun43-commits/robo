"""Download the author-linked public academic dataset; never execute its files."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse
import sys
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_facility_target import sha,save,read

FILE_ID='1PQ50QKfy2vnDOHSmw5bpBFi33hZsSXuM'
SOURCE='https://github.com/ben-z-original/s2ds'
URL='https://drive.usercontent.google.com/download'


class DownloadForm(HTMLParser):
    def __init__(self):super().__init__();self.active=False;self.action=None;self.fields={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':self.active=True;self.action=a.get('action')
        if self.active and tag=='input' and a.get('name'):self.fields[a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.active=False


def main():
    pinned=read(ROOT/'datasets/s2ds_source.json')
    archive=ROOT/'data/facility-archives/s2ds.zip';archive.parent.mkdir(parents=True,exist_ok=True)
    audit=ROOT/'reports/facility-target-s2ds-download.json'
    if archive.exists():
        if not audit.exists() or read(audit)['sha256']!=sha(archive):raise ValueError('Existing archive provenance mismatch')
        print('Previously downloaded S2DS archive hash confirmed');return
    response=requests.get(URL,params={'id':FILE_ID,'export':'download'},timeout=60)
    response.raise_for_status()
    form=DownloadForm();form.feed(response.text)
    if form.action!=URL or form.fields.get('id')!=FILE_ID or form.fields.get('export')!='download':
        raise ValueError('No ordinary author-linked download confirmation form; human access may be required')
    # This is the public large-file virus-scan warning, not a CAPTCHA solver.
    response=requests.get(form.action,params=form.fields,stream=True,timeout=(30,60))
    response.raise_for_status()
    if 'text/html' in response.headers.get('content-type','').lower():
        raise ValueError('Download still requires human access; do not count it as training data')
    partial=archive.with_suffix('.zip.partial');total=0;next_report=128*1024**2
    with partial.open('wb') as handle:
        for block in response.iter_content(4*1024**2):
            if not block:continue
            if total==0 and not block.startswith(b'PK\x03\x04'):raise ValueError('Response is not a ZIP file')
            total+=len(block)
            if total>3*1024**3:raise ValueError('Download exceeds declared safety limit')
            handle.write(block)
            if total>=next_report:print(f'S2DS downloaded {total//1024**2} MiB',flush=True);next_report+=128*1024**2
    expected=response.headers.get('content-length')
    if expected and total!=int(expected):raise ValueError('Download content length mismatch')
    with zipfile.ZipFile(partial) as bundle:
        if bundle.testzip() is not None:raise ValueError('S2DS archive CRC mismatch')
        count=len(bundle.infolist())
    if total!=pinned['bytes'] or sha(partial)!=pinned['sha256']:
        raise ValueError('Author-linked payload differs from audited local snapshot; inspect revision before training')
    partial.replace(archive)
    save(audit,{'source':SOURCE,'author_linked_file_id':FILE_ID,'bytes':total,'sha256':sha(archive),
                'zip_entries':count,'archive_crc_checked':True,'author_published_checksum':None,
                'license':'Academic use only, no dataset redistribution; see author README, not repository GPL code license',
                'used_for_training_at_download':False,'next_gate':'Verify labels, parent groups and overlap with all existing validation/test data before training'})
    print('Author-linked archive downloaded and CRC checked; no training-data claim yet',flush=True)


if __name__=='__main__':main()
