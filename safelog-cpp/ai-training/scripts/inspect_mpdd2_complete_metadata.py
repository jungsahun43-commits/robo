"""Complete public embedded listings, no Google account/cookies or image requests."""
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
import requests

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.fetch_equipment_expansion import sha

OUT=ROOT/'data/industrial-mpdd2-complete-v38'
ROOT_ID='1KAJfPPuusS4fEbPocHtIjLehCkpSEUSS'


class PublicEntries(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.entries=[];self.entry_ids=[];self.anchor=None
    def handle_starttag(self,tag,attrs):
        data=dict(attrs)
        if tag=='div' and data.get('id','').startswith('entry-'):self.entry_ids.append(data['id'][6:])
        if tag=='a':self.anchor={'href':data.get('href',''),'text':[]}
    def handle_data(self,data):
        if self.anchor is not None:self.anchor['text'].append(data)
    def handle_endtag(self,tag):
        if tag!='a' or self.anchor is None:return
        value=self.anchor;self.anchor=None;url=urlsplit(value['href'])
        if url.scheme!='https' or url.hostname!='drive.google.com':return
        file=re.fullmatch(r'/file/d/([A-Za-z0-9_-]{10,100})/view',url.path)
        folder=re.fullmatch(r'/drive/folders/([A-Za-z0-9_-]{10,100})',url.path)
        if file or folder:
            name=''.join(value['text']).strip();assert name and '/' not in name and '\\' not in name
            self.entries.append({'id':(file or folder)[1],'name':name,'kind':'file' if file else 'folder'})
    def result(self):
        assert len(self.entry_ids)==len(set(self.entry_ids))
        assert set(self.entry_ids)=={row['id'] for row in self.entries}
        assert len(self.entries)==len(self.entry_ids)
        return self.entries


def main():
    assert not (OUT/'inventory.json').exists();OUT.mkdir(exist_ok=True)
    new(OUT/'intent-before-complete-listing.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'author_root_id':ROOT_ID,'scope':'all publicly visible MPDD2 folders, metadata only; no script/browser account/cookies/images',
        'source_protocol':'embeddedfolderview public HTML as documented by gdownv6.0.0; own HTMLParser reads links only, no JS executed',
        'minimum_per_category_TEST_each_label':20,'publisher_TEST_or_masks_not_training':True,
        'old_partial_inventory_SHA256':sha(ROOT/'data/industrial-mpdd2-source-v37/public-inventory.json'),
        'no_model_or_photograph_download':True})
    receipts=[];items=[];seen=set()
    def crawl(folder,path,depth=0):
        assert folder not in seen and len(seen)<48 and depth<=6;seen.add(folder)
        url='https://drive.google.com/embeddedfolderview?id='+folder
        cached=OUT/'probe-bracket-grey-train-embedded.html'
        if folder=='1YmzEb3hRDurI4FeJGqqDY252cArxwVkS' and cached.exists():
            raw=cached.read_bytes();origin='earlier_same_turn_public_response';status=200
        else:
            response=requests.get(url,timeout=(20,40));response.raise_for_status()
            assert response.url.startswith('https://drive.google.com/embeddedfolderview')
            raw=response.content;origin='public_HTTP_no_browser_credentials';status=response.status_code
        assert len(raw)<8*1024**2
        filename=OUT/f'embedded-{len(receipts):02d}.html'
        with filename.open('xb') as output:output.write(raw)
        parser=PublicEntries();parser.feed(raw.decode('utf-8'));entries=parser.result()
        receipts.append({'folder_id':folder,'path':path,'url':url,'origin':origin,'status':status,'bytes':len(raw),'sha256':sha(filename),'entries':len(entries),'entry_links_correspond_one_to_one':True})
        for entry in entries:
            relative=path+'/'+entry['name'] if path else entry['name'];items.append({**entry,'relative_path':relative,'parent_id':folder})
            if entry['kind']=='folder':crawl(entry['id'],relative,depth+1)
        print({'folder':path or 'root','visible_entries':len(entries)},flush=True)
    try:crawl(ROOT_ID,'')
    except BaseException as error:
        new(OUT/'failure.json',{'error':repr(error),'receipts':receipts,'items':items,'images_downloaded':0});raise
    files=[row for row in items if row['kind']=='file'];counts=dict(Counter(row['relative_path'].rsplit('/',1)[0] for row in files))
    old=read(ROOT/'data/industrial-mpdd2-source-v37/public-inventory.json')
    old_files={row['id']:row for row in old['items'] if row['mime']!='application/vnd.google-apps.folder'}
    new_files={row['id']:row for row in files};intersection=set(old_files)&set(new_files)
    unmatched=[{'id':key,'path':old_files[key]['relative_path']} for key in sorted(set(old_files)-set(new_files))]
    assert not unmatched,'Earlier public IDs absent; preserve and investigate, not merge inventories'
    assert all(old_files[key]['relative_path']==new_files[key]['relative_path'] for key in intersection)
    sizes={key:int(old_files[key]['raw_metadata'][13]) for key in intersection}
    result={'UTC':datetime.now(timezone.utc).isoformat(),'files':files,'receipts':receipts,'paths_by_parent':counts,
        'listed_public_files':len(files),'earlier_partial_files_all_retained':True,'old_partial_files':len(old_files),
        'newly_listed_metadata_not_new_photos':len(files)-len(old_files),'known_file_byte_sizes_from_prior_metadata':sizes,
        'normal_TEST_below20_by_category':[category for category in ('bracket_brown','bracket_grey','bracket_white','rosette') if counts.get(category+'/test/normal',0)<20],
        'images_downloaded_decoded_viewed':0,'new_model_features_gradients_optimizer':0,
        'account_personal_information_or_cookies_used':False,'whole_publisher_hidden_assets_not_asserted':True,
        'original_TEST_reserved':True,'license':'author MPDD2 CC BY-NC-SA4.0, DOI10.1109/ICUMT57764.2022.9943437'}
    new(OUT/'inventory.json',result)
    summary={key:value for key,value in result.items() if key not in ('files','receipts','known_file_byte_sizes_from_prior_metadata')}
    summary['inventory_SHA256']=sha(OUT/'inventory.json')
    new(ROOT/'reports/industrial-mpdd2-v38-complete-metadata.json',summary)
    print(summary,flush=True)


if __name__=='__main__':main()
