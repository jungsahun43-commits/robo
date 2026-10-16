"""Fetch only a publicly observed publisher page; never source images or code."""
from datetime import datetime,timezone
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.request
import urllib.parse

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/industrial-corrosion-background-v75-source'
URL='https://data.mendeley.com/datasets/kcyn4nhv2c/1'


class Scripts(HTMLParser):
    def __init__(self):super().__init__();self.active=None;self.items=[]
    def handle_starttag(self,tag,attrs):
        if tag=='script':self.active={'attrs':dict(attrs),'text':''}
    def handle_data(self,data):
        if self.active is not None:self.active['text']+=data
    def handle_endtag(self,tag):
        if tag=='script' and self.active is not None:self.items.append(self.active);self.active=None


def main():
    OUT.mkdir(exist_ok=False)
    req=urllib.request.Request(URL,headers={'User-Agent':'SafeLog-academic-metadata-review/1.0'})
    with urllib.request.urlopen(req,timeout=45) as response:
        assert response.status==200;raw=response.read(4*1024**2+1);assert len(raw)<=4*1024**2
        headers=dict(response.headers);final=response.url
    (OUT/'publisher-page.html').write_bytes(raw)
    parser=Scripts();parser.feed(raw.decode('utf-8'))
    result={'UTC':datetime.now(timezone.utc).isoformat(),'requested_url':URL,'final_url':final,'bytes':len(raw),'local_sha256':hashlib.sha256(raw).hexdigest(),'headers':headers,
            'script_elements':[{'attrs':v['attrs'],'text_characters':len(v['text'])} for v in parser.items],
            'source_images_downloaded_or_decoded':0,'model_features_scores_gradient_optimizer':0,'scripts_executed':False,'dataset_files_accessed':False}
    for i,v in enumerate(parser.items):
        if v['text']:(OUT/f'page-inline-script-{i}.txt').write_text(v['text'],encoding='utf-8')
    (OUT/'metadata-fetch.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='headers'})


def inventory():
    # Public anonymous route templates were checked in the previously preserved
    # Mendeley browser client; dataset/version are the observed publisher page.
    state=json.loads((OUT/'publisher-initial-state.json').read_text(encoding='utf-8'))
    snapshot=state['dataset']['snapshot'];assert snapshot['id']=='kcyn4nhv2c' and snapshot['version']==1 and snapshot['licence']['short_name']=='CC BY 4.0'
    root='https://data.mendeley.com/public-api/datasets/kcyn4nhv2c'
    for name,url in (('publisher-folders',root+'/folders/1'),('publisher-root-files',root+'/files?'+urllib.parse.urlencode({'folder_id':'root','version':1,'$start':0,'$limit':1000}))):
        request=urllib.request.Request(url,headers={'Accept':'application/vnd.mendeley-public-dataset.1+json','User-Agent':'SafeLog-academic-metadata-review/1.0'})
        with urllib.request.urlopen(request,timeout=45) as response:
            assert response.status==200;raw=response.read(4*1024**2+1);assert len(raw)<=4*1024**2
        value=json.loads(raw)
        with (OUT/(name+'.json')).open('xb') as out:out.write(raw)
        receipt={'UTC':datetime.now(timezone.utc).isoformat(),'URL':url,'bytes':len(raw),'local_SHA256':hashlib.sha256(raw).hexdigest(),'public_anonymous_template_from_preserved_official_client':True,'image_annotation_files_downloaded':0}
        with (OUT/(name+'-receipt.json')).open('x',encoding='utf-8') as out:json.dump(receipt,out,indent=2)
        print({'kind':name,'metadata':value})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('page','inventory'));args=parser.parse_args()
    if args.stage=='page':main()
    else:inventory()
