"""v94 bounded anonymous author-project metadata/static text; never execute it."""
import argparse,base64,hashlib,json,os,re,shutil,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
from urllib.parse import urlparse
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.process_probe import process_exists
OUT=ROOT/'data/industrial-liaci-v94-public-code';RUN=ROOT/'runs/industrial-liaci-public-code-v94'
POLICY=ROOT/'reports/industrial-liaci-v94-public-code-policy.json';CODE='scripts/inspect_liaci_public_code.py'
REPO='SINTEF/LIACI_context';API='https://api.github.com/repos/'+REPO
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    assert shutil.disk_usage(ROOT).free>=1024**3+8*1024**2
    OUT.mkdir();RUN.mkdir()
    save(POLICY,{'UTC':utc(),'scope':'Public SINTEF LIACI_context official related repository only; inspect static evidence, not assume dataset exporter',
        'repository':REPO,'API':API,'code_SHA256':sha(ROOT/CODE),'process_probe_SHA256':sha(ROOT/'safelog_ai/process_probe.py'),
        'upstream_report_SHA256':sha(ROOT/'reports/LIACI_V90_V93_RESULTS_KO.md'),
        'network':'Anonymous HTTPS GET repo, default-branch commit, exact recursive tree; then listed git blobs only. Pin immutable commit/tree/blob SHA; no clone/archive/model/image/notebook download. Exact Git blob SHA1/length and localSHA256. No code/import/eval/notebook output execution.',
        'selection':'RootREADME/LICENSE plus at most10listed py/md/rst/txt blobs <=128KiB, ranked by path terms coco20/dataset16/annotat12/label12/mask10/segment8, tie lexical. Exclude ipynb and symlinks/submodules; retain skipped candidate list.',
        'max_HTTP_response_bytes':2*1024**2,'max_total_network_bytes':4*1024**2,'max_text_member_bytes':128*1024,'max_selected_other_files':10,
        'reserve_bytes':1024**3,'storage_bound_bytes':8*1024**2,
        'new_source_RGB_masks_pixels_views_features_scores_gradients_optimizer':0,'GT_repaired':False,'ready_for_training':False,
        'authority':'Related author-project code is not evidence of current dataset production version. Record exact provenance/limitations; codeMIT cannot replace LIACi dataCC BY-NC-SA4.0. No auto-contact/PII/authentication or access bypass.',
        'preserve':'All v90-v93 failures, six originalRGB, masks/COCO, roles/held exclusions and original files remain; no complete-source decoding, distance replay, model/test or threshold work.'})
    print(json.dumps({'policy_SHA256':sha(POLICY),'max_network_bytes':4*1024**2}),flush=True)
def run():
    p=read(POLICY);git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in [CODE,POLICY.relative_to(ROOT).as_posix()]:assert subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])==(ROOT/n).read_bytes()
    assert sha(ROOT/CODE)==p['code_SHA256'] and sha(ROOT/'safelog_ai/process_probe.py')==p['process_probe_SHA256']
    assert sha(ROOT/'reports/LIACI_V90_V93_RESULTS_KO.md')==p['upstream_report_SHA256']
    assert shutil.disk_usage(ROOT).free>=p['reserve_bytes']+p['storage_bound_bytes']
    for lock in (ROOT/'runs').glob('**/*.lock'):
        if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(lock))
    with (RUN/'operation.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        session=requests.Session();session.trust_env=False
        session.headers.update({'Accept':'application/vnd.github+json','User-Agent':'SafeLog-public-static-metadata-audit','Accept-Encoding':'identity'})
        received=0;receipts=[]
        def get(url,name):
            nonlocal received
            assert urlparse(url).scheme=='https' and urlparse(url).netloc=='api.github.com' and (url.startswith(API+'/') or url==API)
            size=0;dest=OUT/name
            with session.get(url,stream=True,timeout=(15,40),allow_redirects=False) as response:
                assert response.status_code==200 and response.headers.get('Content-Encoding','identity')=='identity'
                with dest.open('xb') as f:
                    for b in response.iter_content(65536):
                        size+=len(b);received+=len(b)
                        assert size<=p['max_HTTP_response_bytes'] and received<=p['max_total_network_bytes']
                        assert shutil.disk_usage(ROOT).free>=p['reserve_bytes']+len(b)
                        f.write(b)
                receipts.append({'URL':url,'bytes':size,'SHA256':sha(dest),'ETag':response.headers.get('ETag')})
            return json.loads(dest.read_text(encoding='utf-8'))
        repo=get(API,'repository.json');assert repo['full_name'].lower()==REPO.lower() and repo['private'] is False
        branch=repo['default_branch'];assert re.fullmatch(r'[A-Za-z0-9._/-]+',branch) and '..' not in PurePosixPath(branch).parts
        commit=get(API+'/commits/'+branch,'commit.json');revision=commit['sha'];assert re.fullmatch('[a-f0-9]{40}',revision)
        tree_sha=commit['commit']['tree']['sha'];assert re.fullmatch('[a-f0-9]{40}',tree_sha)
        tree=get(API+'/git/trees/'+tree_sha+'?recursive=1','tree.json');assert tree['truncated'] is False and tree['sha']==tree_sha
        mandatory=[];candidates=[]
        for item in tree['tree']:
            path=PurePosixPath(item['path']);assert not path.is_absolute() and '..' not in path.parts and '\\' not in item['path']
            if item['type']!='blob' or item['mode'] not in ('100644','100755') or item['size']>p['max_text_member_bytes']:continue
            low=item['path'].lower()
            if len(path.parts)==1 and (low.startswith('readme.') or low in ('license','license.txt','license.md')):mandatory.append(item);continue
            if path.suffix.lower() not in ('.py','.md','.rst','.txt'):continue
            score=sum(weight for term,weight in [('coco',20),('dataset',16),('annotat',12),('label',12),('mask',10),('segment',8)] if term in low)
            if score:candidates.append((score,item))
        candidates.sort(key=lambda x:(-x[0],x[1]['path']))
        chosen=sorted(mandatory,key=lambda x:x['path'])+[x[1] for x in candidates[:p['max_selected_other_files']]]
        selected=[]
        for i,item in enumerate(chosen):
            blob=get(API+'/git/blobs/'+item['sha'],f'blob-{i:02d}.json');assert blob['encoding']=='base64' and blob['sha']==item['sha']
            raw=base64.b64decode(''.join(blob['content'].split()),validate=True)
            assert len(raw)==item['size']==blob['size'] and len(raw)<=p['max_text_member_bytes']
            assert hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()==item['sha']
            text=raw.decode('utf-8-sig');assert '\0' not in text
            dest=OUT/f'static-{i:02d}.txt'
            with dest.open('xb') as f:f.write(raw)
            selected.append({'path':item['path'],'bytes':len(raw),'git_blob_SHA1':item['sha'],'local_SHA256':sha(dest),'saved_static_text':dest.relative_to(ROOT).as_posix(),
                'source_url':'https://github.com/'+REPO+'/blob/'+revision+'/'+item['path']})
        result={'UTC':utc(),'policy_SHA256':sha(POLICY),'repository':REPO,'revision':revision,'tree_SHA':tree_sha,'tree_entries':len(tree['tree']),
            'selected_static_files':selected,'unread_relevant_candidates':[{'score':x[0],'path':x[1]['path'],'bytes':x[1]['size']} for x in candidates[p['max_selected_other_files']:]],
            'actual_network_bytes':received,'receipts':receipts,'new_source_pixels_features_scores_gradients_optimizer':0,'external_code_executed':False,
            'ready_for_training':False,'authoritative_dataset_exporter_or_GT_relationship':'Not established by metadata retrieval alone; human/static review pending',
            'disk_free_bytes':shutil.disk_usage(ROOT).free}
        save(OUT/'static-code-metadata.json',result);print(json.dumps({k:result[k] for k in ('revision','tree_entries','selected_static_files','actual_network_bytes')}),flush=True)
    except BaseException as exc:
        save(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(exc),'traceback':traceback.format_exc()});raise
    finally:(RUN/'operation.lock').unlink()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));globals()[parser.parse_args().stage]()
