"""v95 five known README blobs at the v94 commit, static text only."""
import argparse,base64,hashlib,json,os,shutil,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.process_probe import process_exists
OUT=ROOT/'data/industrial-liaci-v95-referenced-docs';RUN=ROOT/'runs/industrial-liaci-referenced-docs-v95'
POLICY=ROOT/'reports/industrial-liaci-v95-referenced-docs-policy.json';CODE='scripts/inspect_liaci_referenced_docs.py'
PRIOR=ROOT/'data/industrial-liaci-v94-public-code';API='https://api.github.com/repos/SINTEF/LIACI_context/git/blobs/'
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    metadata=read(PRIOR/'static-code-metadata.json');tree=read(PRIOR/'tree.json')
    wanted=['data/README.md','pipeline/README.md','pipeline/computer_vision/README.md','pipeline/video_input/README.md','demonstrator/README.md']
    rows=[x for name in wanted for x in tree['tree'] if x['path']==name]
    assert len(rows)==5 and all(x['type']=='blob' and x['size']<=16*1024 for x in rows)
    assert not set(wanted)&{x['path'] for x in metadata['selected_static_files']}
    OUT.mkdir();RUN.mkdir()
    save(POLICY,{'UTC':utc(),'scope':'Five already-listed README static texts only; no repo/tree/prior-file/source re-download',
        'revision':metadata['revision'],'rows':rows,'code_SHA256':sha(ROOT/CODE),
        'prior_metadata_SHA256':sha(PRIOR/'static-code-metadata.json'),'prior_tree_SHA256':sha(PRIOR/'tree.json'),
        'max_response_bytes':64*1024,'max_total_network_bytes':1024**2,'reserve_bytes':1024**3,'storage_bound_bytes':2*1024**2,
        'contract':'Anonymous HTTPS GET exact listed gitblobSHA at pinned v94 tree; verify Git blob SHA1/size and localSHA256. Base64 transport text only. Do not execute code/Markdown/embedded scripts or follow image/model/notebook links. Preserve originals/partial/failures.',
        'authority':'README/inference labels may establish related model use only; never inferred GT authority/version or physical capture mapping. No auto-contact/PII/login or data-license replacement.',
        'new_source_RGB_masks_pixels_views_features_scores_gradients_optimizer':0,'ready_for_training':False})
    print(json.dumps({'policy_SHA256':sha(POLICY),'files':5}),flush=True)
def run():
    p=read(POLICY);git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in [CODE,POLICY.relative_to(ROOT).as_posix()]:assert subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])==(ROOT/n).read_bytes()
    assert sha(ROOT/CODE)==p['code_SHA256'] and sha(PRIOR/'static-code-metadata.json')==p['prior_metadata_SHA256'] and sha(PRIOR/'tree.json')==p['prior_tree_SHA256']
    assert shutil.disk_usage(ROOT).free>=p['reserve_bytes']+p['storage_bound_bytes']
    for lock in (ROOT/'runs').glob('**/*.lock'):
        if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(lock))
    with (RUN/'operation.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        session=requests.Session();session.trust_env=False
        session.headers.update({'Accept':'application/vnd.github+json','Accept-Encoding':'identity','User-Agent':'SafeLog-public-static-metadata-audit'})
        received=0;rows=[]
        for i,item in enumerate(p['rows']):
            dest=OUT/f'blob-{i:02d}.json';count=0
            with session.get(API+item['sha'],stream=True,allow_redirects=False,timeout=(15,40)) as response:
                assert response.status_code==200 and response.headers.get('Content-Encoding','identity')=='identity'
                with dest.open('xb') as f:
                    for b in response.iter_content(8192):
                        count+=len(b);received+=len(b);assert count<=p['max_response_bytes'] and received<=p['max_total_network_bytes']
                        assert shutil.disk_usage(ROOT).free>=p['reserve_bytes']+len(b);f.write(b)
            obj=read(dest);assert obj['encoding']=='base64' and obj['sha']==item['sha']
            raw=base64.b64decode(''.join(obj['content'].split()),validate=True)
            assert len(raw)==obj['size']==item['size'] and hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()==item['sha']
            assert '\0' not in raw.decode('utf-8-sig')
            text=OUT/f'static-{i:02d}.txt'
            with text.open('xb') as f:f.write(raw)
            rows.append({'path':item['path'],'git_blob_SHA1':item['sha'],'local_SHA256':sha(text),'bytes':len(raw),'saved_static_text':text.relative_to(ROOT).as_posix(),
                'source_url':'https://github.com/SINTEF/LIACI_context/blob/'+p['revision']+'/'+item['path']})
        result={'UTC':utc(),'policy_SHA256':sha(POLICY),'revision':p['revision'],'rows':rows,'actual_network_bytes':received,
            'new_source_pixels_features_scores_gradients_optimizer':0,'external_code_executed':False,'ready_for_training':False,'static_review_pending':True}
        save(OUT/'referenced-docs-metadata.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        save(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(exc),'traceback':traceback.format_exc()});raise
    finally:(RUN/'operation.lock').unlink()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));globals()[parser.parse_args().stage]()
