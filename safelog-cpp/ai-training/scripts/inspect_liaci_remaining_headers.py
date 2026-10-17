"""v92: remaining already-downloaded FIT headers only; no pixel load/GT repair."""
import argparse,hashlib,json,os,shutil,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.process_probe import process_exists
SOURCE=ROOT/'data/industrial-liaci-v90-coordinates'
PRIOR=ROOT/'data/industrial-liaci-v91-bitmap-diagnosis'
OUT=ROOT/'data/industrial-liaci-v92-header-diagnosis'
RUN=ROOT/'runs/industrial-liaci-header-v92'
POLICY=ROOT/'reports/industrial-liaci-v92-header-policy.json'
CODE='scripts/inspect_liaci_remaining_headers.py'
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    p=read(ROOT/'reports/industrial-liaci-v90-coordinate-policy.json');r=read(SOURCE/'member-receipts.json')
    done={read(f)['image_id']:sha(f) for f in PRIOR.glob('decoded-*.json')};assert sorted(done)==[52,107,178]
    rows=[x for x in p['probe_rows'] if x['image_id'] not in done];assert [x['image_id'] for x in rows]==[1,3,8]
    inputs=[SOURCE/'member-receipts.json',ROOT/'reports/industrial-liaci-v90-coordinate-policy.json',*PRIOR.glob('decoded-*.json')]
    files={x['path']:x['encoded_SHA256'] for x in r['rows'] if x['image_id'] in {1,3,8}}
    OUT.mkdir();RUN.mkdir()
    save(POLICY,{'UTC':utc(),'scope':'Remaining21 local file headers only after v91 format assertion; no source pixel load',
        'code_SHA256':sha(ROOT/CODE),'inputs_SHA256':{x.relative_to(ROOT).as_posix():sha(x) for x in inputs},'files_SHA256':files,'probe_rows':rows,
        'method':'Pillow open metadata/getexif/getpalette only; never load/copy/convert/array/show or follow paths/URLs in EXIF. Record actual format/size/mode/nframes and selected scalar EXIF, BMP signature/depth via original bytes. No coordinate or label repair.',
        'preservation':'Reuse three completed v91 numeric rows unchanged. v90 pairing and v91 format failures remain failed; never redownload/restart completed probes. No VAL_CAL_TEST content or selection changes.',
        'new_RGB_pixel_decodes':0,'new_network':0,'new_visual_views':0,'new_features_scores_gradients_optimizer':0,'ready_for_training':False,
        'reserve_bytes':1024**3,'storage_bound_bytes':1024**2,'data_license':'LIACi CC BY-NC-SA4.0; all parent rights retained.'})
    print(json.dumps({'policy_SHA256':sha(POLICY),'headers':len(files)}),flush=True)
def run():
    p=read(POLICY);git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in [CODE,POLICY.relative_to(ROOT).as_posix()]:assert subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])==(ROOT/n).read_bytes()
    assert sha(ROOT/CODE)==p['code_SHA256']
    for n,h in {**p['inputs_SHA256'],**p['files_SHA256']}.items():assert sha(ROOT/n)==h
    assert shutil.disk_usage(ROOT).free>=p['reserve_bytes']+p['storage_bound_bytes']
    for lock in (ROOT/'runs').glob('**/*.lock'):
        if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(lock))
    with (RUN/'operation.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        expected={r['image_id']:r for r in p['probe_rows']};rows=[]
        for receipt in read(SOURCE/'member-receipts.json')['rows']:
            if receipt['image_id'] not in expected:continue
            path=ROOT/receipt['path']
            with Image.open(path) as im:
                exif=im.getexif();row={'image_id':receipt['image_id'],'role':receipt['role'],'expected_size':expected[receipt['image_id']]['size'],
                    'format':im.format,'mode':im.mode,'size':list(im.size),'n_frames':getattr(im,'n_frames',1),
                    'orientation':exif.get(274),'datetime':str(exif.get(306)) if exif.get(306) is not None else None,
                    'native_size_matches':list(im.size)==expected[receipt['image_id']]['size'],'encoded_SHA256':receipt['encoded_SHA256']}
            rows.append(row)
        result={'UTC':utc(),'policy_SHA256':sha(POLICY),'headers_read':len(rows),'rows':rows,'new_pixel_decodes':0,'new_network_bytes':0,
            'new_visual_views':0,'new_features_scores_gradients_optimizer':0,'ready_for_training':False,'old_failed_contracts_remain_failed':True}
        save(OUT/'remaining-headers.json',result);save(ROOT/'reports/industrial-liaci-v92-remaining-headers.json',result)
        print(json.dumps([r for r in rows if r['role']=='RGB']),flush=True)
    except BaseException as exc:
        save(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(exc),'traceback':traceback.format_exc()});raise
    finally:(RUN/'operation.lock').unlink()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));globals()[parser.parse_args().stage]()
