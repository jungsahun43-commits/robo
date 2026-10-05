"""One fixed six-plus-six native ROI pair with durable logs and no overwrite."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_native_roi_study import PROTOCOL,NAMES,REFERENCE,validate_protocol,write
from scripts.train_facility_target import read,sha


def main():
    p=validate_protocol(read(ROOT/PROTOCOL),ROOT)
    preflight=read(ROOT/'runs/facility-native-roi-preflight.json')
    snapshot=read(ROOT/'runs/facility-native-roi-source-before-training.json')
    if (preflight.get('status')!='passed' or preflight['protocol_sha256']!=sha(ROOT/PROTOCOL)
            or snapshot['protocol_sha256']!=sha(ROOT/PROTOCOL)
            or snapshot['preflight_sha256']!=sha(ROOT/'runs/facility-native-roi-preflight.json')):
        raise ValueError('Before-training technical/source proof must match')
    if any((ROOT/'runs'/name).exists() for name in NAMES.values()):
        raise ValueError('Preserve existing training runs')
    directory=ROOT/'runs/facility-native-roi-background-pair';directory.mkdir(exist_ok=False)
    state={'schema':'facility_native_roi_background_pair_v1','pid':os.getpid(),
           'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
           'protocol_sha256':sha(ROOT/PROTOCOL),'source_sha256':p['source_sha256'],'stages':[]}
    def save():
        temporary=directory/'status.tmp';write(temporary,state);temporary.replace(directory/'status.json')
    save()
    for variant,name in NAMES.items():
        command=[sys.executable,'-u','scripts/train_facility_native_roi.py','--name',name,
                 '--variant',variant,'--initial',f'runs/{REFERENCE}/best.pt',
                 '--auxiliary-manifest','data/facility-auxiliary-training/train.json',
                 '--study-protocol',PROTOCOL]
        stage={'variant':variant,'run':name,'status':'running','started_utc':datetime.now(timezone.utc).isoformat()}
        state['stages'].append(stage);save()
        with (directory/f'{variant}.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            stage['child_pid']=child.pid;save();result=child.wait()
        stage.update(exit_code=result,finished_utc=datetime.now(timezone.utc).isoformat())
        if result:
            state['status']='failed';stage['status']='failed';save();raise SystemExit(result)
        training=read(ROOT/'runs'/name/'TRAINING.json')
        if training.get('status')!='complete' or training.get('actual_epochs')!=p['requested_epochs']:
            state['status']='failed';stage['status']='failed';save();raise ValueError('Incomplete fixed training stage')
        stage.update(status='complete',actual_epochs=training['actual_epochs'],weights_sha256=training['weights_sha256']);save()
    state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()


if __name__=='__main__':main()
