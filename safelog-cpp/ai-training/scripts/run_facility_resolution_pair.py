"""Run the fixed resolution pair sequentially, with durable progress files.

Launch the helper as a hidden background process on Windows. This is a one-off
training job, not a scheduler. Failed runs are preserved and never overwritten.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.facility_resolution_study import validate_protocol


def main():
    directory = ROOT/'runs/facility-resolution-background-pair'
    directory.mkdir(exist_ok=False)
    protocol_path = ROOT/'reports/facility-resolution-study-protocol.json'
    protocol = validate_protocol(json.loads(protocol_path.read_text()), ROOT)
    state = {'schema':'facility_resolution_background_pair_v1', 'pid':os.getpid(),
             'started_utc':datetime.now(timezone.utc).isoformat(), 'status':'running',
             'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
             'source_sha256':protocol['source_sha256'], 'stages':[]}
    status_path = directory/'status.json'
    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        temporary.replace(status_path)
    save()
    for variant, name in (('control',protocol['control']),('highres',protocol['treatment'])):
        command = [sys.executable,'-u','scripts/train_facility_resolution.py',
                   '--name',name,'--variant',variant,
                   '--initial',f'runs/{protocol["reference"]}/best.pt',
                   '--auxiliary-manifest','data/facility-auxiliary-training/train.json',
                   '--study-protocol','reports/facility-resolution-study-protocol.json']
        stage = {'variant':variant,'run':name,'status':'running',
                 'started_utc':datetime.now(timezone.utc).isoformat()}
        state['stages'].append(stage); save()
        with (directory/f'{variant}.log').open('x',encoding='utf-8') as log:
            child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            stage['child_pid'] = child.pid; save()
            result = child.wait()
        stage.update(exit_code=result,finished_utc=datetime.now(timezone.utc).isoformat())
        if result:
            stage['status'] = 'failed'; state['status'] = 'failed'; save()
            raise SystemExit(result)
        training = json.loads((ROOT/'runs'/name/'TRAINING.json').read_text())
        if training.get('status') != 'complete' or training.get('actual_epochs') != 6:
            stage['status'] = 'failed'; state['status'] = 'failed'; save()
            raise ValueError('The declared six-epoch stage did not complete')
        stage.update(status='complete',actual_epochs=training['actual_epochs'],
                     weights_sha256=training['weights_sha256']); save()
    state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == '__main__': main()
