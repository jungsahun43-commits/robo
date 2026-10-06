"""One fixed six-epoch retention-strength candidate with reused control with durable logs and no overwrite."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_retention_strength_study import PROTOCOL,NAMES,REFERENCE,validate_protocol,write
from scripts.train_facility_target import read,sha


def main():
    p=validate_protocol(read(ROOT/PROTOCOL),ROOT)
    preflight=read(ROOT/'runs/facility-retention-strength-preflight.json')
    snapshot=read(ROOT/'runs/facility-retention-strength-source-before-training.json')
    if (preflight.get('status')!='passed' or preflight['protocol_sha256']!=sha(ROOT/PROTOCOL)
            or snapshot['protocol_sha256']!=sha(ROOT/PROTOCOL)
            or snapshot['preflight_sha256']!=sha(ROOT/'runs/facility-retention-strength-preflight.json')):
        raise ValueError('Before-training technical/source proof must match')
    if any((ROOT/'runs'/name).exists() for name in NAMES.values()):
        raise ValueError('Preserve existing training runs')
    directory=ROOT/'runs/facility-retention-strength-background-candidate';directory.mkdir(exist_ok=False)
    state={'schema':'facility_retention_strength_background_candidate_v1','pid':os.getpid(),
           'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
           'protocol_sha256':sha(ROOT/PROTOCOL),'source_sha256':p['source_sha256'],'stages':[]}
    def save():
        temporary=directory/'status.tmp';write(temporary,state);temporary.replace(directory/'status.json')
    save()
    for variant,name in NAMES.items():
        command=[sys.executable,'-u','scripts/train_facility_retention_strength.py','--name',name,
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
    # Run the fixed held-in source-VAL and technical checks after the actual
    # six epochs. Each command preserves its own original evidence contract.
    followups=[
        ('source-val',['scripts/evaluate_facility_target.py','select','--name',NAMES['strong'],'--grids','1']),
        ('small-region-audit',['scripts/analyze_facility_target.py','--name',NAMES['strong'],'--aggregate-only']),
        ('technical-verification',['scripts/verify_facility_retention_strength.py','--test-results','runs/facility-retention-strength-test-results.json']),
        ('result-report',['scripts/report_facility_retention_strength.py']),
        ('result-plot',['scripts/plot_facility_retention_strength.py']),
    ]
    for title,arguments in followups:
        stage={'stage':title,'status':'running','started_utc':datetime.now(timezone.utc).isoformat()}
        state['stages'].append(stage);save()
        with (directory/f'{title}.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u',*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            stage['child_pid']=child.pid;save();result=child.wait()
        stage.update(exit_code=result,finished_utc=datetime.now(timezone.utc).isoformat(),status='complete' if result==0 else 'failed')
        if result:
            state['status']='failed';save();raise SystemExit(result)
        save()
    state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()


if __name__=='__main__':main()
