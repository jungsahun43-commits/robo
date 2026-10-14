"""Invoke the isolated ZIP runtime on every known LOCO TEST photo."""
import base64
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0,str(Path.cwd()))
import safelog_ai.loco_context_server as runtime


def main():
    root=Path(sys.argv[1]); report=Path(sys.argv[2])
    assert Path(runtime.__file__).resolve()==Path.cwd()/'safelog_ai/loco_context_server.py'
    torch.set_num_threads(4)
    spec=json.loads(Path('model.json').read_text(encoding='utf-8'))
    engine=runtime.ContextEngine('model.pt',spec['sha256'],'evaluation.json')
    selected=json.loads((root/'reports/industrial-loco-context-v13-selection.json').read_text(encoding='utf-8'))['selected']
    gpu=json.loads((root/'reports/industrial-loco-context-v13-validation.json').read_text(encoding='utf-8'))['candidates'][selected]
    manifest=json.loads((root/'data/industrial-loco-context-v13/manifest.json').read_text(encoding='utf-8'))
    rows=[r for r in manifest['rows'] if r['phase']=='test']; expected={r['id']:r for r in gpu['observations']}
    mismatch=[]; scores=[]; biggest=0; tp=fp=tn=fn=0; start=time.monotonic()
    for i,row in enumerate(rows):
        encoded=base64.b64encode((root/row['image']).read_bytes()).decode()
        result=engine.analyze('loco_splicing_connectors',encoded)
        score=result['classResults'][0]['score'];decision=result['candidateDetected'];prior=expected[row['id']]
        assert result['modelSha256']==spec['sha256'] and result['scoreIsCalibratedProbability'] is False
        assert result['originalLocalizationChanged'] is False and result['semanticClampWireRelationsVerified'] is False
        biggest=max(biggest,abs(score-prior['score']))
        if decision!=prior['candidate_detected']:mismatch.append(row['id'])
        if row['label']:
            if decision:tp+=1
            else:fn+=1
        elif decision:fp+=1
        else:tn+=1
        scores.append({'id':row['id'],'score':score,'candidate_detected':decision})
        if i%50==0:print({'isolated_CPU_API_photos':i+1,'total':len(rows)},flush=True)
    record={'status':'passed' if not mismatch else 'device_mismatch','isolated_ZIP_runtime_imported':True,'model_sha256':spec['sha256'],
        'photos':len(rows),'all_CPU_GPU_photo_decisions_equal':not mismatch,'decision_mismatches':mismatch,'max_absolute_score_difference':biggest,
        'CPU_photo_rates':{'tp':tp,'fn':fn,'fp':fp,'tn':tn,'positive_photos':tp+fn,'negative_photos':fp+tn,'fnr':fn/(tp+fn),'fpr':fp/(fp+tn)},
        'GPU_photo_rates':gpu['photo_rates'],'actual_full_packaged_API_pipeline_invoked':True,'model_and_thresholds_unchanged':True,
        'same_known_TEST_reused_for_device_verification':True,'independent_field_validated':False,'total_seconds_not_latency_benchmark':time.monotonic()-start}
    with report.open('x',encoding='utf-8') as out:json.dump(record,out,indent=2)
    with Path('CPU-known-test-observations.json').open('x',encoding='utf-8') as out:json.dump(scores,out)
    print(record,flush=True)


if __name__=='__main__':main()
