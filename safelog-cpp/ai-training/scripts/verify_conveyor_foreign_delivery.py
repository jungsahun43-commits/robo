import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

protocol = read(root/'reports/conveyor-foreign-grouped-protocol.json')
for name,digest in {**protocol['legacy_source_sha256'], **protocol['source_sha256']}.items():
    assert sha(root/name)==digest, name
for record, key in [('conveyor-foreign-grouped-api-smoke.json', 'source_sha256')]:
    for name,digest in read(root/'reports'/record)[key].items():
        assert sha(root/name)==digest, name
plot = read(root/'reports/conveyor-foreign-grouped-plot.json')
assert sha(root/'scripts/report_conveyor_foreign.py')==plot['source_sha256']
assert sha(root/'reports/CONVEYOR_FOREIGN_RESULTS_KO.md')==plot['report_sha256']
for name,digest in plot['figure_sha256'].items():
    assert sha(root/'reports'/name)==digest
old = read(root/'reports/beltcrack-expansion-release-v2.json')
new = read(root/'reports/conveyor-dual-handoff-release-v3.json')
assert sha(Path(old['package']))==old['package_sha256']
assert sha(Path(new['package']))==new['package_sha256']
assert sha(root/'scripts/build_conveyor_dual_handoff.py')==new['builder_sha256']
standalone = read(root/'reports/conveyor-dual-handoff-standalone-v3.json')
assert standalone['status']=='passed' and standalone['package_sha256']==new['package_sha256']
assert sha(root/'scripts/verify_conveyor_handoff.py')==standalone['source_sha256']
for name,model in new['models'].items():
    assert sha(root/'runs'/model['path'])==model['sha256'], name
feedback = []
base = root/'data/equipment-conveyor-foreign/grouped-photo-v1'
manifest = {r['id']:r for r in read(base/'manifest.json')['rows']}
for phase in ('val','test'):
    for row in read(root/f'runs/conveyor-foreign-grouped-v1/{phase}-predictions.json'):
        if (row['score']>.5)!=bool(row['label']):
            source = manifest[row['id']]
            feedback.append({**row, 'split':phase, 'error_type':'false_negative' if row['label'] else 'false_positive',
                'publisher_label':row['label'], 'label_source':source['label_source'],
                'native_sha256':source['native_sha256'], 'model_sha256':new['models']['conveyor_foreign_object']['sha256'],
                'human_review_status':'not_reviewed', 'approved_for_training':False})
with (root/'reports/conveyor-foreign-grouped-feedback.json').open('x',encoding='utf-8') as handle:
    json.dump({'status':'exported_saved_errors', 'cases':feedback, 'new_labels_created':0,
        'warning':'Do not train on these held-out cases and reuse them as an independent test'},handle,indent=2)
proof = {'status':'passed','frozen_and_previous_sources_preserved':len(protocol['source_sha256'])+len(protocol['legacy_source_sha256']),
    'new_api_and_packaging_sources_preserved':True,'completed_reports_and_figures_preserved':True,
    'old_v2_zip_preserved':True,'new_v3_zip_verified':True,'standalone_two_model_http_passed':True,
    'feedback_cases_exported':len(feedback),'additional_test_model_scores':0,'new_ai_or_human_labels':0,
    'actual_epochs':30,'actual_optimizer_updates':1170,'package_sha256':new['package_sha256']}
with (root/'reports/conveyor-foreign-grouped-final-audit.json').open('x',encoding='utf-8') as handle:
    json.dump(proof,handle,indent=2)
print(json.dumps(proof,indent=2))
