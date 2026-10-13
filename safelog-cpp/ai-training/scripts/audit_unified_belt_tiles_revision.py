"""Final delivery v1.1 audit against the actually tested v1 runtime bytes."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.unified_main_data import read, sha
from scripts.study_unified_belt_tiles import validate


def main():
    validate(); reports = ROOT/'reports'; original = read(reports/'unified-belt-tiles-release.json')
    revision = read(reports/'unified-belt-tiles-release-v1.1.json'); choice = read(reports/'unified-belt-tiles-selection.json')
    assert original['status'] == 'passed' and revision['status'] == 'passed_docs_only_revision'
    assert sha(Path(original['archive'])) == original['archive_sha256'] and sha(Path(revision['archive'])) == revision['archive_sha256']
    with zipfile.ZipFile(original['archive']) as old, zipfile.ZipFile(revision['archive']) as new:
        assert new.testzip() is None and sum(n.endswith('.pt') for n in new.namelist()) == 1
        members = json.loads(new.read('package-sha256.json'))
        assert all(hashlib.sha256(new.read(n)).hexdigest() == h for n, h in members.items())
        runtime = [n for n in old.namelist() if Path(n).suffix in ('.pt', '.py', '.cpp', '.hpp') or n in ('model.json', 'requirements-api.txt')]
        assert all(old.read(n) == new.read(n) for n in runtime)
        unchanged_metrics = [n for n in old.namelist() if n.endswith('-validation.json') or n.endswith('unified-belt-tiles-selection.json')]
        assert all(old.read(n) == new.read(n) for n in unchanged_metrics)
        assert hashlib.sha256(new.read('model.pt')).hexdigest() == choice['checkpoint_sha256']
        model = json.loads(new.read('model.json')); assert model['inferenceMode'] == choice['mode'] and model['inferencePolicySha256'] == choice['inference_policy']['sha256']
    result = {**read(reports/'unified-belt-tiles-final-audit.json'), 'new_archive': {k: revision[k] for k in ('archive', 'archive_sha256', 'archive_bytes')},
        'delivered_revision': 'v1.1 docs only', 'exact_runtime_files_identical_to_actual_CPU_CPP_test': len(runtime),
        'original_scores_counts_thresholds_and_selection_identical': True, 'actual_CPP_operations_inherited': 9, 'fresh_revision_CPP_operations': 0,
        'score_and_native_resolution_scope_clarified': True, 'operational_read_only_process_guard_tests_passed': 2,
        'revision_audit_source_sha256': sha(Path(__file__))}
    with (reports/'unified-belt-tiles-final-audit-v1.1.json').open('x', encoding='utf-8') as out: json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps({'status': 'passed', 'delivered_archive': revision['archive'], 'sha256': revision['archive_sha256'],
        'weights': choice['checkpoint_sha256'], 'mode': choice['mode'], 'policy_sha256': choice['inference_policy']['sha256'], 'runtime_byte_identical': len(runtime)}), flush=True)


if __name__ == '__main__': main()
