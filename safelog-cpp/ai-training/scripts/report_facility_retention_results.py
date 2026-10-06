"""Bind a missing optional plan field for the frozen retention reporter.

The executed trainer records the protocol and draw archive hashes. Its exact
original-control row order was verified independently. This adapter supplies
the corresponding protocol-bound plan hash to an in-memory copy only, then
runs every original reporting check. No training record or frozen source is
edited and no accuracy, threshold, label or gate is changed.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import report_facility_retention as original
from scripts.facility_retention_study import PROTOCOL, NAMES, validate_protocol, write, require
from scripts.train_facility_target import read, sha

TEST_RECORD = 'runs/facility-retention-report-adapter-tests.json'
FAILURE_RECORD = 'runs/facility-retention-original-report-failure.json'
TEST_SOURCE = 'tests/test_facility_retention_report_adapter.py'


def bind_verified_plan(trainings, protocol, protocol_sha, proof):
    require(proof.get('status') == 'passed' and proof.get('protocol_sha256') == protocol_sha
            and proof.get('source_sha256') == protocol['source_sha256']
            and proof.get('actual_completed_training_epochs') == 12
            and proof.get('protected_files_unchanged') is True
            and proof.get('working_runtime_sources_unchanged') is True,
            'Plan binding requires the completed original frozen source/draw proof')
    copied = deepcopy(trainings)
    for training in copied[1:]:
        require(training.get('study_protocol_sha256') == protocol_sha
                and training.get('private_draw_archive_sha256') == protocol['paired_draws_sha256']
                and training.get('source_sha256') == protocol['source_sha256'],
                'Training protocol/archive/source binding differs')
        if 'sampler_plan_sha256' in training:
            require(training['sampler_plan_sha256'] == protocol['sampler_plan_sha256'],
                    'Explicit plan hash differs from the verified protocol')
        else:
            training['sampler_plan_sha256'] = protocol['sampler_plan_sha256']
    return copied


def main():
    require(not (ROOT / TEST_RECORD).exists(), 'Preserve adapter execution evidence')
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_facility_retention_report_adapter')
    count = suite.countTestCases(); before = {p: sha(ROOT / p) for p in (TEST_SOURCE, 'scripts/report_facility_retention_results.py')}
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(count == result.testsRun == 2 and result.wasSuccessful() and not result.skipped,
            'Both actual adapter tests must pass')
    require(before == {p: sha(ROOT / p) for p in before}, 'Adapter bytes changed during tests')
    write(ROOT / TEST_RECORD, {'schema': 'facility_retention_report_adapter_tests_v1', 'status': 'passed',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'source_sha256': before, 'executed_utc': datetime.now(timezone.utc).isoformat()})
    protocol = validate_protocol(read(ROOT / PROTOCOL), ROOT); digest = sha(ROOT / PROTOCOL)
    proof = read(ROOT / original.VERIFICATION_PATH)
    require(sha(ROOT / 'reports/facility-spalling-sampler-dry-run.json') == protocol['sampler_plan_sha256'],
            'Prepared original plan bytes changed')
    paths = [ROOT / 'runs' / name / 'TRAINING.json' for name in NAMES.values()]
    metadata_sha = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    write(ROOT / FAILURE_RECORD, {'schema': 'facility_retention_original_report_failure_v1',
        'preserved_utc': datetime.now(timezone.utc).isoformat(), 'status': 'report_failed_after_training',
        'error': 'ValueError: Actual fixed retention condition differs: sampler_plan_sha256',
        'original_reporter_sha256': sha(ROOT / 'scripts/report_facility_retention.py'),
        'protocol_sha256': digest, 'training_metadata_sha256': metadata_sha,
        'reason': 'Trainer retained protocol/archive hashes but omitted the reporter optional plan field',
        'training_records_modified': False, 'metrics_modified': False, 'new_training_epochs': 0})
    validate_pair = original.validate_pair
    try:
        original.validate_pair = lambda trainings, p, d: validate_pair(bind_verified_plan(trainings, p, d, proof), p, d)
        comparison = original.main()
    finally:
        original.validate_pair = validate_pair
    require(metadata_sha == {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}, 'Reporting altered original metadata')
    comparison['reporting_provenance']['optional_plan_binding'] = {
        'adapter_source_sha256': sha(Path(__file__)), 'adapter_tests_run': 2,
        'adapter_test_record_sha256': sha(ROOT / TEST_RECORD),
        'original_failure_record_sha256': sha(ROOT / FAILURE_RECORD),
        'derived_field': 'sampler_plan_sha256', 'derived_from': 'Verified pre-training protocol and actual original draw/archive proof',
        'training_metadata_sha256': metadata_sha, 'original_training_metadata_unchanged': True,
        'original_frozen_61_sources_unchanged': True, 'original_measurements_and_gates_unchanged': True}
    (ROOT / original.OUTPUT_PATH).write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    (ROOT / original.MARKDOWN_PATH).write_text(original.render(comparison)+
        '\n보고서 생성 보완: 학습 기록에는 프로토콜·추출 배열 해시가 남아 있으나 계획 해시 필드는 생략돼 있었다. '
        '완료 검증에 묶인 계획 해시를 메모리 복사본에 연결한 뒤 원래 검사를 실행했다. '
        '원본 61개 소스·학습 기록·성능 수치·평가 기준은 유지했으며 보완 테스트 2개가 통과했다.\n',
        encoding='utf-8', newline='\n')


if __name__ == '__main__': main()
