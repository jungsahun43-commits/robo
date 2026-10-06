"""Run and record the complete focused retention study suites with byte binding."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.facility_batchnorm_study import SOURCE_FILES, MIN_TEST_COUNTS, require, write
from scripts.verify_facility_resolution import sha

TESTS = tuple(MIN_TEST_COUNTS)


def main():
    output = ROOT / 'runs/facility-batchnorm-test-results.json'
    require(not output.exists(), 'Preserve the actual previous focused test evidence')
    paths = set(TESTS) | set(SOURCE_FILES) | {'scripts/test_facility_batchnorm.py'}
    before = {p: sha(ROOT / p) for p in paths}
    modules = {p: unittest.defaultTestLoader.loadTestsFromName(Path(p).with_suffix('').as_posix().replace('/', '.')) for p in TESTS}
    counts = {p: suite.countTestCases() for p, suite in modules.items()}; expected = sum(counts.values())
    require(all(counts.get(p, 0) >= n for p, n in MIN_TEST_COUNTS.items()), 'Focused full-suite test collection is incomplete')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(modules.values()))
    after = {p: sha(ROOT / p) for p in paths}; require(before == after, 'Focused test/runtime bytes changed during testing')
    passed = result.wasSuccessful() and result.testsRun == expected and not result.skipped
    write(output, {'schema': 'facility_batchnorm_test_results_v1', 'executed_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed' if passed else 'failed', 'tests_run': result.testsRun, 'expected_tests_collected': expected,
        'tests_by_module': counts, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
        'test_source_sha256': {p: before[p] for p in TESTS}, 'source_sha256': {p: before[p] for p in SOURCE_FILES},
        'runner_sha256': sha(Path(__file__)), 'before_after_sources_equal': True, 'scope': 'Code invariants; not model accuracy'})
    require(passed, 'Actual focused retention tests failed or incomplete')
    print({'status': 'passed', 'tests_run': result.testsRun}, flush=True)


if __name__ == '__main__': main()
