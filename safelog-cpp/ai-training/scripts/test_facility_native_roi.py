"""Record actual focused native ROI tests and preserve source bytes across execution."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_native_roi_study import SOURCE_FILES,MIN_TEST_COUNTS,write,require
from scripts.train_facility_target import sha

TESTS=('tests/test_facility_native_roi_data.py','tests/test_facility_native_roi_study.py',
       'tests/test_facility_native_roi_report.py')


def main():
    output=ROOT/'runs/facility-native-roi-test-results.json'
    require(not output.exists(),'Preserve the actual previous test execution')
    paths=TESTS+SOURCE_FILES+('scripts/test_facility_native_roi.py',)
    before={path:sha(ROOT/path) for path in paths}
    modules={path:unittest.defaultTestLoader.loadTestsFromName(Path(path).with_suffix('').as_posix().replace('/','.'))
             for path in TESTS}
    counts={path:suite.countTestCases() for path,suite in modules.items()}
    expected=sum(counts.values())
    require(all(counts.get(path,0)>=count for path,count in MIN_TEST_COUNTS.items()),'Focused module test collection is incomplete')
    suite=unittest.TestSuite(modules.values())
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    after={path:sha(ROOT/path) for path in paths}
    require(before==after,'Executed test/source files changed during testing')
    passed=result.wasSuccessful() and result.testsRun==expected and not result.skipped
    value={'schema':'facility_native_roi_test_results_v1','executed_utc':datetime.now(timezone.utc).isoformat(),
           'status':'passed' if passed else 'failed','tests_run':result.testsRun,
           'expected_tests_collected':expected,'tests_by_module':counts,
           'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
           'test_source_sha256':{path:before[path] for path in TESTS},
           'source_sha256':{path:before[path] for path in SOURCE_FILES},
           'runner_sha256':sha(Path(__file__)),'before_after_sources_equal':True,
           'scope':'Code invariants; not model accuracy'}
    write(output,value)
    require(passed,'Focused native ROI tests failed or incomplete')
    print({'status':'passed','tests_run':result.testsRun})


if __name__=='__main__':main()
