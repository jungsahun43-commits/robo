from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
paths = sorted(root.glob('tests/test_facility_resolution*.py'))
verifier = root/'scripts/verify_facility_resolution.py'
sources = paths+[verifier]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
before = {path.relative_to(root).as_posix():digest(path) for path in sources}
suite = unittest.defaultTestLoader.discover(str(root/'tests'), pattern='test_facility_resolution*.py')
start = time.perf_counter()
result = unittest.TextTestRunner(verbosity=2).run(suite)
after = {path.relative_to(root).as_posix():digest(path) for path in sources}
passed = result.wasSuccessful() and not result.skipped and before == after
record = {'schema':'facility_resolution_test_results_v1',
          'created_utc':datetime.now(timezone.utc).isoformat(),
          'status':'passed' if passed else 'failed', 'tests_run':result.testsRun,
          'failures':len(result.failures), 'errors':len(result.errors), 'skipped':len(result.skipped),
          'elapsed_seconds':time.perf_counter()-start,
          'test_source_sha256':{p.relative_to(root).as_posix():before[p.relative_to(root).as_posix()] for p in paths},
          'source_sha256':{'scripts/verify_facility_resolution.py':before['scripts/verify_facility_resolution.py']},
          'before_after_sources_equal':before == after,
          'scope':'Focused code behavior tests; synthetic fixtures are not measured accuracy'}
output = root/'runs/facility-resolution-test-results.json'
with output.open('x',encoding='utf-8') as stream: json.dump(record,stream,indent=2)
print(json.dumps({'status':record['status'], 'tests_run':result.testsRun}))
if not passed: raise SystemExit(1)
