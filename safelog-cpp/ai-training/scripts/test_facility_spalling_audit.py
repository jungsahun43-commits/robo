"""Record a complete focused source-bound execution of spalling audit tests."""
import importlib
import json
from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_facility_spalling_train import SOURCE_PATHS, snapshot, require, write

MODULES = {"tests.test_spalling_geometry": 8, "tests.test_facility_spalling_train_audit": 6}
OUTPUT = ROOT / "reports/facility-spalling-train-audit-tests.json"


def main():
    require(not OUTPUT.exists(), "Preserve existing executed test evidence")
    test_sources = {name.replace(".", "/") + ".py": count for name, count in MODULES.items()}
    sources = list(SOURCE_PATHS) + ["scripts/test_facility_spalling_audit.py"] + list(test_sources)
    before = {p: snapshot(ROOT / p) for p in sources}
    suite = unittest.TestSuite(); collection = {}
    for name, expected in MODULES.items():
        module = importlib.import_module(name)
        tests = unittest.defaultTestLoader.loadTestsFromModule(module)
        collection[name] = tests.countTestCases()
        require(collection[name] == expected, "Complete focused module collection required")
        suite.addTests(tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {p: snapshot(ROOT / p) for p in sources}
    passed = result.wasSuccessful() and not result.skipped and before == after and result.testsRun == sum(collection.values())
    record = {"schema": "facility_spalling_train_audit_tests_v1", "status": "passed" if passed else "failed",
        "executed_utc": datetime.now(timezone.utc).isoformat(), "tests_run": result.testsRun,
        "expected_tests_collected": sum(collection.values()), "tests_by_module": collection,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "source_sha256": {p: x["sha256"] for p, x in before.items() if p not in test_sources},
        "test_source_sha256": {p: before[p]["sha256"] for p in test_sources},
        "before_after_sources_equal": before == after, "scope": "Synthetic replay/provenance tests; no accuracy measurement or training"}
    write(OUTPUT, record)
    print(json.dumps({"status": record["status"], "tests_run": result.testsRun}))
    require(passed, "Focused audit tests failed")


if __name__ == "__main__":
    main()
