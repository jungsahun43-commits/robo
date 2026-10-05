"""Post-completion reporting adapter for integral floating AP support counts.

The original frozen reporter, protocol, labels, probabilities, checkpoints and
metric/gate code remain unchanged. This module only adapts the loader's count
representation and appends an explicit reporting provenance record.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import report_facility_native_roi as frozen
from scripts.facility_native_roi_study import PROTOCOL, SOURCE_FILES, protected_hashes
from scripts.verify_facility_native_roi import TEST_SOURCES, hash_map, local_path, read, require, sha

ADAPTER_PATH = "scripts/report_facility_native_roi_results.py"
ADAPTER_TEST_PATH = "tests/test_facility_native_roi_report_adapter.py"
ADAPTER_TEST_RECORD = "reports/facility-native-roi-report-adapter-tests.json"
ORIGINAL_FAILURE_RECORD = "runs/facility-native-roi-original-report-failure.json"
MIN_ADAPTER_TESTS = 4


def normalize_ap_positive_support(entry):
    """Copy an already validated entry; change only exact integral float counts."""
    require(isinstance(entry, dict) and isinstance(entry.get("ranking_ap"), dict),
            "Loaded ranking AP aggregates are required")
    result = deepcopy(entry); converted = 0
    for source in result["ranking_ap"].values():
        require(isinstance(source, dict), "Ranking AP source aggregates must be objects")
        for point in source.values():
            require(isinstance(point, dict), "Ranking AP class aggregates must be objects")
            known, positive = point.get("known_photos"), point.get("positive_photos")
            require(type(known) is int and known >= 0, "Known AP support must remain a nonnegative integer")
            require(type(positive) in (int, float), "Boolean/non-numeric AP positive support is invalid")
            if type(positive) is float:
                require(math.isfinite(positive) and positive >= 0 and positive.is_integer(),
                        "Only finite nonnegative exact-integral floating support can be adapted")
            require(0 <= positive <= known, "AP positive support exceeds its original known support")
            if type(positive) is float:
                point["positive_photos"] = int(positive); converted += 1
    return result, converted


def adapted_load_run(loader, name):
    # Run every original file/score/label/provenance check before normalizing.
    training, entry = loader(name)
    normalized, count = normalize_ap_positive_support(entry)
    return training, normalized, count


@contextmanager
def patched_load_run(module, normalizations):
    """Patch the loader only for this invocation, restoring it on any exception."""
    original = module.load_run
    def load_run(name):
        training, entry, count = adapted_load_run(original, name)
        normalizations.append({"run": name, "normalized_support_fields": count})
        return training, entry
    module.load_run = load_run
    try:
        yield
    finally:
        module.load_run = original


def metric_payload_sha256(result):
    payload = {key: value for key, value in result.items() if key != "reporting_provenance"}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def add_reporting_provenance(result, normalizations, evidence):
    require(isinstance(result, dict) and "reporting_provenance" not in result,
            "Preserve an existing report's reporting provenance")
    require(isinstance(evidence, dict) and isinstance(normalizations, list)
            and all(isinstance(row, dict) and isinstance(row.get("run"), str)
                    and type(row.get("normalized_support_fields")) is int
                    and row["normalized_support_fields"] >= 0 for row in normalizations),
            "Actual normalization counts must be recorded")
    original_digest = metric_payload_sha256(result)
    updated = deepcopy(result)
    updated["reporting_provenance"] = {
        **deepcopy(evidence), "schema": "facility_native_roi_reporting_adapter_v1",
        "reporting_only_after_training": True,
        "normalized_field": "experiments[*].ranking_ap[*][*].positive_photos",
        "normalization": "finite nonnegative exact-integral float to int; bounded by original known_photos",
        "normalizations_by_run": deepcopy(normalizations),
        "normalized_support_fields": sum(row["normalized_support_fields"] for row in normalizations),
        "confusion_ap_threshold_gate_code_unchanged": True,
        "metrics_and_labels_unchanged": True, "training_recipe_changed": False,
        "original_metric_payload_sha256": original_digest,
        "metric_payload_sha256_after_reporting_metadata": metric_payload_sha256(updated),
        "original_frozen_reporter_source_modified": False,
    }
    require(metric_payload_sha256(updated) == original_digest, "Reporting metadata changed a measured result")
    return updated


def validate_adapter_tests(record, root):
    require(record.get("schema") == "facility_native_roi_report_adapter_tests_v1"
            and record.get("status") == "passed" and record.get("before_after_sources_equal") is True,
            "Actual passed reporting adapter tests are required")
    require(type(record.get("tests_run")) is int and record["tests_run"] >= MIN_ADAPTER_TESTS
            and type(record.get("expected_tests_collected")) is int
            and record["expected_tests_collected"] == record["tests_run"]
            and all(type(record.get(k)) is int and record[k] == 0 for k in ("failures", "errors", "skipped")),
            "The complete adapter suite must execute with no failures/errors/skips")
    hash_map(record.get("test_source_sha256"), (ADAPTER_TEST_PATH,))
    hash_map(record.get("frozen_test_source_sha256"), TEST_SOURCES)
    require(record.get("adapter_sha256") == sha(local_path(root, ADAPTER_PATH)),
            "Executed adapter source bytes changed")
    for relative, expected in {**record["test_source_sha256"], **record["frozen_test_source_sha256"]}.items():
        require(sha(local_path(root, relative)) == expected, "Executed adapter/original focused test bytes changed")
    return record


def capture_frozen_inputs(root, protocol):
    sources = {path: sha(local_path(root, path)) for path in SOURCE_FILES}
    require(sources == protocol["source_sha256"], "Original frozen runtime source bytes changed")
    result = {**sources, **protected_hashes(root)}
    for relative in TEST_SOURCES:
        result[relative] = sha(local_path(root, relative))
    for name in (protocol["reference"], protocol["control"], protocol["treatment"]):
        relative = f"runs/{name}/best.pt"; result[relative] = sha(local_path(root, relative))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-test-record", type=Path, default=ROOT / ADAPTER_TEST_RECORD)
    args = parser.parse_args(argv); test_path = args.adapter_test_record.resolve()
    require(test_path.is_relative_to((ROOT / "reports").resolve()) and test_path.is_file(),
            "Adapter test evidence must be an existing local aggregate report")
    tests = validate_adapter_tests(read(test_path), ROOT)
    protocol = read(ROOT / PROTOCOL); before = capture_frozen_inputs(ROOT, protocol)
    failure_path = local_path(ROOT, ORIGINAL_FAILURE_RECORD)
    failure_sha = sha(failure_path); test_sha = sha(test_path)
    normalizations = []
    with patched_load_run(frozen, normalizations):
        # All original count/rate/AP/gate/protocol/technical checks execute here.
        result = frozen.main()
    require(capture_frozen_inputs(ROOT, protocol) == before
            and sha(test_path) == test_sha and sha(failure_path) == failure_sha,
            "Reporting changed frozen source/test/protocol/data/weights or evidence bytes")
    evidence = {
        "reported_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_source": ADAPTER_PATH, "adapter_source_sha256": sha(ROOT / ADAPTER_PATH),
        "frozen_reporter_sha256": sha(ROOT / "scripts/report_facility_native_roi.py"),
        "adapter_test_record_sha256": test_sha, "adapter_tests_run": tests["tests_run"],
        "adapter_test_source_sha256": dict(tests["test_source_sha256"]),
        "original_reporting_failure_record_sha256": failure_sha,
        "frozen_protocol_sha256": sha(ROOT / PROTOCOL),
        "frozen_training_source_count": len(SOURCE_FILES),
        "frozen_sources_tests_protocol_data_weights_before_after_equal": True,
        "source_git_commit": result["technical_verification"]["source_git_commit"],
    }
    updated = add_reporting_provenance(result, normalizations, evidence)
    output_path = ROOT / frozen.OUTPUT_PATH; markdown_path = ROOT / frozen.MARKDOWN_PATH
    # The original main has just created these two reports. Add metadata only.
    require(read(output_path) == result, "Original measured output differs from returned frozen result")
    output_path.write_bytes((json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))
    note = (
        "\n## 학습 완료 후 보고 표현형 보정\n\n"
        "동결 reporter의 최초 보고 실행은 AP 양성 분모가 `211.0` 같은 정수값 float로 저장되어 "
        "새 정수 타입 검사에서 중단됐다. 해당 실패 기록의 SHA를 보존했다. 학습 소스30개·기존 테스트3개·프로토콜·정답·확률·가중치를 변경하지 않았다.\n\n"
        "별도 보고 adapter가 원래 loader의 검증을 먼저 실행하고, `ranking_ap.positive_photos`에서 "
        "유한·비음수·정확한 정수값 float만 원래 known 분모 이내에서 int로 바꿨다. "
        "미탐·오탐·AP·임계값·후보 기준 계산과 원래 frozen main은 그대로 실행했다. "
        "bool·NaN·소수·음수는 허용하지 않는다. 이 수정은 새 학습·정답 수정·성능 개선이 아니다.\n\n"
        f"추가 adapter 테스트 {tests['tests_run']}개가 실제 통과했다. 원래 학습 전30개 테스트와 별도이며 "
        "정확도 평가 사례 수가 아니다. JSON의 `reporting_provenance`에 adapter·테스트·실패 증거 SHA와 "
        "메타데이터 추가 전후 동일한 계측 payload SHA를 기록했다.\n\n"
        "실행 명령: `python scripts/report_facility_native_roi_results.py`\n\n"
        "[추가 adapter 테스트 실측](facility-native-roi-report-adapter-tests.json)\n"
    )
    with markdown_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(note)
    print(json.dumps({"reporting_adapter": "passed", "normalized_support_fields": updated["reporting_provenance"]["normalized_support_fields"],
                      "adapter_tests_run": tests["tests_run"], "measured_metrics_changed": False,
                      "training_sources_modified": False, "new_training_epochs": 0}))
    return updated


if __name__ == "__main__":
    main()
