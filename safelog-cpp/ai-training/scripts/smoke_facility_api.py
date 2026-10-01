"""Exercise the real HTTP API with trained facility weights (not accuracy scoring)."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    args = parser.parse_args()
    def request(endpoint, body=None):
        encoded = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.url + endpoint, data=encoded,
                                     headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.load(response)
            return response.status, data, round(time.perf_counter() - started, 3)
    health_code, health, _ = request("/health")
    assert health_code == 200 and health["status"] == "ok"
    assert "facility-dacl" in health["model"] and "facility-corrosion" in health["model"]
    assert health["inferenceSizes"]["facility-dacl"] == 960
    fixed = json.loads((ROOT / "runs/fixed-facility-dacl-test.json").read_text())
    # First held-out image with a matched defect is a FUNCTIONAL fixture, not an accuracy sample.
    fixture = next(item["image"] for item in fixed["per_image"] if any(c["tp"] for c in item["counts"].values()))
    image_path = ROOT / "data/dacl10k-yolo/images/test" / fixture
    image = "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode()
    code, analysis, analysis_seconds = request("/v1/analyze-hazard", {"image": image, "userMemo": "시설 표면 점검", "promptVersion": "facility-smoke-v1"})
    assert code == 200 and analysis["requiresHumanReview"] is True
    assert 1 <= analysis["riskLevel"] <= 5
    assert any(d["model"] == "facility-dacl" for d in analysis["detections"])
    code, comparison, comparison_seconds = request("/v1/compare-action", {"beforeImage": image, "afterImage": image,
                                 "actionNote": "같은 사진 기능 시험", "promptVersion": "facility-smoke-v1"})
    assert code == 200 and comparison["likelyResolved"] is False and comparison["requiresHumanReview"] is True
    code, summary, _ = request("/v1/summarize", {"inspectionContext": "시설 표면 | 점검 제안 | 사람 확인 필요", "promptVersion": "facility-smoke-v1"})
    assert code == 200 and summary["modelName"] == "safelog-summary-template-v1"
    # Do not publish absolute workstation paths in this proof.
    health["models"] = [Path(value).name for value in health["models"]]
    result = {"status": "passed", "device": "CPU", "fixture": fixture,
              "fixture_selection": "first held-out test image with TP>0; functional API check only, not accuracy",
              "health": health, "analysis": analysis, "comparison": comparison, "summary": summary,
              "timing_seconds": {"analysis_first_request_including_load": analysis_seconds, "comparison_two_images": comparison_seconds},
              "scope": "PC HTTP endpoints; Android APK/physical phone not tested"}
    (ROOT / "reports/facility-api-smoke-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
