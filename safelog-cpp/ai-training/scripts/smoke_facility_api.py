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
    parser.add_argument("--optimized", action="store_true")
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
    dacl_id = "facility-dacl-optimized" if args.optimized else "facility-dacl"
    if args.optimized:
        profile = json.loads((ROOT / "reports/facility-inference-profile.json").read_text())
        assert health["facilityProfile"] == profile["version"]
        assert health["facilityThresholds"][dacl_id] == profile["models"][dacl_id]["thresholds"]
        assert health["inferenceSizes"][dacl_id] == profile["models"][dacl_id]["imgsz"]
    else:
        assert health["inferenceSizes"][dacl_id] == 960
    fixed = json.loads((ROOT / "runs/fixed-facility-dacl-test.json").read_text())
    # First held-out image with a matched defect is a FUNCTIONAL fixture, not an accuracy sample.
    fixture = next(item["image"] for item in fixed["per_image"] if any(c["tp"] for c in item["counts"].values()))
    image_path = ROOT / "data/dacl10k-yolo/images/test" / fixture
    image = "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode()
    code, analysis, analysis_seconds = request("/v1/analyze-hazard", {"image": image, "userMemo": "시설 표면 점검", "promptVersion": "facility-smoke-v1"})
    assert code == 200 and analysis["requiresHumanReview"] is True
    assert 1 <= analysis["riskLevel"] <= 5
    assert any(d["model"] == dacl_id for d in analysis["detections"])
    code, comparison, comparison_seconds = request("/v1/compare-action", {"beforeImage": image, "afterImage": image,
                                 "actionNote": "같은 사진 기능 시험", "promptVersion": "facility-smoke-v1"})
    assert code == 200 and comparison["likelyResolved"] is False and comparison["requiresHumanReview"] is True
    code, summary, _ = request("/v1/summarize", {"inspectionContext": "시설 표면 | 점검 제안 | 사람 확인 필요", "promptVersion": "facility-smoke-v1"})
    assert code == 200 and summary["modelName"] == "safelog-summary-template-v1"
    photo_proof = None
    if args.optimized and profile.get("photo_classifier"):
        assert health["photoClassifier"] == profile["photo_classifier"]
        import sys
        sys.path.insert(0, str(ROOT))
        import numpy as np
        from scripts.optimize_facility_presence import baseline_presence
        dacl_entry = profile["models"][dacl_id]
        cache = json.loads((ROOT / f"runs/proposals-{dacl_id}-test-{dacl_entry['imgsz']}.json").read_text())
        classifier = json.loads((ROOT / "runs/presence-scores-test.json").read_text())
        for i, label in enumerate(classifier["classes"]):
            cutoff = profile["photo_classifier"]["thresholds"][label]
            base = baseline_presence(cache, i, dacl_entry["thresholds"][label], classifier["images"])
            scores = np.array(classifier["probabilities"])[:, i]
            positives = np.array(classifier["targets"])[:, i] > 0
            indices = np.where(~base & positives & (scores >= cutoff + .01))[0]
            if cutoff < 1 and len(indices):
                fixture2 = classifier["images"][int(indices[0])]
                data = (ROOT / "data/dacl10k-yolo/images/test" / fixture2).read_bytes()
                code, added, seconds = request("/v1/analyze-hazard", {"image": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
                    "promptVersion": "presence-smoke-v1"})
                assert code == 200 and added["requiresHumanReview"]
                opinions = [d for d in added["detections"] if d["model"] == "facility-presence"]
                assert opinions and all(d["box"] is None and d["evidence_scope"] == "photo_presence" for d in opinions)
                photo_proof = {"fixture": fixture2, "fixture_selection": "first additional true-positive photo with score margin >=0.01; functional check only",
                               "response": added, "seconds": seconds}
                break
        assert photo_proof is not None, "No positive classifier fixture; check whether classifier improves validation/test"
    # Do not publish absolute workstation paths in this proof.
    health["models"] = [Path(value).name for value in health["models"]]
    result = {"status": "passed", "device": "CPU", "fixture": fixture,
              "fixture_selection": "first held-out test image with TP>0; functional API check only, not accuracy",
              "health": health, "analysis": analysis, "comparison": comparison, "summary": summary,
              "timing_seconds": {"analysis_first_request_including_load": analysis_seconds, "comparison_two_images": comparison_seconds},
              "photo_classifier_functional_proof": photo_proof,
              "scope": "PC HTTP endpoints; Android APK/physical phone not tested"}
    filename = "facility-optimized-api-smoke-result.json" if args.optimized else "facility-api-smoke-result.json"
    (ROOT / "reports" / filename).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
