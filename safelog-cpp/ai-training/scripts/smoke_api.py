from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="실행 중인 SafeLog 서버 API 확인")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    image = "data:image/jpeg;base64," + base64.b64encode(args.image.read_bytes()).decode()
    base = args.url.rstrip("/")

    def request(path: str, body: dict | None = None) -> dict:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(base + path, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)

    health = request("/health")
    assert health["modelExists"], "학습 모델 파일이 없습니다."
    analysis = request("/v1/analyze-hazard", {"image": image, "userMemo": "연결 확인", "promptVersion": "smoke-hazard-v1"})
    assert 1 <= analysis["riskLevel"] <= 5 and 0 <= analysis["confidence"] <= 1
    assert analysis["promptVersion"] == "smoke-hazard-v1"
    comparison = request("/v1/compare-action", {
        "beforeImage": image, "afterImage": image, "actionNote": "동일 사진 비교", "promptVersion": "smoke-compare-v1",
    })
    assert not comparison["likelyResolved"], "동일 사진을 조치 완료로 판정했습니다."
    summary = request("/v1/summarize", {
        "inspectionContext": "출입구 | 연결 시험 | 확인 대기 | Open\n", "promptVersion": "smoke-summary-v1",
    })
    assert summary["modelName"] == "safelog-summary-template-v1"
    print(json.dumps({"health": health, "analysis": analysis, "comparison": comparison, "summary": summary},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
