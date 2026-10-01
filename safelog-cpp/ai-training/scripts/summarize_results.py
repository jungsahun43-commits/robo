from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    reports = []
    for path in sorted((ROOT / "runs").glob("evaluation-*.json")):
        if path.name == "evaluation-summary.json":
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append({"run": path.stem.removeprefix("evaluation-"), **report})

    if not reports:
        raise SystemExit("평가 결과가 없습니다. evaluate.py를 먼저 실행하세요.")

    output_json = ROOT / "runs" / "evaluation-summary.json"
    output_json.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SafeLog AI 테스트 결과",
        "",
        "| 모델 | 평가 분할 | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            f"| {report['run']} | {report.get('split', 'test')} | {report['precision']:.3f} | {report['recall']:.3f} | "
            f"{report['map50']:.3f} | {report['map50_95']:.3f} |"
        )
    lines += [
        "",
        "> test는 학습과 모델 선택에 사용하지 않은 분할입니다. SH17 val은 학습 중 모델 선택에도 사용했으므로 test 결과와 구분합니다.",
    ]
    output_markdown = ROOT / "runs" / "evaluation-summary.md"
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_markdown.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
