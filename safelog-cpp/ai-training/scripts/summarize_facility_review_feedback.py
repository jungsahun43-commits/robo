"""Import separate TRAIN review opinions and write a new local count report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.review_feedback import aggregate_feedback, read_json


def render_summary(summary):
    lines = ["# TRAIN 검수 의견 집계", "", summary["policy"], "",
        "이 집계는 학습 사진의 검수 의견이며 현장·앱 정확도 평가가 아닙니다.", "",
        "## 현재 집계", "",
        f'- 원본 검수 사례: {summary["package_cases"]}건',
        f'- 가져온 의견 파일: {summary["submitted_unique_feedback_documents"]}개',
        f'- 동일 내용 중복 제외: {summary["duplicate_files_ignored"]}개',
        f'- 검수 의견이 있는 고유 사례: {summary["reviewed_unique_cases"]}건',
        f'- 검수 의견 수: {summary["reviewed_task_opinions"]}개 (사람 × 사례 × 균열/박락)',
        f'- 판단 불확실 의견: {summary["uncertain_task_opinions"]}개',
        f'- 상반된 있음/없음 의견이 있는 사례·항목: {summary["cross_reviewer_conflicting_case_tasks"]}개',
        f'- 미검수·불확실·상반 의견으로 남아 있는 사례·항목: {summary["unresolved_unique_case_tasks"]}개',
        "- 전문가 확인 정답·정답 변경·학습 반영: 모두 0건", "",
        "## 검수자 구분", "",
        "| 자기 신고 역할 | 검수자 수 | 있음 | 없음 | 불확실 | 검수 전 | 원본 정답과 다른 의견 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for role, label in (("team_observer", "팀원 관찰"), ("domain_expert", "분야 전문가 자기 신고")):
        counts = summary["judgements_by_role"][role]
        lines.append(f'| {label} | {summary["reviewer_count_by_role"][role]} | {counts["present"]} | {counts["absent"]} | {counts["uncertain"]} | {counts["unreviewed"]} | {summary["disagreement_with_original_by_role"][role]} |')
    lines += ["", "검수자 수는 자기 신고한 ID별 수입니다. 독립된 사람 수나 전문가 신원·자격을 인증하지 않습니다.",
        "전문가 역할·전문성 입력은 본인의 신고입니다. 신원·자격 검증이나 정답 확정으로 계산하지 않습니다.",
        "원본 정답과 다른 의견은 정답 오류 확정을 뜻하지 않습니다. 상반된 의견도 다수결로 적용하지 않습니다.", "",
        "## 이유·자료·항목별 의견", "",
        "| 역할 | 이유 | 자료 | 항목 | 의견 수 |", "|---|---|---|---|---:|"]
    for row in summary["counts_by_role_reason_source_task"]:
        lines.append(f'| {row["role"]} | {row["reason"]} | {row["source"]} | {row["task"]} | {row["count"]} |')
    if not summary["counts_by_role_reason_source_task"]:
        lines.append("| 입력된 항목별 의견 없음 | | | | 0 |")
    old = summary["legacy"]
    lines += ["", "## 이전 형식의 관찰 메모", "",
        f'- 이전 의견 파일 {old["documents"]}개, 사례 단위 관찰 {old["case_level_reviewed_observations"]}건, 검수 전 {old["unreviewed_rows"]}건.',
        "- 이전 형식에는 검수자 신원과 균열·박락별 판단이 없어 항목별 판정·전문가 의견에 합치지 않습니다.", "",
        "## 파일 보관", "",
        "- `feedback-summary.json`: 사례 ID·검수자 이름·메모·근거·사진 경로 없는 집계.",
        "- `feedback-details.json`: 정규화한 의견, 상반된 판단, 미해결 사례 ID가 있는 로컬 상세 자료. 외부 배포 전 개인정보·자료 조건을 확인하세요.",
        "- 기존 사진·주석·모델·정답 파일을 수정하지 않았습니다.", "",
        "패키지 내용 해시는 입력 내용의 일치를 확인합니다. 출판자 출처 인증이나 모델 실파일 검증은 하지 않습니다.", "",
        f'원본 패키지 내용 SHA256: `{summary["source_package_sha256"]}`',
        f'모델 SHA256: `{summary["weights_sha256"]}`', ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, nargs="*", default=[])
    parser.add_argument("--allow-legacy", action="store_true",
                        help="Explicitly import old case-level proposals separately from v2 task opinions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    runs = (ROOT / "runs").resolve()
    if output == runs or not output.is_relative_to(runs):
        raise ValueError("Feedback output must be a new directory inside ignored ai-training/runs")
    if output.exists():
        raise ValueError("Preserve existing review evidence; choose a new output directory")
    package = read_json(args.package)
    documents = [read_json(path) for path in args.feedback]
    # All documents and all conflicts are validated before creating any output.
    result = aggregate_feedback(package, documents, allow_legacy=args.allow_legacy)
    report = render_summary(result["summary"])
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("feedback-summary.json", result["summary"]),
                        ("feedback-details.json", result["details"])):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                            allow_nan=False) + "\n", encoding="utf-8")
    (output / "README_KO.md").write_text(report, encoding="utf-8")
    # Console output is aggregate-only; no source paths or reviewer identifiers.
    print(json.dumps({"schema": result["summary"]["schema"],
        "package_cases": result["summary"]["package_cases"],
        "reviewed_unique_cases": result["summary"]["reviewed_unique_cases"],
        "reviewed_task_opinions": result["summary"]["reviewed_task_opinions"],
        "label_changes": 0, "training_applied": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
