"""Count unconfirmed AI notes and original input dimensions without changing labels."""
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.review_feedback import read_json, validate_package
from safelog_ai.review_observations import summarize_ai_observations, validate_ai_observations
from scripts.build_facility_review_workbench import REASON_LABELS, safe_reference


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_dimensions(package, directory):
    from PIL import Image
    rows, groups = [], defaultdict(list)
    for case in package["cases"]:
        path = safe_reference(case["image_url"], directory)
        if sha(path) != case["image_sha256"]:
            raise ValueError("Input photo bytes changed before the metadata audit")
        with Image.open(path) as image:
            width, height = image.size
        if min(width, height) <= 0:
            raise ValueError("Invalid source image dimensions")
        row = {"case_id": case["case_id"], "domain": case["domain"],
               "image_sha256": case["image_sha256"], "width": width, "height": height,
               "min_side_below_64": min(width, height) < 64,
               "aspect_above_8": max(width, height) / min(width, height) > 8}
        rows.append(row); groups[case["domain"]].append(row)
    summary = {}
    for domain, items in sorted(groups.items()):
        sides = [min(i["width"], i["height"]) for i in items]
        summary[domain] = {"cases": len(items), "min_input_side": min(sides),
                           "median_min_input_side": statistics.median(sides),
                           "min_side_below_64": sum(i["min_side_below_64"] for i in items),
                           "aspect_above_8": sum(i["aspect_above_8"] for i in items)}
    return summary, rows


def prior_photo_context(prior, observations, file_sha256):
    """Count shared photo identities, without copying historical free text."""
    if (not isinstance(prior, dict)
            or prior.get("schema") != "facility_train_AI_visual_observations_v1"
            or prior.get("split") != "train"
            or not isinstance(prior.get("cases"), list) or not prior["cases"]):
        raise ValueError("Expected the historical TRAIN AI observation record")
    prior_hashes = []
    for case in prior["cases"]:
        value = case.get("image_sha256") if isinstance(case, dict) else None
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("Invalid prior photo identity")
        prior_hashes.append(value)
    if len(set(prior_hashes)) != len(prior_hashes):
        raise ValueError("Prior record has duplicate photo identities")
    identities = set(prior_hashes)
    overlap = sum(row["image_sha256"] in identities for row in observations["cases"])
    return {"prior_observed_cases": len(prior_hashes), "same_photo_sha256_cases": overlap,
            "not_in_prior_observation_cases": len(observations["cases"]) - overlap,
            "prior_record_file_sha256": file_sha256, "prior_record_changed": False}


def render_report(result):
    summary = result["ai_observations"]
    package_count = sum(row["cases"] for row in result["input_dimensions"].values())
    source_counts = " / ".join(f"{source} {count}건" for source, count in summary["by_source"].items())
    lines = ["# 최근 ROI 모델 TRAIN 사례의 AI 관찰과 입력 크기", "",
             "**AI의 미확정 관찰·가설이다. 전문가 판정, 원본 정답 수정, 새 학습과 성능 개선은 이 작업에서 0건이다.**", "",
             f'검수 묶음 {package_count}건 중 선택한 {summary["observed_cases"]}건의 사진을 확인했다고 기록하고, 항목별 AI 의견 {summary["observed_task_opinions"]}개를 따로 보관했다.',
             f'{source_counts}. 선택한 TRAIN 표본이므로 대표 표본이나 전체 오류율이 아니다.',
             "한 사진에 최대 두 항목의 의견을 기록한다. 의견 수와 사진 수가 다르다. 사진 확인 선언과 SHA 연결은 전문가 자격이나 관찰의 정확성을 인증하지 않는다.", "",
             "## AI 항목 의견의 사유", "", "| 관찰 사유 | 항목 의견 수 |", "|---|---:|"]
    for reason, count in summary["by_reason_task_opinions"].items():
        lines.append(f'| {REASON_LABELS[reason]} | {count} |')
    prior = result.get("prior_ai_observation_context")
    if prior:
        lines += ["", f'기존 AI 기록 {prior["prior_observed_cases"]}건과 같은 사진은 {prior["same_photo_sha256_cases"]}건이다. 나머지 {prior["not_in_prior_observation_cases"]}건은 그 기록에 미포함이며, 같은 사진의 재관찰을 새로운 전문가 판정으로 세지 않는다.']
    lines += ["", "원인을 입증한 건수가 아니라 AI가 기록한 항목별 참고 의견이다. 같은 사진의 두 항목과 사유가 다를 수 있다.", "",
              f"## 원본 {package_count}건의 입력 크기 감사", "",
              f'선택 {summary["observed_cases"]}건의 시각 관찰과 별도로 {package_count}개 입력 파일의 크기를 읽고 파일 SHA를 확인했다. 나머지 {result["metadata_only_case_count"]}건을 새로 육안 판정한 것은 아니다.', "",
              "| 자료 | 사진 수 | 최소 짧은 변 | 짧은 변 중앙값 | 짧은 변 64px 미만 | 비율 8:1 초과 |",
              "|---|---:|---:|---:|---:|---:|"]
    for source, row in result["input_dimensions"].items():
        lines.append(f'| {source} | {row["cases"]} | {row["min_input_side"]} | {row["median_min_input_side"]} | {row["min_side_below_64"]} | {row["aspect_above_8"]} |')
    lines += ["", "64px와 8:1은 입력 자료를 살펴보기 위한 진단 기준이다. 학습 제외 기준·오류율 기준·업계 표준이 아니다. 크기·비율만으로 오류 원인을 확정하지 않는다.", "",
              "## 확인할 원인 후보와 다음 개발", "",
              "1. 작은·낮은 대비 균열: 원본 주석 범위와 사진 정보를 함께 확인한다. 주석 면적은 모델이 본 위치나 실패 원인을 입증하지 않는다.",
              "2. 패치 해상도·범위: 더 넓은 원본을 확보할 수 있는지 확인한다. 작은 사진을 높은 해상도로 늘렸다는 이유로 세부 정보가 복구됐다고 간주하지 않는다.",
              "3. 손상 정의 경계: 이음·표면 흔적과 균열, 거친 결손·구멍·철근 노출과 박락의 구분 질문을 전문가에게 전달한다. AI가 다르게 보았다는 이유로 native 태그를 바꾸지 않는다.",
              "4. 모델에 없는 정상·안전 라벨을 추가하지 않고, 원본 태그와 unknown을 보존한다. 승인된 근거가 생기면 보강 TRAIN 버전·평가 조건을 먼저 고정하고 다음 대조 학습을 진행한다.", "",
              "이번 도구는 관찰·메타데이터 집계용이며 오류율을 새로 측정하거나 학습하지 않는다. 모델 성능은 별도의 고정 평가 결과를 참조한다. 공장 현장 성능은 측정하지 않았다.", "",
              "## 검수 화면 연결", "",
              "새 AI 보조 화면에서 `AI 보조 관찰 보기`를 켜면 관찰한 사실·원인 후보·질문을 읽을 수 있다. 기본은 꺼져 있으며 사람이 입력하는 의견란은 비어 있다. AI 참고 의견은 팀원·전문가 의견 수와 자동 합산하지 않는다.",
              "개별 사진·주석·AI 메모·경로는 로컬에 보관한다. 이 공개 문서에는 집계와 파일 SHA만 남긴다.", "",
              f'원본 검수 묶음 내용 SHA256: `{summary["source_package_sha256"]}`',
              f'AI 관찰 내용 SHA256: `{summary["observation_content_sha256"]}`', ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--prior-observations", type=Path, help="Optional old TRAIN AI record; compare photo digests only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packagepath = args.package.resolve(); output = args.output.resolve()
    if not packagepath.is_relative_to((ROOT / "runs").resolve()) or not output.is_relative_to((ROOT / "runs").resolve()) or output == (ROOT / "runs").resolve():
        raise ValueError("Local package and new output must remain inside ignored runs")
    if output.exists():
        raise ValueError("Preserve prior evidence; choose a fresh output directory")
    package = validate_package(read_json(packagepath))
    observations = validate_ai_observations(read_json(args.observations), package)
    prior_context = None
    if args.prior_observations:
        priorpath = args.prior_observations.resolve()
        if not priorpath.is_relative_to((ROOT / "runs").resolve()):
            raise ValueError("Prior AI record must be local inside ignored runs")
        prior_context = prior_photo_context(read_json(priorpath), observations, sha(priorpath))
    dimensions, detail = audit_dimensions(package, packagepath.parent)
    result = {"schema": "facility_ai_roi_review_report_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "ai_observations": summarize_ai_observations(observations, package), "input_dimensions": dimensions,
              "package_file_sha256": sha(packagepath), "observations_file_sha256": sha(args.observations),
              "metadata_only_case_count": len(package["cases"]) - len(observations["cases"]),
              "visual_observation_is_full_package": len(package["cases"]) == len(observations["cases"]),
              "error_rate_measured": False,
              "app_model_promoted": False, "training_epochs": 0,
              "script_sha256": sha(__file__),
              "prior_ai_observation_context": prior_context,
              "limitations": ["Selected TRAIN observations do not estimate error prevalence or field accuracy.",
                              "Dimensions are original input headers; these flags do not exclude cases or confirm causes.",
                              "AI opinions are unconfirmed and are not human or expert feedback."]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README_KO.md").write_text(render_report(result), encoding="utf-8")
    (output / "input-dimensions-details.json").write_text(json.dumps({"source_package_sha256": package["package_content_sha256"], "cases": detail}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"observed_cases": len(observations["cases"]), "observed_task_opinions": result["ai_observations"]["observed_task_opinions"], "metadata_cases": len(detail), "expert_confirmed_labels": 0, "training_epochs": 0}, ensure_ascii=False))


if __name__ == "__main__": main()
