"""Create a local opinion workbench next to an immutable TRAIN review package."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safelog_ai.review_feedback import CLASSES, TASKS, REASONS, read_json, validate_package
from safelog_ai.review_observations import validate_ai_observations
from scripts.build_facility_train_review import LABELS, overlay

REASON_LABELS = {"unreviewed": "검수 전", "small_damage": "작은 손상 / 세부 정보 부족", "texture_confusion": "질감·그림자·이음매와 혼동 의심", "other_damage_confusion": "다른 손상과 혼동 의심", "capture_quality": "촬영 상태 문제", "annotation_uncertain": "원본 정답 기준 질문", "model_error": "판단 가능한 사진의 모델 오류 의심", "correct_comparison": "정답 일치 비교 사례", "other": "기타"}


def safe_reference(url, directory):
    if not isinstance(url, str) or not url or "\\" in url:
        raise ValueError("Review references must be relative local URLs")
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or parts.query or parts.fragment or url.startswith("/"):
        raise ValueError("Remote or absolute references are forbidden")
    path = (directory / unquote(parts.path)).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("Review reference leaves the training workspace or is missing")
    return path


def verify_asset(url, expected_sha, directory, cache):
    path = safe_reference(url, directory)
    if path not in cache:
        with path.open("rb") as stream:
            cache[path] = hashlib.file_digest(stream, "sha256").hexdigest()
    if cache[path] != expected_sha:
        raise ValueError("Source photo or annotation bytes changed after the review package")


def render_ai_notes(row):
    if row is None:
        return ""
    e = lambda value: html.escape(str(value), quote=True)
    labels = {"present": "보이는 손상 후보 있음", "absent": "해당 손상 뚜렷하지 않음", "uncertain": "사진으로 판단 어려움"}
    tasks = "".join(f'<li>{e(LABELS[t["task"]])}: {e(labels[t["visual_judgement"]])} · {e(REASON_LABELS[t["reason"]])}</li>' for t in row["tasks"])
    groups = "".join(f'<h4>{title}</h4><ul>{"".join(f"<li>{e(value)}</li>" for value in row[key])}</ul>' for key, title in (("facts", "관찰한 사실"), ("hypotheses", "원인 후보"), ("questions", "추가 확인 질문")) if row[key])
    return f'<section class="ai-observation"><h3>AI 보조 관찰 · 정답 미확정</h3><p>전문가 판정이 아닌 참고 의견입니다. 아래 내용은 사람 의견 입력란에 자동 반영되지 않습니다.</p><ul>{tasks}</ul>{groups}</section>'


def render_html(package, ai_observations=None):
    e = lambda value: html.escape(str(value), quote=True)
    cards = []
    ai_by_id = {row["case_id"]: row for row in (ai_observations or {}).get("cases", [])}
    for case in package["cases"]:
        annotation = case.get("annotation", {})
        fields = []
        for task in TASKS:
            reasons = "".join(f'<option value="{e(reason)}">{e(REASON_LABELS[reason])}</option>' for reason in REASONS)
            fields.append(f'''<fieldset data-task="{e(task)}"><legend>{e(LABELS[task])} 의견</legend><div class="fields">
<label>사진에서의 판단<select class="judgement" aria-label="{e(LABELS[task])} 판단"><option value="unreviewed">검수 전</option><option value="present">손상이 보임</option><option value="absent">손상이 보이지 않음</option><option value="uncertain">사진으로 판단하기 어려움</option></select></label>
<label>관찰 사유<select class="reason" aria-label="{e(LABELS[task])} 사유">{reasons}</select></label>
<label class="wide">관찰 메모 <small>보이는 사실과 원인 추정을 구분해 적으세요.</small><textarea class="note" maxlength="4000" aria-label="{e(LABELS[task])} 관찰 메모"></textarea></label>
<label class="wide">판단 근거 <small>전문가 의견이면 필수입니다. 출판자 정의·주석·관련 점검 기준 등의 근거를 적으세요.</small><textarea class="evidence" maxlength="4000" aria-label="{e(LABELS[task])} 판단 근거"></textarea></label></div>
<button class="secondary reset-task" type="button">이 항목 의견 지우기</button></fieldset>''')
        outcomes = " · ".join(f'{LABELS[s["task"]]} {s["outcome"]}' for s in case.get("selected_for", []))
        rows = "".join(f'<tr><td>{e(LABELS[c])}</td><td class="reference">{ {1:"있음",0:"없음",-1:"미확인"}[case["original_targets"][k]] }</td><td class="model">{case["probabilities"][k]:.4f}</td></tr>' for k, c in enumerate(CLASSES))
        source_link = f'<a href="{e(case["original_source_url"])}">출판자 원본 사진</a>' if case.get("original_source_url") else ""
        annotation_link = f'<a href="{e(annotation["url"])}">원본 주석 파일</a>' if annotation.get("url") else "위치 주석 없음"
        cards.append(f'''<article id="{e(case["case_id"])}" data-case-id="{e(case["case_id"])}" data-domain="{e(case["domain"])}">
<h2>{e(case["case_id"])} · {e(case["domain"])} <span class="review-state">검수 전</span></h2>
<figure class="photo"><img loading="lazy" src="{e(case["image_url"])}" alt="{e(case["case_id"])} 원본 TRAIN 사진"><figcaption>먼저 사진을 관찰하고 두 손상 항목의 의견을 적으세요.</figcaption></figure>
<section class="reference"><h3>출판자 정답·주석 참고</h3><div class="reference-grid"><figure>{overlay(case)}<figcaption>{annotation_link}</figcaption></figure><div><p>{source_link}</p><p>원본 태그: {e(", ".join(case.get("source_tags", [])) or "없음")}</p><p>태그가 함께 있다는 사실로 오류 원인이나 다른 손상을 확정하지 않습니다.</p></div></div></section>
<p class="model">모델 기준 선택 사유: {e(outcomes)}. FP/FN은 원본 정답과의 비교이며 전문가 판정이 아닙니다.</p>
<div class="label-table"><table><thead><tr><th>항목</th><th class="reference">원본 정답</th><th class="model">모델 확률</th></tr></thead><tbody>{rows}</tbody></table></div>
{render_ai_notes(ai_by_id.get(case["case_id"]))}{"".join(fields)}<details><summary>사진·주석 기록 식별 정보</summary><p class="hashes">사진 SHA256 {e(case["image_sha256"])}<br>주석 SHA256 {e(annotation.get("sha256", "해당 없음"))}</p></details></article>''')
    metadata = {"source_package_sha256": package["package_content_sha256"], "weights_sha256": package["weights_sha256"], "case_ids": [c["case_id"] for c in package["cases"]], "domains": {c["case_id"]: c["domain"] for c in package["cases"]}}
    embedded = json.dumps(metadata, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeLog TRAIN 검수 의견</title><link rel="stylesheet" href="review-workbench.css"></head><body>
<header><h1>균열·박락 검수 의견</h1><p class="intro">사진별 의견을 기록하고 JSON으로 저장·불러오는 로컬 작업 화면입니다. 기존 정답·모델은 그대로 두며, 의견을 학습에 자동 반영하지 않습니다.</p>
<p class="notice">이 {len(package["cases"])}장은 TRAIN에서 고른 검수 자료입니다. 원인 의견의 빈도는 전체 오류율이 아닙니다. 전문가 구분은 작성자의 자기 신고이며, 자격 검증이나 정답 승인 기능이 아닙니다.</p>
<div class="reviewer"><label>검수자 ID <small>같은 사람은 같은 ID를 사용하세요.</small><input id="reviewer-id" type="text" maxlength="200" autocomplete="off"></label><label>이름 또는 표시명<input id="reviewer-name" type="text" maxlength="200" autocomplete="off"></label>
<label>작성자 구분<select id="reviewer-role"><option value="team_observer">팀원 관찰</option><option value="domain_expert">관련 분야 전문가 의견</option></select></label><label class="expert-field">전문 분야·관련 경험<input id="reviewer-expertise" type="text" maxlength="500" autocomplete="off"></label></div>
<div class="actions"><button id="export" type="button">의견 JSON 저장</button><label class="file-label">v2 의견 불러오기<input id="import" type="file" accept=".json,application/json"></label></div>
<p><small>입력은 이 창의 메모리에만 있습니다. 창을 닫기 전에 저장하세요. 불러오기는 현재 입력을 교체하며, 여러 사람의 의견은 CLI에서 집계합니다.</small></p>
<p id="status" role="status" aria-live="polite"></p><div id="progress" class="progress"></div><p id="reason-counts"></p>
<details id="export-preview"><summary>저장할 JSON 확인·복사</summary><p>브라우저가 다운로드를 지원하지 않으면 내용을 복사해 메모장 등에서 UTF-8 JSON 파일로 저장하세요. 수정하기 전 이 내용을 먼저 저장할 수 있습니다.</p><textarea id="export-json" readonly aria-label="저장할 의견 JSON"></textarea><button id="select-json" class="secondary" type="button">JSON 전체 선택</button></details>
<div class="controls"><label>자료 출처 <select id="domain-filter"><option value="all">전체</option><option value="dacl">DACL</option><option value="damsegment">DamSegment</option><option value="codebrim">CODEBRIM</option></select></label><label><input id="pending-only" type="checkbox"> 미완료 사진만 보기</label><label><input id="show-reference" type="checkbox"> 원본 정답·주석 보기</label><label><input id="show-model" type="checkbox"> 모델 제안 보기</label><label><input id="show-ai" type="checkbox" {"disabled" if not ai_by_id else ""}> AI 보조 관찰 보기 ({len(ai_by_id)}건)</label></div>
<details><summary>입력·집계 방법</summary><p>검수한 항목은 판단·사유·관찰 메모를 입력하세요. 전문가 의견에는 전문 분야와 항목별 근거도 필요합니다. 불확실한 사진은 ‘사진으로 판단하기 어려움’으로 기록하세요. 하나의 사진이 여러 사유에 포함될 수 있습니다.</p><p>서로 다른 사람의 있음/없음 의견은 충돌로 기록하고 자동 다수결·정답 변경은 하지 않습니다. 같은 사람의 수정본은 가장 최근에 저장한 한 파일을 집계에 사용하세요. 기존 v1 사유·메모 파일은 CLI에서 사진 단위의 별도 관찰로 집계합니다.</p><p>{e(package.get("license", "원본 데이터 이용 조건을 확인하세요."))}</p></details></header>
{"".join(cards)}<script>window.reviewMetadata={embedded};</script><script src="review-workbench.js"></script></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New HTML inside ignored runs; relative source links are rebased")
    parser.add_argument("--ai-observations", type=Path, help="Optional separate, unconfirmed AI observation sidecar")
    args = parser.parse_args(argv)
    source = args.package.resolve()
    output = (args.output or source.with_name("review-workbench.html")).resolve()
    if not source.is_relative_to((ROOT / "runs").resolve()) or not source.is_file():
        raise ValueError("Package must be in local ignored runs")
    if not output.is_relative_to((ROOT / "runs").resolve()) or output.suffix != ".html":
        raise ValueError("Generated HTML must remain inside ignored runs")
    targets = [output, output.with_name("review-workbench.js"), output.with_name("review-workbench.css")]
    if any(path.exists() for path in targets):
        raise ValueError("Workbench already exists; preserve it rather than overwrite opinions")
    package = validate_package(read_json(source))
    ai = validate_ai_observations(read_json(args.ai_observations), package) if args.ai_observations else None
    cache = {}
    for case in package["cases"]:
        verify_asset(case["image_url"], case.get("image_sha256"), source.parent, cache)
        if case.get("original_source_url"):
            verify_asset(case["original_source_url"], case.get("original_source_sha256"), source.parent, cache)
        annotation = case.get("annotation", {})
        if annotation.get("url"):
            verify_asset(annotation["url"], annotation.get("sha256"), source.parent, cache)
    presentation = deepcopy(package)
    for case in presentation["cases"]:
        references = [(case, "image_url"), (case, "original_source_url"), (case.get("annotation", {}), "url")]
        for obj, key in references:
            if obj.get(key):
                target = safe_reference(obj[key], source.parent)
                obj[key] = quote(Path(os.path.relpath(target, output.parent)).as_posix(), safe="/")
    page = render_html(presentation, ai)
    # No source image, annotation, model or immutable package is written here.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    for suffix in ("js", "css"):
        shutil.copyfile(ROOT / "review-ui" / f"review-workbench.{suffix}", output.with_name(f"review-workbench.{suffix}"))
    print(json.dumps({"output": str(output), "cases": len(package["cases"]), "verified_unique_assets": len(cache), "ai_observed_cases": len(ai["cases"]) if ai else 0, "source_package_sha256": package["package_content_sha256"], "original_labels_changed": 0, "expert_confirmed_labels": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
