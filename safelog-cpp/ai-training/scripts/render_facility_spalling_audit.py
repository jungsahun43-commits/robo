"""Render a read-only, local TRAIN spalling audit without changing source data."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "facility_spalling_train_audit_v1"
AUDIT_DIRECTORY = Path("runs/facility-spalling-train-audit")
UNKNOWN = "미확인 / 계측불가"
OUTCOMES = {
    "FP": "원본 라벨 기준 오탐",
    "FN": "원본 라벨 기준 미탐",
    "TP": "원본 라벨과 양성 일치",
    "TN": "원본 라벨과 음성 일치",
}
SOURCE_LABELS = {"dacl": "DACL", "damsegment": "DamSegment", "codebrim": "CODEBRIM"}
TAG_LABELS = {
    "concrete_crack": "균열", "concrete_spalling": "박락", "rust_stain": "녹 흔적",
    "exposed_rebar": "철근 노출", "wet_surface": "젖은 표면",
    "efflorescence": "백화", "surface_cavity": "공동",
}
GEOMETRY_LABELS = {
    "fine_area_fraction": "640 출처 박락 라벨 면적 비율",
    "coarse_any_area_fraction": "80 any 셀 면적 비율",
    "occupied80cells": "박락 표시가 있는 80 셀 수",
    "expansion_ratio": "80 any 면적 / 640 라벨 면적",
    "thin_cells_le_eighth": "fine 픽셀 ≤8인 80 셀 수",
    "thin_cells_le_eighth_fraction": "표시된 80 셀 중 fine 픽셀 ≤8 비율",
    "occupied80_cell_fill": "표시된 80 셀 안의 fine 픽셀 수",
    "ambiguous_tag_overlap": "다른 출처 태그와의 라벨 영역 겹침",
    "annotation_union_area_fraction": "출처 주석 전체 합집합 면적 비율",
    "source_unmarked_area_fraction": "출처 미표기 면적 비율",
    "tag_fine_area_fraction": "해당 태그 fine 라벨 면적 비율",
    "overlap_area_fraction": "겹친 면적 비율",
    "overlap_fine_area_fraction": "겹친 fine 면적 비율",
    "spalling_overlap_fraction": "박락 라벨 중 겹친 비율",
    "tag_overlap_fraction": "해당 태그 라벨 중 겹친 비율",
    "min": "최소", "mean": "평균", "median": "중앙값", "max": "최대",
    "histogram_fine_pixels": "fine 픽셀 수별 셀 분포",
}
TEXTURE_LABELS = {
    "pixel_count": "계측 픽셀 수",
    "luminance_mean": "휘도 평균 (0–1)",
    "luminance_std": "휘도 표준편차 (0–1)",
    "finite_difference_pair_count": "인접 픽셀 계측 쌍 수",
    "finite_difference_energy": "가로·세로 인접 휘도 절대차 평균",
    "laplacian_pixel_count": "라플라시안 계측 픽셀 수",
    "laplacian_variance": "라플라시안 분산",
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _local_path(base: Path, value: Any) -> Path:
    """Interpret filesystem references, never URLs, within their specified base."""
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(character) < 32 for character in value)):
        raise ValueError("Asset references must be nonempty relative POSIX filesystem paths")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Asset reference escapes its local base")
    base = base.resolve()
    candidate = (base / Path(*relative.parts)).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError("Asset reference resolves outside its local base")
    return candidate


def _local_url(base: Path, value: Any, output_directory: Path) -> str:
    path = _local_path(base, value)
    relative = Path(os.path.relpath(path, output_directory)).as_posix()
    return quote(relative, safe="/")


def _number(value: Any, *, fraction: bool = False) -> str:
    if isinstance(value, bool):
        return "있음" if value else "없음"
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return UNKNOWN
    if fraction:
        return f"{value * 100:.6g}%"
    return f"{value:,}" if isinstance(value, int) else f"{value:.6g}"


def _value(value: Any, key: str = "") -> str:
    if value is None:
        return UNKNOWN
    if isinstance(value, dict):
        if not value:
            return UNKNOWN
        return '<dl class="nested">' + "".join(
            f'<dt>{_escape(GEOMETRY_LABELS.get(str(k), TAG_LABELS.get(str(k), str(k))))}</dt>'
            f'<dd>{_value(v, str(k))}</dd>' for k, v in value.items()
        ) + "</dl>"
    if isinstance(value, list):
        return ", ".join(_value(item, key) for item in value) or UNKNOWN
    if isinstance(value, (int, float, bool)):
        return _escape(_number(value, fraction=key.endswith("_fraction")))
    return _escape(value)


def _metric_table(metrics: Any, labels: dict[str, str], *, geometry: bool = False) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return f'<p class="unknown">{UNKNOWN}</p>'
    unavailable = geometry and metrics.get("raster_available") is False
    rows = []
    for key, value in metrics.items():
        if key == "raster_available":
            continue
        rendered = UNKNOWN if unavailable else _value(value, str(key))
        rows.append(f'<tr><th scope="row">{_escape(labels.get(str(key), str(key)))}</th>'
                    f'<td>{rendered}</td></tr>')
    return '<table class="metrics"><tbody>' + "".join(rows) + "</tbody></table>"


def _figure(url: str | None, title: str, caption: str, *, coarse: bool = False,
            missing: str = "출처 마스크 자산 미확인") -> str:
    picture = (f'<a href="{_escape(url)}"><img loading="lazy" src="{_escape(url)}" '
               f'alt="{_escape(title)}"></a>' if url else
               f'<div class="missing">{_escape(missing)}</div>')
    return f'<figure class="{"coarse" if coarse else ""}"><h3>{_escape(title)}</h3>' \
           f'{picture}<figcaption>{_escape(caption)}</figcaption></figure>'


def _case_html(case: dict[str, Any], output_directory: Path, root: Path,
               workbench_url: str | None, global_threshold: Any) -> str:
    case_id, domain, outcome = case["case_id"], case["domain"], case["outcome"]
    error = outcome in ("FP", "FN")
    selection = ("박락 오류로 직접 선택" if case["direct_spalling_selection"] else
                 "다른 과제에서 선택된 추가 박락 오류") if error else "정답 일치 비교 사례"
    geometry = case.get("geometry")
    pixel = case.get("pixel_supervision") or {}
    known = pixel.get("known") == 1
    spatial_available = isinstance(geometry, dict) and geometry.get("raster_available") is True
    if not spatial_available:
        spatial_notice = "사진 태그만 있음: 위치 정답 미확인"
    else:
        spatial_notice = "출처 위치 라벨에서 계산한 기하 계측"
    image_url = _local_url(root, case["image"], output_directory)
    fine_url = (_local_url(output_directory, case["fine_mask_asset"], output_directory)
                if spatial_available and case.get("fine_mask_asset") else None)
    coarse_url = (_local_url(output_directory, case["coarse_mask_asset"], output_directory)
                  if known and case.get("coarse_mask_asset") else None)
    photos = _figure(image_url, "기존 TRAIN 처리 사진", "기존 패키지의 TRAIN 이미지 · 파일을 열어 확대 가능")
    photos += _figure(fine_url, "640 출처 라벨 마스크", "출처 라벨에서 재구성한 박락 위치 정답",
                      missing=spatial_notice if not spatial_available else "640 출처 마스크 자산 미확인")
    photos += _figure(coarse_url, "저장된 80 마스크", "기존 픽셀 정답 · 최근접 방식으로 표시만 확대",
                      coarse=True, missing="위치 정답 미확인" if not known else "80 마스크 자산 미확인")
    tags = case.get("source_tags") or []
    tags_text = ", ".join(TAG_LABELS.get(str(tag), str(tag)) for tag in tags) or "미확인"
    target = case.get("original_target")
    target_text = {1: "출처 라벨에 박락 표시 있음", 0: "출처 라벨에 박락 표시 없음"}.get(target, "출처 박락 태그 미확인")
    workbench = (f'<a class="workbench" href="{_escape(workbench_url + "#" + quote(str(case_id), safe=""))}">'
                 '기존 검수 화면에서 이 사례 열기 ↗</a>' if workbench_url else "")
    supervision = (_number(pixel.get("positive_cells")) + " / 6,400 셀" if known else
                   "미확인 · 위치 정답 없음")
    observations = case.get("observations") or []
    observations_html = ('<section class="observations"><h3>감사 관찰 기록 · 판정 미확정</h3><ul>' +
                         "".join(f'<li>{_escape(item)}</li>' for item in observations) +
                         '</ul></section>') if observations else ""
    texture = case.get("texture") or {}
    threshold = case.get("threshold", global_threshold)
    return f'''<article class="case" data-domain="{_escape(domain)}" data-outcome="{_escape(outcome)}" id="{_escape(case_id)}"{' hidden' if not error else ''}>
<div class="case-heading"><div><p class="eyebrow">{_escape(SOURCE_LABELS.get(domain, domain))} · {_escape(selection)}</p>
<h2>{_escape(case_id)}</h2></div><span class="outcome outcome-{outcome.lower()}">{outcome} · {OUTCOMES[outcome]}</span></div>
<div class="reference-line"><span>{_escape(target_text)}</span><span>기존 박락 점수 {_escape(_number(case.get('probability')))}</span><span>임계값 {_escape(_number(threshold))}</span></div>
<div class="photos">{photos}</div>
<p class="label-note">마스크는 출처 라벨·기존 픽셀 정답입니다. 출처 미표기 영역 ≠ 정상·건전. 표시 영역은 모델의 주목 영역이나 예측 위치를 뜻하지 않습니다.</p>
<div class="case-facts"><p><b>출처 주석 종류</b> {_escape(case.get('annotation_kind') or '미확인')}</p><p><b>함께 있는 출처 태그</b> {_escape(tags_text)}</p><p><b>기존 픽셀 정답 박락 셀</b> {_escape(supervision)}</p></div>
<details class="measurements"><summary>출처 라벨 기하 계측 <span>{_escape(spatial_notice)}</span></summary>
<p class="measurement-note">80 any 셀은 포함된 640 픽셀 중 하나라도 라벨이 있으면 표시됩니다. 셀 채움 수와 면적 비율은 이 축소 표현을 설명하는 값입니다.</p>
{_metric_table(geometry, GEOMETRY_LABELS, geometry=True)}</details>
<details class="measurements"><summary>사진 질감 계측 <span>수치 기록 · 오류 원인 미확정</span></summary>
<p class="measurement-note">휘도·인접 차이·라플라시안은 기술적 계측값입니다. 출처 미표기 영역에도 라벨이 없는 손상이 포함될 수 있습니다.</p>
<div class="texture-grid"><section><h3>처리 사진 전체</h3>{_metric_table(texture.get('whole_image'), TEXTURE_LABELS)}</section>
<section><h3>출처 주석에서 미표기된 영역</h3>{_metric_table(texture.get('source_unmarked'), TEXTURE_LABELS)}</section></div></details>
{observations_html}{workbench}</article>'''


CSS = r"""
:root{font-family:Segoe UI,Malgun Gothic,Apple SD Gothic Neo,sans-serif;color:#192b38;background:#eef2f3;font-size:15px;line-height:1.55;color-scheme:light}*{box-sizing:border-box}body{margin:0}a{color:#175e78;text-decoration-thickness:1px;text-underline-offset:3px}button,select{font:inherit}main{max-width:1440px;margin:auto;padding:32px 32px 64px}header{background:#fff;border:1px solid #d2dce0;border-top:5px solid #244e5d;border-radius:12px;padding:28px 32px;margin-bottom:24px}.eyebrow{color:#586d78;font-size:12px;letter-spacing:.04em;margin:0 0 7px}h1{font-size:clamp(25px,3vw,36px);line-height:1.25;margin:0 0 12px;letter-spacing:-.04em}h2{font-size:20px;margin:0;word-break:break-word}h3{font-size:14px;margin:0 0 10px}p{margin:10px 0}.intro{max-width:920px;color:#4b626f}.scope-notice{padding:12px 15px;background:#edf4f5;border-left:3px solid #386b77;font-size:13px}.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:22px}.stat{border:1px solid #dce4e7;border-radius:7px;padding:12px}.stat b{display:block;font-size:25px;line-height:1.3;font-variant-numeric:tabular-nums}.stat span{color:#526875;font-size:12px}.provenance{margin-top:16px;font-size:12px;color:#4d6572}.provenance summary{cursor:pointer}.hashes{font-family:Consolas,monospace;word-break:break-all}.controls{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:18px 0}.filter-group{display:flex;gap:5px;align-items:center;flex-wrap:wrap}.filter-label{font-size:12px;color:#536c78;margin-right:3px}.filter-group button{border:1px solid #bdcdd4;border-radius:6px;padding:7px 11px;background:#fff;color:#2c4b5a;cursor:pointer}.filter-group button[aria-pressed=true]{background:#244e5d;border-color:#244e5d;color:#fff}.comparison-control{font-size:13px;display:flex;gap:7px;align-items:center}.counter{margin-left:auto;color:#506b77;font-size:13px;font-variant-numeric:tabular-nums}.case{background:#fff;border:1px solid #d2dce0;border-radius:10px;padding:24px 26px;margin:0 0 22px}.case[hidden]{display:none}.case-heading{display:flex;justify-content:space-between;align-items:center;gap:16px}.outcome{font-size:12px;font-weight:650;border-radius:6px;padding:7px 10px;white-space:nowrap}.outcome-fp{background:#fbeade;color:#8b4822}.outcome-fn{background:#f6e4e6;color:#8e3641}.outcome-tp,.outcome-tn{background:#e8f1ec;color:#3c6850}.reference-line{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#566d78;padding:13px 0}.photos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.photos figure{margin:0;min-width:0}.photos figure h3{color:#405c68;font-weight:600;font-size:12px}.photos figure a,.missing{display:flex;aspect-ratio:1;align-items:center;justify-content:center;background:#111a1e;border-radius:7px;overflow:hidden}.photos img{display:block;width:100%;height:100%;object-fit:contain}.photos .coarse img{image-rendering:pixelated;image-rendering:crisp-edges}.missing{color:#d8e2e7;font-size:13px;padding:25px;text-align:center}.photos figcaption{color:#596e77;font-size:11px;margin:8px 0}.label-note{font-size:12px;color:#596d77;background:#f4f7f8;padding:10px 12px;border-radius:5px}.case-facts{display:flex;flex-wrap:wrap;column-gap:24px;row-gap:2px;font-size:12px;color:#516977;margin:14px 0}.case-facts p{margin:2px 0}.case-facts b{color:#284855;margin-right:5px}.measurements{margin-top:10px;border:1px solid #dce4e8;border-radius:7px;padding:12px 14px}.measurements summary{font-size:13px;font-weight:650;cursor:pointer}.measurements summary span{font-weight:400;font-size:11px;color:#607582;margin-left:12px}.measurement-note{font-size:12px;color:#627682;max-width:1000px}.metrics{width:100%;border-collapse:collapse;font-size:12px}.metrics th,.metrics td{text-align:left;vertical-align:top;padding:7px 8px;border-bottom:1px solid #e8eef0}.metrics th{font-weight:500;color:#4b6471;width:56%}.metrics td{font-variant-numeric:tabular-nums;word-break:break-word}.metrics tr:last-child th,.metrics tr:last-child td{border-bottom:0}.nested{display:grid;grid-template-columns:minmax(120px,1fr) minmax(100px,1fr);gap:5px 12px;margin:0}.nested dt{color:#556c76}.nested dd{margin:0}.texture-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.texture-grid h3{color:#455f6a;margin-top:10px}.unknown{color:#697a83;font-size:12px}.observations{margin-top:15px;font-size:13px}.observations li{margin:4px 0}.workbench{display:inline-block;margin-top:16px;font-size:12px}.empty{padding:30px;background:#fff;border-radius:8px;border:1px solid #d2dce0;color:#516976;text-align:center}.footer{font-size:12px;color:#5e717c;margin:24px 0 0}button:focus-visible,a:focus-visible,summary:focus-visible,input:focus-visible{outline:3px solid #d9912f;outline-offset:3px}@media(max-width:950px){main{padding:20px}.stats{grid-template-columns:repeat(3,1fr)}.case-heading{align-items:flex-start}.outcome{white-space:normal}.texture-grid{grid-template-columns:1fr}}@media(max-width:650px){main{padding:12px}header,.case{padding:18px}.photos{grid-template-columns:1fr}.photos figure a,.missing{aspect-ratio:4/3}.case-heading{flex-direction:column;gap:10px}.stats{grid-template-columns:repeat(2,1fr)}.counter{margin-left:0}.controls{gap:10px}.measurements summary span{display:block;margin:4px 0}.nested{grid-template-columns:1fr}.nested dd{padding-left:10px}}@media print{.controls{display:none}.case{break-inside:avoid}.photos{grid-template-columns:repeat(3,1fr)}body{background:#fff}.measurements:not([open]){display:none}}
"""


JS = r"""
(() => {
  const cards = Array.from(document.querySelectorAll('.case'));
  const comparison = document.getElementById('show-comparisons');
  const counter = document.getElementById('visible-count');
  const empty = document.getElementById('empty-state');
  let domain = 'all';
  let outcome = 'all';
  function update() {
    let visible = 0;
    let errors = 0;
    for (const card of cards) {
      const isError = card.dataset.outcome === 'FP' || card.dataset.outcome === 'FN';
      const show = (isError || comparison.checked) &&
        (domain === 'all' || card.dataset.domain === domain) &&
        (outcome === 'all' || card.dataset.outcome === outcome);
      card.hidden = !show;
      if (show) { visible++; if (isError) errors++; }
    }
    counter.textContent = '표시 ' + visible + '장 · 오류 ' + errors + '장';
    empty.hidden = visible !== 0;
  }
  for (const group of document.querySelectorAll('[data-filter-group]')) {
    group.addEventListener('click', event => {
      const button = event.target.closest('button[data-value]');
      if (!button || !group.contains(button)) return;
      for (const sibling of group.querySelectorAll('button')) {
        sibling.setAttribute('aria-pressed', sibling === button ? 'true' : 'false');
      }
      if (group.dataset.filterGroup === 'domain') domain = button.dataset.value;
      else outcome = button.dataset.value;
      update();
    });
  }
  comparison.addEventListener('change', update);
  update();
})();
"""


def render_audit(details: dict[str, Any], output_directory: Path, root: Path) -> str:
    """Return escaped local HTML. This function reads no dataset files and writes nothing.

    ``image`` and ``review_workbench`` are filesystem paths relative to ``root``.
    Mask asset paths are relative to ``output_directory``. References are validated
    and quoted into local URLs; masks and images are not copied or embedded.
    """
    if not isinstance(details, dict) or details.get("schema") != SCHEMA:
        raise ValueError("Unsupported spalling audit schema")
    root, output_directory = Path(root).resolve(), Path(output_directory).resolve()
    if not output_directory.is_relative_to(root):
        raise ValueError("Output directory must remain inside the training workspace")
    cases = details.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Audit cases must be a list")
    case_ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case["case_id"]:
            raise ValueError("Each audit case needs a nonempty case_id")
        if case["case_id"] in case_ids:
            raise ValueError("Duplicate audit case_id")
        case_ids.add(case["case_id"])
        if not isinstance(case.get("domain"), str) or not case["domain"]:
            raise ValueError("Each audit case needs a source domain")
        if case.get("outcome") not in OUTCOMES:
            raise ValueError("Audit outcome must be FP, FN, TP or TN")
        if not isinstance(case.get("direct_spalling_selection"), bool):
            raise ValueError("direct_spalling_selection must be a boolean")
        if case.get("geometry") is not None and not isinstance(case["geometry"], dict):
            raise ValueError("geometry must be a dictionary or null")
        if isinstance(case.get("geometry"), dict) and "raster_available" in case["geometry"] and not isinstance(case["geometry"]["raster_available"], bool):
            raise ValueError("raster_available must be a boolean")
        if case.get("texture") is not None and not isinstance(case["texture"], dict):
            raise ValueError("texture must be a dictionary or null")
        if case.get("pixel_supervision") is not None and not isinstance(case["pixel_supervision"], dict):
            raise ValueError("pixel_supervision must be a dictionary")
        if isinstance(case.get("pixel_supervision"), dict):
            known = case["pixel_supervision"].get("known")
            if known is not None and (isinstance(known, bool) or known not in (0, 1)):
                raise ValueError("Pixel supervision known must retain numeric 0 or 1")
        if case.get("observations") is not None and (not isinstance(case["observations"], list) or
                any(not isinstance(item, str) for item in case["observations"])):
            raise ValueError("observations must be a list of strings")
        if case.get("source_tags") is not None and not isinstance(case["source_tags"], list):
            raise ValueError("source_tags must be a list")
        _local_path(root, case.get("image"))
        for key in ("fine_mask_asset", "coarse_mask_asset"):
            if case.get(key) is not None:
                _local_path(output_directory, case[key])
    counts = Counter(case["outcome"] for case in cases)
    error_cases = [case for case in cases if case["outcome"] in ("FP", "FN")]
    direct_count = sum(case["direct_spalling_selection"] for case in error_cases)
    computed = {"all": len(cases), "errors": len(error_cases), "direct_spalling_errors": direct_count,
                "additional_other_task_errors": len(error_cases) - direct_count}
    supplied_counts = details.get("case_counts")
    if supplied_counts is not None:
        if not isinstance(supplied_counts, dict) or any(supplied_counts.get(key) != value for key, value in computed.items()):
            raise ValueError("case_counts does not match audit cases")
    workbench_url = (_local_url(root, details["review_workbench"], output_directory)
                     if details.get("review_workbench") else None)
    cards = "".join(_case_html(case, output_directory, root, workbench_url, details.get("threshold")) for case in cases)
    domain_counts = Counter(case["domain"] for case in cases)
    domain_buttons = '<button type="button" data-value="all" aria-pressed="true">전체 출처</button>'
    domain_buttons += "".join(f'<button type="button" data-value="{_escape(domain)}" aria-pressed="false">'
                              f'{_escape(SOURCE_LABELS.get(domain, domain))} {count}</button>'
                              for domain, count in sorted(domain_counts.items()))
    outcome_buttons = '<button type="button" data-value="all" aria-pressed="true">전체 유형</button>'
    outcome_buttons += "".join(f'<button type="button" data-value="{outcome}" aria-pressed="false">'
                               f'{outcome} {counts[outcome]}</button>' for outcome in OUTCOMES)
    stats = "".join(f'<div class="stat"><b>{count}</b><span>{label}</span></div>' for count, label in (
        (len(cases), "선택 TRAIN 사진"), (len(error_cases), "박락 FP/FN"),
        (direct_count, "박락 오류로 직접 선택"), (len(error_cases) - direct_count, "다른 과제에서 추가 확인"),
        (0, "이 감사의 관찰자 판정"), (0, "이 감사의 전문가 판정")))
    workbench_header = (f'<p><a href="{_escape(workbench_url)}">기존 검수 의견 화면 열기 ↗</a></p>'
                        if workbench_url else "")
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>박락 TRAIN 라벨·축소 계측 감사</title><style>{CSS}</style></head><body><main>
<header><p class="eyebrow">LOCAL TRAIN AUDIT · SOURCE LABELS &amp; MEASUREMENTS</p>
<h1>박락 TRAIN 라벨·축소 계측 감사</h1>
<p class="intro">기존 모델의 박락 FP/FN과 정답 일치 사례를 출처 라벨·기존 80 픽셀 정답·사진 계측으로 비교합니다. 기본 화면에는 FP/FN만 표시됩니다.</p>
<p class="scope-notice">이 {len(cases)}장은 편향된 TRAIN 선별 표본입니다. 현장 정확도: 산출 대상 아님. 원본 라벨 변경 0건. 계측값으로 오류 원인이나 향후 개선 효과를 확정할 수 없습니다.</p>
<div class="stats">{stats}</div>
<p class="intro">FP/FN/TP/TN은 기존 박락 점수와 출처 태그를 비교한 결과입니다. 출처 미표기 영역 ≠ 정상·건전. 위치 라벨이 없는 사진은 면적 0으로 해석하지 않습니다.</p>
{workbench_header}<details class="provenance"><summary>감사 범위·기존 패키지 식별 정보</summary>
<p>{_value(details.get('scope'))}</p><p>기존 박락 임계값: {_escape(_number(details.get('threshold')))}</p>
<p class="hashes">패키지 SHA256: {_escape(details.get('package_sha256') or '미확인')}<br>가중치 SHA256: {_escape(details.get('weights_sha256') or '미확인')}</p>
<p>로컬 검수용 링크와 계측 결과입니다. 사진·출처 주석·파생 마스크의 이용은 각 데이터의 원본 이용 조건을 따릅니다.</p></details></header>
<div class="controls"><div class="filter-group" data-filter-group="domain" aria-label="자료 출처 필터"><span class="filter-label">출처</span>{domain_buttons}</div>
<div class="filter-group" data-filter-group="outcome" aria-label="박락 결과 필터"><span class="filter-label">결과</span>{outcome_buttons}</div>
<label class="comparison-control"><input type="checkbox" id="show-comparisons"> TP/TN 비교 사례 포함</label>
<span id="visible-count" class="counter" role="status" aria-live="polite">표시 {len(error_cases)}장 · 오류 {len(error_cases)}장</span></div>
<noscript><p class="scope-notice">필터에는 JavaScript가 필요합니다. 현재 화면에는 FP/FN만 표시됩니다.</p></noscript>
<p id="empty-state" class="empty" hidden>선택한 필터에 맞는 사례가 없습니다.</p>{cards}
<p class="footer">출처 라벨과 계측을 읽기 위한 화면 · 의견 입력과 사람 판정은 기존 검수 화면에서 진행합니다.</p>
</main><script>{JS}</script></body></html>'''


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local SPALLING-AUDIT.json in the ignored audit run")
    parser.add_argument("--output", type=Path, help="New local HTML path or directory; defaults beside the input")
    args = parser.parse_args(argv)
    allowed = (ROOT / AUDIT_DIRECTORY).resolve()
    if not allowed.is_relative_to(ROOT.resolve()):
        raise ValueError("Audit run resolves outside the training workspace")
    source = args.input.resolve()
    if not source.is_relative_to(allowed) or source.suffix.lower() != ".json" or not source.is_file():
        raise ValueError("Input must be local JSON inside runs/facility-spalling-train-audit")
    destination = args.output if args.output is not None else source.parent
    if destination.suffix.lower() != ".html":
        destination = destination / "SPALLING-AUDIT.html"
    destination = destination.resolve()
    if not destination.is_relative_to(allowed):
        raise ValueError("HTML must remain inside runs/facility-spalling-train-audit")
    if destination.exists():
        raise ValueError("Audit HTML already exists; refusing overwrite")
    details = json.loads(source.read_text(encoding="utf-8"))
    page = render_audit(details, destination.parent, ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against an output appearing after validation.
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(page)
    print(json.dumps({"output": str(destination), "cases": len(details["cases"]),
                      "original_labels_changed": 0, "observer_decisions": 0, "expert_decisions": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
