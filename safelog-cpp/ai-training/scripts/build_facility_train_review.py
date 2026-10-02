"""Build a local, read-only TRAIN error review; never edits or exports source images."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
from urllib.parse import quote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar",
           "wet_surface", "efflorescence", "surface_cavity"]
TASKS = CLASSES[:2]
LABELS = dict(zip(CLASSES, ["균열", "박락", "녹 흔적", "철근 노출", "젖은 표면", "백화", "공동"]))
POLICY = "TRAIN 검수 제안만 기록. 원래 7종 targets, 원본 주석, 검증·시험 정답은 변경하지 않음. 검수 제안은 자동 학습에 반영하지 않음."
LICENSE = "로컬 검수용. 원본·가공 사진 및 주석을 GitHub/ZIP/외부에 재배포하지 마세요. CODEBRIM은 비상업 연구·교육 조건이며 데이터 재배포를 허용하지 않습니다. 다른 자료도 각 원본 라이선스를 따릅니다."


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(root, value):
    """No absolute source paths, traversal or links outside the dataset workspace."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Source path must be a nonempty relative POSIX path")
    candidate = (root / value).resolve()
    if Path(value).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError("Source path leaves the training workspace")
    return candidate


def vector(value, length, probability=False):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("Wrong seven-label vector shape")
    if probability:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in value):
            raise ValueError("Invalid probability")
    elif any(isinstance(v, bool) or v not in (-1, 0, 1) for v in value):
        raise ValueError("Original targets must retain -1/0/1")


def canonical_training(root):
    """Reconstruct the original full TRAIN membership, including the fixed Dam resplit."""
    sources = {}
    def load(relative):
        path = local_path(root, relative)
        sources[relative] = sha(path)
        return read(path)
    records = load("data/dacl10k-yolo/records.json")
    canonical, metadata = {}, {}
    for record in records:
        if record["split"] != "train":
            continue
        name = record["stem"]
        relative = f"data/dacl10k-yolo/images/train/{name}.jpg"
        target = [0] * len(CLASSES)
        label_path = local_path(root, f"data/dacl10k-yolo/labels/train/{name}.txt")
        for line in label_path.read_text(encoding="utf-8").splitlines():
            k = int(line.split()[0])
            if not 0 <= k < len(CLASSES):
                raise ValueError("Unexpected publisher-derived DACL class")
            target[k] = 1
        canonical[relative] = {"image": relative, "targets": target, "domain": "dacl",
                               "source": "data/" + record["source"].replace("\\", "/")}
        metadata[relative] = record
    dam = load("data/damsegment-training/train.json")
    if dam["split"] != "train" or dam["classes"] != CLASSES:
        raise ValueError("Original Dam train manifest changed")
    reserved_dam = load("data/damsegment-training/test.json")
    if reserved_dam["split"] != "test" or reserved_dam["classes"] != CLASSES:
        raise ValueError("Reserved Dam test membership metadata changed")
    reserved_groups = {i["group_id"] for i in reserved_dam["items"]}
    for item in dam["items"]:
        if item["split"] != "train" or item["group_id"] in reserved_groups:
            raise ValueError("Dam original TRAIN/test group leakage")
        digest = hashlib.sha256(("facility-target-seed43:" + item["group_id"]).encode()).hexdigest()
        if int(digest[:8], 16) % 5 == 0:
            continue  # This old TRAIN record is now reserved validation, not review TRAIN.
        canonical[item["image"]] = {**item, "domain": "damsegment"}
    code = load("data/codebrim-training/train.json")
    heldout_code = [load(f"data/codebrim-training/{split}.json") for split in ("val", "test")]
    if code["split"] != "train" or code["classes"] != CLASSES:
        raise ValueError("CODEBRIM TRAIN classes/split changed")
    if any(doc["split"] != split or doc["classes"] != CLASSES for doc, split in zip(heldout_code, ("val", "test"))):
        raise ValueError("Reserved CODEBRIM split membership metadata changed")
    heldout_groups = {i["group_id"] for doc in heldout_code for i in doc["items"]}
    for item in code["items"]:
        if item["split"] != "train" or item["domain"] != "codebrim" or item["group_id"] in heldout_groups:
            raise ValueError("CODEBRIM TRAIN/held-out parent leakage")
        canonical[item["image"]] = item
    return canonical, metadata, sources


def validate_training_manifest(manifest, canonical, detail_manifest=None):
    if manifest.get("split") != "train" or manifest.get("classes") != CLASSES:
        raise ValueError("Review manifest must be original seven-class TRAIN")
    items = manifest.get("items", [])
    full_count = manifest.get("full_count", len(items))
    if not isinstance(full_count, int) or not 0 < full_count <= len(items):
        raise ValueError("Invalid original full TRAIN count")
    originals = items[:full_count]
    images = [i["image"] for i in items]
    if len(set(images)) != len(images):
        raise ValueError("Duplicate TRAIN image path")
    if {i["image"] for i in originals} != set(canonical):
        raise ValueError("Full review rows must exactly match original permitted TRAIN")
    for item in originals:
        reference = canonical[item["image"]]
        if item["targets"] != reference["targets"] or item.get("domain") != reference["domain"]:
            raise ValueError("Original TRAIN targets or domain changed")
        if item.get("split", "train") != "train" or item.get("target_split", "train") != "train":
            raise ValueError("Held-out row in TRAIN manifest")
        vector(item["targets"], len(CLASSES))
    if items[full_count:] and (detail_manifest or {}).get("split") != "train":
        raise ValueError("Detail provenance manifest is not TRAIN")
    details = {i["image"]: i for i in (detail_manifest or {}).get("items", [])}
    for item in items[full_count:]:
        parent = item.get("parent_image")
        if item.get("parent_split") != "train" or parent not in canonical:
            raise ValueError("Detail row has a validation/test or unknown original parent")
        reference = details.get(item["image"])
        keys = ("targets", "parent_image", "parent_split", "box_in_parent", "domain")
        if reference is None or any(item.get(k) != reference.get(k) for k in keys):
            raise ValueError("Detail crop provenance or targets changed")
    return originals, items


def verify_cache(cache, all_items, classes, expected_weights_sha):
    if cache.get("split") != "train":
        raise ValueError("VAL/TEST scoring caches are forbidden for TRAIN review")
    if cache.get("classes", classes) != classes:
        raise ValueError("Scoring cache class order changed")
    actual_hash = cache.get("weights_sha256", cache.get("initial_weights_sha256"))
    if actual_hash != expected_weights_sha or not re.fullmatch(r"[a-f0-9]{64}", str(actual_hash)):
        raise ValueError("Scoring cache is not from the specified immutable weights")
    by_image = {i["image"]: i for i in all_items}
    images, targets, probabilities = (cache.get(k) for k in ("images", "targets", "probabilities"))
    if not all(isinstance(v, list) for v in (images, targets, probabilities)) or not len(images) == len(targets) == len(probabilities):
        raise ValueError("Cache row count mismatch")
    if len(images) != len(set(images)) or set(images) != set(by_image):
        raise ValueError("Cache must cover permitted manifest rows exactly once")
    result = {}
    for image, target, probability in zip(images, targets, probabilities):
        vector(target, len(classes))
        vector(probability, len(classes), probability=True)
        if target != by_image[image]["targets"]:
            raise ValueError("Cache targets differ from unchanged source TRAIN labels")
        result[image] = probability
    return result


def sample_cases(originals, scores, thresholds, per_error, per_correct, seed):
    """Source/task/outcome stratification, with hard and random examples, no duplication."""
    if min(per_error, per_correct) < 0:
        raise ValueError("Review sample limits must be nonnegative")
    cohorts = defaultdict(list)
    for item in originals:
        image = item["image"]
        for k, task in enumerate(TASKS):
            target = item["targets"][k]
            if target < 0:
                continue
            probability = scores[image][k]
            prediction = int(probability >= thresholds[task])
            outcome = ("TP" if prediction else "FN") if target else ("FP" if prediction else "TN")
            cohorts[(item["domain"], task, outcome)].append((item, probability))
    selected, counts = {}, []
    rng = random.Random(seed)
    for (domain, task, outcome), rows in sorted(cohorts.items()):
        limit = per_error if outcome in ("FP", "FN") else per_correct
        # Half of each error cohort shows severe mistakes; the rest covers the cohort.
        severe = sorted(rows, key=lambda pair: ((-pair[1]) if outcome in ("FP", "TP") else pair[1], pair[0]["image"]))
        hard_count = min(len(rows), (limit + 1) // 2) if outcome in ("FP", "FN") else 0
        chosen = severe[:hard_count]
        chosen_names = {i["image"] for i, _ in chosen}
        remaining = sorted([p for p in rows if p[0]["image"] not in chosen_names], key=lambda p: p[0]["image"])
        chosen += rng.sample(remaining, min(len(remaining), max(0, limit - len(chosen))))
        counts.append({"domain": domain, "task": task, "outcome": outcome, "available": len(rows), "sampled": len(chosen)})
        for item, probability in chosen:
            entry = selected.setdefault(item["image"], {"image": item["image"], "domain": domain,
                "original_targets": list(item["targets"]), "probabilities": list(scores[item["image"]]), "selected_for": []})
            entry["selected_for"].append({"task": task, "outcome": outcome, "probability": probability, "threshold": thresholds[task]})
    return list(selected.values()), counts


def relative_url(path, output):
    return quote(Path(os.path.relpath(path, output)).as_posix(), safe="/")


def dacl_annotation(root, record):
    source = local_path(root, "data/" + record["source"].replace("\\", "/"))
    annotation = source.parent.parent.parent / "annotations/train" / f"{source.stem}.json"
    if not annotation.resolve().is_relative_to(root.resolve()) or not annotation.is_file():
        raise ValueError("Missing original TRAIN DACL polygon annotation")
    return annotation, read(annotation)


def enrich_cases(cases, canonical, records, root, output):
    code_tags = {}
    for category in ("background", "defects"):
        path = root / "data/codebrim/source/classification_dataset/metadata" / f"{category}.xml"
        if path.is_file():
            for node in ET.parse(path).getroot():
                if node.tag == "Defect":
                    code_tags[(category, node.attrib["name"])] = (path, {c.tag: int(c.text) for c in node})
    for index, case in enumerate(cases, 1):
        image = local_path(root, case["image"])
        if not image.is_file():
            raise ValueError("Missing selected TRAIN image")
        case.update(case_id=f"train-review-{index:04d}", image_sha256=sha(image), image_url=relative_url(image, output),
                    original_source=canonical[case["image"]].get("source"), proposed_review={"reason": "unreviewed", "note": ""})
        if case["original_source"]:
            original_source = local_path(root, case["original_source"])
            if not original_source.is_file():
                raise ValueError("Missing selected publisher TRAIN photo")
            case["original_source_sha256"] = sha(original_source)
            case["original_source_url"] = relative_url(original_source, output)
        case["source_tags"] = []
        if case["domain"] == "dacl":
            derived_label = root / "data/dacl10k-yolo/labels/train" / (image.stem + ".txt")
            case["derived_training_labels"] = {"path": derived_label.relative_to(root).as_posix(), "sha256": sha(derived_label)}
            annotation, document = dacl_annotation(root, records[case["image"]])
            case["annotation"] = {"path": annotation.relative_to(root).as_posix(), "sha256": sha(annotation),
                "url": relative_url(annotation, output), "kind": "publisher_polygons",
                "imageWidth": document["imageWidth"], "imageHeight": document["imageHeight"],
                "shapes": [{"label": shape["label"], "points": shape["points"]} for shape in document["shapes"]]}
            case["source_tags"] = sorted({s["label"] for s in document["shapes"]})
            width, height = document["imageWidth"], document["imageHeight"]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (width, height)):
                raise ValueError("Invalid source annotation dimensions")
            for shape in case["annotation"]["shapes"]:
                if any(len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) for p in shape["points"]):
                    raise ValueError("Invalid publisher polygon coordinate")
        elif case["domain"] == "damsegment":
            source = Path(canonical[case["image"]]["source"])
            if "Damage Detection" in source.as_posix():
                level = {"E": "Easy", "M": "Medium", "H": "Hard"}[source.stem[0]]
                mask = root / "data/damsegment/source/Damage Segmentaion" / level / "Labels/Mask" / f"{source.stem}_mask.png"
                if not mask.is_file():
                    raise ValueError("Missing original Dam TRAIN mask")
                case["annotation"] = {"path": mask.relative_to(root).as_posix(), "sha256": sha(mask),
                                      "url": relative_url(mask, output), "kind": "publisher_mask"}
            else:
                case["annotation"] = {"kind": "publisher_photo_category_only", "note": "원본 사진 분류 정답만 있으며 양성 위치 주석 없음"}
        elif case["domain"] == "codebrim":
            source = Path(canonical[case["image"]]["source"])
            path, tags = code_tags[(source.parent.name, source.name)]
            case["annotation"] = {"path": path.relative_to(root).as_posix(), "sha256": sha(path),
                                  "url": relative_url(path, output), "kind": "publisher_xml_tags", "tags": tags}
            case["source_tags"] = sorted(k for k, v in tags.items() if v)
    return cases


def overlay(case):
    annotation = case.get("annotation", {})
    source = html.escape(case["image_url"], quote=True)
    if annotation.get("kind") == "publisher_polygons":
        shapes = []
        colors = {"Crack": "#ff5b5b", "ACrack": "#ff5b5b", "Spalling": "#33ddff"}
        for shape in annotation["shapes"]:
            if len(shape["points"]) < 3:
                continue
            points = " ".join(f"{x:g},{y:g}" for x, y in shape["points"])
            color = colors.get(shape["label"], "#ffd36c")
            shapes.append(f'<polygon points="{points}" fill="{color}" fill-opacity=".13" stroke="{color}" stroke-width="2"><title>{html.escape(shape["label"])}</title></polygon>')
        width, height = annotation["imageWidth"], annotation["imageHeight"]
        return f'<svg viewBox="0 0 {width:g} {height:g}" role="img" aria-label="원본 주석 겹쳐 보기"><image href="{source}" width="{width:g}" height="{height:g}" preserveAspectRatio="none"/>{"".join(shapes)}</svg><small>빨강: 균열, 하늘색: 박락, 노랑: 다른 원본 태그. 폴리곤은 원본 정답이며 예측 위치가 아닙니다.</small>'
    if annotation.get("kind") == "publisher_mask":
        return f'<img loading="lazy" src="{html.escape(annotation["url"], quote=True)}" alt="원본 손상 마스크"><small>원본 색상 마스크: 빨강 균열, 파랑 박락. 예측 위치가 아닙니다.</small>'
    return f'<p>{html.escape(annotation.get("note", "위치 주석 없음. 원본 사진 태그만 제공됩니다."))}</p>'


def render_html(package):
    cards = []
    for case in package["cases"]:
        e = lambda value: html.escape(str(value), quote=True)
        outcomes = " · ".join(f'{LABELS[s["task"]]} {s["outcome"]}' for s in case["selected_for"])
        rows = "".join(f'<tr><td>{e(LABELS[c])}</td><td>{ {1:"있음",0:"없음",-1:"미확인"}[case["original_targets"][k]] }</td><td>{case["probabilities"][k]:.4f}</td><td>{package["thresholds"].get(c, "미설정")}</td></tr>' for k, c in enumerate(CLASSES))
        annotation = case.get("annotation", {})
        link = f'<a href="{e(annotation["url"])}">원본 주석 열기</a>' if "url" in annotation else "원본 위치 주석 없음"
        original_link = f' · <a href="{e(case["original_source_url"])}">출판자 원본 사진 열기</a>' if case.get("original_source_url") else ""
        cards.append(f'''<article data-domain="{e(case["domain"])}" data-status="{e(outcomes)}" id="{e(case["case_id"])}">
<h2>{e(case["case_id"])} · {e(case["domain"])} · {e(outcomes)}</h2><p class="path">{e(case["image"])}{original_link}</p>
<div class="images"><figure><img loading="lazy" src="{e(case["image_url"])}" alt="변경하지 않은 TRAIN 사진"><figcaption>모델에 입력한 원본 TRAIN 사진</figcaption></figure><figure>{overlay(case)}<figcaption>{link}</figcaption></figure></div>
<table><thead><tr><th>항목</th><th>원래 정답</th><th>모델 확률</th><th>오류 분류 기준</th></tr></thead><tbody>{rows}</tbody></table>
<p>원본 태그: {e(", ".join(case["source_tags"]) or "별도 세부 태그 없음")}<br><small>태그가 함께 나타난다는 사실만으로 오류 원인을 확정할 수 없습니다.</small></p>
<label>검수 제안 <select class="reason"><option value="unreviewed">검수 전</option><option value="small_damage">작은 손상 / 세부 정보 부족</option><option value="other_damage_confusion">다른 손상과 혼동 의심</option><option value="capture_quality">촬영 상태 문제</option><option value="annotation_uncertain">정답 기준 불명확 / 추가 검토</option><option value="model_error">명확한 모델 오류</option><option value="correct_comparison">정답 일치 비교 사례</option><option value="other">기타</option></select></label><label> 근거 메모 <textarea class="note" placeholder="정답을 자동 변경하지 않습니다. 관찰한 사실과 의견을 구분해 기록하세요."></textarea></label>
<small>사진 SHA256 {e(case["image_sha256"])}<br>원본 주석 SHA256 {e(annotation.get("sha256", "해당 없음"))}</small></article>''')
    summary = "".join(f'<tr><td>{html.escape(r["domain"])}</td><td>{LABELS[r["task"]]}</td><td>{r["outcome"]}</td><td>{r["available"]}</td><td>{r["sampled"]}</td></tr>' for r in package["sampling"])
    # No source images in the HTML; original relative references remain local.
    metadata = json.dumps({"schema": "facility_train_review_proposals_v1", "source_package_sha256": package["package_content_sha256"], "weights_sha256": package["weights_sha256"], "policy": POLICY}, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TRAIN 오류 검수</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;margin:24px auto;max-width:1200px;background:#f3f5f8;color:#17232e;padding:0 16px}}article,header{{background:white;padding:22px;border:1px solid #c8d1da;border-radius:12px;margin:20px 0}}h1{{font-size:28px}}h2{{font-size:21px}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0}}img,svg{{width:100%;max-height:520px;object-fit:contain;background:#13232e}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:6px}}small{{color:#485d70;overflow-wrap:anywhere}}.path{{overflow-wrap:anywhere}}textarea{{display:block;width:95%;min-height:70px}}select,button{{padding:8px}}.warning{{border-left:5px solid #bd5b09;padding-left:12px}}@media(max-width:700px){{.images{{grid-template-columns:1fr}}}}</style>
<header><h1>균열·박락 TRAIN 오류 검수</h1><p>{POLICY}</p><p class="warning">{LICENSE}</p><p>이 사진들은 학습에 사용된 자료입니다. 이 검수표로 현장 성능·시험 정확도를 주장할 수 없습니다. FP=없는데 있다고 제안, FN=있는데 미검출, TP/TN=정답 일치 비교.</p><p>확률은 모델 출력이며, 정확도를 보장하는 신뢰도 수치가 아닙니다. 다른 5종에는 오류 분류 기준을 설정하지 않았습니다.</p><p>선택된 사진 {len(package["cases"])}장 · 모델 SHA256 <small>{package["weights_sha256"]}</small></p>
<details><summary>표본별 전체 TRAIN 수와 검수 선택 수</summary><table><tr><th>자료</th><th>항목</th><th>종류</th><th>전체 TRAIN</th><th>선택</th></tr>{summary}</table><p>오류 집단은 절반을 강한 오답, 나머지를 고정 난수로 선택합니다. 정답 일치 비교는 난수 선택입니다. 동일 사진은 한 번만 보여 주며 여러 선택 사유를 기록합니다. 이것은 전체 TRAIN 오류율 계산표가 아닙니다.</p></details>
<button id="export">검수 의견 JSON 저장</button><p><small>입력은 브라우저 메모리에만 있습니다. 창을 닫기 전에 저장하세요. 저장 파일은 의견이며 자동 정답 변경·학습은 하지 않습니다.</small></p></header>
{"".join(cards)}
<script>const metadata={metadata};document.getElementById('export').onclick=()=>{{const cases=Array.from(document.querySelectorAll('article')).map(a=>({{case_id:a.id,reason:a.querySelector('.reason').value,note:a.querySelector('.note').value}}));const blob=new Blob([JSON.stringify({{...metadata,cases}},null,2)],{{type:'application/json;charset=utf-8'}});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='TRAIN-review-proposals.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}};</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/facility-spatial-training/train.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/facility-presence-target-auxiliary/best.pt")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "runs/facility-presence-target-auxiliary/TARGET-SELECTION.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/facility-train-review")
    parser.add_argument("--per-error", type=int, default=12)
    parser.add_argument("--per-correct", type=int, default=3)
    parser.add_argument("--seed", type=int, default=49)
    args = parser.parse_args()
    for name in ("cache", "manifest", "weights", "thresholds", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Local review output must remain inside ignored ai-training/runs")
    if args.output.exists():
        raise ValueError("Preserve existing review material; choose a new output directory")
    canonical, records, sources = canonical_training(ROOT)
    details_path = ROOT / "data/facility-detail-training/manifest.json"
    sources[details_path.relative_to(ROOT).as_posix()] = sha(details_path)
    originals, all_items = validate_training_manifest(read(args.manifest), canonical, read(details_path))
    cache = read(args.cache)
    if cache.get("manifest_sha256", sha(args.manifest)) != sha(args.manifest):
        raise ValueError("Scoring manifest provenance changed")
    if "split_sha256" in cache and cache["split_sha256"] != sha(args.weights.parent / "SPLIT.json"):
        raise ValueError("Scoring source TRAIN split provenance changed")
    # The scoring cache may intentionally score only full original records.
    permitted = originals if set(cache.get("images", [])) == {i["image"] for i in originals} else all_items
    weights_sha = sha(args.weights)
    scores = verify_cache(cache, permitted, CLASSES, weights_sha)
    selection = read(args.thresholds)
    if selection.get("weights_sha256") != weights_sha or selection.get("classes") != CLASSES or selection.get("selection_split") != "val" or selection["selected"]["grid"] != 1:
        raise ValueError("Threshold provenance/model/view mismatch; never refit on review rows")
    thresholds = {task: selection["selected"]["per_class"][task]["threshold"] for task in TASKS}
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in thresholds.values()):
        raise ValueError("Invalid frozen review thresholds")
    cases, sampling = sample_cases(originals, scores, thresholds, args.per_error, args.per_correct, args.seed)
    cases = enrich_cases(cases, canonical, records, ROOT, args.output)
    package = {"schema": "facility_train_review_v1", "split": "train", "created_utc": datetime.now(timezone.utc).isoformat(),
        "classes": CLASSES, "weights_sha256": weights_sha, "thresholds": thresholds, "policy": POLICY, "license": LICENSE,
        "scope": "TRAIN review only; frozen VAL cutoffs label review errors, no VAL/TEST prediction rows or original annotation changes",
        "original_full_train_count": len(originals), "scored_cache_count": len(scores), "detail_rows_not_reviewed": len(scores)-len(originals),
        "cases": cases, "sampling": sampling, "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "provenance": {"cache_sha256": sha(args.cache), "manifest_sha256": sha(args.manifest), "thresholds_sha256": sha(args.thresholds),
                       "script_sha256": sha(Path(__file__)), "membership_sources": sources},
        "review_note": "Original tags overlap; co-occurrence is not a demonstrated cause. Proposed reviews remain separate from original_targets."}
    package["package_content_sha256"] = hashlib.sha256(json.dumps(package, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    args.output.mkdir(parents=True)
    json_path = args.output / "TRAIN-REVIEW.json"
    json_path.write_text(json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "index.html").write_text(render_html(package), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_photos": len(cases), "TRAIN-REVIEW.json_sha256": sha(json_path),
                      "index.html_sha256": sha(args.output / "index.html"), "policy": POLICY}, ensure_ascii=False))


if __name__ == "__main__":
    main()
