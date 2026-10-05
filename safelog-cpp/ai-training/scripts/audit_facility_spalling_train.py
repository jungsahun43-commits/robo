"""Read-only TRAIN spalling evidence audit, with private cases and public counts.

This reuses saved 0773 photo scores, publisher annotations and existing masks.
No model inference, training, new labels or expert decisions are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.review_feedback import CLASSES, TASKS, read_json, validate_package
from safelog_ai.spalling_geometry import (
    AMBIGUOUS_TAGS, SourceRaster, dacl_source_raster, dam_source_raster,
    unavailable_source_raster, geometry_metrics, texture_metrics,
)
from scripts.build_facility_train_review import (
    canonical_training, validate_training_manifest, verify_cache, vector,
)
from scripts.render_facility_spalling_audit import render_audit

INITIAL_SHA = "0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a"
PACKAGE = "runs/facility-train-review-roi-0773/TRAIN-REVIEW.json"
MANIFEST = "data/facility-spatial-training/train.json"
CACHE = "runs/facility-roi-0773-train-review-predictions.json"
WEIGHTS = "runs/facility-presence-target-roi-control/best.pt"
SELECTION = "runs/facility-presence-target-roi-control/TARGET-SELECTION.json"
OUTCOME_ORDER = ("FP", "FN", "TP", "TN", "UNKNOWN")
DOMAINS = ("dacl", "damsegment", "codebrim")
SOURCE_PATHS = ("scripts/audit_facility_spalling_train.py", "safelog_ai/spalling_geometry.py",
                "scripts/render_facility_spalling_audit.py", "scripts/prepare_facility_spatial.py",
                "scripts/build_facility_train_review.py", "safelog_ai/review_feedback.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative
            and not Path(relative).is_absolute() and ":" not in relative
            and not {"", ".", ".."}.intersection(relative.split("/")), "Invalid local relative path")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(), "Missing or escaping local input")
    return path


def snapshot(path):
    a = path.stat(); digest = sha(path); b = path.stat()
    require((a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns), "Input changed during hashing")
    return {"sha256": digest, "size_bytes": b.st_size, "mtime_ns": b.st_mtime_ns}


def outcome(target, probability, cutoff):
    if target == -1:
        return "UNKNOWN"
    found = probability >= cutoff
    return ("TP" if found else "FN") if target else ("FP" if found else "TN")


def cohort_rows(package, original_items, scores):
    """Bind all selected cases to unchanged full TRAIN rows and cached scores."""
    p = validate_package(package)
    require(p["weights_sha256"] == INITIAL_SHA, "Use the original immutable 0773 review model")
    thresholds = p.get("thresholds", {})
    require(set(thresholds) == set(TASKS) and all(type(v) in (int, float) and math.isfinite(v)
            and 0 <= v <= 1 for v in thresholds.values()), "Two frozen finite cutoffs required")
    require(isinstance(original_items, list) and original_items, "Original full TRAIN rows required")
    parents = {}
    for item in original_items:
        require(isinstance(item, dict) and isinstance(item.get("image"), str)
                and item.get("domain") in DOMAINS and item.get("split", "train") == "train"
                and item.get("parent_image") is None, "Only original full TRAIN rows are allowed")
        vector(item.get("targets"), 7)
        require(all(type(v) is int for v in item["targets"]), "TRAIN asserted targets must be integers")
        require(item["image"] not in parents, "Duplicate original TRAIN image")
        parents[item["image"]] = item
    require(isinstance(scores, dict) and set(scores) == set(parents), "Saved scores must cover full TRAIN exactly")
    rows = []; seen = set()
    for c in p["cases"]:
        image = c.get("image"); require(image in parents and image not in seen, "Duplicate or non-TRAIN case")
        seen.add(image); item = parents[image]
        require(c["domain"] == item["domain"] and c["original_targets"] == item["targets"],
                "Case source/labels differ from unchanged TRAIN")
        vector(scores[image], 7, probability=True)
        require(c.get("probabilities") == scores[image], "Case probabilities differ from original saved score")
        selected = c.get("selected_for")
        require(isinstance(selected, list) and selected, "Original selection reasons are required")
        selected_tasks = set()
        for reason in selected:
            require(isinstance(reason, dict) and reason.get("task") in TASKS
                    and reason["task"] not in selected_tasks, "Invalid or duplicate selection task")
            task = reason["task"]; selected_tasks.add(task); k = CLASSES.index(task)
            require(item["targets"][k] != -1 and reason.get("outcome") == outcome(
                    item["targets"][k], scores[image][k], thresholds[task])
                    and type(reason.get("probability")) in (int, float)
                    and type(reason.get("threshold")) in (int, float)
                    and reason["probability"] == scores[image][k]
                    and reason["threshold"] == thresholds[task], "Saved selection outcome/cutoff/score changed")
        rows.append({"case_id": c["case_id"], "image": image, "domain": c["domain"],
            "outcome": outcome(item["targets"][1], scores[image][1], thresholds[TASKS[1]]),
            "direct_spalling_selection": TASKS[1] in selected_tasks,
            "original_target": item["targets"][1], "probability": scores[image][1],
            "threshold": thresholds[TASKS[1]], "case": deepcopy(c), "item": deepcopy(item)})
    return rows


def replay_supervision(source, targets):
    """Reproduce original known flags; source absence and unknown stay distinct."""
    require(isinstance(source, SourceRaster), "SourceRaster required")
    source.__post_init__()  # Also detect an array changed after construction.
    require(isinstance(targets, list) and len(targets) == 7
            and all(type(t) is int and t in (-1, 0, 1) for t in targets), "Seven unchanged integer photo tags required")
    known = np.asarray([int(t == 0) for t in targets], dtype=np.uint8)
    expected = np.zeros((7, 80, 80), dtype=np.uint8); conflicts = []
    for k, target in enumerate(targets):
        if not source.class_available[k] or target == -1:
            continue
        if bool(source.any_masks[k].any()) != bool(target):
            known[k] = 0; conflicts.append(k)
        else:
            known[k] = 1; expected[k] = source.any_masks[k]
    return expected, known, conflicts


def describe(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return {"measured_cases": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None}
    x = np.asarray(finite)
    q = np.quantile(x, [.25, .5, .75], method="linear")
    return {"measured_cases": len(x), "min": float(x.min()), "q1": float(q[0]),
            "median": float(q[1]), "q3": float(q[2]), "max": float(x.max())}


def cohort_counts(rows):
    return {d: {o: sum(r["domain"] == d and r["outcome"] == o for r in rows)
                for o in OUTCOME_ORDER} for d in DOMAINS}


def aggregate(rows):
    errors = [r for r in rows if r["outcome"] in ("FP", "FN")]
    groups = []
    for d in DOMAINS:
        for o in OUTCOME_ORDER:
            group = [r for r in rows if r["domain"] == d and r["outcome"] == o]
            if not group:
                continue
            positive = [r for r in group if r["original_target"] == 1 and r["geometry"]["raster_available"]]
            overlap = {tag: sum(tag in r["source_tags"] for r in group) for tag in AMBIGUOUS_TAGS}
            groups.append({"domain": d, "outcome": o, "selected_cases": len(group),
                "direct_spalling_selected_cases": sum(r["direct_spalling_selection"] for r in group),
                "spalling_pixel_known_cases": sum(r["pixel_supervision"]["known"] for r in group),
                "positive_spalling_source_geometry_cases": len(positive),
                "positive_geometry_small_below_one_percent": sum(r["geometry"]["fine_area_fraction"] < .01 for r in positive),
                "fine_spalling_area_fraction_positive_only": describe([r["geometry"]["fine_area_fraction"] for r in positive]),
                "coarse_any_area_fraction_positive_only": describe([r["geometry"]["coarse_any_area_fraction"] for r in positive]),
                "coarse_expansion_ratio_positive_only": describe([r["geometry"]["expansion_ratio"] for r in positive]),
                "thin_occupied_cell_fraction_positive_only": describe([r["geometry"]["thin_cells_le_eighth_fraction"] for r in positive]),
                "publisher_other_tag_cooccurrence": overlap,
                "whole_image_difference_energy": describe([r["texture"]["whole_image"]["finite_difference_energy"] for r in group]),
                "source_unmarked_difference_energy": describe([(r["texture"]["source_unmarked"] or {}).get("finite_difference_energy") for r in group]),
                "source_unmarked_laplacian_variance": describe([(r["texture"]["source_unmarked"] or {}).get("laplacian_variance") for r in group])})
    return {"case_counts": {"all": len(rows), "errors": len(errors),
                "direct_spalling_errors": sum(r["direct_spalling_selection"] for r in errors),
                "additional_other_task_errors": sum(not r["direct_spalling_selection"] for r in errors)},
            "all_package_case_counts": cohort_counts(rows),
            "direct_spalling_selected_case_counts": cohort_counts([r for r in rows if r["direct_spalling_selection"]]),
            "error_counts_by_domain": cohort_counts(errors), "groups": groups}


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/facility-spalling-train-audit")
    args = parser.parse_args(argv); output = args.output.resolve()
    require(output == (ROOT / "runs/facility-spalling-train-audit").resolve() and not output.exists(),
            "Use a new confined local audit directory; preserve previous evidence")
    public = ROOT / "reports/facility-spalling-train-audit.json"
    require(not public.exists(), "Preserve previous public aggregate")
    test_path = ROOT / "reports/facility-spalling-train-audit-tests.json"
    tests = read_json(test_path)
    require(tests.get("schema") == "facility_spalling_train_audit_tests_v1"
            and tests.get("status") == "passed" and tests.get("tests_run") == 14
            and tests.get("expected_tests_collected") == 14 and tests.get("before_after_sources_equal") is True
            and all(type(tests.get(k)) is int and tests[k] == 0 for k in ("failures", "errors", "skipped")),
            "Actual complete 14-test audit suite must pass")
    for name, expected in {**tests["source_sha256"], **tests["test_source_sha256"]}.items():
        require(sha(local(ROOT, name)) == expected, "Executed audit source/test bytes changed")
    p = validate_package(read_json(ROOT / PACKAGE))
    manifest = read_json(ROOT / MANIFEST); cache = read_json(ROOT / CACHE); selection = read_json(ROOT / SELECTION)
    require(sha(ROOT / WEIGHTS) == p["weights_sha256"] == INITIAL_SHA
            and selection["weights_sha256"] == INITIAL_SHA and selection["selected"]["grid"] == 1,
            "Original 0773 model/full-photo cutoffs required")
    for field, path in (("cache_sha256", CACHE), ("manifest_sha256", MANIFEST), ("thresholds_sha256", SELECTION)):
        require(p["provenance"][field] == sha(ROOT / path), "Package original input bytes changed")
    require(p["thresholds"] == {task: selection["selected"]["per_class"][task]["threshold"] for task in TASKS},
            "Package cutoffs differ from frozen selection")
    canonical, records, membership = canonical_training(ROOT)
    detail = read_json(ROOT / "data/facility-detail-training/manifest.json")
    originals, all_items = validate_training_manifest(manifest, canonical, detail)
    require(len(originals) == p["original_full_train_count"] == p["scored_cache_count"] == 14248,
            "Original full TRAIN membership changed")
    require(cache["manifest_sha256"] == sha(ROOT / MANIFEST), "Cache manifest binding changed")
    scores = verify_cache(cache, originals, CLASSES, INITIAL_SHA)
    for name, expected in p["provenance"]["membership_sources"].items():
        require(sha(local(ROOT, name)) == expected, "Original source membership bytes changed")
    base = cohort_rows(p, originals, scores)
    # Protect old experiment sources and model/application choices, without redoing inference.
    protocol_path = "reports/facility-native-roi-study-protocol.json"
    frozen = read_json(ROOT / protocol_path)
    protected_paths = set(frozen["source_sha256"]) | set(SOURCE_PATHS) | set(membership)
    protected_paths |= {PACKAGE, MANIFEST, CACHE, WEIGHTS, SELECTION, protocol_path,
                        "data/facility-detail-training/manifest.json", "data/facility-auxiliary-training/train.json",
                        "data/dacl10k-yolo/PREPARATION.json", "reports/facility-inference-profile.json"}
    protected_paths |= set(tests["source_sha256"]) | set(tests["test_source_sha256"])
    protected_paths.add("reports/facility-spalling-train-audit-tests.json")
    for name, expected in frozen["source_sha256"].items():
        require(sha(local(ROOT, name)) == expected, "Previous frozen runtime source changed")
    for row in base:
        c, item = row["case"], row["item"]
        protected_paths.add(item["pixel_target"])
        for name, expected in ((c["image"], c["image_sha256"]),
                               (c["original_source"], c["original_source_sha256"])):
            require(sha(local(ROOT, name)) == expected, "Selected source/photo changed")
            protected_paths.add(name)
        for asset in (c.get("annotation", {}), c.get("derived_training_labels", {})):
            if "path" in asset:
                require(sha(local(ROOT, asset["path"])) == asset["sha256"], "Selected annotation/derived label changed")
                protected_paths.add(asset["path"])
    before = {name: snapshot(local(ROOT, name)) for name in sorted(protected_paths)}
    output.mkdir(); (output / "assets").mkdir()
    code_xml = {}; rows = []; pixel_replays = 0; conflicts = Counter()
    for number, row in enumerate(base, 1):
        c, item = row.pop("case"), row.pop("item")
        with Image.open(ROOT / c["image"]) as handle:
            processed_size = handle.size
            rgb = np.asarray(handle.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)).copy()
        annotation = c["annotation"]; kind = annotation["kind"]
        if kind == "publisher_polygons":
            doc = read_json(ROOT / annotation["path"])
            shapes = [{"label": s["label"], "points": s["points"]} for s in doc["shapes"]]
            require(doc.get("split") == "train" and shapes == annotation["shapes"]
                    and doc["imageWidth"] == annotation["imageWidth"] and doc["imageHeight"] == annotation["imageHeight"],
                    "Case polygon copy differs from publisher TRAIN annotation")
            with Image.open(ROOT / c["original_source"]) as handle:
                native_size = ImageOps.exif_transpose(handle).size
            require(native_size == (doc["imageWidth"], doc["imageHeight"]), "EXIF-corrected source annotation dimensions differ")
            source = dacl_source_raster(doc, processed_size)
        elif kind == "publisher_mask":
            with Image.open(ROOT / annotation["path"]) as handle:
                source = dam_source_raster(np.asarray(handle.convert("RGB")).copy())
        elif kind == "publisher_xml_tags":
            key = annotation["path"]
            if key not in code_xml:
                code_xml[key] = {node.attrib["name"]: {child.tag: int(child.text) for child in node}
                                 for node in ET.parse(ROOT / key).getroot() if node.tag == "Defect"}
            actual = code_xml[key].get(Path(c["original_source"]).name)
            require(actual == annotation["tags"] and actual["Spallation"] == row["original_target"],
                    "Original XML photo tags differ from reviewed TRAIN labels")
            source = unavailable_source_raster()
        elif kind == "publisher_photo_category_only":
            source = unavailable_source_raster()
        else:
            raise ValueError("Unknown original annotation kind")
        expected_mask, expected_known, conflict = replay_supervision(source, item["targets"])
        with np.load(ROOT / item["pixel_target"], allow_pickle=False) as archive:
            require(set(archive.files) == {"mask", "known"}, "Unexpected original pixel target fields")
            stored, known = archive["mask"], archive["known"]
            require(stored.shape == (7, 80, 80) and known.shape == (7,)
                    and np.isin(stored, [0, 1]).all() and np.isin(known, [0, 1]).all()
                    and np.array_equal(stored, expected_mask) and np.array_equal(known, expected_known),
                    "Original 80-mask/known flags do not match unchanged source recipe")
        pixel_replays += 1; conflicts.update(CLASSES[k] for k in conflict)
        geometry = geometry_metrics(source)
        textures = {"whole_image": texture_metrics(rgb),
                    "source_unmarked": texture_metrics(rgb, source.source_unmarked) if source.source_unmarked is not None else None}
        row.update(annotation_kind=kind, source_tags=deepcopy(c.get("source_tags", [])), geometry=geometry,
            texture=textures, pixel_supervision={"known": int(known[1]), "positive_cells": int(stored[1].sum())},
            original_source=c["original_source"], annotation_path=annotation.get("path"),
            image_sha256=c["image_sha256"], source_sha256=c["original_source_sha256"],
            original_pixel_target=item["pixel_target"], fine_mask_asset=None, coarse_mask_asset=None)
        observations = []
        if not geometry["raster_available"]:
            observations.append("원본 위치 주석 미확인: 빈 저장 마스크를 박락 면적0으로 해석하지 않음")
        if known[1] == 0:
            observations.append("박락 픽셀 손실 known=0: 위치 지도 학습에 사용하지 않은 항목")
        if row["original_target"] == 1 and geometry["raster_available"] and geometry["fine_area_fraction"] < .01:
            observations.append("이 변환의 원본 주석640 raster 점유율이1% 미만: 실제 물리적 크기나 원인 판정이 아님")
        if row["outcome"] == "FP" and any(tag in row["source_tags"] for tag in AMBIGUOUS_TAGS):
            observations.append("박락 음성 사진에 다른 출판자 손상 태그 공존: 모델 혼동·오라벨 원인 확정이 아님")
        row["observations"] = observations
        if source.class_available[1]:
            name = f"assets/fine-{number:04d}.png"
            Image.fromarray(np.uint8(source.fine_masks[1]) * 255).save(output / name)
            row["fine_mask_asset"] = name
        if known[1] == 1:
            name = f"assets/coarse-{number:04d}.png"
            Image.fromarray(np.uint8(stored[1]) * 255).resize((640, 640), Image.Resampling.NEAREST).save(output / name)
            row["coarse_mask_asset"] = name
        rows.append(row)
        if number % 25 == 0:
            print(json.dumps({"audit_cases_processed": number, "total": len(base)}), flush=True)
    after = {name: snapshot(local(ROOT, name)) for name in before}
    require(before == after, "Read-only audit changed an original input/source/model/application file")
    counts = aggregate(rows)
    details = {"schema": "facility_spalling_train_audit_v1", "split": "train",
        "package_sha256": sha(ROOT / PACKAGE), "package_content_sha256": p["package_content_sha256"],
        "weights_sha256": INITIAL_SHA, "threshold": p["thresholds"][TASKS[1]], **counts,
        "scope": "Biased 0773 TRAIN130 review; source-region/known-mask/descriptive features, not causal or accuracy evidence",
        "review_workbench": "runs/facility-train-review-roi-0773/review-workbench.html", "cases": rows,
        "provenance": {"input_snapshots_before": before, "input_snapshots_after": after}}
    private = output / "SPALLING-AUDIT.json"; write(private, details)
    page = render_audit(details, output, ROOT)
    with (output / "SPALLING-AUDIT.html").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(page)
    result = {"schema": "facility_spalling_train_audit_aggregate_v1", "status": "audited",
        "audited_utc": datetime.now(timezone.utc).isoformat(), "split": "train", **counts,
        "model_weights_sha256": INITIAL_SHA, "frozen_spalling_threshold": p["thresholds"][TASKS[1]],
        "package_file_sha256": sha(ROOT / PACKAGE), "package_content_sha256": p["package_content_sha256"],
        "private_details_sha256": sha(private), "local_html_sha256": sha(output / "SPALLING-AUDIT.html"),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_PATHS},
        "test_results_sha256": sha(test_path), "tests_run": tests["tests_run"],
        "original_inputs_checked": len(before), "all_original_sha_size_mtime_preserved": True,
        "pixel_mask_known_replays_verified": pixel_replays, "photo_pixel_presence_conflicts": dict(conflicts),
        "new_training_epochs": 0, "model_inference_executed": False, "heldout_images_opened": 0,
        "accuracy_measured": False, "labels_modified": 0, "expert_confirmed_labels": 0,
        "observer_decisions": 0, "expert_decisions": 0, "causes_confirmed": 0,
        "individual_paths_ids_coordinates_or_scores_published": False,
        "background_policy": "Source-unmarked pixels are not healthy background; absent photo tags are not expert safety decisions.",
        "raster_policy": "Source at processed-parent resolution -> BOX640 >0 -> any8x8 ->80. Texture on BILINEAR RGB640, no diagnostic cutoffs.",
        "selection_policy": "Existing seed49 source/task/outcome sample; half hard errors, half random; proportions are not population error rates.",
        "unknown_policy": "No location annotation is unknown, not zero area. CODEBRIM positive spalling has pixel-known0.",
        "scope": details["scope"]}
    write(public, result)
    print(json.dumps({"status": "audited", **result["case_counts"], "mask_replays": pixel_replays,
                      "protected_inputs": len(before), "new_training_epochs": 0, "label_changes": 0}))
    return result


if __name__ == "__main__":
    main()
