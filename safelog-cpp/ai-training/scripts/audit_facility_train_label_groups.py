"""Aggregate frozen TRAIN tag composition; no images, inference or label edits."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import combinations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/facility-train-label-groups.json"
TAGS = ("Crack", "ACrack", "Spalling", "Rockpocket", "WConccor", "Hollowareas", "Cavity")
OTHER_FOUR = frozenset(TAGS[3:])
CRACK_TAGS = frozenset(TAGS[:2])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(path):
    before = path.stat()
    value = digest(path.read_bytes())
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "An audit input changed while being read")
    return {"sha256": value, "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns}


def audit(root=ROOT):
    root = Path(root).resolve()
    paths = {
        "records": root / "data/dacl10k-yolo/records.json",
        "preparation": root / "data/dacl10k-yolo/PREPARATION.json",
        "spatial": root / "data/facility-spatial-training/train.json",
        "detail": root / "data/facility-detail-training/manifest.json",
        "auxiliary": root / "data/facility-auxiliary-training/train.json",
        "source_registry": root / "datasets/facility_sources.json",
        "existing_source_audit": root / "reports/facility-resolution-source-audit.json",
        "existing_label_audit": root / "reports/facility-target-negative-data-audit.json",
        "audit_script": Path(__file__).resolve(),
    }
    before = {key: snapshot(path) for key, path in paths.items()}
    source_audit = read(paths["existing_source_audit"])
    label_audit = read(paths["existing_label_audit"])
    require(source_audit.get("status") == "complete" and source_audit.get("train_photos") == 6225,
            "The existing TRAIN source audit must be complete")
    require(label_audit.get("status") == "audited", "The existing native label audit must be complete")
    for key in ("records", "preparation", "spatial", "detail", "auxiliary"):
        require(before[key]["sha256"] == source_audit["manifest_sha256"].get(key),
                "A frozen source or TRAIN manifest changed")
    require(before["source_registry"]["sha256"] == label_audit["input_sha256"]["facility_source_registry"],
            "The frozen source registry changed")
    auxiliary = read(paths["auxiliary"])
    classes = auxiliary.get("classes", [])
    require(auxiliary.get("split") == "train" and classes == source_audit["native_classes"]
            and len(classes) == 19, "Original native19 TRAIN manifest contract changed")
    items = [item for item in auxiliary["items"] if item.get("domain") == "dacl"]
    require(len(items) == 6225 and len({item["annotation"] for item in items}) == 6225,
            "Original DACL TRAIN annotation references must be unique")
    source_root = (root / "data/dacl10k").resolve()

    def verify_annotation(item):
        relative = item["annotation"]
        require(item.get("split") == "train" and isinstance(relative, str)
                and "\\" not in relative and ".." not in Path(relative).parts,
                "Only confined original TRAIN annotation references are allowed")
        path = (root / relative).resolve()
        require(path.is_relative_to(source_root) and path.parent.name == "train"
                and path.parent.parent.name == "annotations", "Annotation is outside original TRAIN")
        evidence = snapshot(path)
        require(evidence["sha256"] == item.get("annotation_sha256"),
                "Original TRAIN annotation bytes changed")
        document = read(path)
        require(document.get("split") == "train", "Held-out native annotations are forbidden")
        tags = {shape["label"] for shape in document["shapes"]}
        require(not tags - set(classes) and [int(label in tags) for label in classes] == item["targets"],
                "Native19 TRAIN photo tags differ from the frozen auxiliary manifest")
        return tags, (relative, evidence["sha256"]), path, evidence

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(verify_annotation, items))
    counts = Counter()
    pairs = Counter()
    for tags, _, _, _ in rows:
        counts.update(tags)
        pairs.update(combinations([tag for tag in TAGS if tag in tags], 2))
    require(all(counts[label] == source_audit["native_label_photo_counts"][label] for label in classes),
            "Native TRAIN composition differs from the existing source audit")
    annotation_digest = digest(json.dumps(sorted(row[1] for row in rows), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    require(annotation_digest == label_audit["dacl_native19"]["source_annotations_aggregate_sha256"],
            "The original TRAIN annotation set differs from the existing label audit")
    tag_rows = [row[0] for row in rows]
    strata = {
        "Crack_without_ACrack": sum("Crack" in tags and "ACrack" not in tags for tags in tag_rows),
        "ACrack_without_Crack": sum("ACrack" in tags and "Crack" not in tags for tags in tag_rows),
        "Crack_and_ACrack": sum(CRACK_TAGS <= tags for tags in tag_rows),
        "Crack_or_ACrack": sum(bool(tags & CRACK_TAGS) for tags in tag_rows),
        "Spalling_without_other_four": sum("Spalling" in tags and not tags & OTHER_FOUR for tags in tag_rows),
        "Spalling_with_any_other_four": sum("Spalling" in tags and bool(tags & OTHER_FOUR) for tags in tag_rows),
        "related_other_four_without_Spalling": sum("Spalling" not in tags and bool(tags & OTHER_FOUR) for tags in tag_rows),
        "related_other_four_without_Spalling_or_crack": sum("Spalling" not in tags and not tags & CRACK_TAGS
            and bool(tags & OTHER_FOUR) for tags in tag_rows),
    }
    require(strata == {"Crack_without_ACrack": 1449, "ACrack_without_Crack": 209,
        "Crack_and_ACrack": 90, "Crack_or_ACrack": 1748,
        "Spalling_without_other_four": 1578, "Spalling_with_any_other_four": 1360,
        "related_other_four_without_Spalling": 666,
        "related_other_four_without_Spalling_or_crack": 421}, "Frozen TRAIN count sanity check failed")
    require(strata["Spalling_without_other_four"] + strata["Spalling_with_any_other_four"] == counts["Spalling"],
            "Spalling strata do not partition the original positive photos")
    after = {key: snapshot(path) for key, path in paths.items()}
    require(before == after, "Audit changed or observed changed manifest inputs")
    with ThreadPoolExecutor(max_workers=8) as pool:
        preserved = list(pool.map(lambda row: snapshot(row[2]) == row[3], rows))
    require(all(preserved), "Original TRAIN annotation SHA256/size/mtime changed during the audit")
    return {
        "schema": "facility_train_label_groups_v1", "status": "audited",
        "scope": "TRAIN dataset composition, not label error, polygon overlap or accuracy",
        "population": "Frozen internal TRAIN original DACL photos; publisher native split=train",
        "train_photos": len(rows), "native_tag_count": len(classes),
        "native_tag_photo_counts": {tag: counts[tag] for tag in TAGS},
        "native_tag_pairwise_photo_counts": {" + ".join(pair): pairs[pair] for pair in combinations(TAGS, 2)},
        "sampling_strata_photo_counts": strata,
        "source_manifest_sha256": {key: before[key]["sha256"] for key in
            ("records", "preparation", "spatial", "detail", "auxiliary", "source_registry")},
        "existing_public_audit_sha256": {key: before[key]["sha256"] for key in
            ("existing_source_audit", "existing_label_audit")},
        "native_train_annotations_rehashed": len(rows),
        "native_train_annotation_set_sha256": annotation_digest,
        "native19_targets_match_original_annotations": True,
        "all_input_sha256_size_mtime_preserved": True,
        "audit_script_sha256": before["audit_script"]["sha256"],
        "policy": {"native_validation_or_test_annotations_opened": False,
            "validation_or_test_images_read": False, "source_images_read": False,
            "source_photo_bytes_rehashed": False, "publisher_archive_bytes_rehashed": False,
            "model_predictions_used": False, "training_executed": False, "gpu_used": False,
            "source_annotations_or_manifests_modified": False,
            "individual_filenames_paths_or_target_arrays_published": False},
        "limitations": ["Pair counts indicate photo-level tag coexistence, not polygon intersection or incorrect labels.",
            "Crack-without-ACrack and ACrack-without-Crack do not assert absence of other defect tags.",
            "Other-four strata use Rockpocket, WConccor, Hollowareas and Cavity; their target absence is not normal or safe ground truth.",
            "Source image byte verification is reused from the bound existing source audit; this audit reads annotation bytes only."],
    }


if __name__ == "__main__":
    require(not OUTPUT.exists(), "Preserve the existing diagnostic report")
    result = audit()
    OUTPUT.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: result[key] for key in ("status", "train_photos",
        "native_train_annotations_rehashed", "all_input_sha256_size_mtime_preserved")}, ensure_ascii=False))
