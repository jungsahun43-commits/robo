"""Read-only TRAIN label/provenance audit; writes only aggregate public reports.

No predictions, photographs, pixel arrays or sample selection are used. A shared
split index contains heldout label summaries, but only TRAIN labels contribute.
Native heldout annotations are not opened. Target absence is not field safety.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar",
           "wet_surface", "efflorescence", "surface_cavity")
AUX_CLASSES = ("ACrack", "Bearing", "Cavity", "Crack", "Drainage", "EJoint", "Efflorescence",
               "ExposedRebars", "Graffiti", "Hollowareas", "JTape", "PEquipment", "Restformwork",
               "Rockpocket", "Rust", "Spalling", "WConccor", "Weathering", "Wetspot")
MAPPING = {"ACrack": "concrete_crack", "Crack": "concrete_crack", "Spalling": "concrete_spalling",
           "Rust": "rust_stain", "ExposedRebars": "exposed_rebar", "Wetspot": "wet_surface",
           "Efflorescence": "efflorescence", "Cavity": "surface_cavity"}
AUTHOR_OBJECTS = {"Bearing", "EJoint", "Drainage", "PEquipment", "JTape", "WConccor"}
PROXY_GROUPS = {"linear_joint_tags": {"EJoint", "JTape"},
                "surface_texture_other_tags": {"Weathering", "Hollowareas", "Rockpocket", "WConccor"},
                "graffiti_tag": {"Graffiti"}}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def validate_targets(rows, classes=CLASSES):
    for row in rows:
        targets = row.get("targets")
        if (not isinstance(targets, list) or len(targets) != len(classes)
                or any(type(t) is not int or t not in (-1, 0, 1) for t in targets)):
            raise ValueError("Invalid partial-label contract; never coerce unknown to normal")


def summarize(rows):
    validate_targets(rows)
    double_negative = [row for row in rows if row["targets"][:2] == [0, 0]]
    return {"rows": len(rows),
            "per_label": {c: {"positive": sum(r["targets"][k] == 1 for r in rows),
                               "negative": sum(r["targets"][k] == 0 for r in rows),
                               "unknown": sum(r["targets"][k] == -1 for r in rows)} for k, c in enumerate(CLASSES)},
            "crack_spall_joint_counts": dict(sorted(Counter(f"{r['targets'][0]},{r['targets'][1]}" for r in rows).items())),
            "both_targets_known_negative": len(double_negative),
            "both_negative_with_other_mapped_positive": sum(any(t == 1 for t in r["targets"][2:]) for r in double_negative),
            "other_mapped_positives_inside_both_negative": {c: sum(r["targets"][k] == 1 for r in double_negative) for k, c in enumerate(CLASSES) if k >= 2},
            "both_negative_with_any_other_unknown": sum(-1 in r["targets"][2:] for r in double_negative),
            "all_seven_known_negative": sum(r["targets"] == [0] * 7 for r in rows),
            "general_normal_or_safe_ground_truth_asserted": False}


def annotation_tags(item):
    raw = item["annotation"]
    if not isinstance(raw, str) or "\\" in raw or ".." in Path(raw).parts:
        raise ValueError("Native TRAIN annotation path must be a confined relative POSIX path")
    path = (ROOT / raw).resolve()
    source_root = (ROOT / "data/dacl10k").resolve()
    if not path.is_relative_to(source_root) or not path.is_file():
        raise ValueError("Native DACL annotation outside existing source")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != item["annotation_sha256"]:
        raise ValueError("Native TRAIN annotation SHA changed")
    doc = json.loads(payload.decode("utf-8"))
    if doc.get("split") != "train" or item.get("split") != "train" or item.get("domain") != "dacl":
        raise ValueError("Only original DACL TRAIN annotations may be audited")
    tags = {shape["label"] for shape in doc["shapes"]}
    if tags - set(AUX_CLASSES) or [int(c in tags) for c in AUX_CLASSES] != item["targets"]:
        raise ValueError("Original nineteen-class photo tags differ from auxiliary manifest")
    mapped = {MAPPING[tag] for tag in tags if tag in MAPPING}
    return tags, [int(c in mapped) for c in CLASSES], (raw, digest)


def audit():
    paths = {"core_spatial_train": ROOT / "data/facility-spatial-training/train.json",
             "detail_train": ROOT / "data/facility-detail-training/manifest.json",
             "auxiliary_train": ROOT / "data/facility-auxiliary-training/train.json",
             "dacl_source_records": ROOT / "data/dacl10k-yolo/records.json",
             "codebrim_train": ROOT / "data/codebrim-training/train.json",
             "codebrim_license": ROOT / "data/codebrim/license.md",
             "damsegment_source_registry": ROOT / "datasets/damsegment_source.json",
             "damsegment_article_xml": ROOT / "data/damsegment/article.xml",
             "facility_source_registry": ROOT / "datasets/facility_sources.json",
             "preparation_spatial_script": ROOT / "scripts/prepare_facility_spatial.py",
             "preparation_auxiliary_script": ROOT / "scripts/prepare_facility_auxiliary.py",
             "preparation_detail_script": ROOT / "scripts/prepare_facility_detail.py"}
    digests = {key: sha(path) for key, path in paths.items()}
    manifest = read(paths["core_spatial_train"])
    auxiliary = read(paths["auxiliary_train"])
    details = read(paths["detail_train"])
    code = read(paths["codebrim_train"])
    if (manifest["split"] != "train" or manifest["classes"] != list(CLASSES)
            or auxiliary["split"] != "train" or auxiliary["classes"] != list(AUX_CLASSES)
            or auxiliary["audit"]["status"] != "prepared"
            or details["split"] != "train" or details["classes"] != list(CLASSES)
            or code["split"] != "train" or code["classes"] != list(CLASSES)):
        raise ValueError("TRAIN class/split contract changed")
    full_count = manifest["full_count"]
    if type(full_count) is not int or not 0 < full_count <= len(manifest["items"]):
        raise ValueError("Invalid base/full row boundary")
    full, crops = manifest["items"][:full_count], manifest["items"][full_count:]
    if any(r["domain"] not in ("dacl", "damsegment", "codebrim") for r in manifest["items"]):
        raise ValueError("Unexpected original source domain")
    if any(r.get("target_split") != "train" for r in full if r["domain"] == "damsegment"):
        raise ValueError("Dam full parent is not TRAIN")
    if not full or len({r["image"] for r in manifest["items"]}) != len(manifest["items"]):
        raise ValueError("TRAIN row identity ambiguous")
    parents = {r["image"]: r for r in full}
    if any(r["parent_split"] != "train" or r["parent_image"] not in parents for r in crops):
        raise ValueError("Crop parent is not an existing full TRAIN row")
    if any(r["domain"] != parents[r["parent_image"]]["domain"] for r in crops):
        raise ValueError("Crop and parent source domain differ")
    if len(crops) != len(details["items"]) or any({k: v for k, v in r.items() if k != "pixel_target"} != d for r, d in zip(crops, details["items"])):
        raise ValueError("Core crops differ from original detail manifest")
    dacl_full = [r for r in full if r["domain"] == "dacl"]
    aux_by_image = {r["image"]: r for r in auxiliary["items"]}
    if len(aux_by_image) != len(auxiliary["items"]) or set(aux_by_image) != {r["image"] for r in dacl_full}:
        raise ValueError("Native auxiliary tags must cover exactly original DACL full TRAIN photos")
    # The shared records JSON is parsed, including heldout boxes/ignored-label
    # summaries. Only TRAIN identities contribute; no heldout summary is used
    # to select rows, count TRAIN labels or change annotations.
    source_records = [r for r in read(paths["dacl_source_records"]) if r["split"] == "train"]
    if {r["stem"] for r in source_records} != {Path(r["image"]).stem for r in dacl_full}:
        raise ValueError("DACL source TRAIN identity mismatch")
    native_items = [aux_by_image[r["image"]] for r in dacl_full]
    with ThreadPoolExecutor(max_workers=4) as pool:
        native = list(pool.map(annotation_tags, native_items))
    if any(r["targets"] != result[1] for r, result in zip(dacl_full, native)):
        raise ValueError("Original native tags disagree with seven-class TRAIN photo labels")
    pairs = [(row, result[0]) for row, result in zip(dacl_full, native)]
    both_negative = [(row, tags) for row, tags in pairs if row["targets"][:2] == [0, 0]]
    tag_counts = {c: {"all_dacl_full_train": sum(c in tags for _, tags in pairs),
                      "crack_known_negative": sum(c in tags and r["targets"][0] == 0 for r, tags in pairs),
                      "spall_known_negative": sum(c in tags and r["targets"][1] == 0 for r, tags in pairs),
                      "both_targets_known_negative": sum(c in tags for _, tags in both_negative),
                      "all_seven_known_negative": sum(c in tags and r["targets"] == [0] * 7 for r, tags in pairs)} for c in AUX_CLASSES}
    proxies = {name: {"native_tags": sorted(tags), "full_train_both_negative_union_rows": sum(bool(tags & native_tags) for _, native_tags in both_negative),
                      "scope": "Existing native tag proxy only; does not assert paint, plaster or normal concrete"} for name, tags in PROXY_GROUPS.items()}
    all_proxy = set.union(*PROXY_GROUPS.values())
    proxies["all_proxy_union"] = {"native_tags": sorted(all_proxy), "full_train_both_negative_union_rows": sum(bool(all_proxy & tags) for _, tags in both_negative)}
    code_rows = [r for r in full if r["domain"] == "codebrim"]
    if [{k: v for k, v in r.items() if k != "pixel_target"} for r in code_rows] != code["items"]:
        raise ValueError("Core CODEBRIM TRAIN rows changed")
    code_background = [r for r in code_rows if "/background/" in r["source"]]
    if any(r["targets"] != [0, 0, 0, 0, -1, 0, -1] for r in code_background):
        raise ValueError("CODEBRIM background contract changed")
    if digests["codebrim_license"] != code["audit"]["license_sha256"]:
        raise ValueError("Original CODEBRIM license changed")
    dam_rows = [r for r in full if r["domain"] == "damsegment"]
    dam_noncrack = [r for r in dam_rows if "/Damage Classification/Non-Crack/" in r.get("source", "")]
    dam_native = [r for r in dam_rows if "/Damage Detection/" in r.get("source", "")]
    if len(dam_noncrack) + len(dam_native) != len(dam_rows):
        raise ValueError("Unexpected historical Dam TRAIN source scope")
    if any(r["targets"] != [0, 0, -1, -1, -1, -1, -1] for r in dam_noncrack):
        raise ValueError("Historical classification Non-Crack mapping changed")
    article = ET.parse(paths["damsegment_article_xml"]).getroot()
    article_dois = [e.text for e in article.iter("article-id") if e.attrib.get("pub-id-type") == "doi"]
    evidence = []
    for section in article.iter("sec"):
        title = section.find("title")
        if title is not None and " ".join(title.itertext()) == "Partitioning and refinement":
            evidence += [" ".join("".join(p.itertext()).split()) for p in section.findall("p")]
    if "10.1016/j.dib.2026.112671" not in article_dois or not any(
            "crack class includes images containing visible surface damage such as cracks and spalling" in p.lower()
            and "non-crack class represents intact concrete surfaces" in p.lower() for p in evidence):
        raise ValueError("Historical Non-Crack spall-negative needs the author's explicit intact definition, not folder-name inference")
    if {key: sha(path) for key, path in paths.items()} != digests:
        raise ValueError("Existing input metadata changed during read-only audit")
    sources = read(paths["facility_source_registry"])
    result = {"status": "audited", "scope": "Existing original TRAIN metadata and native DACL TRAIN annotations only; aggregate public output",
              "input_sha256": digests, "audit_script_sha256": sha(Path(__file__)),
              "classes": list(CLASSES), "full_base_rows": full_count, "derived_crop_rows": len(crops),
              "counts": {"full": summarize(full), "crop": summarize(crops), "all": summarize(manifest["items"])},
              "by_domain_and_row_type": {kind: {domain: summarize([r for r in rows if r["domain"] == domain]) for domain in ("dacl", "damsegment", "codebrim")} for kind, rows in (("full", full), ("crop", crops))},
              "dacl_native19": {"classes": list(AUX_CLASSES), "verified_train_annotations": len(native), "annotation_sha256_mismatches": 0,
                                "native_aux_target_mismatches": 0, "native_to_seven_target_mismatches": 0,
                                "source_annotations_aggregate_sha256": hashlib.sha256(canonical(sorted(result[2] for result in native))).hexdigest(),
                                "aggregate_hash_recipe": "SHA256 of canonical sorted (native TRAIN relative annotation path, SHA256) pairs; no individual paths published",
                                "native_tag_counts": tag_counts, "empty_native19_annotations": sum(not tags for _, tags in pairs),
                                "no_author13_damage_tag": sum(not (tags - AUTHOR_OBJECTS) for _, tags in pairs),
                                "author_object_tags_only_nonempty": sum(bool(tags) and tags <= AUTHOR_OBJECTS for _, tags in pairs),
                                "native_paint_label_available": False, "native_plaster_label_available": False,
                                "auxiliary_scope": "Only original DACL full TRAIN photos. No parent-tag propagation to crop or other domains; no seven-class output change."},
              "train_confusion_proxies": proxies,
              "codebrim_author_background_train": {"patch_rows": len(code_background), "source_parent_ids": len({r["parent_id"] for r in code_background}),
                                                  "five_source_defect_labels": "known absent", "wet_surface_and_surface_cavity": "unknown", "field_normal_or_safe": "not established"},
              "damsegment_train_label_provenance": {"full_base_rows": len(dam_rows), "native_detection_segmentation_rows": len(dam_native),
                                                     "native_detection_segmentation_both_negative": sum(r["targets"][:2] == [0, 0] for r in dam_native),
                                                     "classification_noncrack_rows": len(dam_noncrack), "classification_noncrack_both_negative": len(dam_noncrack),
                                                     "classification_separate_native_spall_annotation_or_mask": False,
                                                     "classification_photo_absence_basis": "Author paper section 4.4 defines Crack as visible damage including cracks and spalling, and Non-Crack as intact concrete; classification predictions manually verified. Both target negatives are a source-supported semantic mapping, not two separate native classification tags.",
                                                     "classification_pixel_negative_basis": "Inherited known photo absence supplies negative grid cells; no released native classification segmentation mask was inspected or claimed.",
                                                     "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13247583/", "paper_doi": "10.1016/j.dib.2026.112671", "paper_section": "4.4 Partitioning and refinement",
                                                     "paper_xml_sha256": digests["damsegment_article_xml"], "paper_license": "CC BY-NC 4.0; distinct from CC BY 4.0 dataset license",
                                                     "historical_train_and_heldout_manifests_changed": False,
                                                     "scope": "Only visible crack/spalling absence in publisher patches; other five targets remain unknown. Does not establish all damage absence or general structural safety."},
              "spatial_supervision_existing_metadata": {"per_label_pixel_cells": manifest["audit"]["per_label_pixel_cells"],
                                                         "photo_pixel_presence_conflicts_masked_unknown": manifest["audit"]["photo_pixel_presence_conflicts_masked_unknown"],
                                                         "verification_scope": "Copied from hashed prepared core audit; no pixel arrays rescanned in this label audit. Native polygons/masks for positives, asserted source-class absence for negatives; unknown supplies no supervision."},
              "source_provenance": {"dacl10k": {k: sources["dacl10k"][k] for k in ("title", "homepage", "license", "citation", "sha256")},
                                    "codebrim": {"source": code["audit"]["source"], "archive_sha256": code["audit"]["source_sha256"],
                                                 "license": code["audit"]["license"], "license_sha256": code["audit"]["license_sha256"]}},
              "policy": {"unknown_is_normal": False, "convid_labels_corrected": False, "new_train_examples_created": 0,
                         "shared_dacl_split_index_loaded": True, "shared_index_contains_heldout_label_summaries": True,
                         "shared_index_heldout_label_summaries_used": False,
                         "shared_index_scope": "Shared TRAIN/val/test records JSON, including boxes and ignored_labels, parsed; only TRAIN identities/labels contribute. No heldout record influences TRAIN counts, row selection or annotations.",
                         "native_validation_or_test_annotations_opened": False,
                         "validation_or_test_labels_used": False, "validation_or_test_images_read": False, "model_predictions_used": False,
                         "source_annotations_or_manifests_modified": False, "training_executed": False},
              "limitations": ["Target-negative means only that the author's named crack/spall label is absent; other defects, objects and unknown categories may be present.",
                              "Dam classification Non-Crack spall absence is supported by the author's intact-concrete definition, not an independent native per-class tag or classification mask. Native detection/segmentation and mapped classification evidence are distinct.",
                              "All-seven negative is narrower than general normal/safe facility ground truth. Even empty native19 annotation is not a prospective expert field safety decision.",
                              "Base full rows include publisher CODEBRIM patches, not 14,248 independent whole-site photographs. Existing crops add no independent photographs.",
                              "Graffiti, joints, weathering and other native tags are possible confusion proxies; they are not a paint/plaster annotation set.",
                              "ConViD's author folder cannot be repaired into paint/plaster labels without authoritative new annotations. Other six classes remain unknown; no corrections performed."]}
    return result


def main():
    result = audit()
    output = ROOT / "reports/facility-target-negative-data-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "full": result["full_base_rows"], "crop": result["derived_crop_rows"],
                      "full_both_targets_negative": result["counts"]["full"]["both_targets_known_negative"],
                      "native_train_annotations_verified": result["dacl_native19"]["verified_train_annotations"],
                      "report_sha256": sha(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
