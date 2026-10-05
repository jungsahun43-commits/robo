"""Unconfirmed AI visual observations remain separate from human review opinions."""
from collections import Counter
from copy import deepcopy

from safelog_ai.review_feedback import (TASKS, REASONS, canonical_sha256, validate_package,
                                       _keys, _text, _utc)

SCHEMA = "facility_ai_train_observations_v1"


def validate_ai_observations(document, package):
    package = validate_package(package)
    canonical_sha256(document)
    _keys(document, ("schema", "source_package_sha256", "weights_sha256", "observer_type",
                    "review_status", "created_utc", "cases", "expert_confirmed_labels",
                    "label_changes", "training_applied"), "AI observations")
    if document["schema"] != SCHEMA or document["observer_type"] != "ai" or document["review_status"] != "unconfirmed":
        raise ValueError("AI observations must remain unconfirmed, separate from human opinions")
    if (document["source_package_sha256"] != package["package_content_sha256"]
            or document["weights_sha256"] != package["weights_sha256"]):
        raise ValueError("AI observations belong to another package or model")
    for key in ("expert_confirmed_labels", "label_changes", "training_applied"):
        if type(document[key]) is not int or document[key] != 0:
            raise ValueError("AI observations cannot confirm labels or apply training")
    _utc(document["created_utc"], "AI observation timestamp")
    cases = document["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("AI observation cases must be a nonempty list")
    source = {case["case_id"]: case for case in package["cases"]}
    seen = set()
    for row in cases:
        _keys(row, ("case_id", "image_sha256", "annotation_sha256", "input_photo_viewed",
                    "annotation_metadata_read", "publisher_mask_viewed", "facts", "hypotheses",
                    "questions", "tasks"), "AI case")
        identifier = row["case_id"]
        if not isinstance(identifier, str) or identifier not in source or identifier in seen:
            raise ValueError("Unknown or duplicate AI observation case")
        seen.add(identifier)
        original = source[identifier]
        annotation = original.get("annotation", {})
        if (row["image_sha256"] != original["image_sha256"]
                or row["annotation_sha256"] != annotation.get("sha256")):
            raise ValueError("AI observation photo or annotation identity changed")
        if row["input_photo_viewed"] is not True or row["annotation_metadata_read"] is not True:
            raise ValueError("AI observations must declare actual photo viewing and metadata reading")
        if type(row["publisher_mask_viewed"]) is not bool:
            raise ValueError("Mask viewing must be a boolean")
        if row["publisher_mask_viewed"] and annotation.get("kind") != "publisher_mask":
            raise ValueError("No publisher pixel mask exists for this case")
        for key in ("facts", "hypotheses", "questions"):
            values = row[key]
            if not isinstance(values, list) or len(values) > 12 or (key == "facts" and not values):
                raise ValueError("AI notes must have observed facts and bounded separate text lists")
            for value in values:
                _text(value, 2000, f"AI {key}", required=True)
        tasks = row["tasks"]
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= 2:
            raise ValueError("AI observation needs one or two separate task notes")
        task_seen = set()
        for task in tasks:
            _keys(task, ("task", "visual_judgement", "reason"), "AI task")
            if task["task"] not in TASKS or task["task"] in task_seen:
                raise ValueError("Unknown or duplicate AI task")
            task_seen.add(task["task"])
            if task["visual_judgement"] not in ("present", "absent", "uncertain"):
                raise ValueError("AI visual opinion is not a confirmed label")
            if task["reason"] not in REASONS or task["reason"] == "unreviewed":
                raise ValueError("Unknown AI observation reason")
    return deepcopy(document)


def summarize_ai_observations(document, package):
    document = validate_ai_observations(document, package)
    source = {case["case_id"]: case for case in package["cases"]}
    domains = Counter(source[row["case_id"]]["domain"] for row in document["cases"])
    reasons, judgements, tasks = Counter(), Counter(), Counter()
    for row in document["cases"]:
        for note in row["tasks"]:
            reasons[note["reason"]] += 1
            judgements[note["visual_judgement"]] += 1
            tasks[note["task"]] += 1
    return {"schema": "facility_ai_train_observation_summary_v1",
            "source_package_sha256": document["source_package_sha256"],
            "weights_sha256": document["weights_sha256"],
            "observation_content_sha256": canonical_sha256(document),
            "review_status": "unconfirmed", "observer_type": "ai",
            "observed_cases": len(document["cases"]), "by_source": dict(sorted(domains.items())),
            "observed_task_opinions": sum(tasks.values()), "by_task": dict(sorted(tasks.items())),
            "by_reason_task_opinions": dict(sorted(reasons.items())),
            "by_visual_judgement": dict(sorted(judgements.items())),
            "native_masks_reported_viewed": sum(row["publisher_mask_viewed"] for row in document["cases"]),
            "cases_with_follow_up_questions": sum(bool(row["questions"]) for row in document["cases"]),
            "human_review_opinions_generated": 0, "expert_confirmed_labels": 0,
            "label_changes": 0, "training_applied": 0,
            "scope": "AI hypotheses from selected TRAIN cases; not error prevalence, expert truth or field accuracy"}
