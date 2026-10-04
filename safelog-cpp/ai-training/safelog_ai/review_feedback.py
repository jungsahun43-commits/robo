"""Validate separate TRAIN review opinions and produce counts, never training labels.

The publisher-derived package remains immutable. A reviewer role is a declaration,
not verified expertise. Multiple reviewers are kept separate; votes never relabel
a case and conflicting opinions are deliberately left unresolved.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

PACKAGE_SCHEMA = "facility_train_review_v1"
FEEDBACK_SCHEMA = "facility_train_review_feedback_v2"
LEGACY_SCHEMA = "facility_train_review_proposals_v1"
CLASSES = ["concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar",
           "wet_surface", "efflorescence", "surface_cavity"]
TASKS = CLASSES[:2]
ROLES = ("team_observer", "domain_expert")
JUDGEMENTS = ("unreviewed", "present", "absent", "uncertain")
REASONS = ("unreviewed", "small_damage", "texture_confusion", "other_damage_confusion",
           "capture_quality", "annotation_uncertain", "model_error", "correct_comparison", "other")
LEGACY_REASONS = tuple(reason for reason in REASONS if reason != "texture_confusion")
SOURCES = ("dacl", "damsegment", "codebrim")
POLICY = "검수 의견을 분리 보관합니다. 원본 정답 변경·자동 학습·앱 모델 교체는 하지 않습니다."
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)\Z")


def _reject_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def read_json(path):
    """Strict UTF-8 JSON: a BOM, repeated keys and NaN/Infinity are rejected."""
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("UTF-8 BOM is not accepted; save the JSON as UTF-8 without BOM")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_pairs,
                          parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid UTF-8 JSON document") from error


def canonical_sha256(value):
    """Match the historical package digest exactly, including default separators."""
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Document is not finite JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Invalid fields at {where}; expected {', '.join(expected)}")


def _text(value, maximum, where, required=False):
    if not isinstance(value, str) or len(value) > maximum or (required and not value.strip()):
        raise ValueError(f"Invalid text at {where}")
    return value


def _hash(value, where):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"Invalid SHA256 at {where}")


def _utc(value, where):
    if not isinstance(value, str) or not _UTC.fullmatch(value):
        raise ValueError(f"Expected a UTC ISO timestamp at {where}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid UTC timestamp at {where}") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"Expected UTC at {where}")
    return parsed


def validate_package(package):
    """Return an unchanged copy after checking the historical immutable package.

    No photo, annotation, split manifest, model or dataset is read by this module.
    The content hash binds all package fields, including its original targets.
    """
    if not isinstance(package, dict) or package.get("schema") != PACKAGE_SCHEMA:
        raise ValueError("Expected facility_train_review_v1 package")
    if package.get("split") != "train" or package.get("classes") != CLASSES:
        raise ValueError("Review package must be original seven-class TRAIN")
    _hash(package.get("weights_sha256"), "package.weights_sha256")
    _hash(package.get("package_content_sha256"), "package.package_content_sha256")
    unsigned = {key: value for key, value in package.items() if key != "package_content_sha256"}
    if canonical_sha256(unsigned) != package["package_content_sha256"]:
        raise ValueError("Review package content hash changed")
    cases = package.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Review package must contain cases")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Invalid package case")
        identifier = _text(case.get("case_id"), 200, "package.case_id", required=True)
        if identifier in seen:
            raise ValueError("Duplicate package case_id")
        seen.add(identifier)
        if case.get("domain") not in SOURCES:
            raise ValueError("Unknown review source")
        targets = case.get("original_targets")
        if (not isinstance(targets, list) or len(targets) != len(CLASSES)
                or any(type(value) is not int or value not in (-1, 0, 1) for value in targets)):
            raise ValueError("Original targets must be seven integer -1/0/1 values")
        if "probabilities" in case:
            probabilities = case["probabilities"]
            if (not isinstance(probabilities, list) or len(probabilities) != len(CLASSES)
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)):
                raise ValueError("Invalid original probabilities")
        if case.get("split", "train") != "train":
            raise ValueError("Held-out case in TRAIN review package")
    return deepcopy(package)


def _validate_metadata(document, package):
    if document.get("source_package_sha256") != package["package_content_sha256"]:
        raise ValueError("Feedback belongs to a different review package")
    if document.get("weights_sha256") != package["weights_sha256"]:
        raise ValueError("Feedback belongs to a different model")


def _validate_cases(cases, package_cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Feedback cases must be a nonempty list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Invalid feedback case")
        identifier = case.get("case_id")
        if not isinstance(identifier, str) or identifier not in package_cases:
            raise ValueError("Unknown feedback case_id")
        if identifier in seen:
            raise ValueError("Duplicate feedback case_id")
        seen.add(identifier)


def validate_feedback(feedback, package, *, allow_legacy=False):
    """Validate every row atomically; legacy observations need explicit opt-in.

    Task-specific opinions never replace the package's original seven targets.
    Unknown fields (including proposed target vectors) are rejected.
    """
    package = validate_package(package)
    if not isinstance(feedback, dict):
        raise ValueError("Feedback must be an object")
    canonical_sha256(feedback)  # Also reject nonfinite/non-JSON direct-call values.
    case_lookup = {case["case_id"]: case for case in package["cases"]}
    if feedback.get("schema") == LEGACY_SCHEMA:
        if not allow_legacy:
            raise ValueError("Legacy case-level proposals are accepted only with explicit opt-in")
        _keys(feedback, ("schema", "source_package_sha256", "weights_sha256", "policy", "cases"), "legacy")
        _validate_metadata(feedback, package)
        _text(feedback["policy"], 4000, "legacy.policy", required=True)
        _validate_cases(feedback["cases"], case_lookup)
        for case in feedback["cases"]:
            _keys(case, ("case_id", "reason", "note"), "legacy.case")
            if case["reason"] not in LEGACY_REASONS:
                raise ValueError("Invalid legacy case-level reason")
            _text(case["note"], 4000, "legacy.note")
            if case["reason"] == "unreviewed" and case["note"].strip():
                raise ValueError("An unreviewed legacy row must have a blank note")
            if case["reason"] != "unreviewed" and not case["note"].strip():
                raise ValueError("A reviewed legacy observation needs a note")
        return deepcopy(feedback)
    if feedback.get("schema") != FEEDBACK_SCHEMA:
        raise ValueError("Unsupported review feedback schema")
    _keys(feedback, ("schema", "source_package_sha256", "weights_sha256", "exported_utc", "reviewer", "cases"), "feedback")
    _validate_metadata(feedback, package)
    exported_at = _utc(feedback["exported_utc"], "feedback.exported_utc")
    reviewer = feedback["reviewer"]
    _keys(reviewer, ("reviewer_id", "name", "role", "expertise"), "reviewer")
    _text(reviewer["reviewer_id"], 200, "reviewer.reviewer_id", required=True)
    if reviewer["reviewer_id"] != reviewer["reviewer_id"].strip():
        raise ValueError("reviewer_id must not have leading or trailing whitespace")
    _text(reviewer["name"], 200, "reviewer.name", required=True)
    _text(reviewer["expertise"], 500, "reviewer.expertise")
    if any(any(ord(character) < 32 or ord(character) == 127 for character in reviewer[field])
           for field in ("reviewer_id", "name", "expertise")):
        raise ValueError("Reviewer single-line fields must not contain ASCII control characters")
    if reviewer["role"] not in ROLES:
        raise ValueError("Unknown reviewer role")
    _validate_cases(feedback["cases"], case_lookup)
    any_reviewed = False
    for case in feedback["cases"]:
        _keys(case, ("case_id", "tasks"), "feedback.case")
        tasks = case["tasks"]
        if not isinstance(tasks, list) or len(tasks) != len(TASKS):
            raise ValueError("Every included case requires exactly the two review tasks")
        seen = set()
        for row in tasks:
            _keys(row, ("task", "judgement", "reason", "note", "evidence", "reviewed_at"), "feedback.task")
            task = row["task"]
            if task not in TASKS or task in seen:
                raise ValueError("Unknown or duplicate feedback task")
            seen.add(task)
            if row["judgement"] not in JUDGEMENTS or row["reason"] not in REASONS:
                raise ValueError("Unknown task judgement or reason")
            _text(row["note"], 4000, "task.note")
            _text(row["evidence"], 4000, "task.evidence")
            if row["judgement"] == "unreviewed":
                if (row["reason"] != "unreviewed" or row["note"].strip()
                        or row["evidence"].strip() or row["reviewed_at"] is not None):
                    raise ValueError("Unreviewed tasks must have no judgement metadata")
            else:
                any_reviewed = True
                if row["reason"] == "unreviewed" or not row["note"].strip():
                    raise ValueError("Reviewed tasks need a reason and a nonblank observation note")
                reviewed_at = _utc(row["reviewed_at"], "task.reviewed_at")
                if reviewed_at > exported_at:
                    raise ValueError("Task reviewed_at cannot be later than exported_utc")
                if reviewer["role"] == "domain_expert" and not row["evidence"].strip():
                    raise ValueError("Declared expert opinions require per-task evidence")
        if seen != set(TASKS):
            raise ValueError("Review tasks must match the two fixed tasks")
    if any_reviewed and reviewer["role"] == "domain_expert" and not reviewer["expertise"].strip():
        raise ValueError("Declared expert opinions require an expertise description")
    return deepcopy(feedback)


def aggregate_feedback(package, feedback_documents, *, allow_legacy=False):
    """Return aggregate-only public counts plus separate private opinion details.

    Entire differing snapshots for the same reviewer are rejected. The caller
    must explicitly select the intended file; no timestamp-based overwrite occurs.
    """
    package = validate_package(package)
    if not isinstance(feedback_documents, (list, tuple)):
        raise ValueError("Feedback documents must be a sequence")
    case_lookup = {case["case_id"]: case for case in package["cases"]}
    normalized, legacy, duplicate_count = [], [], 0
    seen_digests, reviewer_snapshots = set(), {}
    for document in feedback_documents:
        validated = validate_feedback(document, package, allow_legacy=allow_legacy)
        digest = canonical_sha256(validated)
        if digest in seen_digests:
            duplicate_count += 1
            continue
        seen_digests.add(digest)
        if validated["schema"] == LEGACY_SCHEMA:
            legacy.append(validated)
            continue
        identifier = validated["reviewer"]["reviewer_id"]
        if identifier in reviewer_snapshots:
            raise ValueError("Different snapshots share a reviewer_id; explicitly choose one feedback file")
        reviewer_snapshots[identifier] = digest
        normalized.append(validated)

    role_counts = Counter({role: 0 for role in ROLES})
    judgements = {role: {judgement: 0 for judgement in JUDGEMENTS} for role in ROLES}
    disagreements_by_role = Counter({role: 0 for role in ROLES})
    unknown_original_by_role = Counter({role: 0 for role in ROLES})
    breakdown, opinions = Counter(), defaultdict(list)
    reviewed_cases, reviewed_case_tasks = set(), set()
    original_disagreements = []
    for document in normalized:
        reviewer = document["reviewer"]
        role = reviewer["role"]
        role_counts[role] += 1
        for case in document["cases"]:
            identifier = case["case_id"]
            source = case_lookup[identifier]["domain"]
            for row in case["tasks"]:
                judgement, task, reason = row["judgement"], row["task"], row["reason"]
                judgements[role][judgement] += 1
                breakdown[(role, reason, source, task)] += 1
                if judgement == "unreviewed":
                    continue
                reviewed_cases.add(identifier)
                reviewed_case_tasks.add((identifier, task))
                opinion = {"reviewer_id": reviewer["reviewer_id"], "role": role,
                           "judgement": judgement}
                opinions[(identifier, task)].append(opinion)
                target = case_lookup[identifier]["original_targets"][TASKS.index(task)]
                if judgement in ("present", "absent"):
                    if target == -1:
                        unknown_original_by_role[role] += 1
                    elif int(judgement == "present") != target:
                        disagreements_by_role[role] += 1
                        original_disagreements.append({"case_id": identifier, "task": task,
                            "original_target": target, **opinion})

    conflicts, unresolved = [], []
    conflict_roles = Counter({role: 0 for role in ROLES})
    role_pairs = Counter({"team_observer+team_observer": 0,
                          "domain_expert+team_observer": 0,
                          "domain_expert+domain_expert": 0})
    for identifier in sorted(case_lookup):
        for task in TASKS:
            rows = opinions[(identifier, task)]
            present = [row for row in rows if row["judgement"] == "present"]
            absent = [row for row in rows if row["judgement"] == "absent"]
            uncertain = any(row["judgement"] == "uncertain" for row in rows)
            conflicting = bool(present and absent)
            if conflicting:
                decisive = present + absent
                conflicts.append({"case_id": identifier, "task": task, "opinions": decisive,
                                  "resolution": "unresolved"})
                for row in decisive:
                    conflict_roles[row["role"]] += 1
                pairs = {"+".join(sorted((left["role"], right["role"])))
                         for left in present for right in absent}
                role_pairs.update(pairs)
            if conflicting or uncertain or not (present or absent):
                reasons = []
                if conflicting:
                    reasons.append("conflicting_present_absent")
                if uncertain:
                    reasons.append("uncertain_opinion")
                if not rows:
                    reasons.append("no_reviewed_opinion")
                unresolved.append({"case_id": identifier, "task": task, "reasons": reasons})

    legacy_counts, legacy_reviewed_cases = Counter(), set()
    legacy_reviewed, legacy_unreviewed = 0, 0
    for document in legacy:
        for case in document["cases"]:
            source = case_lookup[case["case_id"]]["domain"]
            legacy_counts[(case["reason"], source)] += 1
            if case["reason"] == "unreviewed":
                legacy_unreviewed += 1
            else:
                legacy_reviewed += 1
                legacy_reviewed_cases.add(case["case_id"])

    summary = {
        "schema": "facility_train_review_feedback_summary_v1",
        "source_package_sha256": package["package_content_sha256"],
        "weights_sha256": package["weights_sha256"],
        "scope": "TRAIN review opinions only; these counts are not model or app accuracy",
        "package_cases": len(case_lookup), "package_tasks": len(case_lookup) * len(TASKS),
        "submitted_unique_feedback_documents": len(normalized) + len(legacy),
        "duplicate_files_ignored": duplicate_count,
        "reviewer_count_by_role": dict(role_counts),
        "reviewed_unique_cases": len(reviewed_cases),
        "reviewed_unique_case_tasks": len(reviewed_case_tasks),
        "reviewed_task_opinions": sum(sum(counts[j] for j in JUDGEMENTS if j != "unreviewed")
                                      for counts in judgements.values()),
        "unreviewed_task_rows": sum(counts["unreviewed"] for counts in judgements.values()),
        "uncertain_task_opinions": sum(counts["uncertain"] for counts in judgements.values()),
        "unresolved_unique_case_tasks": len(unresolved),
        "judgements_by_role": judgements,
        "counts_by_role_reason_source_task": [
            {"role": role, "reason": reason, "source": source, "task": task, "count": count}
            for (role, reason, source, task), count in sorted(breakdown.items())],
        "disagreement_with_original_by_role": dict(disagreements_by_role),
        "decisive_opinions_on_unknown_original_by_role": dict(unknown_original_by_role),
        "cross_reviewer_conflicting_case_tasks": len(conflicts),
        "conflict_role_opinions_by_role": dict(conflict_roles),
        "conflicts_by_role_pair": dict(role_pairs),
        "expertise_self_declared": True, "expert_identity_verified": False,
        "reviewer_identity_authenticated": False,
        "package_origin_authenticated": False, "model_weights_file_verified": False,
        "expert_confirmed_labels": 0, "label_changes": 0, "training_applied": 0,
        "legacy": {
            "documents": len(legacy), "case_level_reviewed_observations": legacy_reviewed,
            "unreviewed_rows": legacy_unreviewed,
            "unique_reviewed_cases": len(legacy_reviewed_cases),
            "by_reason_source": [{"reason": reason, "source": source, "count": count}
                                 for (reason, source), count in sorted(legacy_counts.items())],
            "excluded_from_task_opinions": True,
            "reviewer_identity_available": False, "expert_confirmed_labels": 0},
        "policy": POLICY,
        "limitations": [
            "Reviewer roles and expertise descriptions are self-declared, not authenticated.",
            "Reviewer counts are distinct submitted self-declared IDs, not verified independent people.",
            "The package digest binds supplied content, not its publisher origin or an actual weights file.",
            "Opinion disagreement with publisher targets is not proof of a wrong label.",
            "Error-stratified TRAIN examples do not estimate dataset or factory error rates.",
            "Counts are opinions or unique case-task pairs as named; they are not independent photos.",
            "No votes, conflicts, legacy observations or expert declarations change training targets."],
    }
    details = {
        "schema": "facility_train_review_feedback_details_v1", "local_only": True,
        "source_package_sha256": package["package_content_sha256"],
        "weights_sha256": package["weights_sha256"],
        "normalized_reviews": normalized, "legacy_documents": legacy,
        "conflicts": conflicts, "unresolved_case_tasks": unresolved,
        "original_disagreements": original_disagreements, "policy": POLICY,
    }
    return {"summary": summary, "details": details}
