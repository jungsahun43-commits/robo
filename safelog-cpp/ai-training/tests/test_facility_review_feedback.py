"""Synthetic review intake boundaries; no photo, model or real review data reads."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from safelog_ai.review_feedback import (
    CLASSES, TASKS, PACKAGE_SCHEMA, FEEDBACK_SCHEMA, LEGACY_SCHEMA,
    aggregate_feedback, canonical_sha256, read_json, validate_feedback,
    validate_package,
)


class ReviewFeedbackTests(unittest.TestCase):
    def package(self):
        package = {
            "schema": PACKAGE_SCHEMA,
            "split": "train",
            "classes": list(CLASSES),
            "weights_sha256": "a" * 64,
            "policy": "Opinions only; preserve original TRAIN targets.",
            "original_full_train_count": 2,
            "scored_cache_count": 2,
            "detail_rows_not_reviewed": 0,
            "provenance": {"cache_sha256": "b" * 64, "manifest_sha256": "c" * 64},
            "cases": [
                {"case_id": "train-review-0001", "domain": "dacl",
                 "image": "data/private-first-image.jpg", "image_sha256": "d" * 64,
                 "original_targets": [1, 0, -1, -1, -1, -1, -1],
                 "probabilities": [.2, .8, .3, .4, .5, .6, .7],
                 "selected_for": [{"task": TASKS[0], "outcome": "FN",
                                   "probability": .2, "threshold": .5}]},
                {"case_id": "train-review-0002", "domain": "codebrim",
                 "image": "data/private-second-image.jpg", "image_sha256": "e" * 64,
                 "original_targets": [0, 1, -1, -1, -1, -1, -1],
                 "probabilities": [.8, .2, .3, .4, .5, .6, .7],
                 "selected_for": [{"task": TASKS[1], "outcome": "FN",
                                   "probability": .2, "threshold": .5}]},
            ],
        }
        return self.rehash(package)

    @staticmethod
    def rehash(package):
        package.pop("package_content_sha256", None)
        package["package_content_sha256"] = hashlib.sha256(
            json.dumps(package, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return package

    @staticmethod
    def task(task, judgement="unreviewed", reason=None):
        reviewed = judgement != "unreviewed"
        return {"task": task, "judgement": judgement,
                "reason": reason or ("model_error" if reviewed else "unreviewed"),
                "note": "Private observed hairline surface evidence" if reviewed else "",
                "evidence": "Private source polygon observation" if reviewed else "",
                "reviewed_at": "2026-10-04T00:00:00Z" if reviewed else None}

    def feedback(self, package, reviewer_id="private-reviewer-one", role="team_observer"):
        return {"schema": FEEDBACK_SCHEMA,
                "source_package_sha256": package["package_content_sha256"],
                "weights_sha256": package["weights_sha256"],
                "exported_utc": "2026-10-04T01:00:00Z",
                "reviewer": {"reviewer_id": reviewer_id, "name": "Private Reviewer Name",
                             "role": role, "expertise": "Concrete inspection" if role == "domain_expert" else ""},
                "cases": [{"case_id": case["case_id"],
                           "tasks": [self.task(task) for task in TASKS]}
                          for case in package["cases"]]}

    def legacy(self, package):
        return {"schema": LEGACY_SCHEMA,
                "source_package_sha256": package["package_content_sha256"],
                "weights_sha256": package["weights_sha256"], "policy": package["policy"],
                "cases": [{"case_id": case["case_id"], "reason": "unreviewed", "note": ""}
                          for case in package["cases"]]}

    def test_canonical_hash_preserves_historical_spaces_and_unicode(self):
        value = {"나": [1, 2], "a": "검수"}
        expected = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        compact = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.assertEqual(canonical_sha256(value), expected)
        self.assertNotEqual(expected, compact)

    def test_json_reader_rejects_ambiguous_or_invalid_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_bytes('{"검수": [1, 2]}'.encode("utf-8"))
            self.assertEqual(read_json(path), {"검수": [1, 2]})
            for payload in (b'{"a": 1, "a": 2}', b'{"nested": {"a": 1, "a": 2}}',
                            b'{"a": NaN}', b'{"a": Infinity}', b'{"a": -Infinity}',
                            b'\xef\xbb\xbf{"a":1}', b'{"a":"\xff"}'):
                path.write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaises((ValueError, UnicodeError)):
                    read_json(path)

    def test_valid_package_copy_and_hash_guard_preserve_original_targets(self):
        package = self.package()
        snapshot = copy.deepcopy(package)
        accepted = validate_package(package)
        self.assertEqual(accepted, snapshot)
        self.assertIsNot(accepted, package)
        accepted["cases"][0]["original_targets"][0] = 0
        self.assertEqual(package, snapshot)
        mutated = copy.deepcopy(package)
        mutated["cases"][0]["original_targets"][0] = 0
        with self.assertRaises(ValueError):
            validate_package(mutated)

    def test_package_rejects_heldout_duplicate_ids_and_invalid_targets_even_if_rehashed(self):
        for mutation in ("val", "test", "heldout_case", "duplicate_id", "wrong_classes", "short_targets", "bool_target", "out_of_range", "not_finite"):
            package = self.package()
            if mutation in ("val", "test"):
                package["split"] = mutation
            elif mutation == "heldout_case":
                package["cases"][0]["split"] = "test"
            elif mutation == "duplicate_id":
                package["cases"][1]["case_id"] = package["cases"][0]["case_id"]
            elif mutation == "wrong_classes":
                package["classes"][0] = "general_safe"
            elif mutation == "short_targets":
                package["cases"][0]["original_targets"].pop()
            elif mutation == "bool_target":
                package["cases"][0]["original_targets"][0] = True
            elif mutation == "out_of_range":
                package["cases"][0]["original_targets"][0] = 2
            elif mutation == "not_finite":
                package["cases"][0]["probabilities"][0] = float("nan")
            self.rehash(package)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_package(package)

    def test_v2_roundtrip_has_deep_copies_and_never_changes_original_targets(self):
        package = self.package()
        feedback = self.feedback(package)
        feedback["cases"][0]["tasks"][0] = self.task(TASKS[0], "absent")
        package_snapshot, feedback_snapshot = copy.deepcopy(package), copy.deepcopy(feedback)
        accepted = validate_feedback(feedback, package)
        self.assertEqual(accepted, feedback)
        accepted["cases"][0]["tasks"][0]["note"] = "changed copy"
        self.assertEqual(feedback, feedback_snapshot)
        aggregate_feedback(package, [feedback])
        self.assertEqual(package, package_snapshot)
        self.assertEqual(feedback, feedback_snapshot)

    def test_feedback_rejects_invalid_complete_documents_atomically(self):
        mutations = ("unknown_id", "duplicate_case", "duplicate_task", "missing_task", "unknown_task",
                     "model_hash", "package_hash", "invalid_date", "naive_date", "non_utc_date", "export_date",
                     "blank_note", "wrong_role", "wrong_judgement", "wrong_reason", "extra_targets",
                     "extra_top_level", "dirty_unreviewed", "empty_cases")
        package = self.package()
        for mutation in mutations:
            feedback = self.feedback(package)
            row = feedback["cases"][1]["tasks"][0]
            row.update(self.task(TASKS[0], "present"))
            if mutation == "unknown_id": feedback["cases"][1]["case_id"] = "unlisted-case"
            elif mutation == "duplicate_case": feedback["cases"][1]["case_id"] = feedback["cases"][0]["case_id"]
            elif mutation == "duplicate_task": feedback["cases"][1]["tasks"][1]["task"] = TASKS[0]
            elif mutation == "missing_task": feedback["cases"][1]["tasks"].pop()
            elif mutation == "unknown_task": row["task"] = "whole_facility_safe"
            elif mutation == "model_hash": feedback["weights_sha256"] = "f" * 64
            elif mutation == "package_hash": feedback["source_package_sha256"] = "f" * 64
            elif mutation == "invalid_date": row["reviewed_at"] = "2026-02-30T00:00:00Z"
            elif mutation == "naive_date": row["reviewed_at"] = "2026-10-04T00:00:00"
            elif mutation == "non_utc_date": row["reviewed_at"] = "2026-10-04T09:00:00+09:00"
            elif mutation == "export_date": feedback["exported_utc"] = "not-a-date"
            elif mutation == "blank_note": row["note"] = "  "
            elif mutation == "wrong_role": feedback["reviewer"]["role"] = "certified_by_model"
            elif mutation == "wrong_judgement": row["judgement"] = "safe"
            elif mutation == "wrong_reason": row["reason"] = "original_label_wrong"
            elif mutation == "extra_targets": feedback["cases"][1]["original_targets"] = [0] * 7
            elif mutation == "extra_top_level": feedback["is_expert_verified"] = True
            elif mutation == "dirty_unreviewed": feedback["cases"][0]["tasks"][0]["note"] = "not blank"
            elif mutation == "empty_cases": feedback["cases"] = []
            snapshot = copy.deepcopy(feedback)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_feedback(feedback, package)
            self.assertEqual(feedback, snapshot)
            with self.subTest(aggregate_mutation=mutation), self.assertRaises(ValueError):
                aggregate_feedback(package, [self.feedback(package, "valid-other-reviewer"), feedback])

    def test_reviewed_expert_requires_expertise_and_each_task_evidence(self):
        package = self.package()
        feedback = self.feedback(package, role="domain_expert")
        feedback["cases"][0]["tasks"][0] = self.task(TASKS[0], "uncertain", "annotation_uncertain")
        self.assertEqual(validate_feedback(feedback, package), feedback)
        for mutation in ("expertise", "evidence"):
            invalid = copy.deepcopy(feedback)
            if mutation == "expertise": invalid["reviewer"]["expertise"] = "  "
            else: invalid["cases"][0]["tasks"][0]["evidence"] = "  "
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_feedback(invalid, package)

    def test_exact_duplicates_are_idempotent_but_different_reviewer_snapshots_fail(self):
        package = self.package()
        feedback = self.feedback(package)
        feedback["cases"][0]["tasks"][0] = self.task(TASKS[0], "present")
        once = aggregate_feedback(package, [feedback])
        twice = aggregate_feedback(package, [feedback, copy.deepcopy(feedback)])
        self.assertEqual(once["summary"]["submitted_unique_feedback_documents"], 1)
        self.assertEqual(twice["summary"]["duplicate_files_ignored"], 1)
        once_counts, twice_counts = copy.deepcopy(once["summary"]), copy.deepcopy(twice["summary"])
        once_counts.pop("duplicate_files_ignored")
        twice_counts.pop("duplicate_files_ignored")
        self.assertEqual(once_counts, twice_counts)
        self.assertEqual(once["details"], twice["details"])
        changed = copy.deepcopy(feedback)
        changed["cases"][0]["tasks"][0]["note"] += " later snapshot"
        with self.assertRaises(ValueError):
            aggregate_feedback(package, [feedback, changed])

    def test_legacy_requires_opt_in_and_rejects_added_task_truth(self):
        package = self.package()
        legacy = self.legacy(package)
        legacy["cases"][0].update(reason="capture_quality", note="Private blur observation")
        with self.assertRaises(ValueError):
            validate_feedback(legacy, package)
        with self.assertRaises(ValueError):
            aggregate_feedback(package, [legacy])
        self.assertEqual(validate_feedback(legacy, package, allow_legacy=True), legacy)
        for mutation in ("tasks", "reviewer", "targets"):
            invalid = copy.deepcopy(legacy)
            if mutation == "tasks": invalid["cases"][0]["tasks"] = [self.task(TASKS[0], "present")]
            elif mutation == "reviewer": invalid["reviewer"] = {"role": "domain_expert"}
            else: invalid["cases"][0]["targets"] = [1] * 7
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_feedback(invalid, package, allow_legacy=True)

    def test_counts_separate_team_observations_and_declared_experts_without_label_confirmation(self):
        package = self.package()
        team = self.feedback(package)
        team["cases"][0]["tasks"][0] = self.task(TASKS[0], "absent")
        expert = self.feedback(package, "private-declared-expert", "domain_expert")
        expert["cases"][0]["tasks"][0] = self.task(TASKS[0], "uncertain", "annotation_uncertain")
        expert["cases"][1]["tasks"][1] = self.task(TASKS[1], "absent")
        summary = aggregate_feedback(package, [team, expert])["summary"]
        self.assertEqual(summary["reviewer_count_by_role"], {"team_observer": 1, "domain_expert": 1})
        self.assertEqual(summary["reviewed_task_opinions"], 3)
        self.assertEqual(summary["reviewed_unique_cases"], 2)
        self.assertEqual(summary["reviewed_unique_case_tasks"], 2)
        self.assertEqual(summary["uncertain_task_opinions"], 1)
        self.assertEqual(summary["judgements_by_role"]["team_observer"]["absent"], 1)
        self.assertEqual(summary["judgements_by_role"]["domain_expert"]["uncertain"], 1)
        self.assertEqual(summary["disagreement_with_original_by_role"], {"team_observer": 1, "domain_expert": 1})
        self.assertTrue(summary["expertise_self_declared"])
        self.assertFalse(summary["expert_identity_verified"])
        for field in ("reviewer_identity_authenticated", "package_origin_authenticated", "model_weights_file_verified"):
            self.assertFalse(summary[field])
        for field in ("expert_confirmed_labels", "label_changes", "training_applied"):
            self.assertEqual(summary[field], 0)

    def test_decisive_unknown_original_is_an_observation_not_a_false_negative_or_new_label(self):
        package = self.package()
        package["cases"][0]["original_targets"][0] = -1
        self.rehash(package)
        before = copy.deepcopy(package)
        expert = self.feedback(package, "private-declared-expert", "domain_expert")
        expert["cases"][0]["tasks"][0] = self.task(TASKS[0], "present")
        result = aggregate_feedback(package, [expert])
        self.assertEqual(result["summary"]["decisive_opinions_on_unknown_original_by_role"]["domain_expert"], 1)
        self.assertEqual(result["summary"]["disagreement_with_original_by_role"]["domain_expert"], 0)
        self.assertEqual(result["details"]["original_disagreements"], [])
        self.assertEqual(result["summary"]["expert_confirmed_labels"], 0)
        self.assertEqual(package, before)

    def test_independent_reviewers_conflict_without_vote_resolution_or_target_mutation(self):
        package = self.package()
        before = copy.deepcopy(package)
        team = self.feedback(package)
        team["cases"][0]["tasks"][0] = self.task(TASKS[0], "present")
        expert = self.feedback(package, "private-declared-expert", "domain_expert")
        expert["cases"][0]["tasks"][0] = self.task(TASKS[0], "absent")
        result = aggregate_feedback(package, [team, expert])
        self.assertEqual(result["summary"]["cross_reviewer_conflicting_case_tasks"], 1)
        self.assertEqual(result["summary"]["conflict_role_opinions_by_role"], {"team_observer": 1, "domain_expert": 1})
        self.assertEqual(result["summary"]["conflicts_by_role_pair"]["domain_expert+team_observer"], 1)
        conflict = result["details"]["conflicts"][0]
        self.assertEqual((conflict["case_id"], conflict["task"], conflict["resolution"]),
                         (package["cases"][0]["case_id"], TASKS[0], "unresolved"))
        self.assertEqual({row["judgement"] for row in conflict["opinions"]}, {"present", "absent"})
        self.assertEqual(len({row["reviewer_id"] for row in conflict["opinions"]}), 2)
        self.assertEqual(result["summary"]["label_changes"], 0)
        self.assertEqual(package, before)

    def test_partial_case_submission_counts_remaining_package_tasks_as_unresolved(self):
        package = self.package()
        feedback = self.feedback(package)
        feedback["cases"] = feedback["cases"][:1]
        feedback["cases"][0]["tasks"][0] = self.task(TASKS[0], "present")
        summary = aggregate_feedback(package, [feedback])["summary"]
        self.assertEqual(summary["package_tasks"], 4)
        self.assertEqual(summary["reviewed_unique_case_tasks"], 1)
        self.assertEqual(summary["unresolved_unique_case_tasks"], 3)

    def test_legacy_observations_are_never_new_task_or_expert_opinions(self):
        package = self.package()
        legacy = self.legacy(package)
        legacy["cases"][0].update(reason="capture_quality", note="Private blur observation")
        summary = aggregate_feedback(package, [legacy], allow_legacy=True)["summary"]
        self.assertEqual(summary["legacy"]["documents"], 1)
        self.assertEqual(summary["legacy"]["case_level_reviewed_observations"], 1)
        self.assertEqual(summary["legacy"]["unreviewed_rows"], 1)
        self.assertTrue(summary["legacy"]["excluded_from_task_opinions"])
        self.assertFalse(summary["legacy"]["reviewer_identity_available"])
        self.assertEqual(summary["legacy"]["expert_confirmed_labels"], 0)
        self.assertEqual(summary["reviewer_count_by_role"], {"team_observer": 0, "domain_expert": 0})
        self.assertEqual(summary["reviewed_task_opinions"], 0)
        self.assertEqual(summary["reviewed_unique_case_tasks"], 0)
        self.assertEqual(summary["unresolved_unique_case_tasks"], 4)

    def test_public_summary_has_aggregate_counts_and_no_private_identity_notes_or_paths(self):
        package = self.package()
        team = self.feedback(package)
        team["cases"][0]["tasks"][0] = self.task(TASKS[0], "absent")
        expert = self.feedback(package, "private-declared-expert", "domain_expert")
        expert["cases"][0]["tasks"][0] = self.task(TASKS[0], "present")
        result = aggregate_feedback(package, [team, expert])
        public = json.dumps(result["summary"], ensure_ascii=False)
        private = json.dumps(result["details"], ensure_ascii=False)
        sensitive = [case["case_id"] for case in package["cases"]]
        sensitive += [case["image"] for case in package["cases"]]
        sensitive += [team["reviewer"]["name"], team["reviewer"]["reviewer_id"],
                      expert["reviewer"]["reviewer_id"], expert["reviewer"]["expertise"],
                      team["cases"][0]["tasks"][0]["note"], team["cases"][0]["tasks"][0]["evidence"]]
        for value in sensitive:
            with self.subTest(value=value):
                self.assertNotIn(value, public)
        self.assertIn(team["reviewer"]["name"], private)
        self.assertIn(team["cases"][0]["case_id"], private)
        self.assertTrue(result["details"]["local_only"])

    def test_review_timestamps_use_full_microsecond_precision_and_equivalent_utc_forms(self):
        package = self.package()
        feedback = self.feedback(package)
        feedback["cases"][0]["tasks"][0] = self.task(TASKS[0], "present")
        feedback["exported_utc"] = "2026-10-04T00:00:00.000500Z"
        feedback["cases"][0]["tasks"][0]["reviewed_at"] = "2026-10-04T00:00:00.000500+00:00"
        self.assertEqual(validate_feedback(feedback, package), feedback)
        feedback["cases"][0]["tasks"][0]["reviewed_at"] = "2026-10-04T00:00:00.000499Z"
        self.assertEqual(validate_feedback(feedback, package), feedback)
        for mutation in ("later_microsecond", "seven_review_fraction_digits", "seven_export_fraction_digits"):
            invalid = copy.deepcopy(feedback)
            if mutation == "later_microsecond":
                invalid["cases"][0]["tasks"][0]["reviewed_at"] = "2026-10-04T00:00:00.000501Z"
            elif mutation == "seven_review_fraction_digits":
                invalid["cases"][0]["tasks"][0]["reviewed_at"] = "2026-10-04T00:00:00.0004999Z"
            else:
                invalid["exported_utc"] = "2026-10-04T00:00:00.0005000Z"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_feedback(invalid, package)

    def test_reviewer_identity_rejects_padding_and_embedded_ascii_controls(self):
        package = self.package()
        for identifier in (" padded", "padded ", "\tpadded", "padded\n",
                           "private\x00identity", "private\x1fidentity", "private\x7fidentity"):
            feedback = self.feedback(package, reviewer_id=identifier)
            with self.subTest(identifier=repr(identifier)), self.assertRaises(ValueError):
                validate_feedback(feedback, package)
        for field in ("name", "expertise"):
            for control in ("\x00", "\x1f", "\x7f", "\n"):
                feedback = self.feedback(package)
                feedback["reviewer"][field] = "Private" + control + "metadata"
                with self.subTest(field=field, control=repr(control)), self.assertRaises(ValueError):
                    validate_feedback(feedback, package)

    def test_observation_notes_and_evidence_keep_multiline_content(self):
        package = self.package()
        feedback = self.feedback(package, role="domain_expert")
        row = self.task(TASKS[0], "uncertain", "annotation_uncertain")
        row["note"] = "Observed boundary\nMaterial uncertain\nAdditional angle needed"
        row["evidence"] = "Publisher polygon\nBoundary observation"
        feedback["cases"][0]["tasks"][0] = row
        self.assertEqual(validate_feedback(feedback, package), feedback)


if __name__ == "__main__":
    unittest.main()
