"""Measured resolution report contracts using synthetic counts only."""
import copy
import hashlib
import math
import unittest

from scripts.report_facility_resolution import (DOMAINS, TARGETS, error_rows, render,
                                               validate_histories, validate_technical_proof)


class ResolutionReportTests(unittest.TestCase):
    def history(self):
        rows = []
        for epoch in range(1, 7):
            rows.append({
                "epoch": epoch,
                "sampled_row_indices_sha256": hashlib.sha256(f"synthetic epoch {epoch}".encode()).hexdigest(),
                "sampled_domain_counts": {"dacl": 10000, "damsegment": 1400, "codebrim": 2848},
                "sampled_row_type_counts": {"full": 10848, "crop": 3400},
                "sampled_full_target_joint_counts": {
                    "dacl": {"00": 1700, "10": 1700, "01": 1700, "11": 1700, "unknown": 0},
                    "damsegment": {"00": 300, "10": 300, "01": 300, "11": 300, "unknown": 0},
                    "codebrim": {"00": 712, "10": 712, "01": 712, "11": 712, "unknown": 0}},
                "optimizer_step_diagnostics": {"attempted_batches": 1781,
                    "actual_optimizer_steps": 1781 - epoch, "amp_skipped_steps": epoch},
            })
        return rows

    def result(self):
        entries = []
        for index, (run, title, size, error) in enumerate((
                ("reference", "추가 학습 전 ROI 모델", 640, .2228),
                ("control", "640 대조군", 640, .23),
                ("highres", "960 보강군", 960, .225))):
            entry = {"run": run, "title": title, "imgsz": size,
                     "actual_epochs": 6, "best_epoch": index + 1,
                     "worst_error": error, "target_passed": False,
                     "small_dacl_polygon_area_below_one_percent": {
                         TARGETS[0]: {"false_negatives": 29 - index, "positive_photos": 93},
                         TARGETS[1]: {"false_negatives": 42, "positive_photos": 105}},
                     "per_class": {task: {"domains": {domain: {
                         "fn": 2, "positive_photos": 10, "fnr": .2,
                         "fp": 3, "negative_photos": 20, "fpr": .15,
                     } for domain in DOMAINS}} for task in TARGETS}}
            if index:
                entry["resources"] = {"elapsed_training_minutes": 18.5 if index == 1 else 29.25,
                    "peak_cuda_allocated_bytes": (2 if index == 1 else 3) * 1024 ** 3,
                    "optimizer_step_diagnostics": {"attempted_batches": 1781 * 6,
                        "actual_optimizer_steps": 1781 * 6 - 21, "amp_skipped_steps": 21}}
            entries.append(entry)
        return {"experiments": entries, "error_rows": error_rows(entries),
                "comparisons": {"maximum_error_treatment_minus_initializer_pp": .22,
                                "maximum_error_treatment_minus_control_pp": -.5},
                "research_gate": {"research_candidate_nominated": False},
                "deployed": False, "source_test_inference_executed": False,
                "additional_expert_confirmed_labels": 0, "label_changes": 0}

    def test_complete_six_epoch_pair_has_actual_budget_and_allows_different_amp_skips(self):
        control, highres = self.history(), self.history()
        for row in highres:
            row["optimizer_step_diagnostics"].update(actual_optimizer_steps=1771, amp_skipped_steps=10)
        before = copy.deepcopy((control, highres))
        proof = validate_histories(control, highres, 6, 14248)
        self.assertEqual(proof["epochs_compared"], 6)
        self.assertEqual(proof["draws_per_epoch"], 14248)
        self.assertTrue(proof["actual_ordered_row_index_hashes_identical_each_epoch"])
        self.assertTrue(proof["actual_source_full_crop_target_counts_identical_each_epoch"])
        self.assertEqual((control, highres), before)

    def test_incomplete_or_noninteger_epochs_and_corrupt_draw_proof_are_rejected(self):
        for mutation in ("short", "out_of_order", "boolean_epoch", "float_epoch", "bad_sha",
                         "missing_sha", "domain_budget", "crop_budget", "negative", "boolean_count",
                         "wrong_source", "wrong_row_type"):
            control, highres = self.history(), self.history()
            row = highres[0]
            if mutation == "short": highres.pop()
            elif mutation == "out_of_order": row["epoch"] = 2
            elif mutation == "boolean_epoch": row["epoch"] = True
            elif mutation == "float_epoch": row["epoch"] = 1.
            elif mutation == "bad_sha": row["sampled_row_indices_sha256"] = "not-a-sha"
            elif mutation == "missing_sha": row.pop("sampled_row_indices_sha256")
            elif mutation == "domain_budget": row["sampled_domain_counts"]["dacl"] += 1
            elif mutation == "crop_budget": row["sampled_row_type_counts"]["crop"] += 1
            elif mutation == "negative": row["sampled_domain_counts"]["damsegment"] = -1
            elif mutation == "boolean_count": row["sampled_row_type_counts"]["full"] = True
            elif mutation == "wrong_source": row["sampled_domain_counts"]["factory"] = row["sampled_domain_counts"].pop("dacl")
            elif mutation == "wrong_row_type": row["sampled_row_type_counts"]["tile"] = row["sampled_row_type_counts"].pop("crop")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_histories(control, highres, 6, 14248)

    def test_pair_order_source_type_or_original_target_state_drift_is_rejected(self):
        for mutation in ("row_order", "source", "full_crop", "known_joint_states"):
            control, highres = self.history(), self.history()
            row = highres[3]
            if mutation == "row_order": row["sampled_row_indices_sha256"] = "f" * 64
            elif mutation == "source":
                row["sampled_domain_counts"]["dacl"] -= 1
                row["sampled_domain_counts"]["damsegment"] += 1
            elif mutation == "full_crop":
                row["sampled_row_type_counts"]["full"] -= 1
                row["sampled_row_type_counts"]["crop"] += 1
                row["sampled_full_target_joint_counts"]["dacl"]["00"] -= 1
            else:
                row["sampled_full_target_joint_counts"]["dacl"]["00"] -= 1
                row["sampled_full_target_joint_counts"]["dacl"]["10"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_histories(control, highres, 6, 14248)

    def test_matching_unknown_states_or_unprepared_codebrim_crops_do_not_pass_pairing(self):
        for mutation in ("unknown", "missing_joint", "source_full_overflow", "codebrim_crop"):
            control = self.history()
            row = control[2]
            if mutation == "unknown":
                row["sampled_full_target_joint_counts"]["dacl"]["00"] -= 1
                row["sampled_full_target_joint_counts"]["dacl"]["unknown"] = 1
            elif mutation == "missing_joint": row["sampled_full_target_joint_counts"].pop("damsegment")
            elif mutation == "source_full_overflow":
                row["sampled_full_target_joint_counts"]["dacl"]["00"] += 4000
            else:
                row["sampled_full_target_joint_counts"]["codebrim"]["00"] -= 1
                row["sampled_row_type_counts"]["full"] -= 1
                row["sampled_row_type_counts"]["crop"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_histories(control, copy.deepcopy(control), 6, 14248)

    def test_missing_or_impossible_amp_update_accounting_is_rejected(self):
        for mutation in ("missing_block", "missing_count", "wrong_batches", "unaccounted_skip",
                         "negative", "boolean", "float", "over_budget"):
            control, highres = self.history(), self.history()
            row = highres[4]
            proof = row["optimizer_step_diagnostics"]
            if mutation == "missing_block": row.pop("optimizer_step_diagnostics")
            elif mutation == "missing_count": proof.pop("actual_optimizer_steps")
            elif mutation == "wrong_batches": proof["attempted_batches"] += 1
            elif mutation == "unaccounted_skip": proof["amp_skipped_steps"] += 1
            elif mutation == "negative": proof["actual_optimizer_steps"] = -1
            elif mutation == "boolean": proof["amp_skipped_steps"] = True
            elif mutation == "float": proof["actual_optimizer_steps"] = float(proof["actual_optimizer_steps"])
            elif mutation == "over_budget": proof.update(actual_optimizer_steps=1782, amp_skipped_steps=0)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_histories(control, highres, 6, 14248)

    def test_render_uses_measured_rates_intervals_resources_and_keeps_scope(self):
        result = self.result()
        before = copy.deepcopy(result)
        document = render(result)
        self.assertEqual(result, before)
        self.assertIn("| 960 보강군 | 960 | 6 | 3 | 22.50% | 미달 |", document)
        self.assertIn("연구 후보 기준: **미달**", document)
        self.assertIn("+0.22pp", document)
        self.assertIn("-0.50pp", document)
        self.assertIn("양수는 악화", document)
        self.assertIn("2/10 | 20.00%", document)
        self.assertIn("3/20 | 15.00%", document)
        # Independent Wilson calculation checks that the interval fields,
        # rather than measured point estimates, appear in the report.
        z, p, n = 1.96, .2, 10
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        self.assertIn(f"{100*(center-half):.2f}%–{100*(center+half):.2f}%", document)
        self.assertIn("29.25분 | 3.000 GiB | 10665 | 21", document)
        for phrase in ("원본 정답", "area 평균으로 80×80", "새 파라미터", "정밀 위치 정답",
                       "전체 오답 사진 비율이나 앱 정확도가 아니다", "보류 TEST 추론은 실행하지 않았다",
                       "전문가 라벨 수정", "공장 시설 성능은 미측정", "같은 시간·메모리"):
            self.assertIn(phrase, document)
        self.assertEqual(result["additional_expert_confirmed_labels"], 0)
        self.assertEqual(result["label_changes"], 0)
        self.assertFalse(result["deployed"])
        self.assertFalse(result["source_test_inference_executed"])

    def test_report_requires_actual_matching_frozen_checkpoint_and_test_proof(self):
        protocol = {'requested_epochs':6, 'source_sha256':{'runtime.py':'a'*64},
                    'imgsz_by_variant':{'control':640,'highres':960},
                    'architecture_by_variant':{'control':'aux','highres':'resolution'}}
        trainings = [{}, {'weights_sha256':'b'*64}, {'weights_sha256':'c'*64}]
        proof = {'schema':'facility_resolution_study_verification_v1','status':'passed',
                 'protocol_sha256':'d'*64,'source_sha256':protocol['source_sha256'],
                 'source_git_commit':'e'*40,'runtime_source_count':1,
                 'git_blob_bytes_verified':True,'working_runtime_sources_unchanged':True,
                 'protected_files_unchanged':True,'actual_completed_training_epochs':12,
                 'verification_training_epochs':0,'source_test_inference_executed':False,
                 'app_model_promoted':False,'deployed':False,'accuracy_measured_by_verifier':False,
                 'additional_expert_confirmed_labels':0,'label_changes':0,
                 'new_photo_targets':0,'new_pixel_targets':0,
                 'protected_file_sha256':{'app_profile':'f'*64},
                 'tests':{'tests_run':38,'failures':0,'errors':0,'skipped':0},'experiments':[]}
        for variant, training in zip(('control','highres'),trainings[1:]):
            proof['experiments'].append({'variant':variant,
                'weights_sha256':training['weights_sha256'],'actual_epochs':6,
                'imgsz':protocol['imgsz_by_variant'][variant],
                'architecture':protocol['architecture_by_variant'][variant],
                'strict_state_inventory_verified':True,'new_state_tensor_count':0,
                'cpu_reload':{'strict_factory_reload_verified':True,'all_outputs_finite':True,
                              'public_output_equals_training_photo_output':True}})
        before = copy.deepcopy(proof)
        self.assertEqual(validate_technical_proof(proof,protocol,'d'*64,trainings,'f'*64)['tests_run'],38)
        self.assertEqual(proof,before)
        for mutation in ('failed','stale_protocol','source_drift','no_git','incomplete_epochs',
                         'new_training','promoted','new_labels','app_changed','test_failure',
                         'no_second_checkpoint','stale_weights','nonfinite_reload'):
            changed = copy.deepcopy(proof)
            if mutation == 'failed': changed['status'] = 'failed'
            elif mutation == 'stale_protocol': changed['protocol_sha256'] = '0'*64
            elif mutation == 'source_drift': changed['source_sha256'] = {}
            elif mutation == 'no_git': changed['git_blob_bytes_verified'] = False
            elif mutation == 'incomplete_epochs': changed['actual_completed_training_epochs'] = 11
            elif mutation == 'new_training': changed['verification_training_epochs'] = 1
            elif mutation == 'promoted': changed['app_model_promoted'] = True
            elif mutation == 'new_labels': changed['label_changes'] = 1
            elif mutation == 'app_changed': changed['protected_file_sha256']['app_profile'] = '0'*64
            elif mutation == 'test_failure': changed['tests']['failures'] = 1
            elif mutation == 'no_second_checkpoint': changed['experiments'].pop()
            elif mutation == 'stale_weights': changed['experiments'][1]['weights_sha256'] = '0'*64
            else: changed['experiments'][1]['cpu_reload']['all_outputs_finite'] = False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_technical_proof(changed,protocol,'d'*64,trainings,'f'*64)


if __name__ == "__main__":
    unittest.main()
