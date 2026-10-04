"""Aggregate a completed stride-four architecture pair without model/test reads.

The two added runs may differ in their declared architecture only. Source data,
supervision, selected validation truth, sampling and initial base state stay
frozen. This script neither infers on images nor changes deployment or labels.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_facility_small_region import (
    CLASSES, COUNTS, DOMAINS, LABELS, TARGETS, comparisons, load_run,
    number, read, require, sha,
)
from scripts.report_facility_discrimination import (
    GATE as ERROR_GATE, descriptive_interval, error_rows, nonnegative_integer,
    MATCHED_KEYS as PREVIOUS_MATCHED_KEYS,
    research_gate as error_research_gate, validate_expected_sampling,
    validate_train_audit, verify_validation_cache,
)

REFERENCE = 'facility-presence-target-roi-control'
CONTROL = 'facility-presence-target-detail-control'
TREATMENT = 'facility-presence-target-detail-s4'
INITIAL_SHA = '0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a'
TITLES = {REFERENCE: '추가 학습 전 모델', CONTROL: '기존 구조 대조군', TREATMENT: 'stride 4 세부 특징 보강군'}
PROTOCOL_PATH = ROOT / 'reports/facility-detail-architecture-protocol.json'
EXPECTED = {'seed': 54, 'requested_epochs': 8, 'patience': 8, 'imgsz': 640,
            'batch_size': 8, 'draws_per_epoch': 14248, 'backbone_lr': .00004,
            'head_lr': .00025, 'auxiliary_weight': .5, 'domain_proportions': [.7, .1, .2]}
SMALL_GATE = {'minimum_small_combined_fn_reduction_vs_reference_and_control': 2,
              'maximum_any_small_class_fnr_regression_vs_reference_and_control': .02}
GATE = {**ERROR_GATE, **SMALL_GATE}
SMALL_SUPPORT = dict(zip(TARGETS, (93, 105)))
CONTROL_ARCH = 'lraspp_mobilenet_facility_auxiliary_v1'
TREATMENT_ARCH = 'lraspp_mobilenet_facility_detail_s4_v1'
CORE_SHA = '8216d8c1c958e59401b7ec20809afa681eb2b1ccb9640fec953597c5db61c681'
AUXILIARY_SHA = '53544bf83e1b32e3a7b8775c5f8348594359bf6e01eade1c8571242e758f4b61'
PARENT_SHA = 'c60e86aee35cc03ad552f582e1cd801128bd7ee672b4eab51e2a814ce713ec04'
LOADER_RANDOMNESS = {'sampler_seed':54, 'training_worker_seed':55, 'post_model_seed':56,
                     'validation_worker_seeds':{'dacl':154,'damsegment':155,'codebrim':156}}
DETAIL_ARCHITECTURE = {
    'early_backbone_block':'3', 'early_channels':24, 'early_feature_stride':4,
    'early_feature_shape_640':[160,160], 'hidden_channels':48, 'depthwise_kernel':3,
    'group_norm_groups':8, 'residual_projection_zero_initialized':True,
    'fusion':'Adaptive average pool learned stride4 seven-channel residual to original maps, then add before unchanged top32/mix',
    'source_supervision_shape_640':[80,80], 'finer_pixel_gold_asserted':False, 'new_photo_labels':0,
}
MATCHED_KEYS = tuple(key for key in PREVIOUS_MATCHED_KEYS if key != 'architecture') + (
    'source_parent_training_script_sha256', 'detail_model_source_sha256', 'model_factory_source_sha256',
    'loader_randomness', 'target_ranking',
)


def validate_protocol(protocol):
    require(protocol.get('schema') == 'facility_detail_architecture_protocol_v1'
            and protocol.get('declared_before_training') is True, 'Predeclared architecture protocol missing')
    declared = {**EXPECTED, 'reference':REFERENCE, 'control':CONTROL, 'treatment':TREATMENT,
                'initial_weights_sha256':INITIAL_SHA, 'core_spatial_manifest_sha256':CORE_SHA,
                'auxiliary_manifest_sha256':AUXILIARY_SHA, 'classes':CLASSES,
                'control_architecture':CONTROL_ARCH, 'treatment_architecture':TREATMENT_ARCH,
                'target_ranking_weight':0., 'loader_randomness':LOADER_RANDOMNESS,
                'detail_architecture':DETAIL_ARCHITECTURE, 'research_candidate_gate':GATE,
                'implementation_parent_sha256':PARENT_SHA}
    for key, value in declared.items():
        require(protocol.get(key) == value, f'Predeclared architecture condition changed: {key}')
    return protocol


def validate_pair(control, treatment, initializer_sha, protocol, protocol_sha):
    validate_protocol(protocol)
    require(initializer_sha == INITIAL_SHA, 'Original frozen ROI initializer changed')
    for training, architecture, variant in ((control,CONTROL_ARCH,'control'), (treatment,TREATMENT_ARCH,'detail')):
        require(training.get('status') == 'complete', 'Both architecture trainings must be complete')
        expected = {**EXPECTED, 'actual_epochs':8, 'classes':CLASSES, 'validation_domains':DOMAINS,
                    'architecture':architecture, 'model_variant':variant, 'loader_randomness':LOADER_RANDOMNESS,
                    'initial_weights_sha256':INITIAL_SHA, 'spatial_manifest_sha256':CORE_SHA,
                    'core_spatial_manifest_sha256':CORE_SHA, 'auxiliary_manifest_sha256':AUXILIARY_SHA,
                    'source_parent_training_script_sha256':PARENT_SHA, 'study_protocol_sha256':protocol_sha,
                    'study_protocol_path':'reports/facility-detail-architecture-protocol.json',
                    'train_dacl':6225, 'train_damsegment':1585, 'train_codebrim':6438,
                    'val_dacl':710, 'val_damsegment':424}
        for key, value in expected.items():
            require(training.get(key) == value, f'Actual architecture protocol condition changed: {key}')
        require(training.get('spatial_manifest_audit', {}).get('full_photo_or_patch_rows') == 14248,
                'Original full TRAIN supervision count changed')
        require(not any(key in training for key in ('hard_training_sampling','photo_supplement_sha256',
                    'supplement_spatial_sha256','train_convid','train_s2ds','train_peccd')),
                'A source/label/mining intervention entered this architecture comparison')
        ranking = training.get('target_ranking', {})
        require(ranking.get('weight') == 0 and ranking.get('classes') == TARGETS
                and ranking.get('sampling_changed') is False and ranking.get('new_labels_asserted') == 0
                and ranking.get('public_outputs_changed') is False, 'Ranking/public labels changed in architecture pair')
        transfer = training.get('initial_state_transfer', {})
        require(transfer.get('shared_state_tensors_equal') is True
                and nonnegative_integer(transfer.get('shared_state_tensor_count'), 'Shared initializer count missing') > 0,
                'Shared backbone/base tensors were not transferred identically')
        require(nonnegative_integer(transfer.get('new_state_tensor_count'), 'New initializer tensor count missing')
                == (0 if variant == 'control' else 8)
                and transfer.get('new_output_projection_zero') is (None if variant == 'control' else True),
                'Only the fresh zero-initialized detail projection may add initializer tensors')
        require(training.get('detail_architecture') == (None if variant == 'control' else DETAIL_ARCHITECTURE),
                'Actual detail branch recipe differs from declared architecture')
    for key in MATCHED_KEYS:
        require(key in control and key in treatment and control[key] == treatment[key],
                f'Paired source/sampler/loss/RNG/provenance changed: {key}')
    require(control['initial_state_transfer']['shared_state_tensor_count']
            == treatment['initial_state_transfer']['shared_state_tensor_count'], 'Shared original tensor support changed')
    validate_expected_sampling(control)
    return {key: control[key] for key in MATCHED_KEYS if key not in ('auxiliary_audit','additional_test')}


def validate_small_area(entry):
    """Count photo-label cases, including a possible shared photo in both labels."""
    source = entry.get('small_dacl_polygon_area_below_one_percent', {})
    require(set(source) == set(TARGETS), 'Both small-area target measurements required')
    total_fn = 0
    for label in TARGETS:
        point = source[label]
        n, fn = point.get('positive_photos'), point.get('false_negatives')
        require(n == SMALL_SUPPORT[label] and not isinstance(n, bool), 'Frozen DACL small-area support changed')
        nonnegative_integer(fn, 'Measured small-area false negatives missing')
        require(fn <= n and math.isclose(number(point.get('fnr'), 'small-area FNR'), fn / n,
                                        rel_tol=0., abs_tol=1e-12), 'Small-area FNR differs from measured counts')
        total_fn += fn
    return {'photo_label_positive_cases': 198, 'false_negative_cases': total_fn,
            'unique_photo_count_established': False}


def detail_research_gate(reference, control, treatment, declared):
    require(declared == GATE, 'Predeclared detail nomination gate changed')
    common = error_research_gate(reference, control, treatment, ERROR_GATE)
    totals = [validate_small_area(entry) for entry in (reference, control, treatment)]
    checks = {}
    for name, entry, total in (('initializer', reference, totals[0]), ('control', control, totals[1])):
        improvement = total['false_negative_cases'] - totals[2]['false_negative_cases']
        classes = []
        for label in TARGETS:
            original = entry['small_dacl_polygon_area_below_one_percent'][label]
            current = treatment['small_dacl_polygon_area_below_one_percent'][label]
            classes.append({'class': label, 'positive_photo_label_cases': SMALL_SUPPORT[label],
                            'treatment_minus_comparator_fnr': current['fnr'] - original['fnr'],
                            'treatment_minus_comparator_false_negatives': current['false_negatives'] - original['false_negatives']})
        sum_passed = improvement >= 2
        class_passed = all(row['treatment_minus_comparator_fnr'] <= .02 + 1e-12 for row in classes)
        checks[name] = {'small_area_total_fn_improvement': improvement,
                        'minimum_two_cases_improvement_passed': sum_passed,
                        'small_area_per_class_regression_guard_passed': class_passed,
                        'per_class_changes': classes, 'passed': sum_passed and class_passed}
    return {'thresholds': declared, 'error_and_other_ap_gate': common,
            'small_area_photo_label_case_totals': dict(zip(('initializer', 'control', 'treatment'), totals)),
            'small_area_comparisons': checks,
            'research_candidate_nominated': common['research_candidate_nominated'] and all(row['passed'] for row in checks.values()),
            'descriptive_only': True, 'deployment_authorized': False}


def validate_actual_pair(control, treatment):
    """Require the same original row schedule and known full-photo states."""
    histories = [entry.get('actual_sampling_history', []) for entry in (control, treatment)]
    require(all(len(rows) == 8 for rows in histories), 'Eight actual sampling epochs required')
    for rows in histories:
        for epoch, row in enumerate(rows, 1):
            require(row.get('epoch') == epoch, 'Actual epoch sequence changed')
            domains, types, joints = [row.get(key, {}) for key in ('domain_counts', 'row_type_counts', 'full_target_joint_counts')]
            require(set(domains) == set(joints) == set(DOMAINS) and set(types) == {'full', 'crop'},
                    'Original domain/full/crop counts missing')
            require(sum(nonnegative_integer(v, 'Invalid actual source draw count') for v in domains.values()) == 14248
                    and sum(nonnegative_integer(v, 'Invalid actual row-type count') for v in types.values()) == 14248,
                    'Fixed epoch draw budget changed')
            for domain in DOMAINS:
                require(set(joints[domain]) == {'00', '10', '01', '11', 'unknown'}, 'Original joint target states changed')
                full = sum(nonnegative_integer(v, 'Invalid original-full joint count') for v in joints[domain].values())
                require(full <= domains[domain] and joints[domain]['unknown'] == 0,
                        'Original full-photo target assertion changed')
                if domain == 'codebrim':
                    require(full == domains[domain], 'CODEBRIM gained unprepared crop rows')
            require(sum(sum(v.values()) for v in joints.values()) == types['full'], 'Full joint counts differ from full row draws')
            require(isinstance(row.get('row_indices_sha256'), str)
                    and re.fullmatch(r'[0-9a-f]{64}', row['row_indices_sha256']), 'Actual ordered row-index hash missing')
            audit = row.get('ranking_audit', {})
            require(audit.get('pair_count') == 0 and audit.get('contributing_batches') == 0
                    and audit.get('unweighted_mean_batch_loss') == 0 and audit.get('pair_counts_by_domain_target') == {},
                    'A ranking-loss intervention entered the architecture pair')
    for a, b in zip(*histories):
        for key in ('domain_counts', 'row_type_counts', 'full_target_joint_counts', 'row_indices_sha256'):
            require(a[key] == b[key], f'Paired seed actual original row schedule differs: {key}')
    return {'actual_ordered_row_index_hashes_identical_each_epoch': True,
            'actual_source_full_crop_target_joint_counts_identical_each_epoch': True,
            'additional_ranking_supervision_used': False}


def make_sampling_history(history):
    return [{'epoch': row['epoch'], 'domain_counts': row.get('sampled_domain_counts'),
             'row_type_counts': row.get('sampled_row_type_counts'),
             'full_target_joint_counts': row.get('sampled_full_target_joint_counts'),
             'row_indices_sha256': row.get('sampled_row_indices_sha256'),
             'ranking_audit': row.get('target_ranking_audit')} for row in history]


def load_measured_runs():
    names = (REFERENCE, CONTROL, TREATMENT)
    require(not any((ROOT / 'reports' / f'{name}-target-test.json').exists() for name in names),
            'This no-test comparison refuses any saved target-test result before opening it')
    loaded = [load_run(name) for name in names]
    trainings, entries = [row[0] for row in loaded], [row[1] for row in loaded]
    truth_digests, measured_ap_count = [], 0
    for entry in entries:
        run = ROOT / 'runs' / entry['run']
        digests = {}
        for domain in DOMAINS:
            points, digest = verify_validation_cache(read(run / f'validation-{domain}.json'),
                             read(run / f'target-validation-{domain}-grid1.json'), entry, domain)
            entry['ranking_ap'][domain] = points
            digests[domain] = digest
            measured_ap_count += sum(point['ap'] is not None for point in points.values())
        truth_digests.append(digests)
        validate_small_area(entry)
        if entry['run'] != REFERENCE:
            entry['actual_sampling_history'] = make_sampling_history(read(run / 'history.json'))
    require(truth_digests[0] == truth_digests[1] == truth_digests[2], 'Original VAL truth/order changed across runs')
    require(measured_ap_count == 42, 'Expected exactly forty-two originally asserted AP measurements')
    return trainings, entries, truth_digests[0]


def resource_measurements(training, history):
    """Retain actual elapsed wall time and allocated CUDA peak, not estimates."""
    elapsed = number(training.get('elapsed_training_minutes'), 'actual total training minutes', upper=math.inf)
    peak = nonnegative_integer(training.get('peak_cuda_allocated_bytes'), 'Actual peak CUDA allocation missing')
    require(elapsed > 0 and peak > 0 and len(history) == 8, 'Completed resource measurements missing')
    times = [number(row.get('elapsed_minutes'), 'actual cumulative epoch minutes', upper=math.inf) for row in history]
    peaks = [nonnegative_integer(row.get('peak_cuda_allocated_bytes'), 'Actual cumulative epoch CUDA peak missing') for row in history]
    require(all(a <= b for a,b in zip(times,times[1:])) and elapsed >= times[-1]
            and all(a <= b for a,b in zip(peaks,peaks[1:])) and peak >= peaks[-1],
            'Actual elapsed/peak cumulative measurements are inconsistent')
    return {'elapsed_training_minutes':elapsed, 'peak_cuda_allocated_bytes':peak,
            'epoch_cumulative_elapsed_minutes':times, 'epoch_cumulative_cuda_allocated_bytes':peaks,
            'scope':'Observed training plus epoch validation wall time; CUDA allocated peak, not reserved memory or deployment cost'}


def optimizer_measurements(training, history):
    keys = {'attempted_batches','actual_optimizer_steps','amp_skipped_steps'}
    aggregate = {key:0 for key in keys}
    rows = []
    require(len(history) == 8, 'Eight optimizer diagnostic epochs required')
    for epoch,row in enumerate(history,1):
        measured = row.get('optimizer_step_diagnostics', {})
        require(row.get('epoch') == epoch and set(measured) == keys, 'Optimizer diagnostic fields/order missing')
        for key,value in measured.items():
            aggregate[key] += nonnegative_integer(value, 'Invalid optimizer diagnostic count')
        require(measured['attempted_batches'] == 1781
                and measured['actual_optimizer_steps'] + measured['amp_skipped_steps'] == 1781,
                'Actual optimizer steps/skips do not cover attempted batch budget')
        rows.append({'epoch':epoch, **measured})
    require(training.get('optimizer_step_diagnostics') == aggregate, 'Optimizer aggregate differs from epoch actual steps/skips')
    return {'total':aggregate, 'per_epoch':rows,
            'scope':'Optimizer post-step hook counts actual updates; AMP skips may differ despite identical sampled rows; diagnostic only, no added post-hoc performance gate'}


def validate_technical_verification(proof, entries, protocol_sha, profile_sha):
    require(proof.get('status') == 'passed' and proof.get('study_protocol_sha256') == protocol_sha
            and proof.get('app_profile_sha256') == profile_sha and proof.get('app_profile_unchanged') is True
            and proof.get('research_models_deployed') is False and proof.get('paired_sampled_row_indices_equal') is True,
            'Architecture technical proof scope/provenance changed')
    rows = proof.get('experiments', [])
    require(len(rows) == 2 and {row.get('run') for row in rows} == {CONTROL,TREATMENT},
            'Architecture technical checkpoint pair missing')
    for entry, architecture, variant in zip(entries[1:], (CONTROL_ARCH,TREATMENT_ARCH), ('control','detail')):
        row = next(row for row in rows if row['run'] == entry['run'])
        require(row.get('weights_sha256') == entry['weights_sha256'] and row.get('architecture') == architecture
                and row.get('model_variant') == variant and row.get('study_protocol_sha256') == protocol_sha
                and row.get('public_shape') == [1,7] and row.get('private_spatial_shape') == [1,7,80,80]
                and row.get('auxiliary_shape') == [1,19] and row.get('outputs_finite') is True
                and row.get('public_training_logits_equal') is True, 'Architecture technical output/weights differ')
    preflight = proof.get('actual_train_gpu_preflight', {})
    require(preflight.get('status') == 'passed' and all(preflight.get(key) is True for key in (
                'initial_fp32_outputs_equal','finite_loss_and_gradients',
                'detail_upstream_gradient_after_projection_step','augmented_batch_replay_equal')),
            'Actual TRAIN zero-initializer/gradient/augmentation integration proof missing')
    tests = proof.get('actual_test_verification', {})
    require(tests.get('status') == 'passed' and nonnegative_integer(tests.get('tests_run'), 'Technical test count missing') > 0
            and tests.get('failures') == 0 and tests.get('errors') == 0,
            'Architecture implementation unit checks missing')
    freeze = proof.get('pre_training_source_snapshot', {})
    expected_paths = {'scripts/train_facility_detail.py','safelog_ai/detail_classifier.py',
                      'reports/facility-detail-architecture-protocol.json'}
    require(freeze.get('comparison_started_before_outcomes') is True
            and expected_paths <= set(freeze.get('sources', {})), 'Pre-training architecture/protocol snapshot missing')
    for relative,row in freeze['sources'].items():
        resolved = (ROOT / relative).resolve()
        require(resolved.is_relative_to(ROOT) and isinstance(relative,str)
                and not Path(relative).is_absolute()
                and (resolved.suffix == '.py' or relative == 'reports/facility-detail-architecture-protocol.json'),
                'Technical source snapshot must reference local code/protocol only, never data/heldout inputs')
        require(row.get('equal') is True and row.get('git_blob_sha256') == row.get('executed_file_sha256') == sha(resolved),
                'Committed architecture source/protocol revision changed')
    return {'checkpoint_pair_passed':True, 'public_outputs':7, 'private_auxiliary_outputs':19,
            'private_spatial_shape':[1,7,80,80], 'initial_fp32_zero_residual_outputs_equal':True,
            'same_seed_augmented_train_batch_replay_equal':True,
            'detail_upstream_gradient_after_projection_step':True,
            'unit_tests_run':tests['tests_run'], 'pre_training_commit':freeze.get('pre_training_commit'),
            'accuracy_measured_by_technical_check':False}


def render(result):
    pct = lambda value: f'{100 * value:.2f}%'
    intervals = lambda values: f'{pct(values[0])}–{pct(values[1])}'
    entries, gate, changes = result['experiments'], result['research_gate'], result['comparisons']
    lines = ['# 세부 특징 구조 대조 실험 결과', '',
             '실제 완료된 8epoch 대조 실험 결과만 기록한다. 새 시설·사진·정답을 추가하지 않고 같은 초기 모델과 기존 원본 TRAIN로 작은 손상의 세부 특징 구조를 비교했다.',
             '대조군은 기존 LRASPP·7종 사진 출력·19종 보조 출력이다. 보강군은 backbone block 3의 stride 4 특징을 24→48의 1×1 convolution, GroupNorm 8, SiLU, depthwise 3×3, GroupNorm 8, SiLU, 7종 zero-initialized 최종 층으로 처리한다. 80×80 adaptive average pooling 후 기존 spatial map에 residual을 더한다.',
             '7종 출력·19종 보조 태그·기존 사진/픽셀 손실·full/crop·출처 비중을 유지했다. 순위 추가 손실은 두 군 모두 0이다. stride 4 특징을 사용한다는 사실이 정밀 위치 검출 성능을 증명하지 않는다.', '',
             '| 모델 | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한 5% 기준 |',
             '|---|---:|---:|---:|---:|']
    for entry in entries:
        lines.append(f"| {TITLES[entry['run']]} | {entry['actual_epochs']} | {entry['best_epoch']} | {pct(entry['worst_error'])} | {'검증 통과' if entry['target_passed'] else '미달'} |")
    lines += ['', f"보강군 − 초기 모델 최대 오류 {changes['maximum_error_treatment_minus_initializer_pp']:+.2f}pp, 보강군 − 대조군 {changes['maximum_error_treatment_minus_control_pp']:+.2f}pp. 양수는 악화다.",
              '최대 오류는 균열·박락 × 세 출처 × 미탐 FN/(TP+FN)·오탐 FP/(FP+TN)의 12개 비율 중 최대다. 앱의 전체 오답 사진 비율이 아니다.', '',
              '## 사전 선언한 후보 기준', '',
              f"연구 후보 기준: **{'통과' if gate['research_candidate_nominated'] else '미달'}**. 초기 모델과 대조군 각각보다 최대 오류 최소 0.5pp 개선, 각 target FNR/FPR 회귀 최대 2pp, 나머지 알려진 항목 AP 회귀 최대 0.02를 적용했다.",
              '작은 DACL 균열·박락의 FN 합계는 초기 모델과 대조군 각각보다 최소 2건 감소해야 하며 각 항목의 작은 손상 FNR 회귀도 2pp 이하여야 한다. 연구 후보 기준은 모든 항목 개선이나 엄격한 5% 달성을 뜻하지 않는다.',
              '앱 배포는 자동으로 진행하지 않는다. 모든 target 출처 FNR/FPR이 각각 5% 미만이어야 하는 기존 엄격한 기준을 그대로 유지하고, 이번 보고는 보류 시험을 열거나 실행하지 않는다.', '']
    for name, point in gate['small_area_comparisons'].items():
        common = gate['error_and_other_ap_gate']['comparisons'][name]
        lines.append(f"- {'초기 모델' if name == 'initializer' else '대조군'} 대비: 최대 오류 개선 {common['maximum_error_improvement'] * 100:+.2f}pp, 작은 손상 FN {point['small_area_total_fn_improvement']:+d}건 감소, 오류/AP 조건 {'통과' if common['passed'] else '미달'}, 작은 손상 조건 {'통과' if point['passed'] else '미달'}.")
    lines += ['', '## 관측 오류와 기술적 범위', '',
              'Wilson 양측 95% 범위는 독립 사진 이항 가정에 따른 관측 비율의 기술 통계다. 사진 상관과 같은 VAL에서 반복한 epoch·임계값 선택 때문에 확인적 신뢰구간, 미래 현장 오류 보장 또는 모델 차이의 유의성 검정으로 해석할 수 없다.', '',
              '| 모델 | 출처 | 항목 | FN/양성 | 미탐률 | Wilson 95% | FP/음성 | 오탐률 | Wilson 95% |',
              '|---|---|---|---:|---:|---|---:|---:|---|']
    for row in result['measured_error_rows']:
        lines.append(f"| {TITLES[row['run']]} | {row['domain']} | {LABELS[row['class']]} | {row['fn']}/{row['positive_photos']} | {pct(row['fnr'])} | {intervals(row['fnr_wilson95_descriptive'])} | {row['fp']}/{row['negative_photos']} | {pct(row['fpr'])} | {intervals(row['fpr_wilson95_descriptive'])} |")
    lines += ['', '## 작은 손상의 사진 단위 미탐', '',
              'DACL 원본 폴리곤을 640 입력으로 옮긴 면적 비율 1% 미만인 양성 사례다. 균열 93 + 박락 105 = 198개 사진·항목 사례이며 같은 사진이 두 항목에 들어갈 수 있다. 고유 사진 198장이나 물리적 길이·정밀 위치 평가를 뜻하지 않는다.', '',
              '| 항목 | 초기 FN/양성 | 대조 FN/양성 | 보강 FN/양성 | 초기 미탐률 | 대조 미탐률 | 보강 미탐률 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label in TARGETS:
        values = [entry['small_dacl_polygon_area_below_one_percent'][label] for entry in entries]
        cells = [f"{v['false_negatives']}/{v['positive_photos']}" for v in values] + [pct(v['fnr']) for v in values]
        lines.append(f"| {LABELS[label]} | {' | '.join(cells)} |")
    totals = gate['small_area_photo_label_case_totals']
    lines.append(f"| 두 항목 FN 합계/198사례 | {totals['initializer']['false_negative_cases']}/198 | {totals['control']['false_negative_cases']}/198 | {totals['treatment']['false_negative_cases']}/198 | 해당 없음 | 해당 없음 | 해당 없음 |")
    lines += ['', '## 원래 정답이 있는 7종 AP', '',
              'AP는 확률 순위 지표이며 앱 정확도·오탐률과 다르다. 세 모델의 DACL 7종, Dam 2종, CODEBRIM 5종 AP 42개를 원본 VAL 확률에서 다시 계산해 확인했다. float32/float64 산술 차이만 허용하며 기존 저장 AP로 비교한다. 미확인은 0점으로 바꾸지 않는다.', '',
              '| 출처 | 항목 | 초기 AP | 대조 AP | 보강 AP | 보강−대조 | 보강−초기 |', '|---|---|---:|---:|---:|---:|---:|']
    for row in changes['ranking_ap_changes']:
        domain, label = row['domain'], row['class']
        cells = [f"{e['ranking_ap'][domain][label]['ap']:.4f}" if e['ranking_ap'][domain][label]['ap'] is not None else '미확인' for e in entries]
        cells += [f'{row[k]:+.4f}' if row[k] is not None else '해당 없음' for k in ('treatment_minus_control_ap', 'treatment_minus_initializer_ap')]
        lines.append(f"| {domain} | {LABELS[label]} | {' | '.join(cells)} |")
    lines += ['', '## 추출·정답·앱 적용 범위', '',
              'seed 54, epoch마다 복원 추출 14,248회, 640 입력, batch 8, backbone/head 학습률 0.00004/0.00025, 19종 보조 손실 0.5, 출처 비중 DACL/Dam/CODEBRIM 70/10/20%다. 실제 순서가 있는 row index 해시와 full/crop·원래 두 target 조합의 출처별 추출 수를 여덟 epoch 모두 비교했다.',
              '두 군은 같은 초기 backbone/base 상태와 명시적인 loader worker seed를 사용했다. 학습 단계·추출 예산이 같아도 새 구조의 연산·활성 메모리는 달라질 수 있으며 동일한 시간·메모리 비용이라고 주장하지 않는다.',
              'full은 기존 원본 TRAIN 행이라는 뜻이고 CODEBRIM 원저자의 patch를 포함한다. 전체 14,248행을 독립된 전체 현장 사진 수로 해석하지 않는다. 기존 파생 crop도 새 독립 사진이 아니다.',
              '원저자 지정 항목의 음성은 전체 시설이 정상·안전하다는 정답이 아니다. DACL의 다른 태그는 혼동 후보일 뿐 오류 원인 확정이나 도장·미장 전문 정답이 아니다. Dam 원논문 4.4는 Non-Crack을 intact concrete, Crack을 균열·박락 포함으로 설명한다. Classification/Non-Crack의 음성 변환은 이 저자 정의에 따른 사진 정답이며 별도 Spall 음성 태그·mask나 산업 현장 전문가의 재검증을 뜻하지 않는다. [DamSegment 원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC13247583/)',
              '교량·댐 콘크리트 손상 자료로 공통 시각적 특징을 연구한 결과다. 공장 벽·바닥·기둥의 현장 정확도는 측정하지 않았다. 기존 정답을 편집하거나 오답을 제외하거나 미확인을 음성으로 채우지 않았다.',
              '단일 seed와 같은 원저자 VAL의 반복 선택이라는 한계가 있다. 후보 기준 통과만으로 새 현장 일반화 또는 5% 미만 오류를 보장하지 않는다. 최종 판단은 점검자가 한다.',
              f"기본 앱 프로필 `{result['app_profile']['version']}` SHA256은 이전 파일과 동일하고 deployed=false다. 모델의 zero-residual 초기 동등성과 최종 출력 shape 점검은 기술 검증이며 정확도 측정과 분리한다.",
              '공개 보고에는 집계·해시만 포함한다. 원본/가공 사진·개별 경로·원저자 주석·개별 점검 의견은 공개 묶음에 복사하지 않으며 데이터의 재배포 조건을 따른다.', '',
              '[실측 집계 JSON](facility-detail-architecture-comparison.json), [사전 계획](facility-detail-architecture-protocol.json), [기술 점검](facility-detail-architecture-technical-verification.json)', '']
    lines += ['## 실측 자원 비용', '', '| 모델 | 총 학습·epoch 검증 시간 | 최대 CUDA allocated | 시도 batch | 실제 update | AMP skip |', '|---|---:|---:|---:|---:|---:|']
    for name in (CONTROL,TREATMENT):
        cost = result['actual_resources'][name]
        updates = result['actual_optimizer_steps'][name]['total']
        lines.append(f"| {TITLES[name]} | {cost['elapsed_training_minutes']:.2f}분 | {cost['peak_cuda_allocated_bytes'] / (1024 ** 3):.3f} GiB | {updates['attempted_batches']} | {updates['actual_optimizer_steps']} | {updates['amp_skipped_steps']} |")
    lines += ['', '각 run에서 관측한 시간과 allocated 메모리다. CUDA reserved·전체 프로세스 메모리·스마트폰 추론 비용 또는 미래 실행 시간의 추정값이 아니다.',
              '같은 순서의 표본과 batch 예산이라도 AMP가 건너뛴 실제 optimizer update 수는 다를 수 있어 그대로 공개했다. 이 진단에 사후 채택 조건을 추가하지 않았다.', '']
    return '\n'.join(lines)


def main():
    protocol = validate_protocol(read(PROTOCOL_PATH))
    trainings, entries, digests = load_measured_runs()
    conditions = validate_pair(trainings[1], trainings[2], entries[0]['weights_sha256'], protocol, sha(PROTOCOL_PATH))
    for key in ('architecture','classes','imgsz','split_sha256','model_source_sha256',
                'auxiliary_model_source_sha256','auxiliary_manifest_sha256','auxiliary_classes',
                'validation_domains','additional_validation','spatial_manifest_sha256'):
        require(trainings[0].get(key) == trainings[1].get(key), f'Original initializer source/evaluation changed: {key}')
    sources = {'training_script_sha256':'scripts/train_facility_detail.py',
               'source_parent_training_script_sha256':'scripts/train_facility_spatial.py',
               'model_source_sha256':'safelog_ai/spatial_classifier.py',
               'auxiliary_model_source_sha256':'safelog_ai/auxiliary_classifier.py',
               'detail_model_source_sha256':'safelog_ai/detail_classifier.py',
               'model_factory_source_sha256':'safelog_ai/presence_classifier.py',
               'photo_supplement_helper_sha256':'scripts/facility_photo_supplement.py'}
    for key,path in sources.items():
        require(trainings[1][key] == sha(ROOT / path), f'Executed architecture source revision changed: {key}')
    require(trainings[1]['target_ranking']['helper_sha256'] == sha(ROOT / 'scripts/facility_target_ranking.py'),
            'Disabled ranking helper revision changed')
    require(sha(ROOT / 'data/facility-spatial-training/train.json') == CORE_SHA
            and sha(ROOT / 'data/facility-auxiliary-training/train.json') == AUXILIARY_SHA,
            'Original TRAIN supervision changed')
    paired_draws = validate_actual_pair(entries[1], entries[2])
    profile_path = ROOT / 'reports/facility-inference-profile.json'
    previous_path = ROOT / 'reports/facility-inference-profile-round1.json'
    profile = read(profile_path)
    require(profile.get('version') == 'facility-validation-v2' and sha(profile_path) == sha(previous_path),
            'App profile changed during architecture research')
    technical_path = ROOT / 'reports/facility-detail-architecture-technical-verification.json'
    technical = validate_technical_verification(read(technical_path), entries, sha(PROTOCOL_PATH), sha(profile_path))
    audit_path = ROOT / 'reports/facility-target-negative-data-audit.json'
    negative = validate_train_audit(read(audit_path), trainings[1])
    resources = {entry['run']:resource_measurements(training,read(ROOT / 'runs' / entry['run'] / 'history.json'))
                 for training,entry in zip(trainings[1:],entries[1:])}
    updates = {entry['run']:optimizer_measurements(training,read(ROOT / 'runs' / entry['run'] / 'history.json'))
               for training,entry in zip(trainings[1:],entries[1:])}
    result = {'schema':'facility_detail_architecture_comparison_v1', 'split':'val', 'deployed':False,
              'test_executed':False, 'industrial_field_performance_measured':False,
              'new_training_photos':0, 'original_labels_edited':0, 'finer_pixel_gold_asserted':False,
              'protocol_sha256':sha(PROTOCOL_PATH), 'protocol':protocol,
              'matched_pair_conditions':conditions, 'architectures':{'control':CONTROL_ARCH,'treatment':TREATMENT_ARCH},
              'detail_architecture':DETAIL_ARCHITECTURE,
              'initial_state_transfer':{name:training['initial_state_transfer'] for name,training in zip((CONTROL,TREATMENT),trainings[1:])},
              'actual_pair_sampling_checks':paired_draws, 'experiments':entries, 'comparisons':comparisons(*entries),
              'measured_error_rows':error_rows(entries), 'known_ap_recomputed_count':42,
              'validation_target_array_sha256':digests, 'research_gate':detail_research_gate(*entries,protocol['research_candidate_gate']),
              'strict_target':'Every original-source crack/spalling FNR/FPR strictly below .05; not whole-app accuracy',
              'interval_policy':'Wilson95 descriptive only; independent-photo assumption unverified and repeated validation selection prevents confirmatory interpretation',
              'train_negative_audit_sha256':sha(audit_path), 'train_negative_summary':negative,
              'technical_verification_sha256':sha(technical_path), 'technical_verification_summary':technical,
              'actual_resources':resources, 'actual_optimizer_steps':updates,
              'app_profile':{'version':profile['version'],'sha256':sha(profile_path),'previous_sha256':sha(previous_path),'unchanged':True},
              'new_training_epochs':16, 'report_script_sha256':sha(Path(__file__)),
              'limitations':['Single seed; repeated original-source validation epoch/threshold selection',
                             'Stride4 learned features pooled to original80 map; no finer pixel truth or localization performance asserted',
                             '198 small-area photo-label cases may share photos across labels; not198independent photographs',
                             'Publisher CODEBRIM full rows include patches; derived crops add no independent photographs',
                             'Dam classification absence follows author intact-concrete semantics; not independent classification spall masks or field expert gold',
                             'Source labels preserved, unknowns not filled as negatives and errors not excluded',
                             'Source evidence is bridge/dam concrete; factory field performance unmeasured',
                             'Research nomination permits predeclared regression budgets and does not authorize deployment',
                             'Equal epochs/draws are not equal measured compute/memory cost']}
    output = ROOT / 'reports'
    (output / 'facility-detail-architecture-comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (output / 'FACILITY_DETAIL_ARCHITECTURE_RESULTS_KO.md').write_text(render(result),encoding='utf-8')
    print(json.dumps({'comparison':'reports/facility-detail-architecture-comparison.json',
                      'maximum_errors':{entry['run']:entry['worst_error'] for entry in entries},
                      'research_candidate_nominated':result['research_gate']['research_candidate_nominated'],
                      'deployed':False},ensure_ascii=False))


if __name__ == '__main__':
    main()
