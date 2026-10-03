"""Report the frozen TRAIN-only discrimination pair; never read test results.

Only aggregates and hashes are published. A research nomination does not deploy
a model or establish industrial field accuracy. Raw validation scores are read
only to check saved counts/AP; this script performs no model inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_facility_small_region import (
    CLASSES, COUNTS, DOMAINS, LABELS, TARGETS, comparisons, load_run,
    number, read, require, sha,
)

REFERENCE = 'facility-presence-target-roi-control'
CONTROL = 'facility-presence-target-discrimination-control'
TREATMENT = 'facility-presence-target-discrimination-ranking'
INITIAL_SHA = '0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a'
TITLES = {REFERENCE: '추가 학습 전 모델', CONTROL: '기존 손실 대조군', TREATMENT: '손상 구분 순위 손실 보강군'}
KNOWN_INDICES = {'dacl': set(range(7)), 'damsegment': {0, 1}, 'codebrim': {0, 1, 2, 3, 5}}
GATE = {
    'minimum_maximum_error_improvement_vs_reference_and_control': .005,
    'maximum_any_target_fnr_fpr_regression_vs_reference_and_control': .02,
    'maximum_any_known_other_class_ap_regression_vs_reference_and_control': .02,
}
EXPECTED = {'seed': 53, 'requested_epochs': 6, 'patience': 6, 'imgsz': 640,
            'batch_size': 8, 'draws_per_epoch': 14248, 'backbone_lr': .00002,
            'head_lr': .000125, 'auxiliary_weight': .5, 'domain_proportions': [.7, .1, .2]}
MATCHED_KEYS = (
    'architecture', 'classes', 'imgsz', 'batch_size', 'split_sha256', 'initial_weights_sha256',
    'seed', 'requested_epochs', 'actual_epochs', 'patience', 'backbone_lr', 'head_lr',
    'auxiliary_manifest_sha256', 'auxiliary_classes', 'auxiliary_weight', 'auxiliary_audit',
    'training_script_sha256', 'model_source_sha256', 'auxiliary_model_source_sha256',
    'photo_supplement_helper_sha256', 'draws_per_epoch', 'domain_proportions',
    'spatial_manifest_sha256', 'core_spatial_manifest_sha256', 'spatial_manifest_audit',
    'train_dacl', 'train_damsegment', 'train_codebrim', 'val_dacl', 'val_damsegment',
    'validation_domains', 'additional_validation', 'additional_test', 'selection', 'loss', 'total_loss_formula',
    'study_protocol_sha256', 'study_protocol_path',
    'photo_positive_weights', 'pixel_positive_weights', 'auxiliary_positive_weights',
    'expected_sampling', 'expected_label_sampling',
)
PROTOCOL_PATH = ROOT / 'reports/facility-target-discrimination-protocol.json'


def validate_protocol(protocol):
    require(protocol.get('schema') == 'facility_target_discrimination_protocol_v1'
            and protocol.get('declared_before_training') is True, 'Predeclared protocol missing')
    for key, value in {**EXPECTED, 'reference': REFERENCE, 'control': CONTROL, 'treatment': TREATMENT,
                       'initial_weights_sha256': INITIAL_SHA, 'classes': CLASSES,
                       'control_ranking_weight': 0., 'treatment_ranking_weight': .25,
                       'research_candidate_gate': GATE}.items():
        require(protocol.get(key) == value, f'Predeclared condition changed: {key}')
    return protocol


def validate_pair(control, treatment, initializer_sha, protocol, protocol_sha):
    validate_protocol(protocol)
    require(initializer_sha == INITIAL_SHA, 'Initializer checkpoint changed')
    for training in (control, treatment):
        require(training.get('status') == 'complete', 'Both paired trainings must be complete')
        require(training.get('study_protocol_sha256') == protocol_sha
                and training.get('study_protocol_path') == 'reports/facility-target-discrimination-protocol.json',
                'Predeclared protocol changed after training or was not recorded by the trainer')
        for key, value in {**EXPECTED, 'actual_epochs': 6, 'classes': CLASSES,
                           'validation_domains': DOMAINS, 'initial_weights_sha256': INITIAL_SHA,
                           'architecture': 'lraspp_mobilenet_facility_auxiliary_v1',
                           'train_dacl': 6225, 'train_damsegment': 1585, 'train_codebrim': 6438,
                           'val_dacl': 710, 'val_damsegment': 424}.items():
            require(training.get(key) == value, f'Actual predeclared condition changed: {key}')
        require(training.get('spatial_manifest_sha256') == training.get('core_spatial_manifest_sha256'),
                'Only the unchanged original core TRAIN manifest is allowed')
        require(training.get('spatial_manifest_audit', {}).get('full_photo_or_patch_rows') == 14248,
                'Original full TRAIN count changed')
        require(not any(key in training for key in ('hard_training_sampling', 'photo_supplement_sha256',
                    'supplement_spatial_sha256', 'train_convid', 'train_s2ds', 'train_peccd')),
                'A different source/sampling intervention entered this pair')
    for key in MATCHED_KEYS:
        require(key in control and key in treatment and control[key] == treatment[key],
                f'Paired source/sampler/loss/provenance changed or missing: {key}')
    for training, weight in ((control, 0.), (treatment, .25)):
        ranking = training.get('target_ranking', {})
        require(ranking.get('weight') == weight and ranking.get('classes') == TARGETS
                and ranking.get('sampling_changed') is False and ranking.get('new_labels_asserted') == 0
                and ranking.get('public_outputs_changed') is False,
                'Ranking changed public labels, sampling, output or declared strength')
        require(all(isinstance(ranking.get(k), str) and ranking[k] for k in ('helper_sha256', 'scope', 'formula')),
                'Ranking helper provenance/scope/formula missing')
    require({k: v for k, v in control['target_ranking'].items() if k != 'weight'}
            == {k: v for k, v in treatment['target_ranking'].items() if k != 'weight'},
            'Ranking helper definition changed between the pair')
    validate_expected_sampling(control)
    return {key: control[key] for key in MATCHED_KEYS if key not in ('auxiliary_audit', 'additional_test')}


def validate_train_audit(audit, training):
    require(audit.get('status') == 'audited' and audit.get('classes') == CLASSES
            and audit.get('full_base_rows') == 14248 and audit.get('derived_crop_rows') == 12041,
            'TRAIN negative audit scope/counts changed')
    require(audit.get('input_sha256', {}).get('core_spatial_train') == training['core_spatial_manifest_sha256']
            and audit['input_sha256'].get('auxiliary_train') == training['auxiliary_manifest_sha256'],
            'TRAIN negative audit belongs to different source labels')
    policy = audit.get('policy', {})
    require(all(policy.get(key) is False for key in ('native_validation_or_test_annotations_opened',
                'validation_or_test_labels_used', 'validation_or_test_images_read', 'model_predictions_used',
                'source_annotations_or_manifests_modified', 'training_executed')),
            'Negative audit used heldout/model inputs or modified original labels')
    require(audit.get('counts', {}).get('full', {}).get('general_normal_or_safe_ground_truth_asserted') is False,
            'Source absence was reinterpreted as normal/safe ground truth')
    return {'full_base_rows': 14248, 'derived_crop_rows': 12041,
            'full_target_known_negative_counts': {label: audit['counts']['full']['per_label'][label]['negative'] for label in TARGETS},
            'both_targets_known_negative_full_rows': audit['counts']['full']['both_targets_known_negative'],
            'global_normal_or_safe_ground_truth_asserted': False}


def validate_expected_sampling(training):
    sampling = training.get('expected_sampling', {})
    require(set(sampling) == set(DOMAINS), 'Unexpected source sampling domain')
    for domain, mass in zip(DOMAINS, EXPECTED['domain_proportions']):
        require(set(sampling[domain]) == {'full', 'crop'}, 'Full/crop sampling measurements missing')
        values = [number(sampling[domain][key], 'sampling mass') for key in ('full', 'crop')]
        require(math.isclose(sum(values), mass, abs_tol=1e-12), 'Source sampling mass changed')
    require(set(training.get('expected_label_sampling', {})) == set(CLASSES), 'Seven expected label masses missing')
    for point in training['expected_label_sampling'].values():
        require(set(point) == {'positive', 'negative', 'unknown'}, 'Known/unknown sampling measurements missing')
        require(math.isclose(sum(number(v, 'label mass') for v in point.values()), 1., abs_tol=1e-12),
                'Label sampling masses do not sum to one')


def nonnegative_integer(value, message):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, message)
    return value


def validate_actual_pair(control, treatment):
    """Check saved epoch draws and actual ranking use, without individual rows."""
    histories = [entry.get('actual_sampling_history', []) for entry in (control, treatment)]
    require(all(len(history) == 6 for history in histories), 'Six actual epoch sampling records required')
    for history, active in zip(histories, (False, True)):
        for epoch, row in enumerate(history, 1):
            require(row.get('epoch') == epoch, 'Epoch sampling order changed')
            domain, kind, joints = [row.get(key, {}) for key in ('domain_counts', 'row_type_counts', 'full_target_joint_counts')]
            require(set(domain) == set(DOMAINS) and set(kind) == {'full', 'crop'} and set(joints) == set(DOMAINS),
                    'Actual source/full/crop/joint sampling fields missing')
            require(sum(nonnegative_integer(v, 'Invalid source draw count') for v in domain.values()) == 14248
                    and sum(nonnegative_integer(v, 'Invalid full/crop draw count') for v in kind.values()) == 14248,
                    'Actual epoch draw budget changed')
            for source in DOMAINS:
                require(set(joints[source]) == {'00', '10', '01', '11', 'unknown'}, 'Full-target joint states changed')
                total = sum(nonnegative_integer(v, 'Invalid full-target joint count') for v in joints[source].values())
                require(total <= domain[source] and joints[source]['unknown'] == 0,
                        'Target labels or source full membership changed')
                if source == 'codebrim':
                    require(total == domain[source], 'CODEBRIM acquired unprepared derived crop draws')
            require(sum(sum(states.values()) for states in joints.values()) == kind['full'],
                    'Full-target joint counts do not cover actual full draws')
            audit = row.get('ranking_audit', {})
            pairs = nonnegative_integer(audit.get('pair_count'), 'Actual ranking pair count missing')
            batches = nonnegative_integer(audit.get('contributing_batches'), 'Actual ranking batch count missing')
            mean = number(audit.get('unweighted_mean_batch_loss'), 'ranking loss', upper=math.inf)
            groups = audit.get('pair_counts_by_domain_target', {})
            require(isinstance(groups, dict) and set(groups) <= {f'{d}:{k}' for d in range(3) for k in range(2)},
                    'Ranking used a non-original domain or target')
            require(sum(nonnegative_integer(v, 'Invalid ranking group count') for v in groups.values()) == pairs,
                    'Ranking group totals differ from actual pair count')
            require(batches <= math.ceil(14248 / 8), 'Ranking batch count exceeds actual minibatch budget')
            if active:
                require(pairs > 0 and batches > 0 and mean > 0, 'Treatment did not use ranking in every actual epoch')
            else:
                require(pairs == batches == mean == 0 and not groups, 'Control unexpectedly used ranking')
    for a, b in zip(*histories):
        for key in ('domain_counts', 'row_type_counts', 'full_target_joint_counts'):
            require(a[key] == b[key], f'Matched-seed actual draws differ: {key}')
    return {'actual_source_draws_identical_each_epoch': True, 'actual_full_crop_draws_identical_each_epoch': True,
            'actual_full_target_joint_draws_identical_each_epoch': True, 'treatment_ranking_used_all_six_epochs': True}


def average_precision(targets, scores):
    """Noninterpolated AP with tied scores grouped, using CPU arithmetic only."""
    require(len(targets) == len(scores) and all(t in (0, 1) for t in targets), 'Invalid known AP input')
    positives = sum(targets)
    if not positives:
        return None
    rows = sorted(zip(scores, targets), key=lambda row: -row[0])
    tp = seen = 0
    ap = 0.
    while seen < len(rows):
        end = seen + 1
        while end < len(rows) and rows[end][0] == rows[seen][0]:
            end += 1
        new_positives = sum(t for _, t in rows[seen:end])
        tp += new_positives
        ap += (new_positives / positives) * (tp / end)
        seen = end
    return ap


def verify_validation_cache(raw, frozen, entry, domain):
    """Recompute saved confusion counts and AP; return only safe aggregates."""
    require(raw.get('split') == 'val' and raw.get('weights_sha256') == entry['weights_sha256']
            and frozen.get('signature', {}).get('weights_sha256') == entry['weights_sha256']
            and frozen['signature'].get('grid') == 1 and frozen.get('classes') == CLASSES,
            'Raw validation/frozen view provenance changed')
    targets, scores = raw.get('targets', []), raw.get('probabilities', [])
    fixed = frozen.get('probabilities', [])
    require(len(targets) == len(scores) == len(fixed) == COUNTS[domain], 'Validation membership counts changed')
    for target, score, freeze in zip(targets, scores, fixed):
        require(len(target) == len(score) == len(freeze) == 7
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v in (-1, 0, 1) for v in target),
                'Validation label order/shape changed')
        for a, b in zip(score, freeze):
            number(a, 'validation probability'); number(b, 'frozen validation probability')
            require(math.isclose(a, b, abs_tol=1e-5), 'Raw and frozen full-photo probabilities differ')
    measured = {}
    for k, label in enumerate(CLASSES):
        gold = [row[k] for row in targets]
        asserted = k in KNOWN_INDICES[domain]
        require(all(t >= 0 for t in gold) if asserted else all(t == -1 for t in gold),
                'An original source unknown label became an asserted negative')
        point = entry['ranking_ap'][domain][label]
        require(point['known_photos'] == (len(gold) if asserted else 0)
                and point['positive_photos'] == (sum(gold) if asserted else 0), 'AP source support changed')
        ap = average_precision(gold, [row[k] for row in scores]) if asserted else None
        # The trainer accumulated float32 recall. CPU Python arithmetic uses
        # float64: tolerate only its small arithmetic difference, not a changed
        # metric. Preserve the original saved AP for all comparison/gate values.
        require((ap is None and point['ap'] is None) or (ap is not None and point['ap'] is not None
                and math.isclose(ap, point['ap'], rel_tol=0., abs_tol=1e-7)),
                'Recorded AP differs from raw validation probabilities')
        measured[label] = {**point, 'cpu_recomputed_ap': ap,
                           'cpu_recomputed_minus_saved_ap': ap - point['ap'] if ap is not None else None}
        if label in TARGETS:
            threshold = entry['per_class'][label]['threshold']
            predictions = [row[k] >= threshold for row in fixed]
            actual = {key: 0 for key in ('tp', 'fn', 'fp', 'tn')}
            for truth, prediction in zip(gold, predictions):
                key = ('tp' if prediction else 'fn') if truth == 1 else ('fp' if prediction else 'tn')
                actual[key] += 1
            recorded = entry['per_class'][label]['domains'][domain]
            require(all(recorded[key] == value for key, value in actual.items()),
                    'Saved target confusion counts differ from frozen raw probabilities')
    digest = hashlib.sha256(json.dumps(targets, separators=(',', ':')).encode('utf-8')).hexdigest()
    return measured, digest


def descriptive_interval(errors, total):
    nonnegative_integer(errors, 'Invalid interval error count')
    require(isinstance(total, int) and not isinstance(total, bool) and 0 < total and errors <= total,
            'Invalid interval support')
    z = 1.959963984540054
    p = errors / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def error_rows(entries):
    rows = []
    for entry in entries:
        for label in TARGETS:
            for domain in DOMAINS:
                point = entry['per_class'][label]['domains'][domain]
                rows.append({'run': entry['run'], 'class': label, 'domain': domain, **point,
                             'fnr_wilson95_descriptive': descriptive_interval(point['fn'], point['positive_photos']),
                             'fpr_wilson95_descriptive': descriptive_interval(point['fp'], point['negative_photos'])})
    require(len(rows) == 18, 'Exactly eighteen measured model/source/target rows required')
    return rows


def validate_technical_verification(proof, entries, protocol_sha, profile_sha):
    require(proof.get('status') == 'passed' and proof.get('study_protocol_sha256') == protocol_sha
            and proof.get('app_profile_sha256') == profile_sha and proof.get('app_profile_unchanged') is True
            and proof.get('research_models_deployed') is False
            and proof.get('accuracy_measured_by_this_check') is False
            and proof.get('independent_field_safety_verified') is False, 'Technical verification scope/provenance changed')
    rows = proof.get('experiments', [])
    require(len(rows) == 2 and {row.get('run') for row in rows} == {CONTROL, TREATMENT},
            'Technical verification checkpoint pair missing')
    for entry in entries[1:]:
        row = next(row for row in rows if row['run'] == entry['run'])
        require(row.get('weights_sha256') == entry['weights_sha256']
                and row.get('study_protocol_sha256') == protocol_sha
                and row.get('public_shape') == [1, 7] and row.get('private_spatial_shape') == [1, 7, 80, 80]
                and row.get('auxiliary_shape') == [1, 19] and row.get('outputs_finite') is True
                and row.get('public_training_logits_equal') is True, 'Technical checkpoint/output contract differs')
    preflight = proof.get('actual_train_gpu_preflight', {})
    require(preflight.get('status') == 'passed' and preflight.get('finite_loss_and_gradients') is True
            and preflight.get('public_shape') == [8, 7] and preflight.get('ranking_pair_count') == 32,
            'Actual TRAIN ranking integration check missing')
    freeze = proof.get('pre_training_source_snapshot', {})
    expected_paths = {'scripts/train_facility_spatial.py', 'scripts/facility_target_ranking.py',
                      'reports/facility-target-discrimination-protocol.json'}
    require(freeze.get('comparison_started_before_outcomes') is True
            and set(freeze.get('sources', {})) == expected_paths, 'Pre-training source/protocol snapshot missing')
    for relative, row in freeze['sources'].items():
        require(row.get('equal') is True and row.get('git_blob_sha256') == row.get('executed_file_sha256') == sha(ROOT / relative),
                'Committed pre-training source/protocol revision changed')
    return {'checkpoint_pair_passed': True, 'public_outputs': 7, 'private_spatial_shape': [1, 7, 80, 80],
            'private_auxiliary_outputs': 19, 'actual_train_preflight_pair_count': 32,
            'pre_training_commit': freeze.get('pre_training_commit'),
            'accuracy_measured_by_technical_check': False}


def research_gate(reference, control, treatment, declared=GATE):
    require(declared == GATE, 'Research nomination gate changed after declaration')
    comparisons_to = {}
    for name, base in (('initializer', reference), ('control', control)):
        gain = base['worst_error'] - treatment['worst_error']
        target_regressions, ap_regressions = [], []
        for label in TARGETS:
            for domain in DOMAINS:
                for rate in ('fnr', 'fpr'):
                    delta = treatment['per_class'][label]['domains'][domain][rate] - base['per_class'][label]['domains'][domain][rate]
                    target_regressions.append({'class': label, 'domain': domain, 'metric': rate, 'change': delta})
        for domain in DOMAINS:
            for label in CLASSES[2:]:
                a, b = base['ranking_ap'][domain][label], treatment['ranking_ap'][domain][label]
                require((a['known_photos'], a['positive_photos']) == (b['known_photos'], b['positive_photos']),
                        'Other-class AP support changed at research gate')
                require((a['ap'] is None) == (b['ap'] is None), 'Unknown AP acquired a measured value')
                if a['ap'] is not None:
                    ap_regressions.append({'class': label, 'domain': domain, 'ap_drop': a['ap'] - b['ap']})
        require(len(target_regressions) == 12 and len(ap_regressions) == 8, 'Research guard support changed')
        improvement_passed = gain >= .005 - 1e-12
        target_passed = all(row['change'] <= .02 + 1e-12 for row in target_regressions)
        other_passed = all(row['ap_drop'] <= .02 + 1e-12 for row in ap_regressions)
        comparisons_to[name] = {'maximum_error_improvement': gain, 'minimum_improvement_passed': improvement_passed,
                                'target_regression_guard_passed': target_passed,
                                'other_known_class_ap_guard_passed': other_passed,
                                'target_rate_changes': target_regressions, 'other_class_ap_drops': ap_regressions,
                                'passed': improvement_passed and target_passed and other_passed}
    return {'thresholds': declared, 'comparisons': comparisons_to,
            'research_candidate_nominated': all(row['passed'] for row in comparisons_to.values()),
            'descriptive_only': True, 'deployment_authorized': False}


def render(result):
    entries, change = result['experiments'], result['comparisons']
    pct = lambda value: f'{value * 100:.2f}%'
    interval = lambda values: f'{pct(values[0])}–{pct(values[1])}'
    lines = ['# 기존 손상·음성 사진 구분 대조 실험 결과', '',
             '실제 완료된 학습과 고정된 전체 사진 검증 결과만 기록한다. 새 사진·정답 편집 없이 기존 원본 TRAIN의 같은 출처, 같은 항목의 양성/음성 구분을 보조 손실로 학습했다.',
             'seed 53, 각 6epoch, epoch당 복원 추출 14,248회, 640 입력, batch 8. 대조군 순위 손실 0, 보강군 0.25이며 두 군의 학습률은 backbone 0.00002 / head 0.000125로 동일하다.',
             '기존 추출 비중·full/crop·기본 사진/픽셀/19종 보조 손실 가중치를 보존했다. 순위 손실은 원본 full 사진의 known 양성/음성만 같은 미니배치·출처·항목 안에서 비교하며 crop과 미확인 정답은 제외했다.', '',
             '| 모델 | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한 5% 기준 |',
             '|---|---:|---:|---:|---|']
    for entry in entries:
        lines.append(f"| {TITLES[entry['run']]} | {entry['actual_epochs']} | {entry['best_epoch']} | {pct(entry['worst_error'])} | {'검증 통과' if entry['target_passed'] else '미달'} |")
    lines += ['', f"보강군 − 초기 모델 최대 오류 {change['maximum_error_treatment_minus_initializer_pp']:+.2f}pp, 보강군 − 대조군 {change['maximum_error_treatment_minus_control_pp']:+.2f}pp. 양수는 악화다.",
              '최대값은 균열·박락 두 항목, 세 출처의 FN/(TP+FN)와 FP/(FP+TN) 12개 비율 중 최대다. 전체 앱 오답 사진 비율이 아니다.', '', '## 사전 선언한 연구 후보 판단', '']
    nominated = result['research_gate']['research_candidate_nominated']
    lines += [f"연구 후보 기준: **{'통과' if nominated else '미달'}**. 초기 모델과 대조군 각각보다 최대 오류를 최소 0.5pp 줄이고, 각 target FNR/FPR의 악화는 2pp 이하, 정답이 있는 나머지 5종 AP의 하락은 0.02 이하라는 사전 선언을 사용했다.",
              '이는 허용할 회귀 범위를 정한 연구 후보 선별 기준이다. 모든 항목 개선, 통계적 유의성, 앱 성능 인증 또는 엄격한 5% 통과를 뜻하지 않는다.',
              '엄격한 기준은 두 target의 세 출처 FNR/FPR이 각각 모두 5% 미만이다. 이번 비교는 앱에 배포하지 않았고 보류 시험을 실행하지 않았다.', '']
    for name, point in result['research_gate']['comparisons'].items():
        lines.append(f"- {'초기 모델' if name == 'initializer' else '대조군'} 대비: 최대 오류 개선 {point['maximum_error_improvement'] * 100:+.2f}pp, 개선 조건 {'통과' if point['minimum_improvement_passed'] else '미달'}, target 회귀 제한 {'통과' if point['target_regression_guard_passed'] else '미달'}, 기타 AP 제한 {'통과' if point['other_known_class_ap_guard_passed'] else '미달'}.")
    lines += ['', '## 이번 판단과 다음 우선순위', '',
              '이번 비교에서는 초기 모델보다 최대 오류가 줄지 않아 새 가중치를 앱에 적용하지 않는다. 작은 손상은 초기 모델 대비 균열·박락에서 각각 한 장씩 덜 놓쳤지만, 일부 출처·항목은 악화됐다. 이 한 번의 관측을 일반적인 성능 향상으로 주장하지 않는다.',
              '다음 연구는 기존 구조의 반복 횟수를 늘리기보다 작은 손상의 세부 특징을 보존하는 모델 구조 후보 하나를 기존 구조와 같은 예산으로 비교하는 것을 우선한다. 이를 실행하기 전 TRAIN의 정답 범위·같은 부모 사진의 분리·7종 출력 호환성을 확인하고, 새 사전 계획에 시간·메모리·채택 기준을 고정한다. 아직 그 구조를 구현하거나 학습했다는 뜻은 아니다.',
              '검수는 TRAIN 원본과 저자 정의를 사용하고 미확인을 음성으로 바꾸지 않는다. 공장 벽·바닥·기둥의 정상/손상 사진이 확보되면 촬영 장소·회차를 분리한 별도 평가로 적용 범위를 확인한다. 현재 교량·댐 검증값을 공장 현장의 오류율로 환산할 수 없다.']
    lines += ['', '## 관측 미탐·오탐과 오차 범위', '',
              'Wilson 양측 95% 구간은 각 관측 비율의 기술 통계다. 독립 사진이라는 이항 가정을 요구한다. 상관된 사진과 같은 VAL에서 반복한 epoch·임계값 선택 때문에 확인적 신뢰구간, 미래 현장 오류 보장 또는 모델 간 유의성 검정으로 해석할 수 없다.', '',
              '| 모델 | 출처 | 항목 | FN/양성 | 미탐률 | Wilson 95% | FP/음성 | 오탐률 | Wilson 95% |',
              '|---|---|---|---:|---:|---|---:|---:|---|']
    for row in result['measured_error_rows']:
        lines.append(f"| {TITLES[row['run']]} | {row['domain']} | {LABELS[row['class']]} | {row['fn']}/{row['positive_photos']} | {pct(row['fnr'])} | {interval(row['fnr_wilson95_descriptive'])} | {row['fp']}/{row['negative_photos']} | {pct(row['fpr'])} | {interval(row['fpr_wilson95_descriptive'])} |")
    lines += ['', '## 작은 손상과 기존 7종', '',
              'DACL 원본 폴리곤의 640 입력 면적 비율 1% 미만인 기존 양성 사진이다. 실제 물리적 크기·정밀 위치 평가가 아니다.', '',
              '| 항목 | 초기 FN/양성 | 대조 FN/양성 | 보강 FN/양성 | 초기 미탐률 | 대조 미탐률 | 보강 미탐률 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label in TARGETS:
        values = [entry['small_dacl_polygon_area_below_one_percent'][label] for entry in entries]
        cells = [f"{v['false_negatives']}/{v['positive_photos']}" for v in values] + [pct(v['fnr']) for v in values]
        lines.append(f"| {LABELS[label]} | {' | '.join(cells)} |")
    lines += ['', 'AP는 확률 순위 지표이며 오탐률·앱 정확도와 다르다. 원본 VAL 확률에서 정답이 있는 AP 42개(세 모델 × DACL 7종/Dam 2종/CODEBRIM 5종)를 재계산해 저장된 값과 대조했다. 미확인은 0점으로 계산하지 않는다.', '',
              '| 출처 | 항목 | 초기 AP | 대조 AP | 보강 AP | 보강−대조 | 보강−초기 |', '|---|---|---:|---:|---:|---:|---:|']
    for row in change['ranking_ap_changes']:
        domain, label = row['domain'], row['class']
        cells = [f"{e['ranking_ap'][domain][label]['ap']:.4f}" if e['ranking_ap'][domain][label]['ap'] is not None else '미확인' for e in entries]
        cells += [f'{row[k]:+.4f}' if row[k] is not None else '해당 없음' for k in ('treatment_minus_control_ap', 'treatment_minus_initializer_ap')]
        lines.append(f"| {domain} | {LABELS[label]} | {' | '.join(cells)} |")
    lines += ['', '## 추출·학습 실적과 해석 범위', '',
              'DACL/Dam/CODEBRIM 추출 비중은 70/10/20%다. 실제 epoch별 출처, full/crop, 두 target의 full 정답 조합 추출 수는 두 군에서 동일했는지 검증했고 아래 JSON에 기록했다. 복원 추출 횟수는 고유 사진 수가 아니다.',
              '보강군의 실제 순위 pair/기여 batch 수를 여섯 epoch 모두 확인했다. 같은 출처·항목 안의 pair 평균 후 기여 그룹들을 동일 비중으로 평균하므로 추가 손실 영향은 기본 출처 추출 비중과 동일하다고 주장하지 않는다.',
              '현재 연구 자료는 교량·댐 콘크리트 손상 중심이다. 공장 벽·바닥·기둥의 공통 시각적 손상에 연결하려는 단계이며 산업체 현장 정확도는 측정하지 않았다. 새로운 시설 범위나 데이터셋을 이번 학습에 추가하지 않았다.',
              '학습 메타데이터의 full은 기존 원본 행이라는 뜻이며 CODEBRIM 원저자의 patch도 포함한다. 14,248행을 독립된 전체 현장 사진 14,248장으로 해석하지 않는다. 기존 crop도 새 독립 사진이 아니다.',
              '기존 항목의 음성은 원저자 주석 또는 의미 정의에 근거해 변환한 항목별 음성이다. DACL 7종이 없는 사진에도 다른 원본 태그가 있을 수 있고 CODEBRIM background는 저자가 제공한 5종의 음성이다. 이를 전체 시설의 정상·안전 정답으로 바꾸지 않았다. 정답 경계의 모호성은 남아 있다.',
              'DamSegment 분류 Non-Crack의 두 항목 음성은 저자 원논문 4.4의 intact concrete 정의에 근거한 기존 의미 변환이다. 별도로 배포된 Spalling=0 태그나 native classification mask가 아니다. 직접 분할 정답과 이 변환의 차이 및 공식 출처는 [정답 범위](FACILITY_LABEL_SCOPE_KO.md)에 기록했다.',
              '한 seed에서 기존 VAL을 반복 선택한 연구다. 95% 구간과 후보 기준 통과만으로 새 현장 성능 또는 5% 미만 오류를 보장하지 않는다. 원본 라벨·검증 정답을 바꾸거나 오답을 제외하지 않았다.',
              f"앱 기본 프로필 `{result['app_profile']['version']}`의 SHA256은 기존 파일과 동일하다. deployed=false이며 보류 시험 결과를 열람하거나 사용하지 않았다.",
              '개별 사진·주석·검수 의견·절대 경로는 이 공개 보고에 포함하지 않는다. 원본/가공 사진은 원저자 데이터 라이선스의 재배포 조건을 따른다. 검출 없음으로 안전을 확정하지 않으며 최종 판단은 점검자가 한다.', '',
              '[실측 집계 JSON](facility-target-discrimination-comparison.json), [사전 선언](facility-target-discrimination-protocol.json), [기존 TRAIN 음성 감사](facility-target-negative-data-audit.json), [완료 가중치 기술 점검](facility-target-discrimination-technical-verification.json)', '']
    return '\n'.join(lines)


def main():
    names = (REFERENCE, CONTROL, TREATMENT)
    # load_run can read a saved test result: fail before calling it if one exists.
    require(not any((ROOT / 'reports' / f'{name}-target-test.json').exists() for name in names),
            'This no-test report refuses any run with a saved target-test result')
    protocol = validate_protocol(read(PROTOCOL_PATH))
    loaded = [load_run(name) for name in names]
    trainings, entries = [row[0] for row in loaded], [row[1] for row in loaded]
    conditions = validate_pair(trainings[1], trainings[2], entries[0]['weights_sha256'], protocol, sha(PROTOCOL_PATH))
    for key, path in (('training_script_sha256', 'scripts/train_facility_spatial.py'),
                      ('model_source_sha256', 'safelog_ai/spatial_classifier.py'),
                      ('auxiliary_model_source_sha256', 'safelog_ai/auxiliary_classifier.py'),
                      ('photo_supplement_helper_sha256', 'scripts/facility_photo_supplement.py')):
        require(trainings[1][key] == sha(ROOT / path), f'Recorded paired source revision unavailable/changed: {key}')
    require(trainings[1]['target_ranking']['helper_sha256'] == sha(ROOT / 'scripts/facility_target_ranking.py'),
            'Recorded ranking helper revision unavailable/changed')
    require(trainings[1]['core_spatial_manifest_sha256'] == sha(ROOT / 'data/facility-spatial-training/train.json')
            and trainings[1]['auxiliary_manifest_sha256'] == sha(ROOT / 'data/facility-auxiliary-training/train.json'),
            'Original TRAIN supervision changed after the frozen pair')
    for key in ('architecture', 'classes', 'imgsz', 'split_sha256', 'model_source_sha256',
                'auxiliary_model_source_sha256', 'auxiliary_manifest_sha256', 'auxiliary_classes',
                'validation_domains', 'additional_validation', 'spatial_manifest_sha256'):
        require(trainings[0].get(key) == trainings[1].get(key), f'Initializer source/evaluation changed: {key}')
    digests = []
    known_ap_count = 0
    for training, entry in zip(trainings, entries):
        run = ROOT / 'runs' / entry['run']
        target_digests = {}
        for domain in DOMAINS:
            measured, digest = verify_validation_cache(read(run / f'validation-{domain}.json'),
                               read(run / f'target-validation-{domain}-grid1.json'), entry, domain)
            entry['ranking_ap'][domain] = measured
            known_ap_count += sum(point['ap'] is not None for point in measured.values())
            target_digests[domain] = digest
        digests.append(target_digests)
        if entry['run'] != REFERENCE:
            entry['actual_sampling_history'] = [
                {'epoch': row['epoch'], 'domain_counts': row.get('sampled_domain_counts'),
                 'row_type_counts': row.get('sampled_row_type_counts'),
                 'full_target_joint_counts': row.get('sampled_full_target_joint_counts'),
                 'ranking_audit': row.get('target_ranking_audit')}
                for row in read(run / 'history.json')]
    require(digests[0] == digests[1] == digests[2], 'Validation truth/order changed across the models')
    require(known_ap_count == 42, 'Exactly 42 originally known AP measurements required')
    actual_draws = validate_actual_pair(entries[1], entries[2])
    profile_path = ROOT / 'reports/facility-inference-profile.json'
    previous = ROOT / 'reports/facility-inference-profile-round1.json'
    profile = read(profile_path)
    require(profile.get('version') == 'facility-validation-v2' and sha(profile_path) == sha(previous),
            'App profile changed: cannot report unchanged deployment')
    technical_path = ROOT / 'reports/facility-target-discrimination-technical-verification.json'
    technical = validate_technical_verification(read(technical_path), entries, sha(PROTOCOL_PATH), sha(profile_path))
    audit_path = ROOT / 'reports/facility-target-negative-data-audit.json'
    require(audit_path.is_file(), 'TRAIN negative audit missing')
    negative_audit = validate_train_audit(read(audit_path), trainings[1])
    result = {'schema': 'facility_target_discrimination_comparison_v1', 'split': 'val', 'deployed': False,
              'test_executed': False, 'industrial_field_performance_measured': False,
              'new_training_photos': 0, 'original_labels_edited': 0,
              'protocol_sha256': sha(PROTOCOL_PATH), 'protocol': protocol,
              'matched_pair_conditions': conditions, 'actual_pair_sampling_checks': actual_draws,
              'experiments': entries, 'comparisons': comparisons(*entries),
              'measured_error_rows': error_rows(entries), 'known_ap_recomputed_count': known_ap_count,
              'validation_target_array_sha256': digests[0],
              'research_gate': research_gate(*entries, protocol['research_candidate_gate']),
              'strict_target': 'Every crack/spalling source FNR/FPR strictly <0.05; not whole-app accuracy',
              'interval_policy': 'Wilson two-sided 95% descriptive only; independent-photo assumption not validated; repeated validation epoch/cutoff selection and correlated photos prevent confirmatory interpretation',
              'train_negative_audit_sha256': sha(audit_path), 'train_negative_summary': negative_audit,
              'technical_verification_sha256': sha(technical_path), 'technical_verification_summary': technical,
              'app_profile': {'version': profile['version'], 'sha256': sha(profile_path),
                              'previous_sha256': sha(previous), 'unchanged': True},
              'new_training_epochs': sum(e['actual_epochs'] for e in entries[1:]),
              'report_script_sha256': sha(Path(__file__)),
              'limitations': ['Single seed and repeated source-validation epoch/threshold selection',
                              'Bridge/dam source evidence only; industrial workplace accuracy unmeasured',
                              'Source negative means asserted absence of that class, not global facility normal/safe',
                              'Ranking groups have equal loss mass; source draw mass unchanged does not mean unchanged loss influence',
                              'Research nomination tolerates declared regressions and does not authorize deployment',
                              'Unknown source labels retain unknown status; no new classes or truth edits',
                              'Only aggregate measurements/hashes published; source images not redistributed']}
    output = ROOT / 'reports'
    (output / 'facility-target-discrimination-comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'FACILITY_TARGET_DISCRIMINATION_RESULTS_KO.md').write_text(render(result), encoding='utf-8')
    print(json.dumps({'comparison': 'reports/facility-target-discrimination-comparison.json',
                      'research_candidate_nominated': result['research_gate']['research_candidate_nominated'],
                      'maximum_errors': {e['run']: e['worst_error'] for e in entries}, 'deployed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
