"""Report a completed, frozen, paired TRAIN crop-recipe experiment.

Only aggregate measurements are published. Source images, annotation dictionaries,
local absolute paths and individual review cases remain in ignored data/runs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'facility-presence-target-auxiliary'
CONTROL = 'facility-presence-target-roi-control'
TREATMENT = 'facility-presence-target-small-region'
CLASSES = ['concrete_crack', 'concrete_spalling', 'rust_stain', 'exposed_rebar',
           'wet_surface', 'efflorescence', 'surface_cavity']
TARGETS = CLASSES[:2]
DOMAINS = ['dacl', 'damsegment', 'codebrim']
COUNTS = dict(zip(DOMAINS, (710, 424, 611)))
LABELS = dict(zip(CLASSES, ('균열', '박락', '녹 흔적', '철근 노출', '젖은 표면', '백화', '공동')))
TITLES = {REFERENCE: '추가 학습 전 모델', CONTROL: '기존 격자 crop 대조군', TREATMENT: '작은 영역 맥락 crop 보강군'}
PAIR_KEYS = ('architecture', 'classes', 'imgsz', 'split_sha256', 'initial_weights_sha256',
             'seed', 'requested_epochs', 'actual_epochs', 'patience', 'backbone_lr', 'head_lr',
             'auxiliary_manifest_sha256', 'auxiliary_classes', 'auxiliary_weight',
             'training_script_sha256', 'model_source_sha256', 'auxiliary_model_source_sha256',
             'draws_per_epoch', 'domain_proportions', 'core_spatial_manifest_sha256',
             'train_dacl', 'train_damsegment', 'train_codebrim', 'val_dacl', 'val_damsegment',
             'validation_domains', 'additional_validation', 'additional_test', 'loss')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, lower=0., upper=1.):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and lower <= value <= upper, f'Invalid measured {name}')
    return value


def validate_pair(control, treatment, initializer_sha):
    """Same initializer/configuration; crop-derived labels and weights may change."""
    for key in PAIR_KEYS:
        require(key in control and key in treatment and control[key] == treatment[key],
                f'Paired condition changed or missing: {key}')
    require(control['classes'] == CLASSES and control['architecture'] == 'lraspp_mobilenet_facility_auxiliary_v1',
            'Pair must preserve the seven-label auxiliary architecture')
    expected = {'seed': 51, 'requested_epochs': 6, 'actual_epochs': 6, 'patience': 6,
                'imgsz': 640, 'draws_per_epoch': 14248, 'backbone_lr': .00004,
                'head_lr': .00025, 'auxiliary_weight': .5, 'domain_proportions': [.7, .1, .2],
                'validation_domains': DOMAINS}
    for key, value in expected.items():
        require(control[key] == value, f'Predeclared paired condition changed: {key}')
    require(control['initial_weights_sha256'] == initializer_sha, 'Pair initializer is not frozen auxiliary weights')
    require(control['spatial_manifest_sha256'] == control['core_spatial_manifest_sha256'],
            'Control must use the original core crop manifest')
    require(treatment['spatial_manifest_sha256'] != control['spatial_manifest_sha256'],
            'Treatment did not use a distinct prepared crop manifest')
    audit = treatment.get('spatial_manifest_audit', {})
    require(audit.get('status') == 'prepared' and audit.get('recipe') == 'small_source_region_context_v1'
            and audit.get('source_manifest_sha256') == control['core_spatial_manifest_sha256']
            and audit.get('unchanged_full_count') == 14248 and audit.get('replaced_rows', 0) > 0,
            'Treatment crop preparation provenance changed or missing')
    return {key: control[key] for key in PAIR_KEYS}


def validate_frozen(training, selection, weights_sha, full_only=True):
    require(training.get('status') == 'complete', 'Measurements require completed training')
    require(training.get('weights_sha256') == weights_sha and selection.get('weights_sha256') == weights_sha,
            'Training/selection weights are not the same frozen checkpoint')
    require(training.get('classes') == CLASSES and selection.get('classes') == CLASSES
            and selection.get('selection_split') == 'val', 'Validation classes/split changed')
    selected = selection.get('selected', {})
    require(selected.get('grid') == 1 and selected.get('views') == 1, 'Compare original full-photo inference only')
    if full_only:
        require(selection.get('configured_grids') == [1], 'Paired runs must not select a different evaluation view')
    require(selection.get('validation_counts') == COUNTS, 'Validation source membership/counts changed')
    return selected


def measured_metrics(selected):
    """Retain counts, denominators and rates; reject unmeasured or inconsistent rows."""
    points = selected.get('per_class', {})
    require(set(points) == set(TARGETS), 'Both target classes must have actual measurements')
    result, all_errors = {}, []
    for label in TARGETS:
        point = points[label]
        threshold = number(point.get('threshold'), 'threshold')
        require(set(point.get('domains', {})) == set(DOMAINS), 'All three source domains must be measured')
        rows = {}
        for domain in DOMAINS:
            metrics = point['domains'][domain]
            counts = {k: metrics.get(k) for k in ('tp', 'fn', 'fp', 'tn')}
            require(all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts.values()),
                    'Missing actual confusion counts')
            positive, negative = counts['tp'] + counts['fn'], counts['fp'] + counts['tn']
            require(positive > 0 and negative > 0 and positive + negative == COUNTS[domain],
                    'Unsupported/missing target denominator')
            fnr, fpr = counts['fn'] / positive, counts['fp'] / negative
            for key, actual in (('fnr', fnr), ('fpr', fpr)):
                recorded = number(metrics.get(key), key)
                require(math.isclose(recorded, actual, abs_tol=1e-12), 'Rate differs from recorded confusion counts')
            rows[domain] = {**counts, 'positive_photos': positive, 'negative_photos': negative, 'fnr': fnr, 'fpr': fpr}
            all_errors.extend((fnr, fpr))
        result[label] = {'threshold': threshold, 'domains': rows}
    worst = max(all_errors)
    passed = all(value < .05 for value in all_errors)
    require(math.isclose(number(selected.get('worst_error'), 'worst error'), worst, abs_tol=1e-12)
            and selected.get('target_passed') is passed, 'Headline target result differs from actual rates')
    return {'per_class': result, 'worst_error': worst, 'target_passed': passed}


def size_summary(audit, weights_sha, metrics):
    require(audit.get('split') == 'val' and audit.get('weights_sha256') == weights_sha, 'Size audit is not matching frozen validation')
    result = {}
    for label in TARGETS:
        point = audit.get('per_class', {}).get(label, {})
        require(point.get('threshold') == metrics['per_class'][label]['threshold'], 'Size audit threshold changed')
        bins = point.get('positive_size_bins', [])
        require(len(bins) == 4 and [b.get('polygon_fraction') for b in bins[:2]] == [[0., .001], [.001, .01]],
                'Small-area source bins changed')
        clean = []
        for row in bins:
            n, fn = row.get('positive_photos'), row.get('false_negatives')
            require(isinstance(n, int) and isinstance(fn, int) and 0 <= fn <= n, 'Invalid size-audit counts')
            rate = fn / n if n else None
            require(row.get('fnr') == rate, 'Size-audit rate differs from counts')
            clean.append({'polygon_fraction': row['polygon_fraction'], 'positive_photos': n, 'false_negatives': fn, 'fnr': rate})
        reference = metrics['per_class'][label]['domains']['dacl']
        require(sum(b['positive_photos'] for b in clean) == reference['positive_photos']
                and sum(b['false_negatives'] for b in clean) == reference['fn'], 'Size-audit subset does not cover original DACL positives')
        small_n = sum(b['positive_photos'] for b in clean[:2]); small_fn = sum(b['false_negatives'] for b in clean[:2])
        result[label] = {'positive_photos': small_n, 'false_negatives': small_fn,
                         'fnr': small_fn / small_n if small_n else None, 'bins': clean}
    return result


def ranking_summary(validation, run, weights_sha):
    result = {}
    for domain in DOMAINS:
        raw = read(run / f'validation-{domain}.json')
        cache = read(run / f'target-validation-{domain}-grid1.json')
        require(raw.get('split') == 'val' and raw.get('weights_sha256') == weights_sha
                and cache.get('signature', {}).get('weights_sha256') == weights_sha
                and cache['signature'].get('grid') == 1 and cache.get('classes') == CLASSES,
                'Validation cache checkpoint/classes/view provenance changed')
        targets, scores, frozen_scores = raw.get('targets', []), raw.get('probabilities', []), cache.get('probabilities', [])
        require(len(targets) == len(scores) == len(frozen_scores) == COUNTS[domain], 'Validation cache counts changed')
        for target, score, frozen in zip(targets, scores, frozen_scores):
            require(len(target) == len(score) == len(frozen) == 7 and all(v in (-1, 0, 1) for v in target), 'Wrong public label order/shape')
            for a, b in zip(score, frozen):
                number(a, 'validation probability'); number(b, 'frozen validation probability')
                require(math.isclose(a, b, abs_tol=1e-5), 'Training and frozen full-photo cache order/probabilities differ')
        result[domain] = {}
        for k, label in enumerate(CLASSES):
            known = [row[k] for row in targets if row[k] >= 0]
            ap = validation.get('ranking_ap', {}).get(domain, {}).get(label)
            if len(known) == len(targets):
                number(ap, 'known-class ranking AP')
                state = 'measured_on_asserted_source_labels'
            else:
                require(ap is None, 'An unasserted source class acquired a measured ranking AP')
                state = 'not_measured_source_labels_unknown'
            result[domain][label] = {'ap': ap, 'known_photos': len(known), 'positive_photos': sum(known), 'status': state}
    return result


def load_run(name, full_only=True):
    run = ROOT / 'runs' / name
    training, validation, history, selection = [read(run / file) for file in ('TRAINING.json', 'VALIDATION.json', 'history.json', 'TARGET-SELECTION.json')]
    weights_sha = sha(run / 'best.pt')
    selected = validate_frozen(training, selection, weights_sha, full_only)
    require(training.get('actual_epochs') == len(history) > 0 and [h.get('epoch') for h in history] == list(range(1, len(history) + 1)),
            'Actual epoch history missing or incomplete')
    epoch = validation.get('epoch')
    require(isinstance(epoch, int) and 1 <= epoch <= len(history) and history[epoch - 1] == validation, 'Selected best-epoch measurement differs from actual history')
    require(sha(run / 'SPLIT.json') == training.get('split_sha256'), 'Frozen source split provenance changed')
    metrics = measured_metrics(selected)
    require(math.isclose(training.get('best_worst_target_error', -1), metrics['worst_error'], abs_tol=1e-12)
            and math.isclose(validation.get('worst_target_error', -1), metrics['worst_error'], abs_tol=1e-12), 'Best training and frozen full-photo scores differ')
    sizes = size_summary(read(ROOT / 'reports' / f'{name}-error-size-audit.json'), weights_sha, metrics)
    entry = {'run': name, 'best_epoch': epoch, 'actual_epochs': len(history), 'weights_sha256': weights_sha,
             'configured_grids': selection.get('configured_grids'), 'selected_grid': 1, **metrics,
             'small_dacl_polygon_area_below_one_percent': sizes, 'ranking_ap': ranking_summary(validation, run, weights_sha),
             'expected_sampling': training.get('expected_sampling'), 'expected_label_sampling': training.get('expected_label_sampling'),
             'actual_sampling_history': [{'epoch': h['epoch'], 'domain_counts': h.get('sampled_domain_counts'), 'row_type_counts': h.get('sampled_row_type_counts')} for h in history],
             'provenance': {file: sha(run / file) for file in ('TRAINING.json', 'VALIDATION.json', 'history.json', 'TARGET-SELECTION.json', 'SPLIT.json')}}
    test_path = ROOT / 'reports' / f'{name}-target-test.json'
    entry['test_executed'] = test_path.exists()
    entry['test_result'] = None
    if test_path.exists():
        test = read(test_path)
        require(metrics['target_passed'] and test.get('run') == name and test.get('split') == 'test'
                and test.get('weights_sha256') == weights_sha and test.get('selection_sha256') == sha(run / 'TARGET-SELECTION.json'),
                'Test report is not a gate-permitted result of this frozen candidate')
        entry['test_result'] = {'report': test_path.relative_to(ROOT).as_posix(), 'sha256': sha(test_path), 'target_passed': test['target_passed']}
    return training, entry


def comparisons(reference, control, treatment):
    errors, ap, sizes = [], [], []
    for label in TARGETS:
        for domain in DOMAINS:
            a, b, c = [e['per_class'][label]['domains'][domain] for e in (reference, control, treatment)]
            require((a['positive_photos'], a['negative_photos']) == (b['positive_photos'], b['negative_photos']) == (c['positive_photos'], c['negative_photos']),
                    'Comparisons use different target support counts')
            errors.append({'class': label, 'domain': domain, 'treatment_minus_control_fnr_pp': 100 * (c['fnr'] - b['fnr']),
                           'treatment_minus_control_fpr_pp': 100 * (c['fpr'] - b['fpr']),
                           'control_minus_initializer_fnr_pp': 100 * (b['fnr'] - a['fnr']), 'control_minus_initializer_fpr_pp': 100 * (b['fpr'] - a['fpr']),
                           'treatment_minus_initializer_fnr_pp': 100 * (c['fnr'] - a['fnr']), 'treatment_minus_initializer_fpr_pp': 100 * (c['fpr'] - a['fpr'])})
        values = [e['small_dacl_polygon_area_below_one_percent'][label] for e in (reference, control, treatment)]
        require(len({v['positive_photos'] for v in values}) == 1, 'Small-area subset denominator changed')
        sizes.append({'class': label, 'treatment_minus_control_fnr_pp': 100 * (values[2]['fnr'] - values[1]['fnr']),
                      'control_minus_initializer_fnr_pp': 100 * (values[1]['fnr'] - values[0]['fnr']),
                      'treatment_minus_initializer_fnr_pp': 100 * (values[2]['fnr'] - values[0]['fnr'])})
    for domain in DOMAINS:
        for label in CLASSES:
            values = [e['ranking_ap'][domain][label] for e in (reference, control, treatment)]
            require(len({(v['known_photos'], v['positive_photos']) for v in values}) == 1, 'Ranking AP label support changed')
            measured = all(v['ap'] is not None for v in values)
            ap.append({'class': label, 'domain': domain, 'status': values[0]['status'],
                       'treatment_minus_control_ap': values[2]['ap'] - values[1]['ap'] if measured else None,
                       'control_minus_initializer_ap': values[1]['ap'] - values[0]['ap'] if measured else None,
                       'treatment_minus_initializer_ap': values[2]['ap'] - values[0]['ap'] if measured else None})
    return {'error_rate_changes': errors, 'small_area_changes': sizes, 'ranking_ap_changes': ap,
            'maximum_error_treatment_minus_control_pp': 100 * (treatment['worst_error'] - control['worst_error']),
            'maximum_error_control_minus_initializer_pp': 100 * (control['worst_error'] - reference['worst_error']),
            'maximum_error_treatment_minus_initializer_pp': 100 * (treatment['worst_error'] - reference['worst_error'])}


def train_review_evidence(verification, package, proposals, file_hashes, expected_weights_sha, expected_manifest_sha):
    """Check real saved proof; publish only counts and hashes, never review rows."""
    required_hashes = ('verification_sha256', 'package_sha256', 'html_sha256', 'cache_sha256', 'manifest_sha256', 'proposals_sha256')
    require(all(isinstance(file_hashes.get(key), str) and file_hashes[key] for key in required_hashes),
            'Required saved review evidence hashes missing')
    require(verification.get('browser_photo_annotation_display_verified') is True
            and verification.get('browser_export_saved_json_verified') is True
            and verification.get('user_proposals_automatically_applied') is False,
            'Required browser/export/no-automatic-edit proof missing')
    require(package.get('schema') == 'facility_train_review_v1' and package.get('split') == 'train'
            and package.get('classes') == CLASSES
            and proposals.get('schema') == 'facility_train_review_proposals_v1', 'Wrong TRAIN review evidence schema')
    require(all(value == expected_weights_sha for value in (verification.get('weights_sha256'), package.get('weights_sha256'),
                                                            proposals.get('weights_sha256'))), 'TRAIN review checkpoint differs')
    expected_hashes = {'package_sha256': verification.get('local_package_sha256'),
                       'html_sha256': verification.get('local_html_sha256'),
                       'cache_sha256': verification.get('cache_sha256')}
    require(all(expected is not None and file_hashes.get(key) == expected for key, expected in expected_hashes.items()),
            'Saved TRAIN review files differ from browser verification hashes')
    require(package.get('provenance', {}).get('cache_sha256') == file_hashes.get('cache_sha256'),
            'Review package and scoring cache evidence differ')
    require(package.get('provenance', {}).get('manifest_sha256') == expected_manifest_sha
            and file_hashes.get('manifest_sha256') == expected_manifest_sha, 'Original TRAIN manifest changed after review')
    payload = {key: value for key, value in package.items() if key != 'package_content_sha256'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    require(package.get('package_content_sha256') == digest and proposals.get('source_package_sha256') == digest,
            'Exported proposals belong to a different or modified review package')
    require(proposals.get('policy') == package.get('policy'), 'Exported review policy changed')
    cases, exported = package.get('cases'), proposals.get('cases')
    require(isinstance(cases, list) and isinstance(exported, list) and len(cases) > 0
            and all(isinstance(row, dict) for row in cases + exported), 'Actual TRAIN review rows missing')
    case_ids = [row.get('case_id') for row in cases]
    export_ids = [row.get('case_id') for row in exported]
    require(all(isinstance(value, str) and value for value in case_ids + export_ids) and len(set(case_ids)) == len(case_ids)
            and len(set(export_ids)) == len(export_ids) and set(case_ids) == set(export_ids), 'Export row IDs/counts differ from original package')
    require(verification.get('selected_photos') == verification.get('export_rows') == len(cases) == len(exported)
            and verification.get('original_full_train_count') == package.get('original_full_train_count') == 14248,
            'Review verification counts differ from actual saved files')
    reasons = {'unreviewed', 'small_damage', 'other_damage_confusion', 'capture_quality',
               'annotation_uncertain', 'model_error', 'correct_comparison', 'other'}
    for row in exported:
        require(set(row) == {'case_id', 'reason', 'note'} and row.get('reason') in reasons
                and isinstance(row.get('note'), str), 'Export must contain review opinions only, not rewritten targets')
    with_opinion = sum(row['reason'] != 'unreviewed' or bool(row['note'].strip()) for row in exported)
    return {'selected_photos': len(cases), 'export_rows': len(exported),
            'export_proposals_verified': True, 'proposals_with_reason_or_note': with_opinion,
            'proposals_without_reason_or_note': len(exported) - with_opinion,
            'manual_review_completion_verified': False, 'automatic_original_label_edits': 0,
            'original_train_manifest_unchanged': True,
            'weights_sha256': expected_weights_sha, 'package_content_sha256': digest,
            'evidence_hashes': {key: file_hashes[key] for key in ('verification_sha256', 'package_sha256', 'html_sha256',
                                                               'cache_sha256', 'manifest_sha256', 'proposals_sha256')},
            'scope': 'Local TRAIN review tool/export verification only; source cases and opinions not published; not completed expert review or field accuracy'}


def load_train_review(expected_weights_sha, expected_manifest_sha):
    """Read locally saved proof and download; never disclose private input paths."""
    verification_path = ROOT / 'reports/facility-train-review-tool-verification.json'
    package_path = ROOT / 'runs/facility-train-review-round9-final/TRAIN-REVIEW.json'
    html_path = package_path.parent / 'index.html'
    proposals_path = Path.home() / 'Downloads/TRAIN-review-proposals.json'
    try:
        verification, package, proposals = map(read, (verification_path, package_path, proposals_path))
        cache_path = Path(package['options']['cache']).resolve()
        manifest_path = Path(package['options']['manifest']).resolve()
        require(cache_path.is_relative_to(ROOT / 'runs') and manifest_path == ROOT / 'data/facility-spatial-training/train.json',
                'Review evidence inputs must be the recorded local TRAIN cache and original manifest')
        files = {'verification_sha256': sha(verification_path), 'package_sha256': sha(package_path),
                 'html_sha256': sha(html_path), 'proposals_sha256': sha(proposals_path),
                 'cache_sha256': sha(cache_path), 'manifest_sha256': sha(manifest_path)}
    except (OSError, KeyError) as error:
        raise ValueError('Required locally saved TRAIN review evidence missing') from error
    return train_review_evidence(verification, package, proposals, files, expected_weights_sha, expected_manifest_sha)


def render(result):
    entries = result['experiments']; base, control, treatment = entries
    pct = lambda value: f'{value * 100:.2f}%'
    delta = lambda value: f'{value:+.2f}pp'
    lines = ['# TRAIN 오류 검수·작은 영역 crop 비교 결과', '',
             '이 보고서는 완료된 실측 결과만 기록한다. 두 추가 학습은 같은 초기 모델·seed 51·6epoch·전체 사진 추론 조건을 사용했다.',
             '작은 손상 주변 맥락 crop으로 TRAIN 증강 자료를 교체하는 방법 전체를 비교했다. 다른 항목의 crop 정답과 추출 비중·픽셀 가중치도 달라질 수 있으므로 순수 확대 효과로 해석하지 않는다.', '',
             '| 모델 | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한 5% 목표 | 보류 시험 |',
             '|---|---:|---:|---:|---|---|']
    for e in entries:
        lines.append(f"| {TITLES[e['run']]} | {e['actual_epochs']} | {e['best_epoch']} | {pct(e['worst_error'])} | {'검증 통과' if e['target_passed'] else '미달'} | {'실행됨' if e['test_executed'] else '미실행'} |")
    change = result['comparisons']
    lines += ['', f"추가 학습 전 대비 대조군 최대 오류 변화: {delta(change['maximum_error_control_minus_initializer_pp'])}. 보강군 변화: {delta(change['maximum_error_treatment_minus_initializer_pp'])}.",
              f"보강군 − 대조군 최대 오류 변화: {delta(change['maximum_error_treatment_minus_control_pp'])}. 양수는 악화, 음수는 개선이다.",
              '최대 오류 한 항목의 개선으로 모든 출처·손상 항목이 개선됐다고 판단하지 않는다. 아래 표에 악화도 함께 기록한다.', '',
              '미탐률은 FN/(TP+FN), 오탐률은 FP/(FP+TN). 균열·박락 각각에 대해 세 출처의 두 비율이 모두 5% 미만이어야 통과한다. 전체 앱 오답 사진 비율은 아니다.', '']
    if change['maximum_error_treatment_minus_control_pp'] >= 0:
        lines += ['**이번 작은 영역 crop 보강은 채택을 보류한다.** 대조군보다 최대 오류가 줄지 않았다. 작은 손상의 미탐은 아래의 동일한 양성 사진 수로 비교한다.', '',
                  '후속 작업은 준비한 TRAIN 검수 표본으로 오류 유형을 확인하고 산업체 벽·바닥·기둥과 가까운 자료를 검토하는 것이다. 이번 결과에 근거해 같은 보강군의 epoch를 계속 늘리는 방법은 우선순위에서 낮춘다.', '']
    for e in entries:
        lines += [f"## {TITLES[e['run']]} 상세", '', '| 자료 | 항목 | 미탐/양성 | 미탐률 | 오탐/음성 | 오탐률 |', '|---|---|---:|---:|---:|---:|']
        for label in TARGETS:
            for domain in DOMAINS:
                m = e['per_class'][label]['domains'][domain]
                lines.append(f"| {domain} | {LABELS[label]} | {m['fn']}/{m['positive_photos']} | {pct(m['fnr'])} | {m['fp']}/{m['negative_photos']} | {pct(m['fpr'])} |")
        lines.append('')
    lines += ['## 작은 부위 미탐: DACL 검증', '', '원본 폴리곤을 640×640에 옮겨 계산한 면적 비율 1% 미만의 기존 양성 사진이다. 실제 손상 크기·정밀 위치 평가가 아니다.', '',
              '| 항목 | 초기 모델 미탐/양성 | 대조군 미탐/양성 | 보강군 미탐/양성 | 초기 미탐률 | 대조 미탐률 | 보강 미탐률 |', '|---|---:|---:|---:|---:|---:|---:|']
    for label in TARGETS:
        a, b, c = [e['small_dacl_polygon_area_below_one_percent'][label] for e in entries]
        lines.append(f"| {LABELS[label]} | {a['false_negatives']}/{a['positive_photos']} | {b['false_negatives']}/{b['positive_photos']} | {c['false_negatives']}/{c['positive_photos']} | {pct(a['fnr'])} | {pct(b['fnr'])} | {pct(c['fnr'])} |")
    lines += ['', '## 기존 7종의 순위 AP', '', '각 출처가 정답을 제공하는 항목만 측정한다. AP는 확률 순위 지표이며 오차율이나 앱 정확도가 아니다. 미확인 항목은 0점으로 계산하지 않는다.', '',
              '| 자료 | 항목 | 초기 AP | 대조 AP | 보강 AP | 보강 − 대조 | 보강 − 초기 |', '|---|---|---:|---:|---:|---:|---:|']
    for row in change['ranking_ap_changes']:
        domain, label = row['domain'], row['class']
        values = [e['ranking_ap'][domain][label]['ap'] for e in entries]
        numbers = [f'{v:.4f}' if v is not None else '미확인' for v in values]
        deltas = [f'{row[k]:+.4f}' if row[k] is not None else '해당 없음' for k in ('treatment_minus_control_ap', 'treatment_minus_initializer_ap')]
        lines.append(f"| {domain} | {LABELS[label]} | {' | '.join(numbers + deltas)} |")
    review = result['train_review']
    lines += ['', '## TRAIN 검수와 추출 조건', '',
              f"로컬 TRAIN 검수 화면에는 {review['selected_photos']}장의 원본 학습 사진과 출판자 주석을 연결했다. 실제 저장한 의견 JSON {review['export_rows']}행의 ID·모델·원본 검수 묶음 해시를 검증했다. 원본 TRAIN manifest 해시는 그대로이며 자동 정답 변경은 {review['automatic_original_label_edits']}건이다.",
              f"현재 의견이 들어 있는 행 {review['proposals_with_reason_or_note']}개, 검수 의견이 없는 행 {review['proposals_without_reason_or_note']}개다. 화면 표시와 JSON 저장 기능을 확인한 것이며, 모든 사진의 수동·전문가 검수가 완료됐다는 뜻이 아니다.",
              '원본·가공 사진, 주석 사전과 개별 검수 사례는 GitHub/ZIP에 재배포하지 않는다. 공개 JSON에는 집계만 기록한다.',
              f"실제 crop 교체: {result['crop_audit']['replaced_rows']}행. 새 독립 사진: 0장. 전체 원본 TRAIN 14,248행, 검증·시험 정답을 보존했다.",
              '출처 전체 비중은 DACL 0.7 / Dam 0.1 / CODEBRIM 0.2. epoch마다 복원 추출 14,248회이며 모든 고유 사진을 한 번씩 읽는다는 뜻이 아니다.',
              '세부 full/crop 기대 비중, 클래스별 positive/negative/unknown 기대 비중과 epoch별 실제 추출 수는 아래 JSON에 기록했다. 초기 모델은 당시 기록되지 않은 세부 기대 비중을 null로 남겼다.', '',
              '## 적용 상태와 해석 범위', '',
              f"기본 앱 프로필 `{result['app_profile']['version']}`는 이전 프로필 파일과 SHA256이 동일하다. 이번 두 후보는 기본 앱에 적용하지 않았다.",
              '현재 시설 손상 학습 자료는 교량·댐 사진 중심이다. 공통 콘크리트 손상을 학습해 산업체 점검 보조에 연결하려는 단계이며, 공장 현장 사진 성능은 검증하지 않았다. 이번 비교에서 자료 범위를 더 늘리지 않았다.',
              '한 seed의 비교이며 epoch·임계값을 같은 검증 자료로 반복 선택했다. 개선 수치는 독립 현장 성능이나 모든 시설의 5% 미만 오류를 보장하지 않는다.',
              '기존 검증/시험 정답과 임계값 평가 기준을 바꾸지 않았다. 검증 목표 미달 후보는 보류 시험으로 반복 선택하지 않는다.',
              '사진에서 검출되지 않았다는 결과로 안전을 확정하지 않는다. 최종 판단은 점검자의 확인이 필요하다.', '',
              '[실측 집계 JSON](facility-small-region-comparison.json), [실제 가중치 기술 점검](facility-small-region-technical-verification.json)', '']
    return '\n'.join(lines)


def main():
    loaded = [load_run(name, full_only=name != REFERENCE) for name in (REFERENCE, CONTROL, TREATMENT)]
    trainings, entries = [x[0] for x in loaded], [x[1] for x in loaded]
    reference, control, treatment = trainings
    conditions = validate_pair(control, treatment, entries[0]['weights_sha256'])
    for key in ('architecture', 'classes', 'imgsz', 'split_sha256', 'model_source_sha256', 'auxiliary_model_source_sha256',
                'auxiliary_manifest_sha256', 'auxiliary_classes', 'validation_domains', 'additional_validation'):
        require(reference.get(key) == control.get(key), f'Initializer reference evaluation differs: {key}')
    require(reference['spatial_manifest_sha256'] == control['core_spatial_manifest_sha256'], 'Initializer core data manifest differs')
    profile_path = ROOT / 'reports/facility-inference-profile.json'
    old_profile_path = ROOT / 'reports/facility-inference-profile-round1.json'
    profile = read(profile_path)
    require(profile.get('version') == 'facility-validation-v2' and sha(profile_path) == sha(old_profile_path), 'App profile changed; cannot claim unchanged deployment')
    audit = read(ROOT / 'reports/facility-target-small-region-data-audit.json')
    require(audit.get('derived_manifest_sha256') == treatment['spatial_manifest_sha256'], 'Published crop aggregate belongs to a different prepared manifest')
    allow = ('status', 'recipe', 'source_manifest_sha256', 'derived_manifest_sha256', 'replaced_rows', 'unchanged_full_count',
             'total_rows', 'new_independent_photos', 'domains', 'replacement_domains', 'replacement_target_classes',
             'source_annotation_count', 'source_annotation_set_sha256', 'per_label_pixel_cells', 'per_label_photo_rows',
             'replacement_photo_pixel_presence_conflicts_masked_unknown', 'label_change_replacement_rows',
             'preparation_script_sha256')
    result = {'schema': 'facility_small_region_comparison_v1', 'split': 'val', 'deployed': False,
              'criterion': 'Crack/spalling FNR and FPR each strictly <.05 in every recorded source; no test selection',
              'matched_pair_conditions': conditions, 'experiments': entries,
              'comparisons': comparisons(*entries), 'crop_audit': {key: audit[key] for key in allow if key in audit},
              'app_profile': {'version': profile['version'], 'sha256': sha(profile_path), 'previous_sha256': sha(old_profile_path), 'unchanged': True},
              'train_review': load_train_review(entries[0]['weights_sha256'], control['core_spatial_manifest_sha256']),
              'new_training_epochs': entries[1]['actual_epochs'] + entries[2]['actual_epochs'],
              'limitations': ['One seed; repeated source-validation checkpoint/cutoff selection, not independent field accuracy',
                              'Bridge/dam images support common visual concrete-defect training; industrial workplace performance is unmeasured; no dataset expansion in this comparison',
                              'Crop recipe changes content, partial photo targets, sampling balance and pixel weights; not a pure zoom intervention',
                              'Derived crops are not new independent photographs; original TRAIN full rows and validation/test truth preserved',
                              '7-class AP recorded only for source-asserted labels; max FNR/FPR criterion covers crack/spalling only'],
              'report_script_sha256': sha(Path(__file__))}
    output = ROOT / 'reports'
    (output / 'facility-small-region-comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'FACILITY_SMALL_REGION_RESULTS_KO.md').write_text(render(result), encoding='utf-8')
    print(json.dumps({'comparison': 'reports/facility-small-region-comparison.json',
                      'maximum_errors': {e['run']: e['worst_error'] for e in entries}, 'deployed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
