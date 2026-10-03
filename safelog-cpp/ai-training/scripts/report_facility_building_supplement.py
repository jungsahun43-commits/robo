"""Publish aggregate results of the completed ConViD TRAIN supplement pair."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_facility_small_region import (
    CLASSES, DOMAINS, TARGETS, LABELS, PAIR_KEYS, read, sha, require, number,
    load_run, comparisons,
)

REFERENCE = 'facility-presence-target-roi-control'
CONTROL = 'facility-presence-target-building-control'
TREATMENT = 'facility-presence-target-building-convid'
INITIAL_SHA = '0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a'
TITLES = {REFERENCE: '추가 학습 전 모델', CONTROL: '기존 자료 추가 학습 대조군', TREATMENT: 'ConViD 콘크리트 사진 보강군'}
MATCHED_KEYS = tuple(k for k in PAIR_KEYS if k != 'domain_proportions') + ('spatial_manifest_sha256', 'batch_size')
DRAW_BUDGET = 14248
CORE_MASS = .9


def validate_building_pair(control, treatment, initializer_sha):
    require(control.get('status') == treatment.get('status') == 'complete', 'Complete both paired runs before comparison')
    for key in MATCHED_KEYS:
        require(key in control and key in treatment and control[key] == treatment[key], f'Paired condition changed or missing: {key}')
    expected = {'classes': CLASSES, 'architecture': 'lraspp_mobilenet_facility_auxiliary_v1',
                'seed': 52, 'requested_epochs': 6, 'actual_epochs': 6, 'patience': 6, 'imgsz': 640,
                'draws_per_epoch': DRAW_BUDGET, 'batch_size': 8, 'backbone_lr': .00004, 'head_lr': .00025,
                'auxiliary_weight': .5, 'validation_domains': DOMAINS,
                'train_dacl': 6225, 'train_damsegment': 1585, 'train_codebrim': 6438,
                'val_dacl': 710, 'val_damsegment': 424}
    for key, value in expected.items():
        require(control.get(key) == value, f'Predeclared paired condition changed: {key}')
    require(initializer_sha == INITIAL_SHA and control.get('initial_weights_sha256') == INITIAL_SHA,
            'Pair must initialize from the frozen ROI control checkpoint')
    require(control.get('domain_proportions') == [.7, .1, .2]
            and treatment.get('domain_proportions') == [.63, .09, .18, .10], 'Intentional source-mass intervention differs')
    require(control['spatial_manifest_sha256'] == control['core_spatial_manifest_sha256'], 'Core spatial manifest was replaced')
    require(not any(key in item for item in (control, treatment) for key in ('train_s2ds', 'hard_training_sampling')),
            'Do not combine other intervention methods with this source comparison')
    require(not any(item.get('train_peccd', 0) for item in (control, treatment)), 'PECCD was not cleared for this training experiment')
    require(not control.get('train_convid') and not control.get('photo_supplement_sha256'), 'Control must not use ConViD photos')
    extra = treatment.get('train_convid')
    require(isinstance(extra, int) and not isinstance(extra, bool) and 1 <= extra <= 200,
            'Record the actual bounded ConViD supplement count')
    require(isinstance(treatment.get('photo_supplement_sha256'), str) and len(treatment['photo_supplement_sha256']) == 64,
            'Prepared ConViD manifest hash missing')
    a, b = control.get('spatial_manifest_audit', {}), treatment.get('spatial_manifest_audit', {})
    require(a == b and a.get('full_photo_or_patch_rows') == DRAW_BUDGET
            and a.get('per_label_pixel_cells'), 'Original spatial supervision/pixel weights changed')
    for key in ('photo_supplement_validator_sha256', 'photo_supplement_helper_sha256'):
        if key in control or key in treatment:
            require(control.get(key) == treatment.get(key), f'Optional paired configuration differs: {key}')
    validate_loss_weights(control, treatment)
    validate_sampling(control, treatment)
    return {**{key: control[key] for key in MATCHED_KEYS},
            **{key: control[key] for key in ('photo_supplement_validator_sha256', 'photo_supplement_helper_sha256') if key in control},
            'control_domain_proportions': control['domain_proportions'],
            'treatment_domain_proportions': treatment['domain_proportions'],
            'pixel_positive_weights': control['pixel_positive_weights'],
            'auxiliary_positive_weights': control['auxiliary_positive_weights'],
            'photo_positive_weights': {'control': control['photo_positive_weights'], 'treatment': treatment['photo_positive_weights']},
            'original_pixel_cell_counts_preserved': True, 'original_conditional_full_crop_sampling_preserved': True,
            'treatment_convid_photo_rows': extra}


def validate_loss_weights(control, treatment):
    for key, length, maximum in (('pixel_positive_weights', 7, 20.), ('auxiliary_positive_weights', 19, 6.)):
        a, b = control.get(key), treatment.get(key)
        require(isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == length,
                f'Actual paired loss weights missing: {key}')
        for before, after in zip(a, b):
            number(before, key, lower=.2, upper=maximum); number(after, key, lower=.2, upper=maximum)
            require(before == after, f'Original paired loss weights changed: {key}')
    a, b = control.get('photo_positive_weights'), treatment.get('photo_positive_weights')
    require(isinstance(a, dict) and isinstance(b, dict) and set(a) == set(DOMAINS)
            and set(b) == set(DOMAINS + ['convid']), 'Actual source photo-loss weights missing')
    for domain, vector in b.items():
        require(isinstance(vector, list) and len(vector) == 7, 'Photo-loss weight public class shape changed')
        for weight in vector: number(weight, 'source photo-positive weight', lower=.2, upper=6.)
        if domain in a:
            require(isinstance(a[domain], list) and len(a[domain]) == 7, 'Old source photo-loss weights missing')
            for before, after in zip(a[domain], vector):
                number(before, 'old photo-positive weight', lower=.2, upper=6.)
                require(math.isclose(before, after, rel_tol=1e-6, abs_tol=1e-6), 'Original source photo-loss weights changed')
    require(all(math.isclose(weight, .2, abs_tol=1e-6) for weight in b['convid'][:2]),
            'Positive-only ConViD photo loss must retain the recorded lower-clamped weights')


def validate_sampling(control, treatment):
    """Old full/crop mass scales by .90; the new source receives photo-only .10."""
    a, b = control.get('expected_sampling', {}), treatment.get('expected_sampling', {})
    require(set(a) == set(DOMAINS) and set(b) == set(DOMAINS + ['convid']), 'Measured sampling mass missing or new domains introduced')
    for domain in DOMAINS:
        for row_type in ('full', 'crop'):
            before = number(a[domain].get(row_type), 'control source sampling mass')
            after = number(b[domain].get(row_type), 'treatment source sampling mass')
            require(math.isclose(after / CORE_MASS, before, rel_tol=1e-10, abs_tol=1e-12),
                    'Old source conditional full/crop weights were not preserved')
    require(math.isclose(number(b['convid'].get('full'), 'ConViD full mass'), .1, abs_tol=1e-12)
            and number(b['convid'].get('crop'), 'ConViD crop mass') == 0, 'ConViD must receive .10 photo-only mass')
    for item in (control, treatment):
        require(math.isclose(sum(sum(parts.values()) for parts in item['expected_sampling'].values()), 1., abs_tol=1e-12),
                'Expected source sampling does not sum to one')
        labels = item.get('expected_label_sampling', {})
        require(set(labels) == set(CLASSES), 'Seven-label positive/negative/unknown sampling measurements missing')
        for values in labels.values():
            require(set(values) == {'positive', 'negative', 'unknown'}, 'Sampling label assertion states changed')
            for state, mass in values.items(): number(mass, f'label {state} sampling mass')
            require(math.isclose(sum(values.values()), 1., abs_tol=1e-12), 'Label sampling states do not sum to one')
    for label in CLASSES[2:]:
        for state in ('positive', 'negative'):
            require(math.isclose(treatment['expected_label_sampling'][label][state], CORE_MASS * control['expected_label_sampling'][label][state], abs_tol=1e-12),
                    'ConViD supplied photo supervision for an unknown facility class')
        require(math.isclose(treatment['expected_label_sampling'][label]['unknown'],
                             .1 + CORE_MASS * control['expected_label_sampling'][label]['unknown'], abs_tol=1e-12),
                'ConViD unknown labels became asserted negatives')
    for label in TARGETS:
        require(math.isclose(treatment['expected_label_sampling'][label]['negative'],
                             CORE_MASS * control['expected_label_sampling'][label]['negative'], abs_tol=1e-12),
                'A ConViD non-target class became an asserted negative')


def validate_actual_draws(entry, treatment=False):
    expected_domains = set(DOMAINS + (['convid'] if treatment else []))
    rows = entry.get('actual_sampling_history', [])
    require(len(rows) == 6, 'Six epochs of actual sampling measurements required')
    for row in rows:
        counts, types = row.get('domain_counts'), row.get('row_type_counts')
        require(isinstance(counts, dict) and set(counts) == expected_domains
                and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts.values())
                and sum(counts.values()) == DRAW_BUDGET, 'Actual source draws differ from fixed epoch budget')
        require(isinstance(types, dict) and set(types) == {'full', 'crop'}
                and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in types.values())
                and sum(types.values()) == DRAW_BUDGET, 'Actual full/crop counts differ from fixed draw budget')
        if treatment:
            require(counts['convid'] > 0, 'No ConViD photo was actually sampled')


def public_supplement(training):
    """Validate local manifest, then exclude individual file/annotation metadata."""
    from scripts.facility_photo_supplement import validate_photo_supplement
    relative = training.get('photo_supplement_path')
    require(isinstance(relative, str), 'Prepared local supplement path missing')
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT / 'data/convid-training') and sha(path) == training['photo_supplement_sha256'], 'ConViD preparation changed after training')
    data = read(path)
    core = read(ROOT / 'data/facility-spatial-training/train.json')
    require(sha(ROOT / 'data/facility-spatial-training/train.json') == training['core_spatial_manifest_sha256'], 'Original TRAIN manifest changed')
    rows = validate_photo_supplement(data, CLASSES, ROOT, {item['image'] for item in core['items']})
    require(len(rows) == training['train_convid'] and data['audit'] == training.get('photo_supplement_audit'), 'Actual supplement count/audit differs from trained preparation')
    positives = [sum(row['targets'][k] == 1 for row in rows) for k in (0, 1)]
    require(all(1 <= count <= 100 for count in positives) and sum(positives) == len(rows),
            'Bounded source must contain at most 100 asserted positives in each target category')
    safe_keys = ('status', 'source_index_sha256', 'source_pick_sha256', 'audit_file_sha256',
                 'source_manifest_sha256', 'preparation_script_sha256',
                 'selected_rows', 'selected_photos', 'requested_limit', 'source_image_count',
                 'invalid_annotation_count', 'exact_overlap_excluded', 'near_overlap_excluded',
                 'source_annotation_count', 'source_annotation_set_sha256')
    return {'train_photos': len(rows), 'manifest_sha256': sha(path),
            'photo_label_counts': {label: {str(t): sum(row['targets'][k] == t for row in rows) for t in (-1, 0, 1)} for k, label in enumerate(CLASSES)},
            'source_audit': {key: data['audit'][key] for key in safe_keys if key in data['audit']},
            'pixel_known_cells': 0, 'auxiliary_photo_tag_labels_asserted': 0,
            'scope': 'Official ConViD V4 crack/Spalling folder positives used for TRAIN only; each photo asserts one positive and leaves all other six labels unknown; no pixel regions, source evaluation or industrial field validation'}


def render(result):
    entries = result['experiments']; change = result['comparisons']
    pct = lambda x: f'{x * 100:.2f}%'
    lines = ['# 콘크리트 사진 보강 대조 실험 결과', '']
    lines += ['실제 완료된 학습·고정 평가만 기록한다. 콘크리트 사진의 사진 단위 균열·박락 정답을 추가하는 방법을 같은 초기 모델·seed 52·6epoch·전체 사진 평가로 비교했다.',
              f"추가 학습 사진은 공식 ConViD V4의 균열·박락 폴더에서 실제 {result['supplement']['train_photos']}장이다. 사진마다 해당 항목 하나만 양성이고 다른 6종은 미확인으로 유지했다. 각 폴더 최대 100장, 전체 최대 200장 범위를 사용했다.",
              'PECCD는 원본 클래스 ID 순서 근거와 추출 검증이 확정되지 않아 이번 실험의 준비·학습 사진 수는 0장이다. 확인되지 않은 숫자 ID를 임의로 해석하여 학습하지 않았다.', '',
              '| 모델 | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한 5% 목표 | 보류 시험 |',
              '|---|---:|---:|---:|---|---|']
    for e in entries:
        lines.append(f"| {TITLES[e['run']]} | {e['actual_epochs']} | {e['best_epoch']} | {pct(e['worst_error'])} | {'검증 통과' if e['target_passed'] else '미달'} | {'실행됨' if e['test_executed'] else '미실행'} |")
    lines += ['', f"대조군 − 초기 모델 최대 오류: {change['maximum_error_control_minus_initializer_pp']:+.2f}pp. 보강군 − 초기 모델: {change['maximum_error_treatment_minus_initializer_pp']:+.2f}pp. 보강군 − 대조군: {change['maximum_error_treatment_minus_control_pp']:+.2f}pp.",
              '양수는 악화, 음수는 개선이다. 대조군의 추가 학습이 초기 모델보다 나빠졌거나 일부 출처·항목이 나빠진 결과도 아래에 그대로 기록한다.',
              '미탐 FN/(TP+FN), 오탐 FP/(FP+TN)이 균열·박락 각각과 세 출처 각각에서 모두 5% 미만이어야 통과한다. 최대값은 전체 앱 오답 사진 비율이 아니다.', '']
    for e in entries:
        lines += [f"## {TITLES[e['run']]} 상세 검증", '', '| 자료 | 항목 | 미탐/양성 | 미탐률 | 오탐/음성 | 오탐률 |', '|---|---|---:|---:|---:|---:|']
        for label in TARGETS:
            for domain in DOMAINS:
                m = e['per_class'][label]['domains'][domain]
                lines.append(f"| {domain} | {LABELS[label]} | {m['fn']}/{m['positive_photos']} | {pct(m['fnr'])} | {m['fp']}/{m['negative_photos']} | {pct(m['fpr'])} |")
        lines.append('')
    lines += ['## 작은 손상과 기존 7종 확인', '', 'DACL 원본 폴리곤을 640×640에 옮긴 면적 비율 1% 미만의 기존 양성 사진에서 확인한 미탐이다. 실제 길이·면적이나 정밀 위치 평가가 아니다.', '',
              '| 항목 | 초기 미탐/양성 | 대조 미탐/양성 | 보강 미탐/양성 | 초기 미탐률 | 대조 미탐률 | 보강 미탐률 |', '|---|---:|---:|---:|---:|---:|---:|']
    for label in TARGETS:
        values = [e['small_dacl_polygon_area_below_one_percent'][label] for e in entries]
        cells = [f"{v['false_negatives']}/{v['positive_photos']}" for v in values] + [pct(v['fnr']) for v in values]
        lines.append(f"| {LABELS[label]} | {' | '.join(cells)} |")
    lines += ['', 'AP는 정답이 있는 항목의 확률 순위 지표다. 미탐·오탐률이나 앱 정확도와 같지 않으며 미확인 항목은 0점으로 계산하지 않는다.', '',
              '| 자료 | 항목 | 초기 AP | 대조 AP | 보강 AP | 보강 − 대조 | 보강 − 초기 |', '|---|---|---:|---:|---:|---:|---:|']
    for row in change['ranking_ap_changes']:
        domain, label = row['domain'], row['class']
        values = [e['ranking_ap'][domain][label]['ap'] for e in entries]
        cells = [f'{v:.4f}' if v is not None else '미확인' for v in values]
        cells += [f'{row[key]:+.4f}' if row[key] is not None else '해당 없음' for key in ('treatment_minus_control_ap', 'treatment_minus_initializer_ap')]
        lines.append(f"| {domain} | {LABELS[label]} | {' | '.join(cells)} |")
    lines += ['', '## 데이터·추출·적용 상태', '',
              '기존 TRAIN 전체 사진과 crop, 픽셀 정답 가중치, 검증·시험 정답은 유지했다. ConViD 폴더 분류에서 픽셀 정답을 만들어 넣지 않았으며, 다른 항목의 음성 정답이나 원본 19종 보조 태그도 부여하지 않았다.',
              '대조군 DACL/Dam/CODEBRIM 비중은 70/10/20%. 보강군은 63/9/18%와 ConViD 10%이며 기존 출처 안의 full/crop 조건부 비중은 동일하다.',
              '양쪽 모두 epoch마다 복원 추출 14,248회를 사용했다. 새로운 사진 수만큼 추가 연산을 늘린 비교가 아니며, 모든 고유 사진을 한 번씩 읽는다는 뜻도 아니다.',
              '실제 출처별 추출 수와 기대 source×full/crop 비중, 클래스별 양성·음성·미확인 비중은 집계 JSON에 기록했다.',
              f"기본 앱 프로필 `{result['app_profile']['version']}` 파일의 SHA256은 기존 프로필과 동일하다. 이번 후보를 기본 앱에 적용하지 않았다.",
              'ConViD 사진의 정확한 시설 유형은 확인되지 않았다. ConViD 자체의 검증·시험과 공장 현장 일반화 성능은 평가하지 않았다. 기존 교량·댐 검증 성능의 변화이며 건물·공장 전체의 성능으로 해석하지 않는다.',
              '원 저자의 Spalling 폴더를 앱의 콘크리트 박락 항목에 대응했으나 클래스 정의의 호환성은 전문가가 검증하지 않았다. 일부 표면의 도장·미장·질감과 박락의 구분에 관한 AI 관찰은 가설이며, 원본 분류를 수정하거나 정제 완료로 처리하지 않았다.',
              '원본 장면 ID가 없으므로 중복·유사사진 점검이 현장 독립성을 증명하지 않는다. 단일 seed, 같은 검증 자료에서 반복한 epoch·임계값 선택이라는 한계를 유지한다.',
              '검증 5% 목표를 통과하지 못한 후보의 보류 시험을 반복하여 모델을 선택하지 않았다. 검출 없음으로 안전을 확정하지 않으며 최종 확인은 점검자가 한다.',
              '공개 문서에는 집계와 해시만 포함했다. 원본/가공 사진·개별 파일 경로·원본 주석·모델 데이터는 Git에서 무시되는 로컬 자료로 유지한다.', '',
              '[실측 집계 JSON](facility-building-supplement-comparison.json)', '']
    return '\n'.join(lines)


def main():
    loaded = [load_run(name) for name in (REFERENCE, CONTROL, TREATMENT)]
    trainings, entries = [row[0] for row in loaded], [row[1] for row in loaded]
    reference, control, treatment = trainings
    conditions = validate_building_pair(control, treatment, entries[0]['weights_sha256'])
    validate_actual_draws(entries[1]); validate_actual_draws(entries[2], treatment=True)
    for key in ('architecture', 'classes', 'imgsz', 'split_sha256', 'model_source_sha256', 'auxiliary_model_source_sha256',
                'auxiliary_manifest_sha256', 'auxiliary_classes', 'validation_domains', 'additional_validation', 'spatial_manifest_sha256'):
        require(reference.get(key) == control.get(key), f'Initializer reference evaluation differs: {key}')
    profile_path = ROOT / 'reports/facility-inference-profile.json'
    old_profile_path = ROOT / 'reports/facility-inference-profile-round1.json'
    profile = read(profile_path)
    require(profile.get('version') == 'facility-validation-v2' and sha(profile_path) == sha(old_profile_path), 'App profile changed; cannot claim unchanged deployment')
    result = {'schema': 'facility_building_supplement_comparison_v1', 'split': 'val', 'deployed': False,
              'criterion': 'Crack/spalling FNR and FPR each strictly <.05 in every recorded original domain; no test selection',
              'matched_pair_conditions': conditions, 'experiments': entries, 'comparisons': comparisons(*entries),
              'supplement': public_supplement(treatment), 'new_training_epochs': entries[1]['actual_epochs'] + entries[2]['actual_epochs'],
              'convid_source_evaluation_executed': False, 'industrial_field_performance_measured': False,
              'peccd_prepared_training_photos': 0, 'peccd_training_photos': 0,
              'peccd_preparation_limitation': 'Native numeric class authority and complete extraction verification unresolved; not used for training',
              'app_profile': {'version': profile['version'], 'sha256': sha(profile_path), 'previous_sha256': sha(old_profile_path), 'unchanged': True},
              'limitations': ['Single seed; repeated original-source validation checkpoint/cutoff selection',
                              'Official V4 folder asserts one photo-level positive only; no segmentation labels inferred',
                              'All other six labels unknown per ConViD photo; no ConViD or industrial field accuracy evaluation',
                              'Author Spalling folder mapped to concrete_spalling; class-definition compatibility not expert-verified and source labels not relabeled',
                              'Scene IDs unavailable; heuristic overlap screening does not prove independent scenes',
                              'Fixed draw budget, old conditional item/crop sampling retained, old domain mass intentionally reduced to .90'],
              'report_script_sha256': sha(Path(__file__))}
    (ROOT / 'reports/facility-building-supplement-comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    (ROOT / 'reports/FACILITY_BUILDING_SUPPLEMENT_RESULTS_KO.md').write_text(render(result), encoding='utf-8')
    print(json.dumps({'comparison': 'reports/facility-building-supplement-comparison.json',
                      'maximum_errors': {e['run']: e['worst_error'] for e in entries}, 'deployed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
