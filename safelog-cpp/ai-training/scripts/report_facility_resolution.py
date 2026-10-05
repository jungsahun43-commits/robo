"""Verify the declared 640/960 pair and publish aggregate VAL measurements."""
from pathlib import Path
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.facility_resolution_study import validate_protocol
from scripts.report_facility_small_region import read, sha, require, load_run, comparisons, DOMAINS, TARGETS, LABELS
from scripts.report_facility_discrimination import verify_validation_cache, error_rows
from scripts.report_facility_detail import detail_research_gate, validate_small_area, GATE

PROTOCOL = ROOT / 'reports/facility-resolution-study-protocol.json'


def validate_histories(control, treatment, epochs, draws):
    require(len(control) == len(treatment) == epochs, 'The declared complete epoch pair is required')
    keys = ('sampled_row_indices_sha256', 'sampled_domain_counts', 'sampled_row_type_counts',
            'sampled_full_target_joint_counts')
    for epoch, (left, right) in enumerate(zip(control, treatment), 1):
        for row in (left, right):
            require(type(row.get('epoch')) is int and row['epoch'] == epoch, 'Actual epoch sequence differs')
            require(isinstance(row.get('sampled_row_indices_sha256'),str)
                    and re.fullmatch(r'[a-f0-9]{64}',row['sampled_row_indices_sha256']),
                    'Actual row order must have a SHA256 digest')
            for key in ('sampled_domain_counts', 'sampled_row_type_counts'):
                values = row.get(key, {})
                require(values and all(type(v) is int and v >= 0 for v in values.values())
                        and sum(values.values()) == draws, 'Actual draw budget differs')
            require(set(row['sampled_domain_counts']) == set(DOMAINS)
                    and set(row['sampled_row_type_counts']) == {'full', 'crop'}, 'Sampling categories changed')
            joints = row.get('sampled_full_target_joint_counts', {})
            require(set(joints) == set(DOMAINS), 'Original full-photo target states missing')
            for domain, counts in joints.items():
                require(set(counts) == {'00','10','01','11','unknown'}
                        and all(type(v) is int and v >= 0 for v in counts.values())
                        and counts['unknown'] == 0
                        and sum(counts.values()) <= row['sampled_domain_counts'][domain], 'Invalid full target state')
                if domain == 'codebrim':
                    require(sum(counts.values()) == row['sampled_domain_counts'][domain],
                            'CODEBRIM cannot acquire unprepared crop rows')
            require(sum(sum(v.values()) for v in joints.values()) == row['sampled_row_type_counts']['full'],
                    'Full target-state total does not match full rows')
            proof = row.get('optimizer_step_diagnostics', {})
            require(all(type(proof.get(k)) is int and proof[k] >= 0 for k in
                        ('attempted_batches','actual_optimizer_steps','amp_skipped_steps'))
                    and proof['attempted_batches'] == math.ceil(draws/8)
                    and proof['actual_optimizer_steps']+proof['amp_skipped_steps'] == proof['attempted_batches'],
                    'Actual optimizer update proof is inconsistent')
        require(all(left.get(key) == right.get(key) and left.get(key) is not None for key in keys),
                'The actual paired photo/label sampling schedules differ')
    return {'actual_ordered_row_index_hashes_identical_each_epoch':True,
            'actual_source_full_crop_target_counts_identical_each_epoch':True,
            'epochs_compared':epochs,'draws_per_epoch':draws}


def render(result):
    lines = ['# 640·960 입력 해상도 대조 학습 결과', '',
             '같은 초기 ROI 모델에서 각 6epoch, 총 12epoch를 실제 추가 학습했다. 원본 정답·자료·표본 순서·손실을 유지하고 입력 해상도를 비교했다.',
             '960의 원시 120×120 logit은 area 평균으로 80×80에 맞춘 뒤 기존 top32·사진/지도 혼합에 사용했다. 픽셀 손실도 기존 80×80 주석을 사용한다. 새 파라미터·정밀 위치 정답·현장 사진은 추가하지 않았다.', '',
             '| 모델 | 입력 | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한 5% 기준 |',
             '|---|---:|---:|---:|---:|---|']
    for entry in result['experiments']:
        lines.append(f'| {entry["title"]} | {entry["imgsz"]} | {entry["actual_epochs"]} | {entry["best_epoch"]} | {100*entry["worst_error"]:.2f}% | {"통과" if entry["target_passed"] else "미달"} |')
    change = result['comparisons']; nominated = result['research_gate']['research_candidate_nominated']
    lines += ['', f'보강군−초기 모델 최대 오류 {change["maximum_error_treatment_minus_initializer_pp"]:+.2f}pp, 보강군−대조군 {change["maximum_error_treatment_minus_control_pp"]:+.2f}pp. 양수는 악화다.',
              '최대값은 균열·박락 × 세 출처 × FNR/FPR의 12개 비율 중 최대다. 전체 오답 사진 비율이나 앱 정확도가 아니다.', '',
              '## 사전 선언한 연구 후보 기준', '',
              f'연구 후보 기준: **{"통과" if nominated else "미달"}**. 초기 모델과 대조군 각각 대비 최대 오류 0.5pp 개선, 항목별 오류 악화 2pp 이하, 다른 알려진 항목 AP 악화 0.02 이하를 요구했다.',
              '작은 DACL 양성 198개 항목·사진 사례의 FN 합계도 각각 최소 2건 줄고 항목별 FNR 악화가 2pp 이하여야 한다. 한 사진이 두 항목에 포함될 수 있다. 연구 기준 통과는 엄격한 5% 기준·현장 검증·앱 배포 통과와 별도다.', '',
              '| 항목 | 초기 FN/양성 | 대조 FN/양성 | 보강 FN/양성 |', '|---|---:|---:|---:|']
    for task in TARGETS:
        values = [e['small_dacl_polygon_area_below_one_percent'][task] for e in result['experiments']]
        lines.append(f'| {LABELS[task]} | ' + ' | '.join(f'{v["false_negatives"]}/{v["positive_photos"]}' for v in values) + ' |')
    lines += ['', '## 출처별 관측 오류', '',
              'Wilson95는 고정 예측·독립 사진 가정의 기술 통계다. 반복 VAL 선택·같은 원본의 파생 crop 상관 때문에 현장 보장이나 모델 차이 유의성 검정이 아니다.', '',
              '| 모델 | 자료 | 항목 | FN/양성 | 미탐률 | Wilson95 | FP/음성 | 오탐률 | Wilson95 |',
              '|---|---|---|---:|---:|---|---:|---:|---|']
    titles = {e['run']:e['title'] for e in result['experiments']}
    for row in result['error_rows']:
        # error_rows has confusion counts plus interval fields from the shared reporter.
        fn_interval, fp_interval = row['fnr_wilson95_descriptive'], row['fpr_wilson95_descriptive']
        fmt = lambda interval: f'{100*interval[0]:.2f}%–{100*interval[1]:.2f}%'
        lines.append(f'| {titles[row["run"]]} | {row["domain"]} | {LABELS[row["class"]]} | {row["fn"]}/{row["positive_photos"]} | {100*row["fnr"]:.2f}% | {fmt(fn_interval)} | {row["fp"]}/{row["negative_photos"]} | {100*row["fpr"]:.2f}% | {fmt(fp_interval)} |')
    lines += ['', '## 자원·재현 확인', '', '| 모델 | 학습·epoch 검증 시간 | allocated peak | 실제 update | AMP skip |', '|---|---:|---:|---:|---:|']
    for entry in result['experiments'][1:]:
        r = entry['resources']; steps = r['optimizer_step_diagnostics']
        lines.append(f'| {entry["title"]} | {r["elapsed_training_minutes"]:.2f}분 | {r["peak_cuda_allocated_bytes"]/1024**3:.3f} GiB | {steps["actual_optimizer_steps"]} | {steps["amp_skipped_steps"]} |')
    lines += ['', '시간에는 자료 읽기·epoch 검증·캐시·병행 작업의 영향이 포함된다. 같은 epoch·표본 수는 같은 시간·메모리·모바일 추론 비용이 아니다.',
              '여섯 epoch의 실제 row 순서 SHA·출처·full/crop·항목 조합 수를 비교했다. 이미지 해상도가 달라 입력 tensor는 같지 않으며, preflight의 세 batch만 라벨·마스크·known·행 순서 재현을 확인했다.', '',
              '## 적용 상태와 한계', '',
              '보류 TEST 추론은 실행하지 않았다. 앱 기본 `facility-validation-v2`와 프로필 SHA는 유지한다. 입력 사진 확대가 native 정보가 없는 작은 사진에 새 세부 정보를 만들지는 않는다.',
              '기존 processed DACL 최대 변1280 사진을 사용했다. 더 큰 native 원본 입력·새 현장 자료·전문가 라벨 수정·이음매 별도 정답을 추가한 실험은 아니다. 원래 19종 보조 태그에는 이음매 관련 태그가 포함되며 이전부터 사용했다.',
              '한 seed와 반복 사용한 공개 자료 VAL의 결과이며 공장 시설 성능은 미측정이다. 이번 결과로 구조 안전이나 법적 점검 완료를 확정하지 않는다.', '',
              '[사전 계획](FACILITY_RESOLUTION_STUDY_PLAN_KO.md), [고정 조건](facility-resolution-study-protocol.json), [실측 집계](facility-resolution-study-comparison.json), [기술 검증](facility-resolution-study-verification.json)', '']
    return '\n'.join(lines)


def main():
    protocol = validate_protocol(read(PROTOCOL), ROOT); protocol_sha = sha(PROTOCOL)
    names = [protocol['reference'], protocol['control'], protocol['treatment']]
    require(not any((ROOT/'reports'/f'{name}-target-test.json').exists() for name in names), 'No-test study refuses test results')
    loaded = [load_run(name) for name in names]; trainings = [v[0] for v in loaded]; entries = [v[1] for v in loaded]
    control, treatment = trainings[1:]
    for variant, training in [('control',control),('highres',treatment)]:
        expected = {'status':'complete','actual_epochs':protocol['requested_epochs'],
                    'architecture':protocol['architecture_by_variant'][variant],
                    'imgsz':protocol['imgsz_by_variant'][variant],
                    'initial_weights_sha256':protocol['initial_weights_sha256'],
                    'core_spatial_manifest_sha256':protocol['core_spatial_manifest_sha256'],
                    'auxiliary_manifest_sha256':protocol['auxiliary_manifest_sha256'],
                    'source_sha256':protocol['source_sha256'],
                    'study_protocol_sha256':protocol_sha}
        for key in ('seed','requested_epochs','patience','batch_size','draws_per_epoch','backbone_lr',
                    'head_lr','auxiliary_weight','loader_randomness','domain_proportions'):
            expected[key] = protocol[key]
        for key,value in expected.items(): require(training.get(key) == value, f'Actual resolution condition differs: {key}')
    require(trainings[0]['weights_sha256'] == protocol['initial_weights_sha256'], 'Reference weights changed')
    matched = ('classes','split_sha256','core_spatial_manifest_sha256','spatial_manifest_sha256',
               'auxiliary_manifest_sha256','expected_sampling','expected_label_sampling','photo_positive_weights',
               'pixel_positive_weights','auxiliary_positive_weights','additional_validation','additional_test')
    for key in matched: require(key in control and control[key] == treatment.get(key), f'Paired data/loss differs: {key}')
    histories = [read(ROOT/'runs'/name/'history.json') for name in names[1:]]
    sampling = validate_histories(*histories, protocol['requested_epochs'], protocol['draws_per_epoch'])
    truth = []; ap_count = 0
    for entry in entries:
        run = ROOT/'runs'/entry['run']; digests = {}
        for domain in DOMAINS:
            points, digest = verify_validation_cache(read(run/f'validation-{domain}.json'),
                      read(run/f'target-validation-{domain}-grid1.json'),entry,domain)
            entry['ranking_ap'][domain] = points; digests[domain] = digest
            ap_count += sum(p['ap'] is not None for p in points.values())
        validate_small_area(entry); truth.append(digests)
    require(truth[0] == truth[1] == truth[2] and ap_count == 42, 'Original VAL truth/AP support changed')
    titles = ['추가 학습 전 ROI 모델','640 대조군','960 보강군']
    for entry, training, title in zip(entries,trainings,titles):
        entry.update(title=title,imgsz=training['imgsz'])
        if entry['run'] != names[0]:
            history = histories[names.index(entry['run'])-1]
            summary = training['optimizer_step_diagnostics']
            require(all(summary.get(key) == sum(row['optimizer_step_diagnostics'][key] for row in history)
                    for key in ('attempted_batches','actual_optimizer_steps','amp_skipped_steps')),
                    'Training summary optimizer counts differ from actual epochs')
            require(training['elapsed_training_minutes'] > 0 and training['peak_cuda_allocated_bytes'] > 0,
                    'Actual resource measurements missing')
            entry['resources'] = {k:training[k] for k in ('elapsed_training_minutes','peak_cuda_allocated_bytes','optimizer_step_diagnostics')}
    profile = ROOT/'reports/facility-inference-profile.json'
    preflight = read(ROOT/'runs/facility-resolution-preflight.json')
    require(preflight['status'] == 'passed' and preflight['protocol_sha256'] == protocol_sha
            and preflight['protected_file_sha256']['app_profile'] == sha(profile), 'Preflight/app profile changed')
    result = {'schema':'facility_resolution_study_comparison_v1','protocol_sha256':protocol_sha,
              'experiments':entries,'comparisons':comparisons(*entries),
              'research_gate':detail_research_gate(*entries, protocol['research_candidate_gate']),
              'error_rows':error_rows(entries),'actual_sampling_verification':sampling,
              'original_validation_truth_sha256':truth[0],'verified_known_class_ap_measurements':ap_count,
              'app_profile_sha256':sha(profile),'deployed':False,'source_test_inference_executed':False,
              'additional_expert_confirmed_labels':0,'label_changes':0,
              'scope':'Source VAL photo presence; repeated selection, not factory accuracy or precise localization'}
    require(protocol['research_candidate_gate'] == GATE, 'Declared research gate differs')
    (ROOT/'reports/facility-resolution-study-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'reports/FACILITY_RESOLUTION_STUDY_RESULTS_KO.md').write_text(render(result),encoding='utf-8')
    print(json.dumps({'actual_new_epochs':sum(e['actual_epochs'] for e in entries[1:]),
          'maximum_validation_error':[e['worst_error'] for e in entries],
          'research_candidate_nominated':result['research_gate']['research_candidate_nominated'],
          'strict_target_passed':[e['target_passed'] for e in entries],'app_promoted':False}))


if __name__ == '__main__': main()
