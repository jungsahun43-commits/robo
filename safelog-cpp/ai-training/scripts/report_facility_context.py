"""Frozen narrow/broad pooling comparison: aggregate VAL proof, no test reads."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_facility_small_region import CLASSES, DOMAINS, LABELS, TARGETS, comparisons, load_run, number, read, require, sha
from scripts.report_facility_discrimination import (
    MATCHED_KEYS as BASE_MATCHED_KEYS, error_rows, nonnegative_integer,
    validate_expected_sampling, validate_train_audit, verify_validation_cache,
)
from scripts.report_facility_detail import (
    CORE_SHA, AUXILIARY_SHA, INITIAL_SHA, GATE, validate_small_area,
    detail_research_gate as research_gate, make_sampling_history,
)

REFERENCE = 'facility-presence-target-roi-control'
CONTROL = 'facility-presence-target-context-control'
TREATMENT = 'facility-presence-target-context-pool'
CONTROL_ARCH = 'lraspp_mobilenet_facility_auxiliary_v1'
TREATMENT_ARCH = 'lraspp_mobilenet_facility_pool_context_v1'
TITLES = {REFERENCE:'추가 학습 전 모델', CONTROL:'기존 pooling 대조군', TREATMENT:'좁은·넓은 pooling 대비 보강군'}
PROTOCOL_PATH = ROOT / 'reports/facility-pool-context-protocol.json'
PARENT_SHA = 'e931284c390aa184142aa4ebc42c1b9d25a3c836565f73784bd68581456a33c4'
EXPECTED = {'seed':55, 'requested_epochs':6, 'patience':6, 'imgsz':640, 'batch_size':8,
            'draws_per_epoch':14248, 'backbone_lr':.00004, 'head_lr':.00025,
            'auxiliary_weight':.5, 'domain_proportions':[.7,.1,.2]}
LOADER_RANDOMNESS = {'sampler_seed':55,'training_worker_seed':56,'post_model_seed':57,
                     'validation_worker_seeds':{'dacl':155,'damsegment':156,'codebrim':157}}
CONTEXT_ARCHITECTURE = {
    'recipe':'zero_initialized_narrow_broad_pool_contrast_v1', 'narrow_topk':32, 'broad_topk':256,
    'gate_transform':'tanh_signed_per_class', 'pool_formula':'narrow + tanh(context_gate) * (broad - narrow)',
    'initial_gate':0., 'additional_parameter_count_for_seven_classes':7,
    'maps_and_auxiliary_unchanged':True, 'backbone_passes':1, 'new_photo_labels':0,
    'spatial_adjacency_or_scene_semantics_asserted':False,
}
MATCHED_KEYS = tuple(key for key in BASE_MATCHED_KEYS if key != 'architecture') + (
    'source_parent_training_script_sha256','context_model_source_sha256','model_factory_source_sha256',
    'loader_randomness','target_ranking',
)


def validate_protocol(protocol):
    require(protocol.get('schema') == 'facility_pool_context_protocol_v1'
            and protocol.get('declared_before_training') is True, 'Predeclared pooling protocol missing')
    values = {**EXPECTED, 'reference':REFERENCE,'control':CONTROL,'treatment':TREATMENT,
              'initial_weights_sha256':INITIAL_SHA,'core_spatial_manifest_sha256':CORE_SHA,
              'auxiliary_manifest_sha256':AUXILIARY_SHA,'classes':CLASSES,
              'control_architecture':CONTROL_ARCH,'treatment_architecture':TREATMENT_ARCH,
              'target_ranking_weight':0.,'loader_randomness':LOADER_RANDOMNESS,
              'context_architecture':CONTEXT_ARCHITECTURE,'research_candidate_gate':GATE,
              'implementation_parent_sha256':PARENT_SHA}
    for key,value in values.items():
        require(protocol.get(key) == value, f'Predeclared pooling condition changed: {key}')
    return protocol


def validate_pair(control, treatment, initializer_sha, protocol, protocol_sha):
    validate_protocol(protocol)
    require(initializer_sha == INITIAL_SHA, 'Original ROI initializer changed')
    for training,arch,variant in ((control,CONTROL_ARCH,'control'),(treatment,TREATMENT_ARCH,'context')):
        require(training.get('status') == 'complete', 'Both pooling trainings must be completed')
        values = {**EXPECTED, 'actual_epochs':6, 'classes':CLASSES,'validation_domains':DOMAINS,
                  'architecture':arch,'model_variant':variant,'loader_randomness':LOADER_RANDOMNESS,
                  'initial_weights_sha256':INITIAL_SHA,'spatial_manifest_sha256':CORE_SHA,
                  'core_spatial_manifest_sha256':CORE_SHA,'auxiliary_manifest_sha256':AUXILIARY_SHA,
                  'source_parent_training_script_sha256':PARENT_SHA,'study_protocol_sha256':protocol_sha,
                  'study_protocol_path':'reports/facility-pool-context-protocol.json',
                  'train_dacl':6225,'train_damsegment':1585,'train_codebrim':6438,'val_dacl':710,'val_damsegment':424}
        for key,value in values.items():
            require(training.get(key) == value, f'Actual pooling condition changed: {key}')
        require(training.get('spatial_manifest_audit', {}).get('full_photo_or_patch_rows') == 14248,
                'Original full TRAIN source count changed')
        require(not any(key in training for key in ('hard_training_sampling','photo_supplement_sha256',
                    'supplement_spatial_sha256','train_convid','train_s2ds','train_peccd','detail_model_source_sha256')),
                'Another source/label/feature intervention entered the pooling comparison')
        ranking = training.get('target_ranking', {})
        require(ranking.get('weight') == 0 and ranking.get('classes') == TARGETS
                and ranking.get('sampling_changed') is False and ranking.get('new_labels_asserted') == 0
                and ranking.get('public_outputs_changed') is False, 'Ranking/source/output intervention entered the pair')
        transfer = training.get('initial_state_transfer', {})
        require(transfer.get('shared_state_tensors_equal') is True
                and nonnegative_integer(transfer.get('shared_state_tensor_count'), 'Shared initializer count missing') > 0,
                'Shared backbone/base states were not transferred identically')
        require(nonnegative_integer(transfer.get('new_state_tensor_count'), 'New initializer state count missing')
                == (0 if variant == 'control' else 1)
                and transfer.get('new_output_projection_zero') is (None if variant == 'control' else True),
                'Only the fresh zero per-class gate tensor may differ at initialization')
        require(training.get('context_architecture') == (None if variant == 'control' else CONTEXT_ARCHITECTURE),
                'Actual pooling recipe differs from the predeclared recipe')
    for key in MATCHED_KEYS:
        require(key in control and key in treatment and control[key] == treatment[key],
                f'Paired data/sampler/loss/RNG/source changed: {key}')
    require(control['initial_state_transfer']['shared_state_tensor_count']
            == treatment['initial_state_transfer']['shared_state_tensor_count'], 'Shared original state support differs')
    validate_expected_sampling(control)
    return {key:control[key] for key in MATCHED_KEYS if key not in ('auxiliary_audit','additional_test')}


def validate_actual_pair(control, treatment):
    histories = [entry.get('actual_sampling_history', []) for entry in (control,treatment)]
    require(all(len(rows) == 6 for rows in histories), 'Six actual sampling epochs required')
    for rows in histories:
        for epoch,row in enumerate(rows,1):
            require(row.get('epoch') == epoch, 'Actual epoch order changed')
            domains,types,joints = [row.get(key,{}) for key in ('domain_counts','row_type_counts','full_target_joint_counts')]
            require(set(domains) == set(joints) == set(DOMAINS) and set(types) == {'full','crop'}, 'Actual source/joint counts missing')
            require(sum(nonnegative_integer(v,'Invalid actual source count') for v in domains.values()) == 14248
                    and sum(nonnegative_integer(v,'Invalid actual row-type count') for v in types.values()) == 14248,
                    'Actual epoch draw budget changed')
            for domain in DOMAINS:
                require(set(joints[domain]) == {'00','10','01','11','unknown'}, 'Full target states changed')
                full = sum(nonnegative_integer(v,'Invalid full target count') for v in joints[domain].values())
                require(full <= domains[domain] and joints[domain]['unknown'] == 0, 'Original full target assertion changed')
                if domain == 'codebrim': require(full == domains[domain], 'CODEBRIM gained unprepared crop rows')
            require(sum(sum(v.values()) for v in joints.values()) == types['full'], 'Joint full counts differ from row-type counts')
            require(isinstance(row.get('row_indices_sha256'),str) and re.fullmatch(r'[0-9a-f]{64}',row['row_indices_sha256']),
                    'Actual ordered row-index hash missing')
            audit = row.get('ranking_audit',{})
            require(audit.get('pair_count') == 0 and audit.get('contributing_batches') == 0
                    and audit.get('unweighted_mean_batch_loss') == 0 and audit.get('pair_counts_by_domain_target') == {},
                    'An additional ranking intervention entered the pooling pair')
    for a,b in zip(*histories):
        for key in ('domain_counts','row_type_counts','full_target_joint_counts','row_indices_sha256'):
            require(a[key] == b[key], f'Paired seed actual schedule differs: {key}')
    return {'actual_ordered_row_index_hashes_identical_each_epoch':True,
            'actual_source_full_crop_target_joint_counts_identical_each_epoch':True,
            'additional_ranking_supervision_used':False}


def load_measured_runs():
    names = (REFERENCE,CONTROL,TREATMENT)
    require(not any((ROOT/'reports'/f'{name}-target-test.json').exists() for name in names),
            'This no-test comparison refuses a saved test result before any run/result loading')
    loaded = [load_run(name) for name in names]
    trainings,entries = [row[0] for row in loaded],[row[1] for row in loaded]
    digests,count = [],0
    for entry in entries:
        run = ROOT/'runs'/entry['run']; truth = {}
        for domain in DOMAINS:
            points,digest = verify_validation_cache(read(run/f'validation-{domain}.json'),
                            read(run/f'target-validation-{domain}-grid1.json'),entry,domain)
            entry['ranking_ap'][domain] = points; truth[domain] = digest
            count += sum(point['ap'] is not None for point in points.values())
        digests.append(truth); validate_small_area(entry)
        if entry['run'] != REFERENCE: entry['actual_sampling_history'] = make_sampling_history(read(run/'history.json'))
    require(digests[0] == digests[1] == digests[2], 'VAL truth/order changed across models')
    require(count == 42, 'Exactly forty-two originally known AP measurements required')
    return trainings,entries,digests[0]


def resource_measurements(training, history):
    elapsed = number(training.get('elapsed_training_minutes'),'actual total minutes',upper=math.inf)
    peak = nonnegative_integer(training.get('peak_cuda_allocated_bytes'),'Actual allocated CUDA peak missing')
    require(elapsed > 0 and peak > 0 and len(history) == 6, 'Completed resource observations missing')
    times = [number(row.get('elapsed_minutes'),'actual cumulative minutes',upper=math.inf) for row in history]
    peaks = [nonnegative_integer(row.get('peak_cuda_allocated_bytes'),'Actual cumulative CUDA peak missing') for row in history]
    require(all(a <= b for a,b in zip(times,times[1:])) and elapsed >= times[-1]
            and all(a <= b for a,b in zip(peaks,peaks[1:])) and peak >= peaks[-1], 'Cumulative resource observations inconsistent')
    return {'elapsed_training_minutes':elapsed,'peak_cuda_allocated_bytes':peak,
            'epoch_cumulative_elapsed_minutes':times,'epoch_cumulative_cuda_allocated_bytes':peaks,
            'scope':'Observed training plus epoch validation time and CUDA allocated peak; not reserved memory or mobile cost'}


def optimizer_measurements(training, history):
    keys = {'attempted_batches','actual_optimizer_steps','amp_skipped_steps'}
    aggregate = {key:0 for key in keys}; rows = []
    require(len(history) == 6, 'Six actual optimizer diagnostic epochs required')
    for epoch,row in enumerate(history,1):
        values = row.get('optimizer_step_diagnostics',{})
        require(row.get('epoch') == epoch and set(values) == keys, 'Actual optimizer diagnostic fields/order missing')
        for key,value in values.items(): aggregate[key] += nonnegative_integer(value,'Invalid optimizer diagnostic count')
        require(values['attempted_batches'] == 1781 and values['actual_optimizer_steps'] + values['amp_skipped_steps'] == 1781,
                'Actual updates/skips do not cover attempted batch budget')
        rows.append({'epoch':epoch,**values})
    require(training.get('optimizer_step_diagnostics') == aggregate, 'Optimizer totals differ from observed epoch counts')
    return {'total':aggregate,'per_epoch':rows,'scope':'Actual optimizer post-step hook counts; skips are diagnostic, not an added performance gate'}


def validate_technical_verification(proof, entries, protocol_sha, profile_sha):
    require(proof.get('schema') == 'facility_pool_context_technical_verification_v1'
            and proof.get('status') == 'passed' and proof.get('study_protocol_sha256') == protocol_sha
            and proof.get('app_profile_sha256') == profile_sha and proof.get('app_profile_unchanged') is True
            and proof.get('research_models_deployed') is False and proof.get('paired_sampled_row_indices_equal') is True
            and proof.get('accuracy_measured_by_this_check') is False
            and proof.get('independent_field_safety_verified') is False,
            'Pooling technical proof scope/provenance changed')
    rows = proof.get('experiments',[])
    require(len(rows) == 2 and {row.get('run') for row in rows} == {CONTROL,TREATMENT}, 'Technical checkpoint pair missing')
    for entry,arch,variant in zip(entries[1:],(CONTROL_ARCH,TREATMENT_ARCH),('control','context')):
        row = next(row for row in rows if row['run'] == entry['run'])
        require(row.get('weights_sha256') == entry['weights_sha256'] and row.get('architecture') == arch
                and row.get('model_variant') == variant and row.get('study_protocol_sha256') == protocol_sha
                and row.get('public_shape') == [1,7] and row.get('private_spatial_shape') == [1,7,80,80]
                and row.get('auxiliary_shape') == [1,19] and row.get('outputs_finite') is True
                and row.get('public_training_logits_equal') is True, 'Technical frozen weights/output contract differs')
    preflight = proof.get('actual_train_gpu_preflight',{})
    require(preflight.get('status') == 'passed' and preflight.get('schema') == 'facility_context_preflight_v1'
            and preflight.get('initial_fp32_outputs_equal') is True
            and preflight.get('finite_loss_and_gradients') is True
            and preflight.get('learned_gate_gradient_nonzero') is True
            and preflight.get('backbone_forward_calls') == 1
            and preflight.get('augmented_batch_replay_equal') is True,
            'Actual TRAIN zero-gate/gradient/single-pass/augmentation integration proof missing')
    tests = proof.get('actual_test_verification',{})
    require(tests.get('status') == 'passed' and nonnegative_integer(tests.get('tests_run'),'Implementation test count missing') > 0
            and tests.get('failures') == tests.get('errors') == 0, 'Actual implementation unit checks missing')
    require(proof.get('new_context_parameter_count') == 7, 'The pooling intervention is not exactly seven learned parameters')
    freeze = proof.get('pre_training_source_snapshot',{})
    required_paths = {'scripts/train_facility_context.py','safelog_ai/context_classifier.py','reports/facility-pool-context-protocol.json'}
    require(freeze.get('comparison_started_before_outcomes') is True
            and required_paths <= set(freeze.get('sources',{})), 'Pre-training source/protocol snapshot missing')
    for relative,row in freeze['sources'].items():
        path = (ROOT/relative).resolve()
        require(path.is_relative_to(ROOT) and not Path(relative).is_absolute()
                and (path.suffix == '.py' or relative == 'reports/facility-pool-context-protocol.json'),
                'Technical snapshot must reference local code/protocol, not data or heldout input')
        require(row.get('equal') is True and row.get('git_blob_sha256') == row.get('executed_file_sha256') == sha(path),
                'Committed pre-training source/protocol differs from executed revision')
    return {'checkpoint_pair_passed':True,'public_outputs':7,'private_spatial_shape':[1,7,80,80],
            'private_auxiliary_outputs':19,'additional_parameter_count':7,'initial_fp32_zero_gate_outputs_equal':True,
            'learned_gate_gradient_nonzero':True,'backbone_forward_calls':1,'augmented_train_batch_replay_equal':True,
            'augmentation_replay_scope':'Three batches over eight original TRAIN photos with four workers; full-epoch augmented tensor equality not measured; actual full-epoch row order checked separately',
            'implementation_unit_tests_run':tests['tests_run'],'pre_training_commit':freeze.get('pre_training_commit'),
            'accuracy_measured_by_technical_check':False}


def render(result):
    pct = lambda value:f'{100*value:.2f}%'
    interval = lambda values:f'{pct(values[0])}–{pct(values[1])}'
    entries,gate,changes = result['experiments'],result['research_gate'],result['comparisons']
    lines = ['# 좁은·넓은 pooling 대비 대조 실험 결과','',
             '대조군과 보강군 각각 6epoch, 총 12epoch를 실제 추가 학습한 결과다. 새 사진·시설·정답을 추가하지 않고 같은 ROI 초기 모델에서 사진 단위 pooling 후보 하나를 비교했다.',
             '기존 top32 map logit 평균에 tanh로 제한한 항목별 gate × (top256 평균 − top32 평균)를 더한다. 추가 파라미터는 7개다. gate 0에서 기존 사진 출력과 같고 maps·global head·mixture·19종 보조 head·backbone의 초기 상태를 그대로 옮겼다.',
             'signed gate는 좁은 피크와 넓은 증거 어느 쪽도 선호할 수 있다. topK 순위 통계는 픽셀 인접 관계를 보존하지 않으므로 공간 맥락·현장 의미 이해나 정밀 위치 성능으로 주장하지 않는다. top256은 학습 전 고정했고 VAL 결과로 K를 바꾸지 않았다.','',
             '| 모델 | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한 5% 기준 |',
             '|---|---:|---:|---:|---|']
    for entry in entries:
        lines.append(f"| {TITLES[entry['run']]} | {entry['actual_epochs']} | {entry['best_epoch']} | {pct(entry['worst_error'])} | {'검증 통과' if entry['target_passed'] else '미달'} |")
    lines += ['',f"보강군 − 초기 모델 최대 오류 {changes['maximum_error_treatment_minus_initializer_pp']:+.2f}pp, 보강군 − 대조군 {changes['maximum_error_treatment_minus_control_pp']:+.2f}pp. 양수는 악화다.",
              '최대값은 균열·박락 × 세 출처 × 미탐 FN/(TP+FN)·오탐 FP/(FP+TN)의 12개 비율 중 최대이며 전체 앱 오답 사진 비율이 아니다.','',
              '## 사전 선언한 연구 후보 판단','',
              f"연구 후보 기준: **{'통과' if gate['research_candidate_nominated'] else '미달'}**. 초기 모델과 대조군 각각보다 최대 오류 최소 0.5pp 감소, 모든 target FNR/FPR 회귀 최대 2pp, 정답이 있는 나머지 항목 AP 하락 최대 0.02를 적용했다.",
              '작은 DACL 균열·박락 FN 합계는 두 비교 대상 각각보다 최소 2건 감소하고 각 작은 항목 FNR 회귀도 2pp 이하여야 한다. 허용 회귀 범위를 둔 연구 기준이며 모든 항목 개선이나 통계적 유의성 인증이 아니다.',
              '엄격한 목표는 모든 target 출처의 FNR/FPR이 각각 5% 미만인 것이다. 후보 선정만으로 앱 모델·프로필을 교체하지 않는다. 보류 시험은 이번 보고에서 열람하거나 실행하지 않았다.','']
    for name,point in gate['small_area_comparisons'].items():
        common = gate['error_and_other_ap_gate']['comparisons'][name]
        lines.append(f"- {'초기 모델' if name == 'initializer' else '대조군'} 대비 최대 오류 개선 {common['maximum_error_improvement']*100:+.2f}pp, 작은 FN 변화 {-point['small_area_total_fn_improvement']:+d}건, 오류/AP 조건 {'통과' if common['passed'] else '미달'}, 작은 사례 조건 {'통과' if point['passed'] else '미달'}.")
    lines += ['', 'FN 변화의 양수는 미탐 증가다.','', '## 판단과 다음 우선순위','',
              '이번 pooling 후보는 최대 오류가 대조군과 같고 기존 최고 후보보다 높아 채택하지 않는다. 작은 손상 FN 합계는 초기 71건·대조 71건·후보 70건으로 한 건 줄었지만, 두 비교 대상 각각보다 최소 두 건 감소한다는 기준을 통과하지 못했다.',
              '다음 우선순위는 [TRAIN 정답 경계 검수 준비](FACILITY_CONTEXT_LABEL_REVIEW_KO.md)의 정의 질문을 전문가에게 확인하고, 승인된 근거가 확보되면 원본과 구분한 새 TRAIN 버전을 검토하는 것이다. 전문가 판정이나 원본 라벨 수정은 이번에 수행하지 않았다. 같은 조건의 추가 반복 학습을 이어가면 5%에 도달한다는 근거도 확인하지 못했다.','',
              '## 관측 오류와 오차 범위','',
              'Wilson 양측 95% 범위는 독립 사진의 이항 가정에 따른 기술 통계다. 사진 상관과 같은 VAL에서 반복한 epoch·임계값 선택 때문에 확인적 신뢰구간·새 현장 오류 보장·모델 차이 유의성 검정으로 해석할 수 없다.','',
              '| 모델 | 출처 | 항목 | FN/양성 | 미탐률 | Wilson 95% | FP/음성 | 오탐률 | Wilson 95% |',
              '|---|---|---|---:|---:|---|---:|---:|---|']
    for row in result['measured_error_rows']:
        lines.append(f"| {TITLES[row['run']]} | {row['domain']} | {LABELS[row['class']]} | {row['fn']}/{row['positive_photos']} | {pct(row['fnr'])} | {interval(row['fnr_wilson95_descriptive'])} | {row['fp']}/{row['negative_photos']} | {pct(row['fpr'])} | {interval(row['fpr_wilson95_descriptive'])} |")
    lines += ['','## 작은 손상의 사진 단위 미탐','',
              'DACL 원본 폴리곤을 640 입력으로 옮긴 면적 비율 1% 미만인 양성 사례다. 균열 93 + 박락 105 = 198개 항목·사진 사례이며 같은 사진이 두 항목에 포함될 수 있다. 고유 사진 198장·물리적 손상 크기·정밀 위치 평가를 뜻하지 않는다.','',
              '| 항목 | 초기 FN/양성 | 대조 FN/양성 | 보강 FN/양성 | 초기 미탐률 | 대조 미탐률 | 보강 미탐률 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label in TARGETS:
        values = [entry['small_dacl_polygon_area_below_one_percent'][label] for entry in entries]
        cells = [f"{p['false_negatives']}/{p['positive_photos']}" for p in values] + [pct(p['fnr']) for p in values]
        lines.append(f"| {LABELS[label]} | {' | '.join(cells)} |")
    totals = gate['small_area_photo_label_case_totals']
    lines += [f"| 두 항목 FN 합계/198사례 | {totals['initializer']['false_negative_cases']}/198 | {totals['control']['false_negative_cases']}/198 | {totals['treatment']['false_negative_cases']}/198 | 해당 없음 | 해당 없음 | 해당 없음 |",'',
              '## 기존 정답이 있는 7종 AP','',
              'AP는 확률 순위 지표이며 오탐률·앱 정확도와 다르다. 세 모델 × DACL 7종/Dam 2종/CODEBRIM 5종의 42개 AP를 원본 VAL 확률에서 재계산해 확인했다. float32/float64 산술 오차만 허용하고 기존 저장 AP로 비교한다. 미확인 항목은 0점으로 계산하지 않는다.','',
              '| 출처 | 항목 | 초기 AP | 대조 AP | 보강 AP | 보강−대조 | 보강−초기 |', '|---|---|---:|---:|---:|---:|---:|']
    for row in changes['ranking_ap_changes']:
        domain,label = row['domain'],row['class']
        cells = [f"{e['ranking_ap'][domain][label]['ap']:.4f}" if e['ranking_ap'][domain][label]['ap'] is not None else '미확인' for e in entries]
        cells += [f'{row[k]:+.4f}' if row[k] is not None else '해당 없음' for k in ('treatment_minus_control_ap','treatment_minus_initializer_ap')]
        lines.append(f"| {domain} | {LABELS[label]} | {' | '.join(cells)} |")
    lines += ['','## 실제 추출·연산 실적','',
              'seed 55, source 비중 DACL/Dam/CODEBRIM 70/10/20%, 640 입력, batch 8, 매 epoch 복원 추출 14,248회다. backbone/head 학습률 0.00004/0.00025, 19종 보조 손실 0.5, 추가 순위 손실 0이다. 기존 사진·픽셀·보조 가중치와 초기 backbone/base 상태를 보존했다.',
              '실제 row index 순서의 SHA, 출처·full/crop·두 target의 full 정답 조합 추출 수를 여섯 epoch 모두 대조했다. loader worker와 post-model seed도 사전 고정했다. draw 수가 같아도 실제 시간·메모리·AMP update 횟수가 같다는 뜻은 아니다.','',
              '| 모델 | 학습·epoch 검증 시간 | CUDA allocated peak | 시도 batch | 실제 update | AMP skip |', '|---|---:|---:|---:|---:|---:|']
    for name in (CONTROL,TREATMENT):
        cost,updates = result['actual_resources'][name],result['actual_optimizer_steps'][name]['total']
        lines.append(f"| {TITLES[name]} | {cost['elapsed_training_minutes']:.2f}분 | {cost['peak_cuda_allocated_bytes']/(1024**3):.3f} GiB | {updates['attempted_batches']} | {updates['actual_optimizer_steps']} | {updates['amp_skipped_steps']} |")
    lines += ['', '시간에는 자료 읽기·epoch 검증·캐시·병행 작업의 영향이 포함되므로 순수 모델 속도 비교로 해석하지 않는다. 관측한 allocated 메모리이며 reserved·전체 프로세스 메모리·스마트폰 추론 비용이나 미래 시간의 추정값이 아니다. 실제 optimizer post-step hook으로 update와 AMP skip을 집계했고 이를 사후 성능 gate로 추가하지 않았다.','',
              '## 정답·현장 범위·앱 적용','',
              'full은 기존 원본 TRAIN 행이며 CODEBRIM 원저자 patch도 포함한다. 14,248행을 독립된 전체 현장 사진 수로 해석하지 않는다. 파생 crop도 독립된 새 사진이 아니다.',
              '지정 항목의 음성은 전체 시설의 정상·안전 정답이 아니다. Dam 원논문 4.4의 Non-Crack=intact concrete 정의에 따른 사진 음성 변환은 별도의 Spall 음성 mask나 산업 현장 전문가의 검증을 뜻하지 않는다. [DamSegment 원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC13247583/)',
              '기존 TRAIN 검수 준비와 저자 정의를 유지했고 전문가 정답 수정은 수행하지 않았다. 이전 초기 모델의 검수 캐시는 이번 최종 모델의 예측 검증이나 검수 완료 증거로 대체하지 않는다. 미확인을 음성으로 바꾸거나 오답을 제외하지 않았다.',
              '교량·댐 콘크리트 손상 자료에서 공통 시각적 특징을 비교했다. 산업체 벽·바닥·기둥의 현장 정확도는 측정하지 않았다. 한 seed, 같은 원저자 VAL의 반복 선택이며 후보 기준 통과가 새 현장 일반화·5% 미만 오류를 보장하지 않는다.',
              f"기본 앱 프로필 `{result['app_profile']['version']}`의 SHA256은 기존 파일과 같으며 deployed=false다. 초기 출력 동등성·한 번의 backbone·학습 가능한 gate·최종 7/19 출력 shape는 기술 점검으로 정확도와 분리했다.",
              '증강 tensor replay는 원본 TRAIN 8장, worker 4개, 3개 batch의 제한된 기술 점검이다. 전체 여섯 epoch의 증강 tensor hash 동등성을 측정했다고 주장하지 않으며 전체 row 순서 해시 검증과 구분한다.',
              '공개 보고에는 집계·해시만 기록한다. 원본/가공 사진·개별 경로·주석·개별 검수 의견을 배포 묶음에 복사하지 않으며 데이터 재배포 조건을 따른다. 검출 없음으로 안전을 확정하지 않고 최종 판단은 점검자가 한다.','',
              '[실측 집계 JSON](facility-pool-context-comparison.json), [사전 계획](facility-pool-context-protocol.json), [기술 점검](facility-pool-context-technical-verification.json)','']
    return '\n'.join(lines)


def main():
    protocol = validate_protocol(read(PROTOCOL_PATH))
    trainings,entries,digests = load_measured_runs()
    conditions = validate_pair(trainings[1],trainings[2],entries[0]['weights_sha256'],protocol,sha(PROTOCOL_PATH))
    for key in ('architecture','classes','imgsz','split_sha256','model_source_sha256','auxiliary_model_source_sha256',
                'auxiliary_manifest_sha256','auxiliary_classes','validation_domains','additional_validation','spatial_manifest_sha256'):
        require(trainings[0].get(key) == trainings[1].get(key), f'Original initializer source/evaluation changed: {key}')
    sources = {'training_script_sha256':'scripts/train_facility_context.py',
               'source_parent_training_script_sha256':'scripts/train_facility_detail.py',
               'model_source_sha256':'safelog_ai/spatial_classifier.py','auxiliary_model_source_sha256':'safelog_ai/auxiliary_classifier.py',
               'context_model_source_sha256':'safelog_ai/context_classifier.py','model_factory_source_sha256':'safelog_ai/presence_classifier.py',
               'photo_supplement_helper_sha256':'scripts/facility_photo_supplement.py'}
    for key,path in sources.items(): require(trainings[1][key] == sha(ROOT/path), f'Executed source revision changed: {key}')
    require(trainings[1]['target_ranking']['helper_sha256'] == sha(ROOT/'scripts/facility_target_ranking.py'), 'Disabled ranking helper revision changed')
    require(sha(ROOT/'data/facility-spatial-training/train.json') == CORE_SHA
            and sha(ROOT/'data/facility-auxiliary-training/train.json') == AUXILIARY_SHA, 'Original TRAIN supervision changed')
    draws = validate_actual_pair(entries[1],entries[2])
    profile_path,previous_path = ROOT/'reports/facility-inference-profile.json',ROOT/'reports/facility-inference-profile-round1.json'
    profile = read(profile_path)
    require(profile.get('version') == 'facility-validation-v2' and sha(profile_path) == sha(previous_path), 'App profile changed during research')
    technical_path = ROOT/'reports/facility-pool-context-technical-verification.json'
    technical = validate_technical_verification(read(technical_path),entries,sha(PROTOCOL_PATH),sha(profile_path))
    audit_path = ROOT/'reports/facility-target-negative-data-audit.json'
    negative = validate_train_audit(read(audit_path),trainings[1])
    histories = {entry['run']:read(ROOT/'runs'/entry['run']/'history.json') for entry in entries[1:]}
    resources = {entry['run']:resource_measurements(training,histories[entry['run']]) for training,entry in zip(trainings[1:],entries[1:])}
    updates = {entry['run']:optimizer_measurements(training,histories[entry['run']]) for training,entry in zip(trainings[1:],entries[1:])}
    result = {'schema':'facility_pool_context_comparison_v1','split':'val','deployed':False,'test_executed':False,
              'industrial_field_performance_measured':False,'new_training_photos':0,'original_labels_edited':0,
              'spatial_adjacency_or_scene_semantics_asserted':False,'finer_pixel_gold_asserted':False,
              'protocol_sha256':sha(PROTOCOL_PATH),'protocol':protocol,'matched_pair_conditions':conditions,
              'architectures':{'control':CONTROL_ARCH,'treatment':TREATMENT_ARCH},'context_architecture':CONTEXT_ARCHITECTURE,
              'initial_state_transfer':{name:training['initial_state_transfer'] for name,training in zip((CONTROL,TREATMENT),trainings[1:])},
              'actual_pair_sampling_checks':draws,'experiments':entries,'comparisons':comparisons(*entries),
              'measured_error_rows':error_rows(entries),'known_ap_recomputed_count':42,'validation_target_array_sha256':digests,
              'research_gate':research_gate(*entries,protocol['research_candidate_gate']),
              'strict_target':'Every original-source crack/spalling FNR/FPR strictly below.05; not whole-app or factory accuracy',
              'interval_policy':'Wilson95 descriptive only; independence unverified and repeated validation selection prevents confirmatory interpretation',
              'train_negative_audit_sha256':sha(audit_path),'train_negative_summary':negative,
              'technical_verification_sha256':sha(technical_path),'technical_verification_summary':technical,
              'actual_resources':resources,'actual_optimizer_steps':updates,
              'app_profile':{'version':profile['version'],'sha256':sha(profile_path),'previous_sha256':sha(previous_path),'unchanged':True},
              'new_training_epochs':12,'report_script_sha256':sha(Path(__file__)),
              'limitations':['Single seed; repeated original-source validation epoch/threshold selection',
                             'Top32/top256 are map-logit order statistics, not spatial adjacency or scene semantics',
                             'Seven signed zero gates add no new gold photo/pixel labels',
                             '198 small-area class-photo cases may share photos across labels',
                             'Source full rows include publisher CODEBRIM patches; crop rows add no independent photographs',
                             'Target negative does not mean general safe/normal facility ground truth',
                             'Dam intact-category conversion follows author semantics, not independent spall masks or field expert validation',
                             'Original labels/masks preserved; unknowns not filled negative and errors not excluded',
                             'Bridge/dam validation does not measure industrial field accuracy',
                             'Research nomination tolerates predeclared regressions and does not deploy the model',
                             'Equal sampled row order does not imply equal actual elapsed cost, CUDA allocation or AMP update count']}
    output = ROOT/'reports'
    (output/'facility-pool-context-comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'FACILITY_POOL_CONTEXT_RESULTS_KO.md').write_text(render(result),encoding='utf-8')
    print(json.dumps({'comparison':'reports/facility-pool-context-comparison.json',
                      'maximum_errors':{entry['run']:entry['worst_error'] for entry in entries},
                      'research_candidate_nominated':result['research_gate']['research_candidate_nominated'],'deployed':False},ensure_ascii=False))


if __name__ == '__main__':main()
