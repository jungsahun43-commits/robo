"""Consolidate actual experiment status; never substitute validation for test."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_facility_target import read,save,sha,TARGETS

RUNS=('facility-presence-target-v2s','facility-presence-target-detail','facility-presence-target-highres',
      'facility-presence-target-codebrim','facility-presence-target-spatial','facility-presence-target-s2ds','facility-presence-target-hard','facility-presence-target-auxiliary',
      'facility-presence-target-roi-control','facility-presence-target-small-region',
      'facility-presence-target-building-control','facility-presence-target-building-convid',
      'facility-presence-target-discrimination-control','facility-presence-target-discrimination-ranking',
      'facility-presence-target-detail-control','facility-presence-target-detail-s4',
      'facility-presence-target-context-control','facility-presence-target-context-pool',
      'facility-presence-target-resolution-control','facility-presence-target-resolution-highres',
      'facility-presence-target-native-roi-control','facility-presence-target-native-roi-native',
      'facility-presence-target-subtype-control','facility-presence-target-subtype-negative',
      'facility-presence-target-retention-control','facility-presence-target-retention-distill',
      'facility-presence-target-retention-strong','facility-presence-target-batchnorm-frozen',
      'facility-presence-target-head-lr-low')
NAMES=('큰 사진 모델','상세 조각·자료 균형','640 해상도','실제 CODEBRIM 추가','사진·위치 동시 학습','S2DS 위치 정답 추가','어려운 TRAIN 사례 보강','원본19종 보조 학습',
       '작은 영역 비교: 기존 자료 대조군','작은 영역 비교: 맥락 crop 보강군',
       '콘크리트 사진 비교: 기존 자료 대조군','콘크리트 사진 비교: ConViD 양성 보강군',
       '균열·박락 구분: 기존 손실 대조군','균열·박락 구분: 양성·음성 순위 학습군',
       '모델 구조 비교: 기존 모델 대조군','모델 구조 비교: stride4 특징 잔차 경로',
       '사진 pooling 비교: 기존 모델 대조군','사진 pooling 비교: 좁은 피크·넓은 증거 대비',
       '입력 해상도 비교: 640 대조군','입력 해상도 비교: 960 보강군',
       '원본 ROI 비교: 먼저 축소한 대조군','원본 ROI 비교: 직접 잘라낸 보강군',
       '박락 음성 태그 비교: 기존 추출 대조군','박락 음성 태그 비교: 관련 표면 손상 보강군',
       '기존 항목 보존 비교: 증류 없는 대조군','기존 항목 보존 비교: 알려진 5항목 증류군',
       '보존 강도 후속 비교: 가중치4 학생','BN 통계 비교: 원본 통계로 학습',
       '학습률 후속 비교: head 초기 학습률 감소')
DOMAINS={'dacl':'기존 교량','damsegment':'추가 댐','codebrim':'CODEBRIM 교량'}
LABELS={'concrete_crack':'균열','concrete_spalling':'박락'}
RESOLUTION_RUNS=('facility-presence-target-resolution-control','facility-presence-target-resolution-highres')
NATIVE_ROI_RUNS=('facility-presence-target-native-roi-control','facility-presence-target-native-roi-native')
SUBTYPE_RUNS=('facility-presence-target-subtype-control','facility-presence-target-subtype-negative')
RETENTION_RUNS=('facility-presence-target-retention-control','facility-presence-target-retention-distill')
RETENTION_STRENGTH_RUNS=('facility-presence-target-retention-strong',)
BATCHNORM_RUNS=('facility-presence-target-batchnorm-frozen',)
HEAD_LR_RUNS=('facility-presence-target-head-lr-low',)


def interrupted_training_attempts(root):
    """Count preserved complete epochs, never unrecorded partial work or scores."""
    path=root/'reports/facility-resolution-interruption.json'
    if not path.exists():return [],None
    evidence=read(path)
    if (evidence.get('schema')!='facility_resolution_interruption_v1'
            or evidence.get('status')!='interrupted' or evidence.get('variant')!='control'
            or any(evidence.get(key) is not False for key in ('experimental_recipe_changed',
                    'included_in_final_model_comparison','app_model_promoted','source_test_inference_executed'))):
        raise ValueError('Interrupted attempt must preserve the declared recipe and remain outside model comparison')
    preserved='facility-resolution-control-interrupted-session'
    archive=root/'runs'/preserved
    history_path=archive/'history.json';metadata_path=archive/'TRAINING.json'
    if (sha(history_path)!=evidence.get('history_sha256')
            or sha(metadata_path)!=evidence.get('training_metadata_sha256')):
        raise ValueError('Preserved interruption evidence changed')
    history=read(history_path);metadata=read(metadata_path)
    completed=evidence.get('completed_training_epochs')
    if (type(completed) is not int or completed<1 or not isinstance(history,list)
            or len(history)!=completed or any(type(row.get('epoch')) is not int
                    or row['epoch']!=index for index,row in enumerate(history,1))
            or metadata.get('study_protocol_sha256')!=evidence.get('study_protocol_sha256')
            or metadata.get('source_sha256')!=evidence.get('source_sha256')):
        raise ValueError('Interrupted complete-epoch history or frozen source provenance differs')
    updates=evidence.get('recorded_completed_epoch_optimizer_steps')
    actual=[row.get('optimizer_step_diagnostics',{}).get('actual_optimizer_steps') for row in history]
    if (type(updates) is not int or updates<0 or any(type(value) is not int or value<0 for value in actual)
            or sum(actual)!=updates):
        raise ValueError('Interrupted complete-epoch optimizer accounting differs')
    current=root/'runs'/RESOLUTION_RUNS[0]/'history.json'
    if current.exists() and sha(current)==evidence['history_sha256']:
        raise ValueError('Interrupted history remains in the active run and would be counted twice')
    attempt={'original_run':RESOLUTION_RUNS[0],'preserved_run':preserved,'status':'interrupted',
             'completed_training_epochs':completed,'recorded_completed_epoch_optimizer_steps':updates,
             'additional_partial_epoch_updates':evidence.get('additional_partial_epoch_updates'),
             'history_sha256':evidence['history_sha256'],'training_metadata_sha256':evidence['training_metadata_sha256'],
             'study_protocol_sha256':evidence['study_protocol_sha256'],
             'experimental_recipe_changed':False,'included_in_final_model_comparison':False,
             'unrecorded_partial_epoch_work_counted':False,
             'report':'reports/facility-resolution-interruption.json','report_sha256':sha(path)}
    return [attempt],sha(path)


def main():
    entries=[]
    for name,title in zip(RUNS,NAMES):
        run=ROOT/'runs'/name
        if not (run/'TRAINING.json').exists():
            entries.append({'run':name,'title':title,'status':'prepared' if name.endswith(('spatial','s2ds')) else 'not_started'});continue
        training=read(run/'TRAINING.json');history=read(run/'history.json') if (run/'history.json').exists() else []
        selection=read(run/'TARGET-SELECTION.json') if (run/'TARGET-SELECTION.json').exists() else None
        validation=read(run/'VALIDATION.json') if (run/'VALIDATION.json').exists() else None
        point=selection['selected'] if selection else validation
        test_path=ROOT/'reports'/f'{name}-target-test.json'
        test=read(test_path) if test_path.exists() else None
        if selection and selection['weights_sha256']!=sha(run/'best.pt'):raise ValueError('Selection weight hash changed')
        entries.append({'run':name,'title':title,'status':training['status'],'epochs':len(history),'imgsz':training['imgsz'],
                        'best_epoch':validation['epoch'] if validation else None,'frozen_validation':bool(selection),
                        'grid':point.get('grid',1) if point else None,'worst_error':point.get('worst_error',point.get('worst_target_error')) if point else None,
                        'target_passed_validation':point['target_passed'] if point else False,
                        'operating_points':point.get('per_class',point.get('operating_points')) if point else None,
                        'test_executed':bool(test),'target_passed_test':test['target_passed'] if test else None,
                        'weights_sha256':selection['weights_sha256'] if selection else training.get('weights_sha256'),
                        'validation_domains':training.get('validation_domains',['dacl','damsegment'])})
    ensemble_name='facility-presence-target-ensemble'
    ensemble_path=ROOT/'reports'/f'{ensemble_name}-target-validation.json'
    if ensemble_path.exists():
        selection=read(ensemble_path);test_path=ROOT/'reports'/f'{ensemble_name}-target-test.json'
        test=read(test_path) if test_path.exists() else None
        entries.append({'run':ensemble_name,'title':'두 모델 확률 평균','status':'validation_only','epochs':0,'imgsz':640,
                        'best_epoch':'해당 없음','frozen_validation':True,'grid':1,'worst_error':selection['worst_error'],
                        'target_passed_validation':selection['target_passed'],'operating_points':selection['per_class'],
                        'test_executed':bool(test),'target_passed_test':test['target_passed'] if test else None,
                        'members':selection['members'],'validation_domains':list(selection['validation_counts'])})
    profile=read(ROOT/'reports/facility-inference-profile.json')
    comparable=[e for e in entries if e.get('frozen_validation') and e.get('validation_domains')==['dacl','damsegment','codebrim'] and e.get('worst_error') is not None]
    best=min(comparable,key=lambda e:e['worst_error']) if comparable else None
    attempts,interruption_sha=interrupted_training_attempts(ROOT)
    model_run_epochs=sum(e.get('epochs',0) for e in entries)
    interrupted_epochs=sum(attempt['completed_training_epochs'] for attempt in attempts)
    total_epochs=model_run_epochs+interrupted_epochs
    paired_epochs=sum(e.get('epochs',0) for e in entries if e['run'] in RESOLUTION_RUNS)
    native_epochs=sum(e.get('epochs',0) for e in entries if e['run'] in NATIVE_ROI_RUNS)
    subtype_epochs=sum(e.get('epochs',0) for e in entries if e['run'] in SUBTYPE_RUNS)
    retention_epochs=sum(e.get('epochs',0) for e in entries if e['run'] in RETENTION_RUNS)
    strength_epochs=sum(e.get('epochs',0) for e in entries if e['run'] in RETENTION_STRENGTH_RUNS)
    batchnorm_epochs=sum(e.get('epochs',0) for e in entries if e['run'] in BATCHNORM_RUNS)
    head_lr_epochs=sum(e.get('epochs',0) for e in entries if e['run'] in HEAD_LR_RUNS)
    running=[{'run':e['run'],'epochs':e['epochs'],'best_full_photo_worst_error':e['worst_error']}
             for e in entries if e.get('status')=='running' and e.get('worst_error') is not None]
    result={'criterion':'Crack and spalling, per-class FNR and FPR each strictly < .05 in every recorded domain; test only after frozen validation passes',
            'experiments':entries,'app_profile':profile['version'],'app_profile_sha256':sha(ROOT/'reports/facility-inference-profile.json'),
            'total_actual_training_epochs':total_epochs,
            'model_run_recorded_training_epochs':model_run_epochs,
            'interrupted_recorded_training_epochs':interrupted_epochs,
            'interrupted_training_attempts':attempts,'interruption_report_sha256':interruption_sha,
            'resolution_training_accounting':{'paired_budget_epochs':12,'paired_recorded_completed_epochs':paired_epochs,
                    'interrupted_recorded_completed_epochs':interrupted_epochs,
                    'recorded_completed_epochs_including_interruption':paired_epochs+interrupted_epochs,
                    'unrecorded_partial_epoch_work_counted':False,
                    'partial_work_scope':'Incomplete epoch batches, updates and time are not quantified or added to completed epochs'},
            'native_roi_training_accounting':{'paired_budget_epochs':12,'paired_recorded_completed_epochs':native_epochs,
                    'scope':'New matched640PNG TRAIN detail pixel intervention; original full rows, masks and labels unchanged'},
            'subtype_training_accounting':{'paired_budget_epochs':12,'paired_recorded_completed_epochs':subtype_epochs,
                    'scope':'Fixed original640 TRAIN pair; related-tag Spalling-negative sampling within original source/full-crop/Crack-Spalling strata; loss weights and labels unchanged'},
            'retention_training_accounting':{'paired_budget_epochs':12,'paired_recorded_completed_epochs':retention_epochs,
                    'scope':'Identical original640 TRAIN draw arrays and teacher forwards; known other-five Bernoulli KL weight0 vs1 at T2; frozen teacher probabilities are regularization signals'},
            'retention_strength_training_accounting':{'new_candidate_budget_epochs':6,'new_candidate_recorded_completed_epochs':strength_epochs,
                    'reused_control_run':RETENTION_RUNS[0],'reused_control_epochs_counted_as_new':0,
                    'scope':'Weight4 follow-up from0773 with identical original draw order; previous weight0 control reused without repeated training'},
            'batchnorm_training_accounting':{'new_candidate_budget_epochs':6,'new_candidate_recorded_completed_epochs':batchnorm_epochs,
                    'reused_control_run':RETENTION_STRENGTH_RUNS[0],'reused_control_epochs_counted_as_new':0,
                    'scope':'Same original teacher/weight4/T2/input/order/loss; student BatchNorm uses fixed0773 running statistics while affine/backbone/head parameters remain trainable'},
            'head_lr_training_accounting':{'new_candidate_budget_epochs':6,'new_candidate_recorded_completed_epochs':head_lr_epochs,
                    'reused_control_run':BATCHNORM_RUNS[0],'reused_control_epochs_counted_as_new':0,
                    'scope':'Same original initializer, frozen BN, teacher weight4/T2, input/order/loss; only initial non-backbone head LR changes .00025 to .0001 with unchanged cosine floor .000005'},
            'best_three_domain_validation_run':best['run'] if best else None,
            'best_three_domain_validation_worst_error':best['worst_error'] if best else None,
            'best_three_domain_scope':'Completed validation view/cutoff selection only; running rows reported separately',
            'running_best_full_photo_validation':running,
            'conditional_extra_once_started':False,
            'release_target_achieved':any(e.get('target_passed_test') is True for e in entries),
            'scope':'Photo presence only; other five facility labels, exact defect location, structural safety and unseen facility error are not claimed below5%',
            'limitation':'Repeatedly selected source validation, not independent field performance; dam scene IDs unavailable; confidence intervals assume fixed predictions and independent examples'}
    save(ROOT/'reports/facility-five-percent-results.json',result)
    lines=['# 시설 항목별 5% 목표 진행·결과','',
           '**5% 미만 목표 미달.**' if not result['release_target_achieved'] else '**기록된 보류 자료의 목표 통과 모델이 있음. 현장 보장은 아님.**','',
           f'기록된 완료 학습은 총{total_epochs}epoch이다. 현재 모델 실행 기록{model_run_epochs}epoch와 보존한 중단 실행 기록{interrupted_epochs}epoch를 합산했다. 모델 확률 평균 검증은 학습 횟수에 더하지 않는다.',
           f'해상도 대조 실험의 계획 예산은 640·960 각각6epoch, 합계12epoch이다. 현재 두 모델의 완료 기록은 {paired_epochs}epoch이며 중단 실행의 {interrupted_epochs}epoch는 모델 비교 예산·성능 표에 포함하지 않고 누적 학습량에만 별도 더한다.',
           f'원본 ROI 대조 실험도 각6epoch·합계12epoch의 고정 예산이며 완료 기록은 {native_epochs}epoch이다. 같은 원본 RGB와 기존 crop 범위에서 사전 축소 유무를 비교하고 양군을 640 PNG로 맞췄다. 독립 사진·정답·full row는 추가하지 않는다.',
           f'박락 음성 태그 대조 실험도 각6epoch·합계12epoch의 고정 예산이며 완료 기록은 {subtype_epochs}epoch이다. 같은 원본640 TRAIN에서 관련4태그 음성의 추출 빈도만 높이고 모든 위치의 출처·full/crop·균열/박락 정답과 손실 가중치를 유지한다. 다른5항목·보조태그 노출 변화는 별도로 평가한다.',
           f'기존 항목 보존 실험도 각6epoch·합계12epoch의 고정 예산이며 완료 기록은 {retention_epochs}epoch이다. 두 조건에서 원본 사진 순서와 교사 추론을 같게 맞추고, 알려진 다른5항목의 증류 손실 가중치0·1만 비교한다. 교사 확률은 새로운 정답이 아니다.',
           f'보존 강도 후속 후보의 신규 예산은6epoch이며 완료 기록은 {strength_epochs}epoch이다. 가중치4만 새로 학습하고 이미 완료한 가중치0 대조군을 재사용하므로 대조군6epoch를 다시 합산하지 않는다.',
           f'BN 통계 고정 후보의 신규 예산은6epoch이며 완료 기록은 {batchnorm_epochs}epoch이다. 같은 가중치4 조건에서 학생 BN을 원본 평균·분산으로 정규화하고 학습 계수는 유지한다. 일반 BN 가중치4 대조군은 재사용해 다시 합산하지 않는다.',
           f'head 학습률 후속 후보의 신규 예산은6epoch이며 완료 기록은 {head_lr_epochs}epoch이다. 같은 원본 초기 모델·BN 정책·가중치4 조건에서 non-backbone 초기 학습률만0.00025→0.0001로 낮춘다. 기존 BN 고정 대조군을 재사용해 학습 횟수를 다시 더하지 않는다.',
           *(['직전 해상도 비교의 대조군은 실행 세션 중단 뒤 optimizer 상태를 복구할 수 없어 같은 초기 모델·고정 조건으로 처음부터 다시 시작했다. 중단 당시 완료하지 못한 epoch의 배치·업데이트·시간은 정량 기록이 없어 추정 합산하지 않는다. [보존한 중단 집계](facility-resolution-interruption.json)를 함께 기록한다.'] if attempts else []),
           f"동일한 세 검증 자료에서 뷰·임계값 선택 고정이 끝난 최대 오류 최소 관측 모델: `{best['run']}`, 최대 미탐·오탐 {best['worst_error']*100:.2f}%. 연구 후보 선정과 앱 배포 기준 통과는 별도다." if best else '동일한 세 검증 자료의 선택 고정 결과가 아직 없다.',
           *[f"진행 중 `{r['run']}`: 실제{r['epochs']}epoch, 지금까지 전체 사진의 최대 미탐·오탐 최저{r['best_full_photo_worst_error']*100:.2f}%. 학습 및 최종 확대 선택이 아직 끝나지 않았다." for r in running],
           '검증 미달 후보를 반복 시험하여 설정을 고르지 않았다. 목표를 통과했을 때 수행할 별도 추가1epoch 조건도 아직 발동하지 않았다.',
           '현재 결과는 추가 반복으로5%달성을 보장하지 않는다. 정답 기준의 전문가 검수와 다른 모델 구조를 포함한 후속 연구가 필요하다.','',
           '균열·박락 각각에 대해 미탐률 FN/(TP+FN), 오탐률 FP/(FP+TN)이 모두 0.05 미만이어야 통과한다.',
           '아래 최대 오류는 이 두 항목의 자료별 미탐/오탐 중 가장 큰 비율이며 전체 오답 사진 비율이나 정확도는 아니다.','',
           '| 시도 | 상태 | 실제 epoch | 입력 | 검증 자료 | 최대 검증 오류 | 목표 |',
           '|---|---|---:|---:|---|---:|---|']
    for e in entries:
        state={'prepared':'준비됨','not_started':'시작 전','running':'학습 중','complete':'학습 완료','validation_only':'결합 검증 완료'}[e['status']]
        error=f"{e['worst_error']*100:.2f}%" if e.get('worst_error') is not None else '-'
        lines.append(f"| {e['title']} | {state} | {e.get('epochs','-')} | {e.get('imgsz','-')} | {', '.join(DOMAINS[d] for d in e.get('validation_domains',[])) or '-'} | {error} | {'검증 통과' if e.get('target_passed_validation') else '미달/미확인'} |")
    lines+=['','학습 중 수치는 지금까지의 가장 좋은 체크포인트이며 최종 결과가 아니다.',
            '검증 자료가 둘인 시도와 셋인 시도의 최대값은 같은 평가 범위가 아니므로 숫자만으로 직접 비교하지 않는다.',
            '두 모델 확률 평균은 추가 학습이 아니며 온라인 앱에 적용된 결과도 아니다.',
            '', '전체 사진 검증의 학습 경과(최종 확대/결합 설정과 구분):', '', '![학습 경과](facility-five-percent-curve.png)', '']
    for e in entries:
        if not e.get('operating_points'):continue
        lines += [f"## {e['title']} 상세 검증",'',
                  f"선택 epoch {e['best_epoch']}, 확대 grid {e['grid']}. {'뷰·임계값 선택 고정 완료.' if e['frozen_validation'] else '전체 사진으로 진행 중인 체크포인트 평가.'}",'',
                  '| 자료 | 항목 | 미탐/양성 | 미탐률 | 오탐/음성 | 오탐률 |','|---|---|---:|---:|---:|---:|']
        for label,point in e['operating_points'].items():
            for domain,m in point['domains'].items():
                lines.append(f"| {DOMAINS[domain]} | {LABELS[label]} | {m['fn']}/{m['tp']+m['fn']} | {m['fnr']*100:.2f}% | {m['fp']}/{m['fp']+m['tn']} | {m['fpr']*100:.2f}% |")
        lines += ['',f"보류 시험: {'실행됨' if e['test_executed'] else '미실행. 검증 목표 미달이면 시험으로 모델을 반복 선택하지 않는다.'}",'']
    lines+=['## 실제 자료와 적용 상태','',
            'DACL 학습6,225/검증710/기존 보류975. 원래 Dam 학습2,009개만 새 train1,585/val424로 분리, 기존 보류491 유지.',
            'CODEBRIM: 공식 MD5 및 7,810파일 CRC 확인. 정답 불명52개 제외 후 train6,438/val611/test628, 고유 부모 사진ID1,522.',
            'CODEBRIM 부모 ID의 분리 교차 및 기존 자료와 정확한 픽셀 겹침은0. 같은 시설/장면 독립성을 증명한 수치는 아니다.',
            '상세 증강12,041개와 위치 지도는 위 학습 부모에서 만든 파생 자료이다. 새 독립 사진 수에 더하지 않는다.',
            'CODEBRIM에 정답이 없는 물기/공동, 위치가 없는 양성 픽셀은 미확인으로 제외한다. 원래 정답은 바꾸지 않았다.','',
            'S2DS: 학술용 저자 원본743패치(train563/val87/test93), CRC 및 색상 수 확인. 중복·유사·부분 겹침 의심을 학습 전에 제외한다. 실제 사용량은 [선별 감사](facility-target-s2ds-screened-data-audit.json)에 기록한다.',
            'S2DS의 저자 TRAIN만 추가하고 원본 장면ID가 없다는 한계를 유지한다. S2DS val/test는 학습과 모델/임계값 선택에 쓰지 않는다. 해당 자료나 독립 현장에 대한5%성능 주장이 아니다.','',
            '추가 반복 학습은 [어려운 TRAIN 사례 보강 계획](FACILITY_HARD_TRAINING_PLAN_KO.md)을 따른다. 기존 최고 모델로 학습 사진만 평가하고, 알려진 균열·박락 정답과 차이가 큰 사진의 추출 비중을 최대3배 높였다. 검증·시험 사진은 비중 계산에 넣지 않았다.',
            '출처별 전체 추출 비중은 이전 공간 모델과 동일하게 유지한다. 새 실험의 실제 epoch와 평가값은 위 표에 반영한다. 이전 최고 후보보다 좋아지지 않았다면 최신 실행이라는 이유로 채택하지 않는다.','',
            '후속 시도는 [원본19종 보조 학습](FACILITY_AUXILIARY_PLAN_KO.md)이다. DACL TRAIN 전체 사진6,225개에만 원래 세부 태그를 부여하고 부분 조각과 다른 출처에는 보조 정답을 미확인으로 둔다. 태그를 원래 균열·박락 정답으로 합치거나 수정하지 않고 기존7항목의 추론 계약을 유지한다.','',
            '최근 작업은 [오류 검수·작은 영역 비교 계획](FACILITY_EFFICIENT_DEVELOPMENT_PLAN_KO.md)이다. 원본 TRAIN 오류 검수 화면을 만들고, 동일 초기 모델·seed·6epoch 조건으로 기존 격자 crop 대조군과 작은 부위 맥락 crop 보강군을 비교한다. 파생 사진은 새 독립 사진이 아니며, 원래 검증·시험 정답은 유지한다.','',
            '새 비교는 [콘크리트 사진 보강 계획](FACILITY_BUILDING_SUPPLEMENT_PLAN_KO.md)이다. ConViD 공식 균열·박락 폴더의 제한된 사진만 각각 해당 항목 양성으로 보강하고 나머지6항목은 미확인으로 유지한다. 공장 사진으로 표시하지 않으며 새 출처의 검증·시험 수치는 산출하지 않는다. PECCD는 숫자 클래스 대응 미확인으로 학습하지 않았다.','',
            '최근 구분 학습은 [고정 대조 계획](FACILITY_TARGET_DISCRIMINATION_PLAN_KO.md)을 따른다. 기존 원본 TRAIN의 같은 출처·항목에서 확인된 양성과 음성 점수 순위를 학습한다. 기존 sampler·정답·기본 손실을 유지하며 crop/unknown을 추가 순위 항에서 제외한다. 연구 후보 기준은 전체5% 목표와 다르며 [결과](FACILITY_TARGET_DISCRIMINATION_RESULTS_KO.md)에 항목별 변화와 서술적 Wilson 구간을 기록한다.','',
            '최신 구조 비교는 [학습 전 고정 계획](FACILITY_DETAIL_ARCHITECTURE_PLAN_KO.md)에 따라 stride 4 세부 특징 residual 분기와 기존 구조를 각각8epoch, 총16epoch 실제 학습했다. 최대 오류는 대조22.53%·보강22.80%로 기존 최고22.28%보다 높아 채택하지 않았다. 작은 손상 FN도 대조67건·보강69건으로 새 분기의 개선 효과를 확인하지 못했다. [실측 결과](FACILITY_DETAIL_ARCHITECTURE_RESULTS_KO.md)에 항목별 집계·143개 코드 테스트·실제 자원 비용을 기록한다.','',
            '후속 사진 pooling 비교는 [고정 계획](FACILITY_POOL_CONTEXT_PLAN_KO.md)을 따른다. 기존 global 경로·map·보조 head를 유지하고 top32/top256 로그잇 대비의 계수7개만 추가한다. 두 군 각각6epoch 예산이며 실제 진행·결과는 위 표와 [별도 집계](FACILITY_POOL_CONTEXT_RESULTS_KO.md)에 기록한다. 정답 검수 준비와 실제 정답 수정·새 산업 현장 검증을 구분한다.','',
            '최신 입력 해상도 비교는 [고정 계획](FACILITY_RESOLUTION_STUDY_PLAN_KO.md)에 따라 기존 processed 사진을 640·960으로 읽어 각6epoch 비교한다. 960의 120×120 지도는 80×80으로 줄인 뒤 기존 top32와 픽셀 정답을 사용한다. 실제 epoch와 [결과](FACILITY_RESOLUTION_STUDY_RESULTS_KO.md)·[집계](facility-resolution-study-comparison.json)에 기록하며 새 native 원본 입력이나 정답 수정은 아니다.','',
            '후속 원본 ROI 비교는 [고정 계획](FACILITY_NATIVE_ROI_STUDY_PLAN_KO.md)에 따라 기존 TRAIN DACL crop 범위와 라벨·80지도·표본 순서를 유지하고, 동일 native RGB에서 먼저 축소한 영역과 원본에서 직접 잘라낸 영역을 비교한다. 양군은 640 PNG이며 역사적 JPEG crop과 다른 전처리다. [실측 결과](FACILITY_NATIVE_ROI_STUDY_RESULTS_KO.md)에 비용·항목별 오류·작은 결함 FN을 기록한다.','',
            '최근 기존 항목 보존 비교는 [고정 계획](FACILITY_RETENTION_STUDY_PLAN_KO.md)에 따라 원본 모델의 알려진 다른5항목 확률을 보존하는 학습을 비교한다. [실측 결과](FACILITY_RETENTION_STUDY_RESULTS_KO.md)에 철근 노출 AP 회복, 나머지 항목 AP, 균열·박락 미탐/오탐 및 작은 손상 FN을 따로 보고한다. 성능 보존 후보 기준과 전체5% 목표는 별도로 판정한다.','',
            '가중치1 비교 결과를 본 후 [가중치4 후속 계획](FACILITY_RETENTION_STRENGTH_STUDY_PLAN_KO.md)을 고정했다. 같은 원본 조건에서 후보6epoch만 추가하며 대조군은 재사용한다. [후속 결과](FACILITY_RETENTION_STRENGTH_STUDY_RESULTS_KO.md)는 반복 source-VAL 탐색으로 보고하며 독립 현장 성능을 주장하지 않는다.','',
            '가중치4의 상태 감사를 근거로 [BN 통계 고정 계획](FACILITY_BATCHNORM_STUDY_PLAN_KO.md)을 추가했다. 같은 가중치4·T2에서 학생 BN의 학습 정규화 정책만 비교한다. [실제 결과](FACILITY_BATCHNORM_STUDY_RESULTS_KO.md)에 신규6epoch와 재사용 대조군, 버퍼 불변·학습 계수 경사·항목별 평가를 구분한다.','',
            'BN 고정 결과 이후 [head 학습률 계획](FACILITY_HEAD_LR_STUDY_PLAN_KO.md)을 고정했다. head 업데이트 크기를 줄이는 후보6epoch를 추가하고 원본 모델 및 기존 BN 고정 대조군과 비교한다. [실측 결과](FACILITY_HEAD_LR_STUDY_RESULTS_KO.md)는 반복 source-VAL 탐색이며 새로운 현장 성능 수치가 아니다.','',
            '보류하는 방법의 별도 진단: [자동 판단 비율·조건부 오류](facility-presence-target-spatial-review-diagnostic_KO.md). 자동으로 판단한 일부 사진만의 오답률이며 기존 전체 사진의 미탐·오탐 기준을 통과했다는 뜻이 아니다. 두 항목을 모두 자동 판단한 사진 비율과 보류 수까지 기록한다. 앱 적용·독립 시험 전이다.','',
            f"기본 앱 프로필: `{profile['version']}`. 실험 프로필로 자동 교체하지 않았다.",
            '현장의 모든 시설, 나머지 다섯 항목, 정밀 위치와 구조 안전에 대한 5% 성능 주장은 하지 않는다.',
            'CODEBRIM 모델의 임계값만 바꿔 5%를 맞출 수 있는지: [미탐·오탐 교환 진단](facility-target-codebrim-threshold-tradeoff.json). 이 고정 모델의 검증 점수에 한정된 진단이다.',
            '조건·후속 방법·엄격한 시험 차단은 [계획](FACILITY_FIVE_PERCENT_PLAN_KO.md), 원자료 사용 조건은 [CODEBRIM](https://zenodo.org/records/2620293)을 확인한다.','']
    (ROOT/'reports/FACILITY_FIVE_PERCENT_RESULTS_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Actual facility target status consolidated; release_target_achieved=',result['release_target_achieved'])


if __name__=='__main__':main()
