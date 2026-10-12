"""Compare saved results, retain regressions and render original missed cases."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from prepare_conveyor_expansion import DATA, read, new, sha
from study_conveyor_expansion import ROOT, RUN, artifact


def pct(value):
    return '산정 불가' if value is None else f'{value*100:.2f}%'


def rate(row, key):
    count, total = ('fp','negative_photos') if key=='fpr' else ('fn','positive_photos')
    return f"{row[count]}/{row[total]} · {pct(row[key])}"


def main():
    result,training,verification = read(artifact('results')),read(artifact('training')),read(artifact('verification'))
    assert verification['status']=='passed' and result['weights_sha256']==sha(RUN/'final.pt')
    report = ROOT/'reports/CONVEYOR_FOREIGN_EXPANSION_RESULTS_KO.md'
    assert not report.exists()
    fig,axes = plt.subplots(2,2,figsize=(10.8,7.2),layout='constrained')
    for axis,(phase,metric) in zip(axes.flat,[('val','fpr'),('val','fnr'),('test','fpr'),('test','fnr')]):
        before,after = result['common_baseline'][phase],result[phase]
        values = [before[metric]*100,after[metric]*100]
        bars = axis.bar(['1230 TRAIN photos','4558 TRAIN photos'],values,color=['#af8c40','#298876'])
        axis.bar_label(bars,labels=[rate(r,metric).replace(' · ',' (')+')' for r in (before,after)],padding=4)
        axis.axhline(5,color='#b24545',linestyle='--',linewidth=1)
        axis.set_ylim(0,max(10,max(values)+3))
        axis.set_title(phase.upper()+' known benchmark | '+('false positives' if metric=='fpr' else 'false negatives'))
        axis.set_ylabel('% of respective labeled photos')
    fig.suptitle('Fixed0.5 and final30 | same original VAL320 / TEST448 | more TRAIN data and updates')
    plot = ROOT/'reports/conveyor-foreign-expansion-rates.png'
    assert not plot.exists()
    fig.savefig(plot,dpi=155);plt.close(fig)

    old = read(ROOT/'runs/conveyor-foreign-grouped-v1/val-predictions.json')
    new_val = {r['id']:r for r in read(RUN/'val-predictions.json')}
    old_misses = sorted((r for r in old if r['label']==1 and r['score']<=.5),key=lambda r:r['id'])
    by_id = {r['id']:r for r in read(DATA/'manifest.json')['rows']}
    fig,axes = plt.subplots(2,4,figsize=(14,6.6),layout='constrained')
    for axis,before in zip(axes.flat,old_misses):
        row = by_id[before['id']]
        after = new_val[before['id']]
        with Image.open(DATA/row['native']) as image:
            axis.imshow(image)
        axis.axis('off')
        outcome = 'CORRECTED' if after['score']>.5 else 'STILL MISSED'
        axis.set_title(f"{row['recording_group'].replace('boardcamera_iron_ore','ore')}/{row['frame_number_hint']} | {outcome}\nAuthor=1 | {before['score']:.3f} -> {after['score']:.3f}",fontsize=9)
    fig.suptitle('ALL original8 VAL misses | original photos and saved scores only | no generated localization')
    examples = ROOT/'reports/conveyor-foreign-expansion-original-misses.png'
    assert not examples.exists()
    fig.savefig(examples,dpi=155);plt.close(fig)
    feedback = []
    for phase in ('val','audit_val','test'):
        for row in read(RUN/f'{phase}-predictions.json'):
            if (row['score']>.5)!=bool(row['label']):
                feedback.append({**row,'split':phase,'error_type':'false_negative' if row['label'] else 'false_positive',
                    'publisher_label':row['label'],'human_review_status':'not_reviewed','approved_for_training':False,
                    'model_sha256':training['weights_sha256']})
    new(artifact('feedback'),{'cases':feedback,'new_labels_created':0,'warning':'Keep benchmark cases out of TRAIN; audit uses same VAL recordings'})
    table = []
    for phase in ('val','audit_val','test'):
        for group,row in result[phase]['per_group'].items():
            table.append(f"| {phase} | {group.replace('boardcamera_iron_ore','ore')} | {rate(row,'fpr')} | {rate(row,'fnr')} |")
    corrected = sum(r['change']=='corrected' for r in result['changed_classifications']['val'])
    introduced = sum(r['change']=='new_error' for r in result['changed_classifications']['val'])
    gate = '통과' if result['common_val_reference_passed'] else '미달'
    audit_gate = '충족' if result['extra_val_fpr_fnr_below_5_percent'] else '미달'
    counts = read(DATA/'manifest.json')['counts']
    text = f'''# 이물질 학습 사진 보강 결과 · 2026-10-12

## 보강한 내용

이전 모델이 VAL의 `canga2`8장을 놓친 문제를 보고, **다른 TRAIN 촬영 구간의 원본 사진**을 추가했다.
미탐 사진이나 같은 VAL/TEST 촬영 구간을 학습에 넣지 않았다.
학습은1,230장(정상718/이물질512)에서 **4,558장(정상3,067/이물질1,491)**으로 증가했다.
추가 TRAIN3,328장과 같은 VAL 촬영명의 미사용309장(정상192/이물질117)을 준비했다.
기존1,998장 전체의 원본 JPEG·384 전처리 PNG는 동일 바이트로 유지했다.
기존 VAL320장·TEST448장의 사진·정답·분모를 바꾸지 않았다.

새 자료를 웹에서 다시 받을 필요 없이 작성자 전체 SHA가 검증된 원본 ZIP에서 선별했다.
TRAIN 촬영명/클래스마다 최대256개의 수치 순서 프레임을 일정 간격으로 뽑고 이전64개 프레임을 합쳤다.
해독RGB 완전 중복0, 촬영명 그룹 분리, sourceTEST의 TRAIN 유입0을 확인했다.
추가 VAL은 기존 평가 사진을 제외하고 촬영명/클래스마다 최대64개를 점수 계산 전에 선택했다.
정답은 작성자 폴더의 사진 단위 이물질 유무이며 AI/사람이 새 라벨을 만들지 않았다.

## 실제 학습과 공정한 비교 범위

ResNet18·입력384·batch32·seed95·임계값0.5·밝기/좌우 변형·AdamW·30epoch는 이전과 같다.
일반 ImageNet과 새 분류 헤드에서 시작했고 **초기 가중치 SHA가 이전 모델과 같음**을 검증했다.
이전 이물질 모델을 이어 학습하거나 건물/벨트 균열 모델을 사용하지 않았다.
클래스 가중치는 새 TRAIN의 정상/이물질 비율을 따른다.

실제 **{training['epochs']}epoch·{training['elapsed_training_minutes']:.2f}분·{training['optimizer_updates']:,}회 업데이트**를 완료했다.
이전 모델은30epoch·1,170회였으므로 **사진 수와 업데이트 수가 함께 증가**했다.
동일 계산량에서 데이터만의 효과를 분리한 실험은 아니다. 사전 GPU 시험1회는 epoch에서 제외했다.
설정·입력·소스를 학습 전 커밋 `{training['frozen_before_fit_commit'][:7]}`으로 고정했고,
마지막30epoch와0.5 기준을 새 점수 계산 전에 저장했다. 점수로 epoch나 임계값을 고르지 않았다.

기존 VAL/TEST 결과를 본 뒤 보강했으므로 아래 기존 사진 결과는 **재사용하는 공개 비교 벤치마크**다.
새로운 독립 최종 시험 성능으로 소개하지 않는다. 추가309장도 처음 점수를 계산한 사진이지만 같은3 VAL 촬영명에 속한다.
독립 공장·다른 카메라·새 물리적 물체 검증은 아니다. 연속 프레임 간 상관은 남아 있다.

## 사진 오탐·미탐

| 자료 | 이전 오탐 | 보강 오탐 | 이전 미탐 | 보강 미탐 |
|---|---:|---:|---:|---:|
| 같은 VAL320장 | {rate(result['common_baseline']['val'],'fpr')} | **{rate(result['val'],'fpr')}** | {rate(result['common_baseline']['val'],'fnr')} | **{rate(result['val'],'fnr')}** |
| 같은 TEST448장 · 알려진 비교 자료 | {rate(result['common_baseline']['test'],'fpr')} | **{rate(result['test'],'fpr')}** | {rate(result['common_baseline']['test'],'fnr')} | **{rate(result['test'],'fnr')}** |
| 추가 VAL309장 · 같은 촬영명 | 이번 이전 모델은 미측정 | **{rate(result['audit_val'],'fpr')}** | 이번 이전 모델은 미측정 | **{rate(result['audit_val'],'fnr')}** |

학습 전에 정한 공통 VAL 기준(오탐 악화 없음·미탐 감소)은 **{gate}**이다.
추가 VAL의 오탐/미탐 각각5% 미만 기준은 **{audit_gate}**이다.
VAL에서 이전 오류{corrected}건을 바로잡았고 새 오류{introduced}건이 생겼다.
기존 모델과 파일을 보존했으며 앱 기본 모델은 자동 교체하지 않았다.

![같은 평가 사진의 이전·보강 모델 비교](conveyor-foreign-expansion-rates.png)

### 모든 촬영명별 분모

| 자료 | 촬영명 | 오탐 | 미탐 |
|---|---|---:|---:|
{chr(10).join(table)}

### 이전 미탐8장 전체

아래8장은 평가 자료로 보존된 동일 원본이다. 저장된 점수만 비교했고 손상 위치·이물질 경계 상자를 생성하지 않았다.
좋아진 사진만 선택하거나 남은 오류의 라벨을 수정하지 않았다.

![이전 누락8장 전체 비교](conveyor-foreign-expansion-original-misses.png)

## 앱 연결과 한계

기존 이물질 서버를 새 체크포인트와 해당 SHA로 명시적으로 실행할 수 있다.
요청/응답은 `conveyor_foreign_object`, `iron_ore_closeup`, 사진 유무 분류, 사람 최종 확인으로 유지한다.
정상은 이물질 없음이고 정상 벨트·정상 가동·기계 고장 없음이 아니다.
이물질 위치 정답이 없어 heatmap/경계 상자는 없다. 벨트 균열 모델의 오차와 이 결과를 합치지 않는다.
이 데이터는 철광석이 흐르는 실험 컨베이어의 **잘린 벨트 근접 사진**이다.
새 현장/운반물/스마트폰 전체 장면 성능은 별도 검증이 필요하다.

- 새 가중치: `runs/conveyor-foreign-expansion-v2/final.pt`
- SHA256: `{training['weights_sha256']}`
- [고정 설정](conveyor-foreign-expansion-protocol.json)
- [실제 학습](conveyor-foreign-expansion-training.json)
- [전체 수치·바로잡은 오류·새 오류](conveyor-foreign-expansion-results.json)
- [실제 학습·보존 확인](conveyor-foreign-expansion-verification.json)
- [남은 오류 피드백 목록](conveyor-foreign-expansion-feedback.json)
- [역할4 전달 설명](../../docs/CONVEYOR_EXPANSION_HANDOFF_KO.md)
- [사진 출처·파생 그림 변경](CONVEYOR_EXPANSION_ATTRIBUTION_KO.md)

검증에서 이전 소스{verification['legacy_sources_sha_preserved']}개,
기존 모델/원본{verification['previous_artifacts_sha_size_mtime_preserved']:,}개,
새 입력{verification['new_inputs_sha_size_mtime_preserved']:,}개의 SHA·크기·수정시간을 확인했다.
데이터 검사4개와 실제GPU 업데이트/strict CPU·GPU 재로드를 통과했다.
후속 보강에서는 원래 알려진 TEST를 독립 성능이라고 재사용하지 않고 새 촬영/현장 정답을 확보한다.
'''
    with report.open('x',encoding='utf-8',newline='\n') as handle:
        handle.write(text)
    new(artifact('plot'),{'status':'created','additional_model_test_scores':0,'saved_predictions_only':True,
        'all_original_val_miss_ids':[r['id'] for r in old_misses],
        'figures':{p.name:sha(p) for p in (plot,examples)},'report_sha256':sha(report),
        'source_sha256':sha(Path(__file__)),'model_sha256':training['weights_sha256']})
    print(json.dumps({'report':str(report),'common_val_reference_passed':result['common_val_reference_passed'],
                      'extra_val_gate':result['extra_val_fpr_fnr_below_5_percent'],'remaining_errors':len(feedback)}))


if __name__=='__main__':
    main()
