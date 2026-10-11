"""Report frozen belt expansion measurements and pre-existing example choices."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from study_beltcrack_expansion import artifact, PROTOCOL, RUN, DATA, BASELINE, read, normalize, sha
from safelog_ai.belt_segmentation import create_model


def main():
    torch.set_num_threads(4)
    result,training,verified = read(artifact('results')),read(artifact('training')),read(artifact('verification'))
    assert verified['status']=='passed' and sha(RUN/'final.pt')==result['weights_sha256']
    baseline,expanded,full = result['common127_baseline'],result['common127_expanded'],result['expanded_val']
    target = ROOT/'reports/BELTCRACK_EXPANSION_RESULTS_KO.md'
    assert not target.exists()
    labels = ['Photo false negatives','Crack pixel false negatives','Damage area IoU']
    old = [baseline['photo_rates']['fnr']*100,baseline['pixel_fnr']*100,baseline['pixel_iou']*100]
    new = [expanded['photo_rates']['fnr']*100,expanded['pixel_fnr']*100,expanded['pixel_iou']*100]
    fig,axes = plt.subplots(1,3,figsize=(11.5,3.8),layout='constrained')
    for i,axis in enumerate(axes):
        bars = axis.bar(['552 training photos','1397 training photos'],[old[i],new[i]],color=['#537ba5','#239978'])
        axis.bar_label(bars,fmt='%.2f%%',padding=4)
        axis.set_ylim(0,100)
        axis.set_title(labels[i]+('\n(lower is better)' if i<2 else '\n(higher is better)'))
        axis.tick_params(axis='x',labelsize=8)
        axis.set_ylabel('%')
    fig.suptitle('Same127 publisher VAL photos | fixed0.5 threshold | normal-photo FPR unknown')
    plot = ROOT/'reports/beltcrack-expansion-comparison.png'
    assert not plot.exists()
    fig.savefig(plot,dpi=160)
    plt.close(fig)

    # Select examples from the old saved scores, without favoring the new model.
    old_predictions = read(ROOT/'runs/beltcrack-polygon-starter-v1/val-predictions.json')
    ordered = sorted(zip(old_predictions['photo_scores'],old_predictions['ids']))
    selected = [ordered[i][1] for i in (0,1,len(ordered)//2,len(ordered)-1)]
    by_id = {r['id']:r for r in read(DATA/'manifest.json')['rows']}
    networks = []
    for checkpoint in (BASELINE,RUN/'final.pt'):
        model = create_model(False).eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'],strict=True)
        networks.append(model)
    fig,axes = plt.subplots(4,4,figsize=(13,10),layout='constrained')
    columns = ['Publisher photo (640 grid)','Author polygon truth','Baseline40epoch mask','Expanded40epoch mask']
    for index,item in enumerate(selected):
        row = by_id[item]
        rgb = np.asarray(Image.open(DATA/row['image']).convert('RGB'))
        valid = np.asarray(Image.open(DATA/row['valid']))>0
        gt = np.asarray(Image.open(DATA/row['mask']))>0
        known_y,known_x = np.where(valid)
        crop = (slice(known_y.min(),known_y.max()+1),slice(known_x.min(),known_x.max()+1))
        with torch.no_grad():
            x = normalize(torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0),device='cpu')
            predictions = [net(x)['out'][0,0].sigmoid().numpy()>0.5 for net in networks]
        maps = [None,gt,*predictions]
        for column,mask in enumerate(maps):
            canvas = rgb.copy()
            if mask is not None:
                active = mask & valid
                canvas[active] = (0.4*canvas[active]+0.6*np.asarray([255,85,35])).astype('uint8')
            axes[index,column].imshow(canvas[crop])
            axes[index,column].axis('off')
            if index==0:
                axes[index,column].set_title(columns[column],fontsize=10)
            if column==0:
                axes[index,column].text(0,-0.05,row['sequence']+'/'+Path(item).name,
                    transform=axes[index,column].transAxes,fontsize=8)
    fig.suptitle('Fixed examples selected by OLD scores | orange = rasterized truth or predicted area')
    examples = ROOT/'reports/beltcrack-expansion-examples.png'
    assert not examples.exists()
    fig.savefig(examples,dpi=140)
    plt.close(fig)

    def pct(value):
        return '산정 불가' if value is None else f'{value*100:.2f}%'
    pp = (expanded['photo_rates']['fnr']-baseline['photo_rates']['fnr'])*100
    iou_pp = (expanded['pixel_iou']-baseline['pixel_iou'])*100
    counts = read(DATA/'manifest.json')['counts']
    tiny_old,tiny_new = result['common_tiny_baseline'],result['common_tiny_expanded']
    tiny_text = (f"작은 균열{tiny_old['photos']}장에서는 사진 미탐 "
        f"{tiny_old['photo_rates']['fn']}→{tiny_new['photo_rates']['fn']}장으로 감소했지만, "
        f"손상 위치 IoU **{pct(tiny_old['pixel_iou'])}→{pct(tiny_new['pixel_iou'])}로 악화**됐고 "
        f"균열 픽셀 미탐도{pct(tiny_old['pixel_fnr'])}→{pct(tiny_new['pixel_fnr'])}로 증가했다. "
        "전체 사진 미탐 개선만으로 작은 손상의 위치 탐지가 해결됐다고 해석하지 않는다. "
        "작은 표본이며 다음 실험에서는 원본 해상도의 국소 영역이나 다중 크기 특징으로 위치 손실을 보강할 필요가 있다.") if tiny_old else ''
    text = f'''# 컨베이어 벨트 원본 사진 확대: 실제 학습 결과

## 변경 내용

공식 [BeltCrack V1](https://doi.org/10.57760/sciencedb.31181)의 이미 확보한 원본에서 **1,071장을 추가 추출**했다. 새로운 출처나 정상 데이터셋을 추가한 것은 아니다. TRAIN552→{counts['train']}장/19구간, VAL127→{counts['val']}장/4구간이다. 기존679장의 RGB·640좌표 사진·마스크·유효 영역은 완전히 동일하며, 선택한1,750장 내 완전 동일 RGB 중복0건이다. 시간상 유사 프레임과 같은 물리적 벨트의 상관관계까지 제거한 것은 아니다.

각 구간에서 완전히 확보된 원본 바이트의 숫자 순서 프레임을 고르게 골라, 새 정답과 점수를 보기 전에 목록을 고정했다. 실제 헤더 일치와 추출 파일별 체크섬을 확인했고, 전체 압축 MD5를 확인한 것은 아니다. sourceTEST 사진·정답은 사용하지 않았다. 데이터 라이선스 CC BY-SA4.0, 네트워크 추가 요청0회.

## 실제 모델 학습

동일한 일반 사전학습 LRASPP MobileNet의 초기 state SHA와 seed87, 입력640, 새1종 head, 가중 BCE+Dice, AdamW1e-4,40회 cosine schedule, 마지막40회 모델, 임계값0.5를 학습 전에 고정했다. 이전 벨트 가중치나 건물 모델로 warm start하지 않았다.

**실제40epoch·{training['elapsed_training_minutes']:.2f}분·optimizer{training['optimizer_updates']:,}회**를 완료했다. 별도 GPU 사전 검사1회 업데이트는40epoch에 포함하지 않는다. 사진 수가 늘어 optimizer 업데이트가2,760→{training['optimizer_updates']:,}회로 늘었다. 같은 연산량의 순수 데이터 효과 실험이라고 해석하지 않는다. 학습 손실은 기록했지만 검증 점수로 epoch·임계값을 고르지 않았다.

## 동일127장 비교

이127장은 앞선 연구에서 이미 확인한 공개 VAL이다. 블라인드 최종 시험이나 독립 공장 성능이 아니다. 이전 모델의 사진·픽셀 혼동 행렬을 재현해 기존 기록과 완전히 같은지 확인했다.

| 지표 | 이전552장 학습 | 확대1,397장 학습 |
|---|---:|---:|
| 균열 사진 미탐 / 양성 사진 | {baseline['photo_rates']['fn']}/127 · {pct(baseline['photo_rates']['fnr'])} | {expanded['photo_rates']['fn']}/127 · {pct(expanded['photo_rates']['fnr'])} |
| 손상 영역 IoU | {pct(baseline['pixel_iou'])} | {pct(expanded['pixel_iou'])} |
| 손상 영역 Dice | {pct(baseline['pixel_dice'])} | {pct(expanded['pixel_dice'])} |
| 균열 픽셀 미탐률 | {pct(baseline['pixel_fnr'])} | {pct(expanded['pixel_fnr'])} |
| 주석 배경 픽셀 오탐률 | {pct(baseline['pixel_fpr'])} | {pct(expanded['pixel_fpr'])} |
| 양성 사진 중 정답 손상과 겹친 예측이 전혀 없는 사진 | {baseline['positive_photos_without_any_gt_overlap']}/127 | {expanded['positive_photos_without_any_gt_overlap']}/127 |
| 정상 사진 오탐 / 정상 사진 | 0/0 · 산정 불가 | 0/0 · 산정 불가 |

사진 미탐 변화 **{pp:+.2f}pp**, IoU 변화 **{iou_pp:+.2f}pp**다. 사진의 어느 위치든0.5보다 큰 점수가 있으면 사진 양성으로 세므로, 사진 미탐률만으로 손상 위치를 정확히 찾았다고 판단하지 않는다. 모델 좌표 픽셀 IoU·미탐도 함께 확인한다.

![동일127장 비교](beltcrack-expansion-comparison.png)

## 확대 VAL353장

사진 미탐 **{full['photo_rates']['fn']}/{full['photos']} · {pct(full['photo_rates']['fnr'])}**, IoU **{pct(full['pixel_iou'])}**, Dice **{pct(full['pixel_dice'])}**, 균열 픽셀 미탐 **{pct(full['pixel_fnr'])}**다. 정상 사진은0장이며 사진 오탐률은 여전히 `null`이다. 추가226장도 기존127장과 같은 네 촬영 구간의 프레임이므로 별도 독립 현장 시험으로 해석하지 않는다.

작은 균열의 기준은 TRAIN정답 면적 비율의 하위10%({result['tiny_gt_area_ratio_cutoff_train_only']:.8f})로 학습 전에 고정했다. 해당 비교와 촬영 구간별 혼동 행렬은 [결과 JSON](beltcrack-expansion-results.json)에 기록했다.

{tiny_text}

**정상 사진 분모가0이므로 사진 오탐·미탐 각각5% 미만을 달성했다고 말할 수 없다.** 로컬화·정상 데이터 검증을 마칠 때까지 연구 후보로 유지하며 앱 기본 모델을 자동으로 바꾸지 않았다.

## 실제 사진 비교

아래 예시는 새 모델을 유리하게 고르지 않도록 이전 모델의 저장된 점수에서 낮은2장·중앙1장·높은1장을 선택했다. 주황색은 각 열의 작성자 주석 또는 모델0.5 예측이다. 보기 좋은 예시가 전체 성능을 대신하지 않는다.

![작성자 사진·주석·두 모델](beltcrack-expansion-examples.png)

## 정상 데이터와 팀 연결

[정상 데이터 조사](BELT_NORMAL_DATA_AUDIT_KO.md)에 실제 접근 조건을 기록했다. Roboflow 후보는 CC BY4.0/normal belt 등6종을 공개하지만 COCO ZIP 다운로드에 로그인이 필요해 이번 학습에서 사용하지 않았다. 원본을 받으면 `audit_belt_normal_coco.py`가 주석 혼합·동일 RGB·증강 부모의 분리 누락을 조사한다. normal belt 사각형을 사진 전체의 정상 가동 정답으로 바꾸지 않는다.

```powershell
.venv/Scripts/python.exe scripts/infer_beltcrack_starter.py --checkpoint runs/beltcrack-cached-expansion-v2/final.pt --image <conveyor-photo.jpg> --heatmap <new-heatmap.png>
.venv/Scripts/python.exe scripts/audit_belt_normal_coco.py --archive <belt-defects-coco.zip> --output data/belt-normal-audit-v1/audit.json
```

모델 파일은 `runs/beltcrack-cached-expansion-v2/final.pt`, SHA256 **{training['weights_sha256']}**다. 데이터·모델은 로컬에 있고 Git에는 재현 코드·프로토콜·수치를 저장한다. 역할4에 전달할 모델은 파일을 별도로 공유해야 한다. 사진의 `belt_crack` 손상 후보·heatmap만 반환하고 실제 내부 고장·정상 가동을 확정하지 않는다. 점검자 확인이 필요하다.

기존 소스{verified['legacy_runtime_sources_sha_preserved']}개 SHA, 이전 아티팩트·입력{verified['previous_artifacts_sha_size_mtime_preserved']:,}개와 확대 입력{verified['expanded_inputs_sha_size_mtime_preserved']:,}개의 SHA/크기/수정시각 보존을 확인했다. CPU/GPU strict 재로드·이전127장 예측 재현·신규 실험 테스트4개와 별도 정상 데이터 검사 테스트3개를 통과했다.

[학습 전 프로토콜](beltcrack-expansion-protocol.json), [실제40회 기록](beltcrack-expansion-training.json), [수치](beltcrack-expansion-results.json), [보존 검증](beltcrack-expansion-verification.json).
'''
    with target.open('x',encoding='utf-8') as handle:
        handle.write(text)
    meta = ROOT/'reports/beltcrack-expansion-plot.json'
    with meta.open('x',encoding='utf-8') as handle:
        json.dump({'labels':labels,'baseline_percent':old,'expanded_percent':new,'example_ids':selected,
            'example_selection':'By old saved scores, not new performance','report_script_sha256':sha(Path(__file__)),
            'protocol_sha256':sha(PROTOCOL),'result_sha256':sha(artifact('results')),
            'comparison_png_sha256':sha(plot),'examples_png_sha256':sha(examples)},handle,indent=2)
    print('Belt expansion report and real-photo comparison created',flush=True)


if __name__=='__main__':
    main()
