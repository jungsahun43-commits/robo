"""Report the actual first belt result, keeping photo and pixel rates separate."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.config/matplotlib'))


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    train = read('reports/beltcrack-starter-training.json')
    result = read('reports/beltcrack-starter-results.json')
    proof = read('reports/beltcrack-starter-verification.json')
    assert proof['status'] == 'passed'
    photo = result['photo_rates']
    lines = ['# 산업 컨베이어 벨트: 실제 첫 모델 학습 결과', '',
        '## 대상과 출처', '',
        '**실제 컨베이어 벨트 표면의 균열·찢어짐 위치**를 표시하는 연구용 모델이다. 콘크리트·교량·댐 균열 모델을 재사용하지 않았다. 롤러·모터·정렬·장력·온도나 내부 고장 상태는 진단하지 않는다.', '',
        '출처는 [BeltCrack 작성자](https://github.com/UESTC-nnLab/BeltCrack)의 [ScienceDB V1](https://doi.org/10.57760/sciencedb.31181), 데이터 라이선스 **CC BY-SA4.0**이다. 인용: Huang, Jianghong; Ji, Luping; Ma, Xin; Ye, Mao. *BeltCrack: the First Sequential-image Industrial Conveyor Belt Crack Detection Dataset and Its Baseline with Triple-domain Feature Learning*, Pattern Recognition,2026,113598, DOI10.1016/j.patcog.2026.113598.', '',
        '압축명은 `BeltCrack14ks`이지만 실제 파일 색인은 **9,645장**, TRAIN5,736장·VAL1,961장·TEST1,948장이다. 실제 촬영 구간은 TRAIN23·VAL10·TEST9개다. 작성자 발표의14,087장/29구간과 일치한다고 주장하지 않는다. 배포된 파일의 실제 내용으로 분석했다.', '',
        '원본에는 **LabelMe `crack` polygon/rectangle와 작성자가 포함한 `imageData` 사진**이 있다. 논문의 COCO bbox 소개와도 형식 차이가 있어 실제 주석으로 별도 위치 학습을 구현했다. embedded 사진과 별도 JPG의 완전 동등성을 검증한 것은 아니다.', '',
        '## 확보·분리·학습', '',
        '처음에는 각 TRAIN/VAL 구간에서 최대32개 프레임을 숫자 순서에 따라 균등하게 선택해1,031장을 계획했다. 제공처429 제한이 발생해 그 계획 중 이미 완전한 원본 바이트를 확보한 **679장(학습552·검증127)**으로 첫 실험을 고정했다. 정답 내용·모델 점수·쉬운 사례 여부로 선택하지 않았다. 사용한 구간은 TRAIN19·VAL4개로 서로 겹치지 않는다. 아직 전체 공개 데이터 학습을 마친 것은 아니다.', '',
        'RAR quick-open 색인의 CRC, 선택한 파일의 실제 헤더와 색인의 일치, 공개 암호를 사용하는 압축 해제 체크섬, 원본 주석·사진 SHA를 확인했다. 전체 압축 MD5는 검증하지 않았다. 남은 전체 전송과1,031장 계획은 부분 파일을 그대로 보존해 재개 가능하다. 요청 제한이 반복된 다운로드 작업은 종료했고 실행 중인 추가 학습은 없다.', '',
        '전체 RGB 완전 동일 사진 중복0건. 균열 정답은640×640 letterbox 좌표에 rasterize하며, 패딩은 손실·평가에서 제외한다. 작은 원본 다각형의 정확한 native 픽셀 성능을 측정한 것은 아니다. 학습·검증에 sourceTEST 사진/정답을 추출하거나 추론하지 않았다. 같은 물리적 벨트·공장의 반복 촬영이므로 독립 공장 검증으로 볼 수 없다.', '',
        f"Torchvision 일반 COCO/VOC 사전학습 LRASPP MobileNet을 시작점으로 새1종 출력 head와 전체 가중치를 **실제40epoch·{train['elapsed_training_minutes']:.2f}분**, optimizer **{train['optimizer_updates']:,}회** 업데이트했다. 별도 사전 GPU 업데이트{proof['disposable_preflight_updates']}회는 epoch에서 제외한다. 원래 건물 모델과 회로기판 모델은 재학습하지 않았다.", '',
        '마지막40epoch 모델과 pixel 임계값0.5를 학습 전에 고정했다. VAL 최적 epoch나 VAL에 맞춘 임계값을 선택하지 않았다. 양성 가중치는 TRAIN 픽셀 수로만 정하고20을 상한으로 사용했다. 손실은 알려진 픽셀의 가중 BCE + 사진별 Dice다.', '',
        '## 실제 VAL127장 결과', '',
        '| 측정 범위 | 결과 |', '|---|---:|',
        f"| 모델 좌표 손상 영역 IoU | {result['pixel_iou']*100:.2f}% |",
        f"| 모델 좌표 손상 영역 Dice | {result['pixel_dice']*100:.2f}% |",
        f"| 균열 픽셀 미탐률 | {result['pixel_fnr']*100:.2f}% |",
        f"| 주석 배경 픽셀 오탐률 | {result['pixel_fpr']*100:.4f}% |",
        f"| 균열 사진 미탐 / 양성 사진 | {photo['fn']}/{photo['positive_photos']} ({photo['fnr']*100:.2f}%) |",
        f"| 정상 사진 오탐 / 정상 사진 | {photo['fp']}/{photo['negative_photos']} · 분모0으로 산정 불가 |", '',
        '**이번679장에는 모두 균열 정답이 있어 정상 벨트 사진이0장이다. 사진 오탐률은0%가 아니라 측정 불가다.** 픽셀 배경 오탐률로 정상 사진의 오탐률을 대신하지 않는다. 사진 미탐률만 낮다고 전체 오차5% 미만 또는 정상 설비 판별이 검증됐다고 주장할 수 없다. 사진의 양성 판별과 정확한 손상 위치 탐지는 별도다.', '',
        '균열 위치 표시의 첫 기준 모델을 확보한 결과다. 기존 건물 오류율과 이 새 벨트 수치를 직접 비교해 성능이 개선됐다고 주장하지 않는다.', '',
        '## 앱 연결과 다음 우선 작업', '',
        '`infer_beltcrack_starter.py`로 실제 사진과 단일 `belt_crack` 체크포인트를 CPU에서 분석하고 원래 사진 크기의 heatmap을 받을 수 있다. 낮은 점수도 정상 가동 확정 결과가 아니다. 모든 결과에 실제 고장 미평가·점검자 확인 필요·정상 벨트 specificity 미검증을 함께 기록한다. 기존 앱 API·기본 모델에 자동 적용하지 않았다.', '',
        '다음 우선순위는 **작성자가 정상으로 주석한 정상 벨트 사진과 손상 비교 자료 확보**다. 촬영 구간을 분리해 사진 오탐률을 검증하고, 작은 균열·이음매·오염·밧줄 같은 혼동을 확인한다. 롤러·모터의 내부 고장은 센서·운전 데이터와 별도 모델을 준비한다.', '',
        '```powershell', '.venv/Scripts/python.exe scripts/infer_beltcrack_starter.py --image <conveyor-photo.jpg> --heatmap <new-heatmap.png>', '```', '',
        f"패딩의 오탐 제외·gradient 제외·형상 오류 검사 테스트{proof['tests_passed']}개를 통과했다. 기존 소스{proof['legacy_runtime_sources_preserved']}개·모델 등 아티팩트{proof['previous_model_artifacts_sha_size_mtime_preserved']}개와 벨트 입력{proof['belt_input_files_sha_size_mtime_preserved']:,}개의 SHA/크기/수정시각 보존을 검증했다. strict CPU/GPU 재로드 비교를 통과했다. 기존 건물·PCB 전체 입력 보존 검증은 별도 [PCB 검증](equipment-visa-verification.json)에 기록했다.", '',
        '[학습 전 고정 프로토콜](beltcrack-starter-protocol.json), [실제 학습 기록](beltcrack-starter-training.json), [사진·픽셀·촬영 구간별 수치](beltcrack-starter-results.json), [보존 검증](beltcrack-starter-verification.json).', '']
    output = ROOT / 'reports/BELTCRACK_STARTER_RESULTS_KO.md'
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    history = read('runs/beltcrack-polygon-starter-v1/history.json')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history], color='#287f87')
    axes[0].set_title('Actual belt neural training: final40 fixed')
    axes[0].set_xlabel('Actual training epoch')
    axes[0].set_ylabel('TRAIN weighted BCE + Dice loss')
    axes[1].bar(['IoU', 'Dice', 'Pixel FNR', 'Pixel FPR'],
                [100*result[k] for k in ('pixel_iou', 'pixel_dice', 'pixel_fnr', 'pixel_fpr')],
                color=['#4489a3', '#6ba185', '#b7686a', '#d59743'])
    axes[1].set_ylabel('At model resolution (%)')
    axes[1].set_title('127 positive VAL photos: photo FPR unavailable')
    axes[1].set_ylim(0, 105)
    for axis in axes:
        axis.grid(axis='y', alpha=0.2)
    fig.suptitle('BeltCrack official archive starter | 552 TRAIN / 127 VAL | Normal belt photos: 0', fontsize=11)
    fig.tight_layout()
    plot = ROOT / 'reports/beltcrack-starter-baseline.png'
    assert not plot.exists()
    fig.savefig(plot, dpi=150)
    plt.close(fig)
    print('Belt actual results reported', flush=True)


if __name__ == '__main__':
    main()
