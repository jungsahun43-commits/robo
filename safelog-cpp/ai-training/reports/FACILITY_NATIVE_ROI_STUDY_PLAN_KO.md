# 원본 ROI의 세부 정보 보존 대조 실험

## 목적

직전 640·960 해상도 비교는 각 6epoch를 실제 완료했지만 최대 검증 미탐·오탐은 초기 22.28%, 대조 22.53%, 보강 22.28%였다. 작은 결함 누락도 960에서 초기 대비 줄지 않았다. 이번에는 학습에 쓰는 기존 영역의 사전 축소가 세부 정보를 잃게 하는지 비교한다.

기존 [TRAIN 원본 조사](facility-resolution-source-audit.json)에서 native 원본 6,225장 중 3,521장은 기존 processed 부모 사진보다 컸다. 이들 부모의 기존 DACL detail row 5,928개를 양군에서 같은 방식으로 준비한다. 원본이 더 크다는 사실은 성능 개선의 증거가 아니다.

## 사전 고정한 개입

같은 native 사진을 EXIF 보정 후 RGB로 한 번 디코딩한다.

| 군 | 처리 |
|---|---|
| 대조 | native RGB를 기존 processed 부모 크기로 LANCZOS 축소 → 기존 정수 box → 640×640 PNG |
| 보강 | native RGB에서 같은 범위의 floating box → 바로 640×640 PNG |

양군 모두 최종 LANCZOS, `reducing_gap=None`, 같은 PNG 저장 설정을 쓴다. 원본 폭·높이는 processed 폭·높이 이상이고 하나 이상이 더 큰 경우만 대상으로 정한다. 기존 음성·양성·미확인 crop을 모두 포함하며 모델의 오류 점수나 VAL·TEST 결과로 준비 대상을 고르지 않는다.

기존 box `(x1,y1,x2,y2)`에 native/processed 축별 비율을 곱해 보강군의 floating box를 계산한다. 좌표는 half-open 범위이고 floor/ceil이나 `x2-1`을 적용하지 않는다. 두 box의 정규화된 범위는 같다. publisher annotation과 EXIF 보정 치수·원본 hash를 확인하며 사진·정답을 자동 교정하지 않는다.

새 대조군은 역사적 `thumbnail(max_side1280) → JPEG95 → 작은 crop JPEG`의 픽셀 재현이 아니다. 양군 모두 같은 native RGB에서 새로 만든 640 PNG이므로 JPEG 재압축 차이를 개입에 섞지 않는다. 이전 초기 모델과의 비교에는 새 crop 해상도·형식·전처리 순서가 함께 달라진다는 한계가 있다. 원본 보존 효과는 새 대조군과 보강군의 차이로 평가한다.

## 유지하는 학습 계약

- 전체 row 26,289개, full 14,248개, detail 12,041개와 순서를 유지한다.
- full 사진, 비대상 crop, 7종 사진 정답, 기존 80×80 위치 정답과 known flag는 그대로 둔다.
- 바뀌는 row의 `image` 경로만 양군 PNG로 대체한다. 원래 box·부모·domain·target·pixel_target와 원본 파일은 유지한다.
- DACL 19종 보조 태그는 기존 full 사진만 사용한다. crop과 다른 출처의 보조 태그는 unknown으로 유지한다.
- 초기 모델은 `facility-presence-target-roi-control/best.pt`, SHA `0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a`다.
- 양군 모두 기존 Auxiliary 구조, 입력 640, batch 8, seed 56, epoch 6, draw/epoch 14,248, 출처 추출 비중 70/10/20%를 쓴다.
- 기존 학습률·손실·지도·sampling weight·positive weight를 유지하고 매 epoch의 실제 row 순서 SHA와 출처·full/crop·항목 조합 수를 비교한다. AMP skip과 실제 optimizer update는 각각 기록한다.

새 독립 사진·전문가 라벨·정밀 위치 정답은 추가하지 않는다. 이번에는 라벨 경계 점검을 TRAIN 자료의 태그 구성 확인으로 제한한다. [원본 태그 공존 집계](facility-train-label-groups.json)의 동시 등장 수는 polygon 겹침이나 오라벨의 증거가 아니며 라벨 수정이나 샘플링 변경에 쓰지 않는다.

## 평가와 비용

기존 공개 자료 VAL 세 출처를 같은 full-photo grid1로 평가한다. 각 항목의 cutoff는 해당 VAL에서 기존 minimax 규칙으로 고정한다. 균열·박락 × 세 출처 × FNR/FPR의 12개 비율 중 최대값과 개별 혼동 수, 42개 알려진 항목 AP, 작은 DACL 균열 93·박락 105개 항목·사진 사례의 FN을 비교한다. 이 198개 사례는 서로 다른 사진 198장이라는 뜻이 아니다.

연구 후보는 초기 모델과 새 대조군 각각에 대해 최대 오류 0.5pp 이상 개선, 개별 오류 악화 2pp 이하, 다른 알려진 항목 AP 악화 0.02 이하, 작은 FN 합계 2건 이상 감소 및 항목별 작은 FNR 악화 2pp 이하를 모두 요구한다. 기존 [기계 판독 기준](facility-native-roi-study-protocol.json)에 고정한다.

엄격한 목표는 12개 비율 모두 각각 5% 미만이다. 연구 후보 선정, 엄격한 목표, 앱 적용과 독립 공장 검증은 별도다. 이번 계획에서는 TEST 추론·앱 자동 교체를 하지 않는다. 한 seed의 반복 사용한 source VAL 결과로 공장 안전이나 현장 오류율을 보장하지 않는다.

자료 준비 시간·실제 PNG 저장 바이트와 모델 학습·epoch 검증 시간·peak allocated CUDA·실제 update/AMP skip을 기록한다. 같게 맞춘 모델 입력 크기·epoch·draw는 같게 맞춘 자료 준비 비용이나 CPU IO 시간이라는 뜻이 아니다. GPU preflight의 disposable update는 완료 학습 epoch에 넣지 않는다.

## 재현 순서

1. `scripts/prepare_facility_native_roi.py`로 TRAIN pair와 집계를 준비한다.
2. 새 데이터·프로토콜·report 경계 테스트를 실제 실행하고 결과 파일을 기록한다.
3. `scripts/facility_native_roi_study.py --declare`로 자료와 코드 SHA를 학습 전에 고정한다.
4. `scripts/preflight_facility_native_roi.py`로 실제 TRAIN 8행의 라벨·지도·known·aux·row 재현, 각 GPU AMP update 1회, 저장·CPU 재로딩을 확인한다.
5. 소스를 Git commit으로 보존하고 before-training snapshot을 만든다. `scripts/run_facility_native_roi_pair.py`를 숨겨진 백그라운드에서 한 번 실행한다.
6. 양군 완료 뒤 기존 select `--grids 1`과 analyze `--aggregate-only`, native verifier와 reporter를 실행한다.
7. 결과와 비용을 기록한다. 결과에 맞춰 이 실험의 box·해상도·epoch·정답을 다시 고르지 않는다.

자료·가중치·상세 경로 ledger·개별 예측은 로컬 ignored 폴더에 둔다. GitHub에는 코드와 집계만 반영한다. raw Git source SHA 재현에는 자동 줄바꿈 변환을 끈 checkout이 필요하며 초기 모델·자료·로컬 실행 증거는 별도로 준비해야 한다.
