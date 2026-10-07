# 데이터 보강 검토와 새로운 특징 학습 계획

작성일: 2026-10-07. 같은 자료에서 작은 학습률 변경만 반복하는 대신, 남은 오류와 추가 자료의 정답 범위를 확인하고 다른 시각 특징을 사용하는 후보를 실제 비교한다.

## 현재 병목

직전 후보의 [고정 source-VAL 결과](FACILITY_HEAD_LR_STUDY_RESULTS_KO.md)에서 DACL 박락 FP는85/386=22.02%, FN은69/324=21.30%다. DACL 균열 FP는107/499=21.44%, FN은45/211=21.33%다. 같은 사진의 여러 항목 판정을 별도 사례로 센 전체 오류450건 중 DACL 오류는306건이다. 고유 오답 사진 수나 앱 전체 정확도를 뜻하지 않는다.

DACL에서 주석 면적1% 미만 양성 사례는198건이며 누락70건이다. 큰 영역은337건 중44건 누락이다. 작은 영역 기준은 사진의 주석 비율이며 물리적 손상 크기가 아니다. 해상도 확대·원본 ROI·관련 음성 태그 추출·기존 항목 증류·BN·학습률 비교에서 큰 오류 감소는 확인하지 못했다. 이 관찰만으로 원인을 모델 용량이나 정답 품질로 확정할 수는 없다.

## 추가 데이터 후보

| 자료 | 공식 설명과 정답 | 이번 판단 |
|---|---|---|
| [CCIC V2](https://data.mendeley.com/datasets/5y9wdsg2zt/2) | METU 건물458개 원본에서227×227 패치40,000개. 균열 양성·음성 각각20,000개, CC BY4.0 | 명시적 균열 음성은 보강 후보다. 박락·다른6항목·픽셀 정답은 미확인이다. 기존 파일 접근 확인 기록은 있지만 이번 작업에서는 취득·학습하지 않았다. |
| [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) | 230개 콘크리트 원본에서256×256 패치56,000개 이상. 균열 유무 및 그림자·거친 표면·모서리·구멍·이물 포함, CC BY4.0 | 혼동하는 균열 음성에 적합한 후보다. 조사에서 공식 다운로드403 응답을 재확인해 이번에 취득하지 못했다. |
| [RC segmentation2119](https://data.mendeley.com/datasets/2vkm6k4cfg/1) | 공개 자료에서 수집한2,119개 철근콘크리트 사진. 균열·박락·철근 노출·부식·파쇄의 색상 픽셀 주석, CC BY4.0. 박락282는 사진 수가 아닌 instance 수 | 두 주요 항목을 함께 보강할 후보다. 기존 DACL/CODEBRIM 등과 중복, 원출처 사용 조건, 실제 파일 및 주석 완전성을 확인한 뒤 사용해야 한다. 이번에는 학습에 섞지 않았다. |

CCIC와 SDNET의 균열 음성은 모든 손상이나 박락이 없다는 정답이 아니다. 이번 최대 오류가 박락에서 발생하므로 균열 사진만 늘려 전체 목표를 해결한다고 가정하지 않는다. 새로운 자료의 미표기 결함·픽셀을 정상으로 만들지 않고, 공식 부모·장면 정보와 기존 분할의 겹침을 확인한다.

## 이번에 실제 비교할 방법

[ConvNeXt-Tiny 공식 ImageNet 가중치](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html)에서 특징을 추출하는 분기를 추가한다. 원래 MobileNet·7종 사진 출력·19종 보조 출력·80×80 위치 지도와 학습된 원본324개 state tensor를 유지한다.

- 고정된 ConvNeXt 저수준 특징192채널·80×80에서 새로운7종 지도 보정값을 학습한다.
- 고정된 고수준 특징768채널과 공식 pooling 정규화에서7종 사진·19종 보조 점수의 보정값을 학습한다.
- 추가한 세 연결층은0으로 초기화한다. 학습 전에 원래 사진 점수·지도·보조 점수가 실제 TRAIN 입력에서 동일한지 확인한다.
- ConvNeXt 특징 추출기는 eval·gradient 없음으로 유지하고 가중치 전체 해시를 학습 전후·매 epoch 검사한다. 원래 모델과 새 연결층을 학습한다.
- 기존 전체 사진640 변환과 RGB 정규화를 유지한다. 공식 분류용224 중앙 crop은 적용하지 않는다.

외부 사전학습 특징을 추가하는 모델 구성 비교다. 새로운 시설 사진이나 전문가 확정 정답을 추가한 실험, ConvNeXt 전체를 재학습한 실험, 모델 용량만의 인과 검증으로 설명하지 않는다.

## 조건과 검증

| 설정 | 재사용 대조군 | 신규 후보 |
|---|---|---|
| 이름 | facility-presence-target-head-lr-low | facility-presence-target-semantic-features |
| 초기 원래 모델 | 원본0773 | 같은 원본0773 + 공식 ImageNet 특징 +0 초기 연결층 |
| 입력·배치·표본 | 640·8·14,248/epoch | 동일 |
| seed·신규 학습 | 기존 seed56·6epoch 완료 | seed56·신규6epoch |
| 학습률·손실 | backbone4e-5·head1e-4·Cosine6·최저5e-6, 기존 손실·교사4/T2 | 동일, 고정 특징 추출기에는 optimizer 업데이트 없음 |
| 감독·BN | 원래 정답·unknown mask·19태그·80지도·BN47개 원본 통계 | 동일 |

대조군을 다시 학습하거나 기존6epoch를 누적 학습량에 중복 합산하지 않는다. 출처·full/crop·7항목 정답·표본 순서·교사 추론·학습 epoch는 같지만 추가 encoder 때문에 매개변수·연산량·시간·추론 비용은 증가한다. 실제 GPU peak, 시간, 기존·새 모델의 TRAIN 단일 사진 추론 비용을 함께 기록한다.

코드·조건을 고정하고 집중 테스트 및 실제 TRAIN8개 입력의 출력 동일성·GPU 업데이트·CPU 재로드를 확인한다. 이어서 후보6epoch와 기존 전체 source-VAL grid1, 작은 손상, 알려진 나머지 항목 AP를 평가한다. 원래 source-TEST는 사용하지 않는다.

이전 AP 유지·철근 회복·연구 후보 기준과 균열·박락의 출처별 FNR/FPR 각각5% 미만 기준은 그대로 적용한다. 반복 공개 VAL 탐색 및 한 seed의 결과이며 독립 공장 현장 성능으로 해석하지 않는다. 기존 앱 기본 프로필의 교체는 별도 검증 조건을 충족한 뒤 판단한다.

## 로컬 재현 순서

저장소에는 코드·공식 가중치 검증 메타데이터·고정 프로토콜이 포함된다. 원본 데이터와 완료된 원본·대조군 가중치 및 로컬 증거는 별도로 필요하다. 새 실험 실행 폴더가 없는 workspace에서 `safelog-cpp/ai-training`을 작업 폴더로 사용한다. 기존 고정 소스는 LF 바이트를 유지한다.

```powershell
.venv/Scripts/python.exe scripts/restore_facility_semantic_weights.py
.venv/Scripts/python.exe scripts/test_facility_semantic.py
.venv/Scripts/python.exe scripts/preflight_facility_semantic.py
.venv/Scripts/python.exe scripts/verify_facility_semantic.py --snapshot-before-training
.venv/Scripts/python.exe scripts/run_facility_semantic.py
```

복원 도구는 Git에 포함된 검증 메타데이터를 보존하면서 제외된 공식 가중치 파일을 취득·검증한다. 이 도구는 동결된112개 학습 실행 소스 외부의 유지보수 도구다. 마지막 명령은 학습·source-VAL grid1·작은 손상 분석·완료 검증·보고서·그래프를 순서대로 실행한다. 각 단계 상태와 로그는 `runs/facility-semantic-background-candidate`에 저장한다. 완료된 현재 실행 파일은 그대로 보관한다.
