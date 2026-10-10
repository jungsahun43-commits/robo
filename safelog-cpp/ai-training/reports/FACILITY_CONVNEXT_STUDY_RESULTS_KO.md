# ConvNeXt 시설 모델 전체 학습 결과

기존 MobileNet 대신 ImageNet으로 초기화한 ConvNeXt-Tiny 특징 추출기 전체를 시설 TRAIN 사진으로 학습했다. 새 FPN 위치 지도와 사진7종·보조19종 head를 함께 학습한다. 이전 ConvNeXt 실험은 특징 추출기가 고정되어 있었으며 이번 후보는 해당 모델의 연속 학습이 아니다.

## 고정한 비교 조건

원본 full 사진14,248장과 파생 영역12,041개, 원래6회 추출 순서와 정답, 출처 가중치,640픽셀 입력, gamma1 사진 focal·위치 focal/Dice·19종 보조 손실을 유지했다. 교사 MobileNet과 알려진 다른5개 항목의 증류4/T2를 유지한다. AI 관찰을 정답으로 쓰지 않았고 사람 피드백0건이다.

새 학생 모델의 초기값·구조·정규화·stochastic depth가 함께 바뀌는 구성 비교다. 원래 MobileNet 가중치를 학생에게 이식하지 않았다. ImageNet 공식 encoder180개 tensor만 정확히 이식하고7종/19종/위치 head는 seed56으로 초기화했다. LayerNorm/GroupNorm을 학습하며 학생 BN 통계는 없다. 교사 BN 통계는 고정한다.

encoder와 pool normalization 학습률4e-5, 나머지 head1e-4, AdamW decay0.0002, cosine6회, batch8을 사전 고정했다. 메모리 절약을 위한 checkpointing은 stochastic-depth RNG를 보존한다. 기존 대조군6회는 재학습하지 않았다.

## 같은 공개 VAL 비교

| 모델 | 실제 epoch | 선택 epoch | 최대 오탐·미탐률 | 각 비율5% 미만 |
|---|---:|---:|---:|---|
| 추가 학습 전 원본 | 6 | 6 | 22.2798% | 미달 |
| 기존 저학습률 대조군 | 6 | 6 | 22.0207% | 미달 |
| ConvNeXt 전체 학습 후보 | 6 | 6 | 22.2798% | 미달 |

최대값은 균열·박락×DACL710/Dam424/CODEBRIM611×오탐·미탐의12개 비율 중 최대다. 앱 전체 또는 독립 산업현장의 오류율이 아니다.

| 작은 DACL 손상 | 원본 FN/양성 | 대조군 FN/양성 | 신규 FN/양성 |
|---|---:|---:|---:|
| 균열 | 29/93 | 30/93 | 28/93 |
| 박락 | 42/105 | 40/105 | 45/105 |

연구 후보 기준 **미달**, 항목 보존 기준 **통과**. 이전 기준의 최대 오류0.5pp 개선·개별 오류2pp 악화 제한·알려진 다른 항목 AP0.02 하락 제한·작은 손상 FN2건 감소를 유지한다.

실제 신규 학습6epoch, 시도10,686batch, 실제 업데이트10,678회. 학습·회차 검증58.40분, GPU 최대 할당2.00GiB.

학습 파라미터28,259,976개, state tensor199개. 기존3,244,151개 MobileNet보다 크므로 앱 연동에는 추가 비용 평가가 필요하다. 기존324개 state checkpoint와 호환되지 않으며 별도 strict 연구용 adapter를 제공한다.

테스트9개, 실제 TRAIN-only GPU 업데이트와 CPU/GPU 저장·재로딩, encoder 실제 변경·고정 교사 보존, 원본72,662개 파일의 SHA·크기·수정시각 보존을 확인했다.

같은 공개 VAL과 한 seed의 반복 탐색이다. sourceTEST 추론과 앱 기본 모델 교체는 수행하지 않았다. 교량·댐·콘크리트 표면 공개 자료의 결과이며 독립 산업체 현장 성능을 확인한 것은 아니다.

## 방법 출처

[Torchvision 공식 모델](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.convnext_tiny.html), [CVPR2022 원 논문](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html). 공식 ImageNet 결과를 이 앱의 안전 판단 성능으로 대체하지 않는다.
