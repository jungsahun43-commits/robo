# 새 RC 자료·양성 손상 위치 학습 결과

새 자료와 양성 위치 손실을 묶어 한 후보6epoch를 실제 학습했다. 과거 대조군6epoch를 다시 학습하거나 누적 학습량에 중복 합산하지 않았다.
원래324개 state·7종 사진 출력·19종 보조 출력·640 입력·80 지도·BN 통계 동결·기존 항목 교사 증류를 유지한다. source-TEST 추론과 앱 기본 모델 교체는 수행하지 않았다.

## 새 자료와 정답 범위

[RC2119](https://data.mendeley.com/datasets/2vkm6k4cfg/1) 원본·JSON·마스크2,119쌍을 취득하고 배포자 SHA256을 대조했다. 전체 사진 및 SIFT 국소 특징 중복 후보를 제외한 뒤, 박락100·균열100개의 그룹 대표 사진을 골랐다. 사진 합계200장에 균열 양성117·박락 양성100개 항목·사진 사례가 있다.
배포자의5종 마스크를 파일 안의 실제 numeric ID와 JSON으로 확인했다. 부식·파쇄를 녹 흔적·박락으로 임의 변경하지 않았다. 선언된 EXIF 방향을 적용하고 좌표 부적합 자료를 제외했다.
새 사진에서 표시가 없는 손상·배경은 미확인으로 유지한다. 새 음성 사진·배경 음성 픽셀·19태그 정답은0개다. 저자가 명시한 균열·박락 foreground 셀에만 추가 양성 focal 손실을 적용했다. 마스크0을 정상으로 바꾸거나 미확인 사진을 음성으로 채우지 않았다.
매 epoch14,248번 표본 추출 중704번(4.94%)을 새 자료로 바꿨다. 나머지 위치의 원래 추출 순서·기존 정답·손실 가중치는 보존한다. 데이터와 추가 양성 위치 손실을 동시에 바꾼 비교이며, 데이터만의 효과나 완벽한 장면 독립성을 입증하지 않는다.

## 실제 source-VAL 결과

| 모델 | 실제 epoch | 선택 epoch | 최대 FNR/FPR | 엄격한5% |
|---|---:|---:|---:|---|
| 추가 학습 전 원본 | 6 | 6 | 22.28% | 미달 |
| 기존 저학습률 대조군 | 6 | 6 | 22.02% | 미달 |
| 새 자료·양성 위치 보강 | 6 | 6 | 22.22% | 미달 |

최대값은 균열·박락×DACL710/Dam424/CODEBRIM611×FNR/FPR의12개 비율 중 최대다. 앱 전체 오답 사진 비율이나 공장 현장 정확도를 뜻하지 않는다.

| 작은 DACL 손상 | 원본 FN/양성 | 기존 대조 FN/양성 | 신규 FN/양성 |
|---|---:|---:|---:|
| 균열 | 29/93 | 30/93 | 30/93 |
| 박락 | 42/105 | 40/105 | 40/105 |

작은 손상은 주석 면적1% 미만인 항목·사진 사례198개(균열93·박락105)다. 같은 사진이 두 항목에 포함될 수 있다.
기존 연구 후보 기준은 **미달**다. 원본·대조군 대비 최대 오류 최소0.5pp 개선, 개별 비율 악화2pp 이하, 다른 알려진 항목 AP 하락0.02 이하, 작은 손상 FN 합계 최소2건 감소를 그대로 확인했다.

## 실제 실행과 해석

실제 신규 학습6epoch, 시도10,686batch, optimizer 업데이트10,679회. TRAIN 전용 일회성 preflight 업데이트는 학습 epoch에 세지 않았다.
학습·epoch 검증 시간16.43분, GPU allocated peak2.079GiB. 집중 테스트26개, 동결 소스123개와 원래 입력59,026개의 SHA·크기·mtime 보존을 실제 검사했다.
새 자료는 학습에만 쓰고 새 시험 자료로 쓰지 않았다. 부모·현장 IDs 및 기존 공개 출처의 상세 연결은 확보하지 못했다. 중복 검사는 발견한 후보를 제외하는 보수적 검사이며 장면 독립성의 증명이 아니다. 같은 공개 VAL에서 반복한 epoch·임계값 선택의 탐색 결과이므로 독립 현장 성능이나5% 보장으로 해석하지 않는다.

## 원출처 인용

Wang, J., & Ueda, T. (2025). Automatic damage detection and segmentation using deep learning algorithms in reinforced concrete structure inspections. Structural Concrete26(5),5511–5534.
Wang, J., Wang, Z., Wang, Y., & Li, Z. (2025). Automated multi-type damage detection framework in reinforced concrete structures via data augmentation and deep segmentation networks. Journal of Civil Structural Health Monitoring15(8),3861–3884.
RC2119 V1, DOI10.17632/2vkm6k4cfg.1, CC BY4.0. 이번 연구에서는 방향 보정·크기 축소·양성 foreground 셀 추출을 적용했다. 원저자의 승인이나 현장 안전 인증을 뜻하지 않는다.
