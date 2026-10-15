# v29 다음 연구: 기어 표면의 공간 특징

## 현재 결론

v26/v27은 각각 실제1,920 optimizer회를 수행했지만 VAL이 초기보다 좋아지지 않아 epoch0 초기 새 판별부를 선택한 실패 연구다. v28 기어 표면만의 별도 작업은 실제1,920회 후 VAL .693147→.587897, epoch1/64회 가중치를 선택했다. 기어 TEST2,179패치 오탐17.93%·미탐37.84%로 두 지표20% 조건에 미달해 앱에는 반영하지 않았다. 전체 부품 TEST3,109개·전체 부품 실패·기존 브래킷/튜브 상태를 보존한다.

## 이번 단계는 FIT 조사만

`industrial-gear-spatial-v29-probe-policy.json`과 코드를 새 특징 전에 고정·commit한다. v28 FIT에서 라벨별16개씩, 정렬된 묶음 순서로 각 묶음의 작은idx를 반복 선택하는32개를 사용한다. VAL/CAL/TEST 이미지나 새 모델 점수는 읽지 않는다. 원래 accepted v16 모델에서 fine24@64×64를 읽고, 기존 low/high의 일반 forward 결과와 bitwise 동일함과 모델 전체 tensor/buffer의 전후 SHA 동일함을 검사한다.

새 fine의 평균/모표준편차/최댓값72차원은 원래 고정 pooled descriptor의 해당 항목과 비교한다. 사전에 `atol=0.002, rtol=0.002`를 선언했다. 불일치하면 원인과 출력만 보존하고 학습 전에 해결하며 허용오차를 결과에 맞춰 바꾸지 않는다.

이 조사에서 optimizer·새 head점수·VAL/CAL/TEST 점수는0이다. 공간 분산이나 좌표 보존은 정확도·향후 개선의 증거가 아니다. 자료의 클래스 라벨은 실제 저자 패치 마모 허용 라벨이며 pixel mask가 아니다.

## 다음 학습 가설 — 아직 미확정

1. **v28 데이터 분리를 그대로 유지**한다. FIT5,466/VAL1,440/CAL1,116/기어 TEST2,179패치, 연결 묶음8/3/3/4개다. 이미 본 FIT예시 묶음1·19도 그대로다.
2. 같은 encoder의 fine24 공간 정보를 작은 convolution 판별부로 학습하는 후보 하나를 구체화한다. 예로24→32의3×3 convolution/GN4/GELU→1의1×1 convolution과 상위10% logit 평균 MIL을 검토할 수 있다. 이 문장은 확정 학습 구조가 아니다. 여러 pooling비율·구조·임계값 조합을 TEST에 맞춰 돌리지 않는다.
3. 실제 OK/nOK 패치 라벨로 사진 조각 BCE만 학습한다. 모든 정상 픽셀·결함 위치의 정답을 만든 것으로 설명하지 않는다. 공간 출력이 생겨도 검증된 결함 위치나 기존 기본 위치 지도로 승격하지 않는다.
4. FIT만의 채널 평균/표준편차 등 정규화 공식을 먼저 고정한다. fine FP16 캐시는 대략2GB 이상을 예상하며 정확한 사진 수·dtype·크기·출력 수·임시 저장 여유를 계산한다. 모든 입력ID·이미지SHA·좌표·기존 encoder·protocolSHA를 묶어 검증한다. CAL/TEST 전체 새 특징은 각각 허용 단계에서 원본RGB로 추출하며 TEST 점수는 CAL 고정 후에만 계산한다.
5. 초기값·학습 대상·회차·lr·loss·샘플링·VAL선택·CAL·채택 조건을 코드/JSON으로 고정하고 commit/push한 뒤 전체 특징·gradient·optimizer를 실행한다. 기존 모든 부모 tensor/buffer/metadata·위치·점수·다른 유형을 유지한다. 새 encoder를 만들지 않는다.
6. 기어 패치의 두 TEST라벨 각각20개 이상·연결묶음3개 이상·선택epoch>0·오탐/미탐 모두20% 이하를 유지한다. 통과 후보만 새로운 패키지 CPU/GPU·실제 C++·기존 alias·기본 위치 PNG 보존 검사 후 연구 전달물에 반영한다. 이전 패키지 검사 수를 새 검사로 세지 않는다.

## 남아 있는 범위와 이용조건

기어만의 결과를 스플라인·동기화 부품이나 전체 기어박스 성능으로 확대하지 않는다. 원래 nonTEST spline OK3개와 FIT의 collar 누락, ring/shaft TEST 부재는 전체 지원의 자료 한계다. 원래 S3fold0 TEST·근접 묶음·제외 TRAIN1,317개·미주석103개를 보존한다.

이전 공개 TEST 개발 이력과 부족한 독립 부품·촬영 그룹을 남긴다. 새 현장 사진에서 독립 검증한 결과가 아니다. 학습 후 출력은 표면의 마모 허용 판별 보조이며 내부 고장·전체 장비 작동 건강의 정답이 아니다.

[원배포본](https://www.kaggle.com/datasets/jhnnsbr/gearbox-components-remanufacturing-inspection) version1 CC BY4.0과 [저자 논문](https://arxiv.org/abs/2511.15440), DOI10.1109/ETFA65518.2025.11205777을 인용하고 기존 다른 자료의 이용조건도 그대로 둔다.
