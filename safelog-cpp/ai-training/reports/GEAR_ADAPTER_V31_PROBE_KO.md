# v31 준비 조사: 연구 저장 방식과 기존 분석층 재사용

이 단계의 실제 새 학습·optimizer·새 사진 판별부 점수·VAL/CAL/TEST 점수는0이다. v29/v30의 실제학습은 완료·실패로 보존했다. 현재 앱용 v16과 전달ZIP도 유지한다.

## 저장 공간 대응

저장 여유가 약4GB이고 연구 체크포인트마다 원래 모델 가중치가 반복 포함된다. 기존 원본·가중치·실패 파일은 수정하거나 삭제하지 않는다. 향후 연구본을 부모파일의 SHA256에 연결하고 새 `gearbox_`/`gear_spatial_` tensor와 새 metadata만 보관하는 delta 형식을 검증한다.

기존 실패 v30을 작은 delta로 **추가 표현**하고, 직렬화된 delta를 다시 읽어 accepted v16 부모와 RAM에서 복원한다. 모든 state key·tensor·buffer·metadata·결합stateSHA가 기존 full v30과 같아야 한다. 부모가 변경되거나 새항목이 기존항목을 덮거나 overlay가 변조되면 거절한다. 원래 full v30도 그대로 보존한다.

이 형식은 부모파일이 필요하므로 독립 앱 전달 파일이 아니다. 향후 채택·패키지 검증 시 단일 full 모델로 합쳐 내보내는 절차와 loader 계약을 별도 확정한다. 저장 형식 변환을 새 학습·새 모델 성능으로 세지 않는다.

## FIT만의 특징 보정 모듈 조사

같은 공통 encoder stage3의 fine24와 기존 후반 모듈을 재사용한다.24→4/GELU/4→24의 작은 residual prototype을 사용하고 마지막층은0으로 초기화한다. 기존 `forward`·가중치·return_layers·hook을 바꾸거나 encoder를 복제하지 않는다. 캐시에서 출발하면 stage3 이전 부분을 다시 실행하지 않고 기존 module 객체들의 후반 부분을 순서대로 실행한다.

이전 v29 조사와 동일한 v28 FIT32개만 사용한다. 새 VAL/CAL/TEST 이미지나 사진 판별 점수는 읽지 않는다.

1. 원래 RGB256 forward의 FP32 fine을0초기화 adapter에 넣고 후반층으로 이어갈 때, 원래 fullforward low/high와 bitwise 같아야 한다.
2. 기존32개 FP16 fine 캐시가 새 fine.half와 bitwise 같은지 검사한다.
3. 캐시 FP16 fine→FP32→후반층 경계는 원래 fullforward와 다른 정밀도 경계다. pooled descriptor3,072차원의 `atol=.02, rtol=.01`을 조사 전에 고정해 비교하고 최대/평균 차이를 남긴다. 불일치하면 기록·중단하며 허용오차를 바꾸지 않는다.
4. 원래 모델 전체 stateSHA와 기존forward 출력이 조사 전후에 같아야 한다.

단위검사에서 작은 모듈로 gradient가 흐르고 frozen 후반층 parameter의 gradient는 생기지 않는 것을 synthetic 입력으로 확인한다. 실제 FIT 조사에는 gradient·optimizer·class점수를 사용하지 않는다.

## 다음 학습 전에 남은 일

성공해도 정확도 개선의 증거는 아니다. 실제 학습의 adapter 구조·seed·초기값·어느 항목을 학습할지·정규화·precision·캐시/원본RGB 여부·loss·회차·VAL/CAL·채택 기준과 delta loader/복원 검증을 코드·JSON으로 고정하고 commit한 뒤 실행해야 한다. 현재 v28 데이터 분리는 계속 보존한다. 클래스 라벨을 위치 mask 정답으로 바꾸지 않는다.

새 학습에서는 기존 모든 accepted v16 가중치·기본 위치·다른 유형을 유지하고 새로운 기어 경로에서만 보정 모듈을 쓸 수 있는지 구체화한다. 기어128표면 패치의 마모 허용을 전체 기어박스 작동 건강·내부 고장·임의 현장 전체사진·검증된 위치 성능으로 확대하지 않는다. 같은 공개TEST개발 이력·적은 TEST묶음4개와 독립 현장 미검증 한계도 남긴다.

Re-GBC version1 CC BY4.0, 원배포본과 DOI10.1109/ETFA65518.2025.11205777을 인용하고 다른자료 이용조건도 유지한다.
