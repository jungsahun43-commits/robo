# v57 다음 자료 단계: 고분자 애자 하부 금속 연결고리

v53 자료 준비·v54 실제 학습과 제외·v55 추가 하위 ZIP 다운로드/CRC·v56 바이트와 프레임 감사는 완료했다. 모두 다시 돌리지 않는다. v54는 정상FP12.61/전체FN20.93%, 새 둥지 유형FN21.74%로 제외됐다. 조건/임계값을 조금 바꿔 구제하지 않고 승인된 v51 세 head overlay를 유지한다.

## 후보와 현재 한계

새 unsupervised 원ZIP의 `polymer-insulator-lower-shackle` 정상 원TRAIN1306장 중 바이트/프레임 기준TEST·기존평가 연결1163장을 제외한143장/135proxy그룹만 후보다. 원TEST 정상327/부식315 전부를 보존한다. 정상만 있는TRAIN을 부식/위치 정답 지도학습으로 바꾸지 않는다. 현재 새 원사진 내용 열람·RGB/근접 감사·분리·특징·모델점수·optimizer는0이다.

## 코드/자료 조건을 먼저 고정할 항목

1. v57 코드/JSON을 새 픽셀 계산 전에 고정·commit/push한다. v56 all-rows.json SHA `a98aa6eb783473f7f6af3f2a9f88d2afd14269448915542180516a9787b83889`, v55 ZIP SHA `baab1d227b408ec6bbe5831cbcae5ea049a9b5132a74bf36f4e3d6051b0cb7e3`, 기존 v44 grouped-all-source/decoded-fingerprints/retained-TEST-fingerprints 및 관련 manifest SHA를 묶는다. 다운로드나 이전 전역 감사를 재실행하지 않는다.
2. 전이적 근접 연결을 놓치지 않도록 새27,620행 전체를 대상으로 하되, encoded rawSHA가 v44 사진과 정확히 같은 행은 기존 검증된 RGB/크기/8방향pHash를 재사용한다. 나머지만 원ZIP→RAM JPEG/PNG 해독한다. 원본 최대32MiB/40MP·EXIF None/1·4workers·실패시 보존/중단을 선언한다. 재사용/새 해독/추출·보기·신경망 특징 수량을 구분하고 원본을 확장하지 않는다.
3. 같은 크기+RGB exact 또는 8방향pHash Hamming<=2의 새/기존 source 연결, 기존 보존8,995TEST 연결, v56의 원래 프레임·바이트·기존그룹 연결을 함께 합친다. 기존 oldsource/oldTEST 지문은 고정 SHA로 재사용한다. 유사성은 동일 부품의 증명이 아니며 비교/제외를 점수에 맞춰 완화하지 않는다.
4. 새 publisherTEST6,481장 전체와 원래 supervised publisherVAL6,417장/기존8,995TEST에 연결되는 새TRAIN은 제외한다. 원래TEST 목록과 지문은 보존하고 TEST를 FIT/VAL/CAL로 옮기지 않는다. 이미 승인된 과거 head의 새TEST 연결 노출은 별도 한계로 남기며 기존 모델을 소급 수정하지 않는다.
5. 최종 targetTRAIN 후보는 원래TRAIN/conditiongood이며 v56에서 rawnovel_unassigned_TRAIN이었던143장으로 한정한다. 픽셀 감사로 추가 제외될 수 있다. 이미지 종류나 TEST 결과를 보고 다른 TRAIN을 보충하지 않는다.
6. v45/v48/v51/실패v54의 FIT/내부VAL/CAL 그룹 역할은 새 연결 그룹에서도 보존한다. 한 역할이면 고정하고 서로 충돌하는 역할 그룹은 새 자료에서 제외한다. 아직 역할 없는 나머지 그룹만 한 번 배정한다: 가장 작은 안정적인 멤버ID에 SHA256(`insplad57:`+ID) 순서, 첫ceil20% 내부VAL·다음ceil20% CAL·나머지FIT. 메타데이터135그룹에서 정상 검증/보정 표본을 확보하려는 사전 규칙이며 모델 점수 기반 선택이 아니다.
7. 최소FIT 정상30/내부VAL정상20/CAL정상20, 각 개발분리proxy그룹>=3, 원TEST 정상/부식 각각20장 이상·각proxy그룹>=3을 학습 전에 고정한다. 미달이면 최소조건·분리·후보집합을 바꾸지 않고 특징/학습0에서 중단한다.
8. 통과한 target 원본만 unchanged bytes·숫자 파일명·최대128MiB로 내보내며1GiB 여유를 유지한다. 원본/실패/기존 ZIP/캐시를 삭제하지 않는다. 내용 열람은 최종FIT 정상 첫3장만 고정하고 원GT를 수정하거나 VAL/CAL/TEST 내용을 골라보지 않는다.

## 이후 모델은 아직 미확정

자료가 통과하고 FIT QA를 마친 다음, 한 공통 encoder를 사용하는 한 가지 새 학습 가설의 정확 구조·입력·실제 정답 범위·초기값·학습 파라미터·회차/lr/loss·VAL 선택·CAL·채택 조건을 별도 코드/JSON으로 고정/commit/push한 뒤 실제 특징/gradient를 실행한다. 이 문서는 학습 프로토콜이 아니다. 기존 v40/v41의 실패 모델·같은 MPDD2 TEST 임계값 실험을 반복하지 않는다.

원래 v16 및 승인된 v45/v48/v51 모든 tensor/buffer/메타데이터·기준·원래 위치 지도를 정확 보존한다. 작은 parent-bound delta의 복원 exact와 모든 보호 상태를 검사한다. 정상 복원을 학습한다면 실제 optimizer를 기록하되 부식 사진 지도학습으로 표시하지 않는다. 현재TEST 부식 라벨·마스크나 자동 추정을 학습 정답으로 가져오지 않는다. 새분기를 위한 가짜 source/category/date embedding이나 별도 encoder를 추가하지 않는다.

채택 조건은 모델 실행 전에 선택epoch>0·평가 각라벨>=20/그룹>=3·사진FPR/FNR 모두<=20%를 최소로 고정하고, 정상CAL85% 순서통계량을 저장한 뒤 고정TEST 한 번만 평가한다. 미검증 위치 지도를 기본으로 승격하지 않는다. 후보 통과 후에만 새 ZIP의 CPU/새GPU/실제C++/기존전체alias/위치PNG 보존을 검사한다. 이전 검사 수를 새 검사로 세지 않는다.

원자료는 InsPLAD MendeleyV1 CC BY-NC3.0 비상업 학술/인용·원저자·출처·버전·변경 표시다. 공개 반복 개발·프레임 proxy·옛 평가 노출을 독립 현장 성능으로 주장하지 않는다. 원래 브래킷/튜브 미탐이 줄었다는 뜻도 아니다. 절전방지 helper는 재시작/연장하지 않는다.
