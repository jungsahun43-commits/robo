# 다음 자료 단계: vari-grip의 부식과 새 둥지 주석

v51 현수 연결구 실제 학습과 v52 세 기능 overlay 검증은 완료했다. 다시 학습하거나 새 TEST 수치에 맞춰 기준을 바꾸지 않는다.

다음 후보는 이미 받은 InsPLAD 원 ZIP과 v44 grouped-all-source.json의 vari-grip이다. 메타데이터의 eligible 원TRAIN은 good165/bird-nest22/rust19, 원래 publisherVAL은 good238/bird-nest23/rust20다. 원래 파일 조건명을 그대로 사용한다. 숫자ID1은 good이며 결함으로 바꾸지 않는다. 새 둥지를 부식이나 금속 파손과 같은 정답으로 표시하지 않는다.

아직 이 자산의 새 FIT/VAL/CAL 분리·원본 내보내기·내용 QA·정식 학습 프로토콜·특징·새 모델 점수·gradient·optimizer는0이다. 이번 수량을 확보한 새 학습 사진/새 성능으로 쓰지 않는다.

1. v53 자료 코드/정책을 먼저 고정하고 commit/push한다. 모든 원 publisherVAL을 평가 예약하고 글로벌 TEST연결TRAIN 및 filename변형 제외와 원본·그룹 SHA를 재사용한다. 다운로드·전역 중복 재계산·기존 분리 변경은 하지 않는다.
2. v45/v48/v51 개발 manifest의 그룹 역할 하나가 있으면 그 역할을 보존하고 서로 충돌하는 역할 그룹은 제외한다. 나머지만 insplad_split.allocate의 기존 라벨집합층 SHA15/15/70 규칙으로 한 번 나눈다. 부식과 새 둥지의 원 condition을 별도 보존한다. 분리 선택에 모델 점수나 내용 열람을 쓰지 않는다.
3. 최소조건을 사전 고정한다: FIT 정상>=30 및 bird-nest>=10/rust>=10, 내부VAL 정상>=20 및 각 결함종류>=2, CAL 정상>=20, 원래 평가 각 condition>=20; 필수 라벨별 연결 그룹 FIT/TEST>=3와 내부VAL 각 결함종류>=2. 전체 부족을 다른 결함종류 합계로 가리지 않는다. 미달이면 변경/이동 없이 특징·학습 전에 중단한다.
4. 원본 ZIP을 전체 확장하지 않고 통과한 구성원만 unchanged bytes로 최대128MiB, >=1GiB 여유를 유지하며 내보낸다. source/globalgroup/rawSHA/ID/order/원 condition/phase/exclusion을 연결한 최종 manifest를 남긴다.
5. 분리 후 원 condition별 첫3FIT만 고정해 AI 시각 QA한다. 원GT를 수정하거나 CAL/평가 내용을 골라보지 않는다. 물리 동일성·날짜/배경·새 둥지 외형 혼동 및 라벨 범위를 기록한다.
6. 자료가 준비된 뒤 한 encoder의 새 분류부 한 후보만 정식 설계한다. good/부식/새 둥지를 구분하는 실제 사진 정답 범위, 기존 세 head와 메타데이터 exact 보존, 작은 delta, 회차/loss/초기값/VAL/CAL/수용조건을 새 코드/JSON에 먼저 고정·commit/push한 다음 실제 특징·gradient를 실행한다. 이 문서는 학습 프로토콜이 아니다.

현재 승인 연구 부모는 v16과 v51 세 head 결합 delta다. v45/v48도 정확 보존해야 한다. 새 위치 GT나 전체설비 작동 정답은 없다. 새 유형 준비가 원래 검정 브래킷·튜브 미탐 감소를 뜻하지 않는다. 사용조건은 InsPLAD CC BY-NC3.0 비상업 학술/인용·버전·변경 표시다. 원본·실패·로그·캐시·모든 ZIP을 보존하고 절전방지 helper를 재시작하지 않는다.
