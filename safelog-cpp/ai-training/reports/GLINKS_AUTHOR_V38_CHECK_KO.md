# GLinks 원저자 설명·정적 데이터 로더 확인

v35의 전체 근접 연결 실패를 보존한 상태에서 추가 공개 텍스트를 확인했다. 사진 재다운로드·재분리·원저자 코드 실행·추가 사진 열람·모델 학습은 하지 않았다.

- [원저자 논문 v1](https://arxiv.org/html/2506.16890v1)은 부품당 여러 촬영 시점과 배경·고정 장치 및 부품 외형에 의한 지름길 학습 문제를 설명한다. 동일 물체의 촬영을 학습과 평가에 나누지 않아야 한다는 문제의식은 현재 보수적 제외 정책과 맞는다. 그러나 이번에 확인한 설명만으로 파일명의 UUID와 물리 부품 ID의 일대일 대응을 입증하지 못했다.
- [재현 저장소](https://github.com/MrShoenel/ecml-pkdd-25-repl/tree/d778d7fb58af482e19d1de4a969607e259e0fe59)의 두 notebook은 **source 셀 텍스트만** 확인했고 저장된 출력·그림을 열람하거나 코드를 실행하지 않았다.
- 그 저장소가 고정한 [cs-flow 하위 저장소의 utils.py](https://github.com/visual-inspection/cs-flow/blob/d4c13f684b950942dcea2af47eac32c99e97656d/utils.py)와 naive 설정을 정적으로 읽었다. 로더는 TRAIN/TEST 디렉터리와 클래스 경로를 사용하며, 읽은 구현에서 별도의 UUID→물리 부품 매핑 표를 확보하지 못했다. 이는 다른 배포 자산에도 그런 표가 없다는 증명은 아니다.

따라서 v35의 제외 그룹을 해제하거나 v36 RGB 차이에 새로운 허용 기준을 붙여 강제로 학습 분리를 만들지 않는다. 원래 TEST 568장과 제외 TRAIN 3424장, v36 개발 전용 두 UUID의 예약은 유지한다. pHash 근접은 물리 동일성 증명이 아니며 이번 조사는 그 불확실성을 해결하지 못했다.

확인한 공개 응답은 `data/industrial-glinks-author-v38`에 보존했다. 논문 HTML 로컬 SHA256 `f0802ec481dc9df83b90bdbf5ebc75e17e37bbab8cf2ca57c4beb8cb161e722a`, 고정 cs-flow tree `19deffc2aebb313be29ecfc241fa5827f97dfccf753ac73bc35c167913b0a432`, utils 텍스트 `87634b2ba57a61a7953701472f311926f9f64c75cb47e1333ffc1e9738db9567`, 설정 텍스트 `8d9550c1fdc9025d3e9912e95f31f0397245830011ac8c65e6fe42a434f1d649`다. 이 값들은 원저자 제공 체크섬이 아닌 로컬 수신 무결성 기록이다.

현재 GLinks로 학습한 새 사진·optimizer는 0이다. 원자료의 CC BY-NC-SA 4.0과 DOI `10.5281/zenodo.15693611`을 유지하며, 논문·코드의 조건과 자료 조건을 혼동하지 않는다.
