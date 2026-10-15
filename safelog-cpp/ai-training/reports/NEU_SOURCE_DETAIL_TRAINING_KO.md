# v23 실제 주석 사진 안의 결함·주석상 음성 영역 구분

2026-10-15. v22는 사전 자료 수량 조건에서 중단됐고 features/모델 점수/optimizer0이다. 그 조건을 낮추거나 실패한 준비를 재실행하지 않는다. 여기서는 **source-only라는 별도 학습 설계**를 코드·JSON에 먼저 고정하고 commit한 뒤 새 특징과 실제 학습을 진행한다. 원래 target 오탐·미탐 채택 기준은 그대로 유지한다.

## 실제 자료와 새 가설

FIT는 SDI20/SPDI15의 실제 배포 mask가 있는 원본35장이고, VAL은 예약된 미열람 Oil Small_defect4/Wipe_crack6의10장이다. 흰 표시가 있었던 Oil Scratch6장은 계속 제외한다. v22에서 고정한9개 native 상세 영역·좌표·mask·작은 양성 제외·근접 원본 그룹 제외를 그대로 상속한다. 새 합성·좌표 생성이나 TEST 점수 기반 분리는 없다.

v22에서 실제 주석 source는 FIT35/VAL10장이 그대로 남았다. 그 source 부분만 쓰면 FIT312crop(양성130/주석상 음성182), VAL88crop(양성51/주석상 음성37)이다. 이 수는 최종 manifest 감사로 다시 확인한다. 새 실사진 수는35/10이며 crop·추출 반복을 새 사진으로 세지 않는다.

같은 원본 사진에서 양성과 주석상 음성을 구분하게 한다. target의 녹색 배경과 새 source의 금속 질감 차이 자체를 정상/결함 구분 단서로 삼는 shortcut을 줄이려는 가설이다. MPDD 정상 사진을 새 분기의 gradient 학습과 VAL 선택에서 뺀다. 기존 모델 분기와 target 정상 CAL30/23·TEST79/101은 유지한다. 주석상 음성 crop이 부품 전체의 정상 기계·전기 작동을 의미하지 않는다.

## 고정 조건

- 자료 요건: FIT 양성 원본20장 이상·SDI/SPDI 각각5장 이상·두 source 각각 주석상 음성 crop이 있어야 한다. VAL 양성 원본8장 이상·주석상 음성 원본8장 이상이며 FIT/VAL 원본은 분리한다. v22의 현재 TEST5746장 및 전체/crop8326개 대조와 연결 그룹 제외를 SHA로 확인해 상속한다. 실패한v22의 target 정상30장 조건을 이 설계에 포함하거나 완화하지 않는다.
- 원본의 절반 크기 영역을 각 축 시작/중간/끝에 둔9개 crop이다. 모든 CAL/TEST에 동일하게 적용하며 GT 위치로 crop을 고르지 않는다. 각 crop은 종횡비 유지512RGB bicubic·mask nearest·회색127 letterbox다. mask양성32픽셀 이상은 양성,0은 주석상 음성,1~31은 제외하며 정상으로 바꾸지 않는다.
- 같은 encoder의 fine24/low40/high960을 사용하는 v21의 VAL 가중치를 새 detail_surface_head로 복사한다. fine projection/fusion만 학습하고 모든 부모 가중치·buffer·메타데이터·high projection은 고정한다. 잔차 채널은 source/target/추론 모두0이고 category/source embedding은 없다. 기존 함수 서명과 기본 위치 지도는 유지한다.
- 예정30epoch×64step=1920 optimizer, seed2323. 배치8은 SDI양성2/SPDI양성2/SDI주석상 음성2/SPDI주석상 음성2다. 원본 사진마다 균등하게 추출한 뒤 해당 라벨 crop 중 균등하게 복원 추출한다. cached 특징/mask/valid에 함께 H/V반전을 적용한다.
- loss는 사진별 양성·음성 평균 BCE(.5/.5,음성crop은음성평균만)+softDice+0.5×top32valid사진BCE다. AdamW lr1e-4/wd1e-4, cosine최소1e-6,clip5,FP32/TF32off를 쓴다.
- VAL은 같은 원본 사진의 crop loss를 먼저 평균하고 양성/주석상 음성 집단마다 사진 평균을 낸 뒤 두 집단을 동등하게 평균한다. epoch0 포함 가장 이른 최솟값을 선택한다. 새 source 자체의 전체 사진 FPR이나 독립 현장 검증으로 설명하지 않는다.
- target 사진 점수는9개 crop별 top32valid sigmoid평균의 최댓값이다. 원본 사진을 한 번씩 집계한다. 정상 CAL30/23의85% 순서 통계량을 저장한 뒤 새 점수 단독/max결합 두 방식만 현재 TEST에서 평가한다.
- 사진 라벨별20장 이상·미탐5%p 이상 감소·오탐 증가5%p 이내이면서20% 이하·표본10장 이상 결함 종류별 미탐 회귀5%p 이내·선택epoch>0 조건을 유지한다. TEST 임계값이나 채택 조건을 반복 변경하지 않는다. 실제 target 위치 검증 없이 새 지도를 기본 위치로 승격하지 않는다.

통과할 때만 새 패키지 CPU/GPU·실제 C++·모든 alias·기본 위치 PNG를 검사한 뒤 반영한다. 실패는 보존하고v16-r1을 유지한다. 공개 TEST 반복 개발·물리 촬영 그룹 미검증·과거 부모 근접 노출의 한계를 유지한다. 학술 연구 전용/상업 이용 금지·NEU 원 제공자와 주석 추가 배포자 인용을 지키고 존재하지 않는 SPDX를 붙이지 않는다. raw·가중치·ZIP을 Git에 넣지 않는다.
