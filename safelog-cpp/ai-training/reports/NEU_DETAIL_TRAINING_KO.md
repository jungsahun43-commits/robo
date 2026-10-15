# v22 실제 주석·고정 상세 영역 학습 조건

2026-10-15. 완료된 v8~v21 모델 실험은 재실행하지 않는다. 이 문서·코드·JSON 조건을 먼저 commit/push하고 자료 준비를 수행한다. 자료 감사가 통과한 최종 manifest의 SHA를 보고서에 담아 다시 commit한 뒤 새 encoder 특징과 실제 optimizer를 실행한다.

## 자료·작은 표시의 범위

학습용 실제 주석 후보는 기존 예약을 지킨 SDI20/SPDI15, 검증용은 미열람 Oil Small_defect4/Wipe_crack6이다. 흰 표시가 발견된 Scratch6장은 계속 제외한다. 원래 source 사진51장의 mask는 배포자가 제공한 실제 표면 주석이며 새 합성 자료를 만들지 않는다. 파일 계열은 물리 부품 ID나 독립 촬영 세션의 증명이 아니다. 실제 고장 진실·주석 전체의 완전성을 모두 확인했다고 주장하지 않는다.

새 source에는 정상 전체 사진이 없다. 따라서 원래 MPDD 정상 사진 FIT70/61·VAL18/15를 함께 쓴다. CAL30/23·TEST79/101은 유지한다. source의 mask 바깥 crop은 **주석상 결함이 없는 영역**으로만 학습하며, 부품 전체의 정상 작동 라벨로 쓰지 않는다. 새 source 전체 사진 FPR·새 앱 유형·독립 현장 통과를 선언하지 않는다.

## 같은 규칙의 상세 영역 9개

원본 가로·세로의 절반을 올림한 크기의 crop을 각 축 시작/중간/끝에 놓은3×3영역이다. GT 위치로 추론 crop을 고르지 않는다. 같은9개 영역을 학습·VAL·CAL·TEST에 사용한다. 각 RGB crop을 종횡비를 유지해512로 bicubic resize하고 회색127로 letterbox하며, 원래 source mask는 같은 좌표에서 crop 후 nearest resize한다.

source crop에서 변환된 양성 픽셀32개 이상은 양성,0개는 주석상 음성,1~31개는 제외한다. 작은 양성 crop을 정상 라벨로 바꾸지 않는다. 원래 정상 사진의 영역은 정상 주석 crop이다. 평가의 사진 점수는9개 crop의 valid top32 sigmoid 평균 점수 중 최댓값이다. 오탐·미탐의 표본 수는 원본 사진을 한 번씩 센다.

## 학습 전 중복 감사·중단 기준

원래5746 TEST 전체 사진뿐 아니라 새로 선언한 VAL/CAL/TEST crop과 전체 사진을 검사한다. raw crop과512모델 입력의 정확 RGB SHA·8방향 pHash를 함께 사용한다. 거리2이하 또는 정확 RGB 일치가 있는 FIT crop의 원래 사진/appearance 그룹을 제외하며, FIT 내부 근접 연결도 합쳐 연결된 그룹 전체를 제외한다. TEST mask나 모델 점수를 자료 분리에 쓰지 않는다.

감사 이후 실제 주석 FIT 원본20장 이상·SDI/SPDI 각각5장 이상과 각 source의 음성 crop, target 정상 FIT유형별30장 이상·VAL10장 이상, 실제 주석 VAL8장 이상과 네 VAL집단의 표본이 필요하다. 충족하지 못하면 자료·오류를 보존하고 새 features/optimizer 전에 멈춘다. 이 기준을 TEST 결과에 맞춰 완화하지 않는다. 과거 부모 모델의 근접 노출이나 미검증 물리 촬영 그룹을 이번 감사가 해소한 것으로 설명하지 않는다.

## 한 모델·한 공통 분석부

원래 encoder의 fine24/low40/high960을 사용한다. v21의 VAL 선택 판별부를 별도 detail_surface_head의 초기값으로 복사하고 fine projection/fusion만 학습한다. high projection과 기존 모든 tensor/buffer/메타데이터는 고정한다. 기존 forward_surface 서명을 가리지 않는다.

입력 잔차 채널은 **모든 source·설비·추론에서 동일하게0**이다. 새 자료를 가짜 MPDD category embedding으로 넣거나 source 식별 신호를 만들지 않는다. 별도 pretrained encoder를 추가하지 않는다. 새 상세 지도는 원래 target 기본 위치 지도나 pixel 기준을 대신하지 않는다.

## 고정한 학습·선택·CAL

30epoch×64step=예정1920 optimizer, seed2222. 배치8은 SDI양성2·SPDI양성2·각 source 주석상 음성1개씩·브래킷 정상1·튜브 정상1이다. 원본 사진별로 균등하게, 그 사진의 해당 라벨 crop 중 균등하게 복원 추출한다. 반복 횟수나 crop 수를 새 실제 사진 수로 세지 않는다.

사진별 양성/배경 평균 BCE + softDice +0.5×top32사진 BCE를 사용한다. AdamW lr1e-4/wd1e-4, cosine 최소1e-6, clip5, FP32/TF32 off. cached 특징·mask·valid에 함께 H/V반전을 적용한다. 실제 RGB를 뒤집어 encoder를 다시 통과한 증강이라고 설명하지 않는다.

VAL은 실제 양성/주석상 음성/브래킷 정상/튜브 정상 네 집단의 loss를 평균한다. 각 집단 안에서는 같은 원본 사진의 crop loss를 먼저 평균하고, 원본 사진들을 평균한다. epoch0 포함 가장 이른 최솟값을 선택한다. 원본 source 계열이 다른 검증 자료이며 독립 현장 검증은 아니다.

학습 후 정상 CAL 원본30/23장의85% 순서 통계량을 저장한다. 새 상세 점수 단독과 현재 v16과의 max결합 두 방법만 사용한다. 사진 라벨별20장 이상·미탐5%p 이상 감소·오탐 증가5%p 이내/오탐20% 이하·종류별 표본10장 이상이면 미탐 회귀5%p 이내·선택 epoch>0 조건을 그대로 적용한다. TEST 임계값을 반복 맞추지 않는다.

통과 후보에만 새 패키지 CPU/GPU·실제 C++·모든 alias·원래 위치 PNG 보존 검사를 수행해 반영한다. 실패는 보존하고 v16-r1을 유지한다. 공개 TEST 반복 개발은 독립 현장 성능이 아니다. raw/가중치/ZIP은 Git에 넣지 않고 코드·보고서만 feature/ai-engine에 일반 push한다.
