# v31 실제 학습 조건: 기어 전용 작은 특징 보정 모듈

## 사전 조사와 범위

저장 형식 검증에서 기존 실패v30 full437,447,129bytes를 부모가 필요한70,847bytes delta로 추가 표현하고, 모든 가중치·buffer·metadata를 정확히 복원했다. 원래 파일은 삭제·변경하지 않았다. FIT32개에서는0초기화 보정 후 기존 후반 분석층 출력과 기존forward를 bitwise 보존했다. cachedfine→후반층의 새 정밀도 경계는 descriptor최대차.015625/평균.00019478로 사전 atol.02/rtol.01을 통과했다. 이것은 정확도 개선의 증거가 아니다.

그 결과를 바탕으로 **새 학습 조건을 확정한 단일 후보**다. v28의 기어 표면만의 자료 분리 FIT5,466/VAL1,440/CAL1,116/TEST2,179, 묶음8/3/3/4와 실제 OK/nOK 라벨을 유지한다. 이전 전체 부품 실패/TEST3,109도 보존한다.

## 학습 구조

기존 공통encoder의 stage3 fine24를 FP16 왕복하고24→4/GELU/4→24 residual adapter를 적용한 뒤, **원래 같은 encoder의 후반 module 객체**로 low40/high960을 계산한다. 기존3,072 pooled descriptor와 v28 연구 선형 판별부를 사용한다. 별도encoder·복제된backbone·hook·return_layers 변경은 없다. 원래 기본forward와 기존 다른 유형의 경로는 그대로다.

accepted v16의 모든 기존parameter/buffer는 고정·eval이다. 새220개 adapter parameter와 이전 v28 연구head의3,073개 parameter만 학습한다. adapter 마지막층은0, 나머지 seed3131 초기값이다. v28 FIT 정규화(mean/std)는 고정한다. 원래 shared weight를 갱신하는 학습과 구분한다.

같은 v29 fine 캐시를 ID/SHA로 재사용하므로 새RGB 특징 추출·전체 캐시는0이다. 이 경계는 원래 fullRGB forward와 rounding 위치가 달라 초기 VAL BCE가 v28 .587897에서.005 이상 벗어나면 optimizer 전에 기록·중단한다. 추론 경로도 원래prefix→같은FP16경계→adapter→원래tail을 한 번씩 실행하도록 정의한다.

## 최적화와 선택

- batch16, OK/nOK8개씩, 라벨별 묶음 패치 수 역수 가중 추출, seed3131, 증강 없음.
- patch BCE + `sum(global_head.weight**2)`, 위치 mask/위치loss 없음.
- AdamW: adapter lr1e-4, head lr1e-5, wd1e-4, clip5. 각lr에 `.01+.99*.5*(1+cos(pi*epoch/30))`를 곱한다. FP32/TF32off.
- **30epoch×64=1,920 optimizer회**. 실제 장부와 매epoch의 adapter/head/optimizer/scheduler/RNG를 보존한다.
- VAL 라벨·묶음 균등 BCE로 epoch0과 매epoch 중 가장 이른 엄격한 최솟값을 선택한다. epoch0은 새 학습 개선·배포로 세지 않는다.
- 선택 후 CAL 라벨의 고정 minimax 오탐/미탐 threshold를 저장하고 TEST를 계산한다. 동률은 합계/0.5까지거리/낮은값, strict `>`다. TEST 반복 threshold·구조 조합 탐색은 없다.

## 저장·검증·채택

새 연구 가중치는 accepted v16 파일SHA에 결합된 추가 `gearbox_`/`gear_task_adapter.` tensor와 새metadata만 delta로 보관한다. 저장한 delta를 다시 읽고 부모와 결합해 모든 tensor/buffer/metadata가 정확히 같은지 검사한 뒤 평가한다. 원래파일을 삭제·압축하지 않는다. 부모가 필요한 연구형식이며, 앱 전달 때에는 채택검사 후 단일fullweight를 만든다.

선택epoch>0, TEST라벨 각각20개 이상·연결그룹3개 이상, 기어 패치 오탐·미탐 모두20% 이하, accepted원래state완전보존·delta복원일치를 요구한다. 이후에도 새패키지CPU/GPU·실제C++·기존alias·기본위치PNG보존을 통과해야 반영한다. 현재앱용 v16-r1은 유지한다.

범위는128×128 기어 표면의 마모 허용이다. 전체 기어박스 작동 건강·내부고장·임의 현장 전체사진·검증된 결함 위치·기존 브래킷/튜브 개선으로 확대하지 않는다. 적은 TEST묶음4개와 이전공개TEST개발·과거부모근접노출 한계를 남기며 독립현장검증을 주장하지 않는다. Re-GBCv1 CC BY4.0, 원배포본과 DOI10.1109/ETFA65518.2025.11205777, 기존 다른자료 조건을 유지한다.
