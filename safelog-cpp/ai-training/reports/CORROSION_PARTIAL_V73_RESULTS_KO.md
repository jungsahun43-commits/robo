# v73 부분주석 부식 학습: 전체 후보 제외

## 고정 조건과 실제 학습

- `ec9bf00`에 코드·조건·v72 자료 SHA를 일반 push한 뒤 실행했다. 기존 336 FIT(60 Ostrava+276 VT), 75 VAL을 재사용했으며 새 자연 사진은 0장이다.
- 같은 공통 encoder는 고정했다. v62 실제 부식 분기와 정규화까지 복제한 새 `corrosion_partial_head` 100,593개 파라미터만 갱신했다. 기존 v16·승인 세 head·원래 v62의 tensor/buffer/metadata는 정확히 보존됐다.
- 2 Ostrava+2 VT 고정 배치에서 BCE 양성/알려진 음성 손실 예산을 .5/.5로 두고 양성 예산을 출처별로 나눴다. VT unknown 및 padding은 양쪽 손실에서 제외했고 Dice는 Ostrava에만 적용했다.
- 실제 30epoch×64=1,920 optimizer를 수행했다. VAL 0.8772400→0.3739328의 epoch 8 / 512 업데이트를 선택했다. 모든 epoch·optimizer·장부를 보존했다.
- 추가 dense encoder 파일 없이 FIT/VAL uint8 RGB·정답 411MiB를 RAM에 두고 online encoder를 사용했다. FIT 반복 입력 7,680회와 원사진 수를 구분한다.

## CAL 저장 후 원 TEST 한 번 평가

원래 Ostrava CAL 15장의 실제 배경/부식 mask에서 사전 고정 19cut 중 .30을 골라 저장했다. VT CAL은 양성 recall 참고값만 기록했고 unknown FP를 만들지 않았다. 원 TEST는 Ostrava 15장과 VT 44장이다. 같은 특징에서 고정 v62 기준(cut .20)과 새 head를 각각 평가했다.

| 원본 좌표 지표 | 고정 v62 | v73 |
|---|---:|---:|
| Ostrava pixel FP | 9.5801% | 14.8031% |
| Ostrava pixel FN | 3.7154% | 5.5707% |
| Ostrava IoU | 70.8030% | 60.6830% |
| VT 명시 양성 pixel recall | 34.4787% | 99.8780% |

VT Fair/Poor/Severe 양성 recall은 각각 99.9106/99.7858/99.9999%다. **VT의 unknown 영역에는 이진 FP 정답이 없으므로 이 높은 recall만으로 위치 정확도를 주장할 수 없다.**

Ostrava FP 증가 5.2230%p가 사전 5%p 한도를 넘었고 IoU 감소 10.1200%p도 사전 5%p 한도를 넘었다. 전체 후보를 제외했다. VT recall만 채택하거나 cut을 다시 맞추지 않았다. source 통과 및 앱 사진 20% 통과는 모두 false다. 전체 정상 TEST 부족, 일부 사진의 큰 오차, 날짜/촬영/물리부품 독립성 미검증과 공개 TEST 반복 개발 한계가 있다.

## 보존과 검증

- 기존 Python 785파일 SHA, 보호 330,663파일 stat, 장부 1..1920, 부모 전체 및 delta 디스크 복원 exact를 검사했다. 보호 원본 전체 SHA를 재계산한 것은 아니다.
- 첫 FIT 4장의 low/high는 일반 encoder forward와 bitwise exact였다. 이 비교에 추가된 encoder 입력 8회도 별도로 남겼다.
- 새 unit 3개는 unknown gradient 0, 알려진 배경 예산·VT 음성 혼입 금지, native 좌표 padding 제거를 확인했다. 학습 runtime 오류 0이다.
- 실패 delta 파일 SHA `d40905d720583c586a0952338b6b4d863276fbf282fbcc4d8824c554d183ab56`, 결합 tensor 상태 SHA `e506b60c2b010f7f2311e9d0fd5e18f285b820542e168841ae94b3dac1cd93c3`는 서로 다른 계약이다. 단독 앱 모델/API 배포 SHA가 아니다.
- 조건부 fresh CPU/GPU·새 API·C++·ZIP·APK 검사는 모두 0이다. 승인 v51 overlay와 기본 v16을 유지했다. 이번 optimizer 1,920, 추적 일부 연구 합 58,960이며 실패 포함 부분합으로 전체 역사 총량은 아니다.

다음은 VAL에만 두 고정 head의 알려진/unknown 영역 예측 비율을 진단한다. unknown의 예측 비율을 FP로 부르거나 원 GT를 바꾸지 않는다. 같은 손실 혼합비·구조·TEST 임계값 탐색은 반복하지 않는다.

VT Bianchi/Hebdon V2 data CC0: DOI 10.7294/16624663.v2 및 10.1061/(ASCE)CP.1943-5487.0001045. Ostrava Matěj Frič Zenodo V2 CC BY 4.0: DOI 10.5281/zenodo.11235637. 원저자·출처·버전·부분주석 투영 변경과 부모 InsPLAD CC BY-NC 3.0 등 기존 조건을 유지한다.
