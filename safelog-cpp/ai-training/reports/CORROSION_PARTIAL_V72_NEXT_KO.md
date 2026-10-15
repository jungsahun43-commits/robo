# 다음 v72: 부식 영역의 부분 정답 계약부터 고정

## 방향과 현재 상태

v69·v71의 4상태 후보는 모두 실패로 보존한다. Severe만 골라 반영하거나 같은 가중치/구조/TEST 임계값 조정을 반복하지 않는다. 학습은 실제로 각1,920회씩 완료됐으나 기본 앱·승인 v51 overlay·연구 v62 기준은 그대로다.

다음은 앱의 **금속 표면 부식 영역 제안** 목적에 맞춘 별도 부분 주석 과제의 자료 단계다. 새 코드/정식 자료 정책/최종 mixed manifest/특징/gradient/optimizer는 아직 없다. 먼저 하나의 정답 계약을 코드·JSON에 고정하고 commit/push한다. 자료 적격성을 확인한 뒤 별도 한 후보 학습 프로토콜을 고정해야 한다.

## 제공된 정답의 의미

| 출처 | 제공된 정답 | 새 과제에서 검토할 투영 |
|---|---|---|
| Ostrava v61 | 실제 L>0 부식 mask, 값0/169 | 원래 전체 픽셀의 알려진 음성/양성 유지 |
| VT v68 polygon | Fair/Poor/Severe 명시 부식 영역 | 명시된 영역만 부식 양성1 |
| VT v68 polygon | Good/배경: 없음 **또는 미미한 부식** | 무부식 음성으로 쓰지 않고 unknown/ignore255 |
| 두 출처의 padding/ignore | 추론 대상이 아닌 영역 | ignore255 |

이는 제공된 상태 주석의 명시된 양성 영역을 사용하는 **부분 supervision**이다. 기존 raw 상태 정답·native PNG·JSON을 변경하지 않는다. Good을 자동 무부식 정답으로 바꾸거나 AI 제안·TEST 주석을 TRAIN으로 넣지 않는다. VT polygon은 넓게 묶인 외관 영역일 수 있으므로 실제 녹의 모든 미세 경계가 검수된 mask로 설명하지 않는다.

## 자료 단계에서 할 일

1. v68 manifest SHA `af40f9f6a0cc91709c0006ec5c9e21c9d7dd94a93f079fe2401dc1dab56888ce`와 v61 manifest SHA `90ab369b28d8ad6955ca0f3114e04c1fd87f7fc464c66a28fbccc763407e4f8c`를 묶는다. 이미 완료된 embedded/copy/Ostrava/외부 TEST 지문과 전이 연결을 재사용한다. 다운로드·기존 RGB 해독·전체 pHash 감사를 다시 돌리지 않는다.
2. 역할을 그대로 잠근다. 기존 후보 풀은 FIT Ostrava60(실제 양성58/전체 주석 음성2)+VT276, 내부VAL15+60, CAL15+60, 원TEST15+44다. 이는 기존 원본의 재사용이지 새로운 자연 사진336장을 추가 확보했다는 뜻이 아니다. 표본/유효픽셀 최소 조건을 새 계산 전에 고정하고 부족하면 이동·완화 없이 중단한다.
3. 같은512 입력에서 두 출처의 실제 positive/knownnegative/unknown/padding 수를 따로 검사한다. VT의 Good 픽셀은 손실과 이진 배경 분모에 들어가지 않아야 한다. source mask와 좌표/ignore가 정확히 보존되는 unit을 준비한다. 원저자 자료·v66 실패 계약은 그대로 둔다.
4. Test/CAL/VAL의 사진 내용을 골라 보거나 GT를 수정하지 않는다. 필요하면 고정 FIT 예시만 확인하고 전문가 전체 검수로 설명하지 않는다.

## 자료가 준비된 뒤 검토할 한 후보

v62 binary mask head의 복제본 하나를 같은 공통 encoder에서 학습하는 방향을 검토한다. v16·승인 3작업·v62의 모든 기존 tensor/buffer/metadata/기준·기본지도는 exact 보존한다. 새 namespace/구별된 메서드를 사용하고 기존 forward 계약을 가리지 않는다. source/category/date embedding이나 별도 encoder를 추가하지 않는다.

부분 양성만으로 모든 영역을 양성으로 예측하는 문제가 생기지 않게 알려진 foreground/background의 손실 예산을 명시해야 한다. 한 가지 검토안은 source별 고정 batch에서 **사진별 알려진 양성 CE 평균과 알려진 음성 CE 평균을 각각 모아 균형을 맞추고**, Dice는 실제 전체 mask가 있는 Ostrava에만 적용하는 것이다. VT unknown을 음성으로 포함하지 않는다. 아직 이 문장은 정식 모델 조건이 아니며 구조·초기값·loss 정확 수식·회차/lr·샘플링·VAL·CAL·채택기준·장치 허용·delta·저장량을 별도 코드/JSON에 확정·사전 push한 뒤 gradient를 실행한다. 여러 수식/배합/가중치/TEST 조합을 동시에 탐색하지 않는다.

CAL 저장 후 원TEST를 한 번 평가한다. VT는 알려진 양성 영역의 recall만 평가할 수 있고 unknown 영역에 대한 정확한 FP를 계산할 수 없다. 모든 영역 양성 예측이 성공처럼 보이지 않도록 Ostrava의 실제 배경 FP/FN/IoU도 유지·검사한다. 이 자료들의 정상 전체 TEST사진은 부족하므로 앱 전체 사진 FP/FN≤20%나 산업현장 독립 성능으로 선언하지 않는다. 기존 v63의 실패한 AND/상위비율/임계값 변형을 다시 돌리지 않는다. 기존 경고를 자동 해제하지 않는다.

## 보존·이용 조건

source-only 후보를 앱에 자동 반영하지 않는다. 통과 후보라도 새 전달물이 필요하면 별도 CPU/새GPU/실제C++/기존alias/기본PNG보존 검사를 수행하고 과거 횟수를 합치지 않는다. 모든 원본·실패모델·로그·캐시·ZIP을 보존하고 추가 dense feature 저장·fullparent 복제 없이 bounded RAM/online 계획과1GiB reserve를 유지한다.

VT는 Bianchi·Hebdon Virginia Tech v2, 데이터CC0/DOI10.7294/16624663.v2 및 논문10.1061/(ASCE)CP.1943-5487.0001045다. Ostrava는 Matěj Frič ZenodoV2/DOI10.5281/zenodo.11235637, CC BY4.0다. 저자·출처·버전·변경을 표시하고 기존 InsPLAD CC BY-NC3.0 및 다른 모든 부모 자료 조건을 유지한다.
