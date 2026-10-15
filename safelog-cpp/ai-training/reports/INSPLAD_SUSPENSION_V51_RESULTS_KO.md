# v51 현수 연결구 부식 학습 및 v52 연결 검증

## 범위와 결과

InsPLAD 원저자의 가공지선 현수 연결구(lightning-rod-suspension) 자른 사진에서 정상 외관과 부식을 분류하는 추가 연구 기능이다. 기존 검정 브래킷·튜브의 미탐 감소를 평가한 실험은 아니다. 기본 v16 앱 모델은 자동 교체하지 않았다.

| 분리 | 정상 사진 | 부식 사진 |
|---|---:|---:|
| FIT |235|21|
| 내부 VAL |51|5|
| CAL |51|5|
| 원래 publisher VAL 평가 |231|20|

분리와 전역 프레임·RGB·8방향 pHash 그룹/제외 결과는 v50 최종 manifest를 그대로 사용했다. 새 다운로드·중복 재감사·재분리는 하지 않았다. 첫 FIT 라벨별3장만 AI로 열람했고 원저자 정답은 수정하지 않았다. FIT 부식21장은 독립 부품21개라는 뜻이 아니다.

## 실제 학습

14e3dd43a833a7c9c92ee94779bcb9ffa3ad04ba에 코드·프로토콜을 먼저 commit/push하고 30epoch×64=1920 optimizer를 실제 실행했다. 같은 encoder fine24/low40/high960의 FP16 경유 통계 특징3072와 새3072→128/GELU/Dropout.2→1 분류부를 사용했다. 새 head만 학습하고 v16 및 승인된 v45 부식/v48 유리 애자의 모든 tensor·buffer·메타데이터를 정확 보존했다.

RGB256 bicubic 입력, FIT만의 정규화, 그룹 크기의 역수로 균형 추출한 배치32, AdamW3e-4/cosine30/weight decay.01, 실제 사진 good/rust BCE를 사전 고정했다. 반복 추출61440회는 새 자연 사진수가 아니다. 초기 VAL .693147에서 최소 .2869375944의 epoch2/128업데이트 가중치를 선택했고 마지막 epoch30을 사후 선택하지 않았다.

정상 CAL51장의 85% 순서통계량 rank45, cut=0.04536491632461548을 먼저 저장하고 고정 평가251장을 한 번 점수화했다. score는 보정된 기계 고장 확률이 아니다.

| 지표 | 결과 |
|---|---:|
| 사진 오탐 |24/231 =10.39%|
| 사진 미탐 |0/20 =0%|
| AUROC |0.997186147|
| 평가 연결 그룹 |정상226/부식20|
| 독립 사진 가정 Wilson95% 오탐 참고 구간 |7.08~14.99%|
| 독립 사진 가정 Wilson95% 미탐 참고 구간 |0.00~16.11%|

선택 epoch>0·라벨별20장 이상·라벨별3그룹 이상·사진 FPR/FNR 각각20% 이하의 사전 **공개 출처 점추정치** 조건만 통과했다. Wilson 구간은 동일 부품/촬영 상관을 조정한 구간이 아니다. 평가 날짜 중 양 라벨이 함께 있는 날짜는0개다. 날짜·배경 상관을 배제하지 못하고 물리 부품·비행·현장 독립성은 검증 전이다. 12월7일 정상5장 중 오탐2장처럼 날짜별20% 초과도 있으므로 모든 하위집단이 통과한 것으로 쓰지 않는다. 사진 미탐0은 보장이나 위치 pixel 미탐0이 아니다.

## 보존과 이번 연결 검증

- 부모와 이전 두 head 전체 tensor/buffer/메타데이터 정확 보존 및 디스크 delta 복원 exact.
- 기존 Python 736개 SHA, 보호 파일 327578개 size/mtime와 캐시 SHA, 실제1920개 optimizer 장부 확인. 모든 보호 원본의 전체 SHA를 다시 계산한 것은 아니다.
- 새 모델 unit2개, 새 서버 unit3개 통과. 서버 unit의 초기 임시폴더 권한 오류는 outputs/INSPLAD_SUSPENSION_V52_UNIT_TEMP_FAILURE_2026-10-16.json에 보존하고 불필요한 임시파일 fixture를 없앴다. 학습·평가 재실행은0회다.
- d452818ba67747ff24dfc8cc6eb794ac805beab2에 패키지 조건을 먼저 고정/commit/push한 뒤 새 ZIP을 실제로 검사했다.
- **이번** 원본 RGB251장의 새 CPU/새 GPU/기존 학습 캐시 판정 일치. 최대 CPU/GPU score 차이 1.32024288177e-05; 사전 atol5e-4/rtol1e-3 및 판정 exact를 통과했다.
- **이번** 기존17개 alias의 CPU classResults/정책 SHA exact 및 기존 위치 PNG 문자열24개 exact. 새 alias1개와 명시 equipment category 및 잘못된 조합422도 검사했다.
- **이번** 별도 Qt 빌드의 실제10tests와 init/cleanup 포함12함수 통과. 기존 KS2/볼스크루/LOCO/검정 브래킷/상부 연결고리/유리 애자/새 현수 연결구7항목×3연산=실제 C++21작업을 수행했다. 이전 v49의59장/18CPP를 이번 검사 수에 합치지 않았다.
- 소유한 테스트 서버는 종료했고 Android APK·새 pip 설치·스마트폰 속도는 검사하지 않았다.

## 연구 전달물

- ZIP: `insplad-three-tasks-role4-research-overlay-2026-10-16-v51.zip`, 9134056bytes, SHA256 `551b66ca3adb02d9b67f3449077f5e7e9b1e991dc24a11e0cbd3972ed1045ff4`.
- 새 결합 delta: `runs/industrial-insplad-suspension-v51/trained/final-delta.pt`, SHA256 `b91b6578eccd1886c5be18805a9475875ff319f77bda019cd0b3f518d84f46dc`.
- 결합 tensor 상태/API modelSha256: `bb388200af167ffde35f6281a5e4626be904f9026411c401151e9bc3b4d7c45a`. 파일 SHA와 다르다.
- 기존 v16-r1 model.pt/evaluation.json이 필요한 작은 overlay이며 한 공통 encoder를 쓰는 한 모델이다. 독립 실행 모델 파일이나 완성 APK가 아니다.
- 실행과 C++ 반영: `docs/INSPLAD_THREE_TASKS_HANDOFF_KO.md`.

기존 KS2 사진12.30/4.55%, 브래킷12.50/63.83%, 튜브3.12/68.12% 및 추가12유형 중8개가 사진20% 이내인 상태는 그대로다. 새 세 가지 InsPLAD 연구 기능을 원래12개 전체 개선으로 합산하지 않는다. 원래 위치 지도·부식·유리 애자 기능의 기준과 가중치를 보존한다.

InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1, 논문10.1080/01431161.2023.2283900, 원저자 revision1b713698d7e97836b41a62dfa3314c20f3acda70. CC BY-NC3.0 비상업 학술/인용·원저자·출처·버전·변경 표시와 다른 데이터의 이용조건을 유지한다. 실제 위치 정답이 없는 사진 분류를 전기/기계 작동 건강이나 결함 위치 정답으로 바꾸지 않는다.

이번 실제 optimizer1920, 추적한 일부 연구 합47440은 실패 포함 부분합이며 전체 역사 학습 총량이 아니다.
