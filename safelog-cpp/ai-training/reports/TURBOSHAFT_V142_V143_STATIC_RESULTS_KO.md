# Turboshaft v142~v143 — 정적 loader 검토, 현재 자연 pixel TRAIN 경로 제외

원 source는 Turboshaft Blade Corrosion Image Collection, Jiří, Pavlas 및 Cejnek, Matous, DOI10.5281/zenodo.17834008/v0.1이다. v140 고정 publisher data-license metadata CC BY4.0/출처·저자·버전·라이선스·변경·제삼자 권리 조건을 유지한다. 코드 권리를 data 허가로 대체하지 않으며 code의 별도 허가/저작권 전체 감사 통과도 주장하지 않는다.

## 실제 취득과 실행 범위

원배포가 연결한 repo의 보존 ref JSON에 나타난 commit `7a63ebc763b94b918b23ab79e3eb3abef4e1b01a`에 정확 경로를 고정했다. v141의 strict guard 실패는 원형 그대로 보존한다. v142는 `343ffdefa9b70b1fa25df845b65c0d5100fa3ed3`, v143은 `a3cb4a2fdb50536164ec735ad4763cd9bd2c5915`의 별도 code/JSON/protocol 일반 push·원격 일치·fresh host 확인 뒤 각각 새 helper1회/GET1만 실행했다.

| 단계 | 고정 원저자 파일 | 실제 bytes | local SHA256 |
|---|---|---:|---|
| v142 | [segmentation/train_seg.py](https://raw.githubusercontent.com/matousc89/synthetic-corrosion-segmentation/7a63ebc763b94b918b23ab79e3eb3abef4e1b01a/segmentation/train_seg.py) | 11408 | cb6c060e4293a8c8c1ec0324e7c83929f836374ac5c64d43d51fcb74ea49f126 |
| v143 | [segmentation/dataset.py](https://raw.githubusercontent.com/matousc89/synthetic-corrosion-segmentation/7a63ebc763b94b918b23ab79e3eb3abef4e1b01a/segmentation/dataset.py) | 6492 | 9e4fef9c297eceeb729fd747563bcc357c75876f8c348fc56973deef0a927178 |

각각 exact200/동일URL/text_plain/identity·길이/64KiB상한/UTF8를 확인했다. 두 helper exit0/stderr0·새 fixture0, 원형 코드 text2/17900bytes 실제 취득이다. v141 raw ref382를 합하면 이번 새 GET3/18282bodybytes다. 원저자 파일은 Git 제외 data `.py.txt`에 바이트 그대로 저장하고 PowerShell에서 텍스트로만 읽었다. import/exec/eval/compile/의존 코드 실행·추가 dependency/API/모델 요청0이다. local SHA는 사전 제공받은 author checksum이 아니고 repo commit도 dataset revision/archive checksum이 아니다. headers/receipt/started/stdout/stderr·원형부분실패를 모두 보존한다.

## 관찰한 실제 코드 범위

`train_seg.py` 112~140행은 patch images와 양성/음성 stem 목록 및 negative stem의 run별 shuffle/split을 사용한다. 145~147행은 TRAIN/VAL `fake=True`, TEST `fake=False`로 `PatchDataset`을 호출한다. 이 실제 연구 split을 SafeLog 역할/촬영 그룹·TEST 독립성으로 채택하지 않았다. substring stem 매칭은 실제 파일목록·사본·물리그룹 감사가 아니다.

`dataset.py` 54행 target은 파일명 `positive` 포함 여부에서 정해진다. 56~73행은 대응 txt의 float 목록을 읽어 labels에 저장하지만, 89~130행의 반환 segmentation mask는 그 labels로 rasterize하지 않는다. 96~106행은 fake 확률 분기에서 `add_corrosion_patch` 반환 mask를 쓰거나 0 mask를 만든다. 120행에서 mask를 64x64로 resize한다. 이 정적 경로의 mask는 자연 손상 native pixel 주석을 입증하지 않는다. 실제 `synthetize.dev`는 취득·실행하지 않았고 합성 세부 알고리즘을 검증한 것도 아니다.

labels는 반환/예제 rectangle 표시 경로에 존재하지만, txt 원문·archive 내용/전체 GT 형식은 읽지 않았다. 그러므로 이 자료군 전체가 bbox-only/GT 부재 또는 원사진 모두 합성이라고 단정하지 않는다. fake=False에서 생기는 zero mask나 `negative` 파일명을 검증된 무부식/wholephoto 정상·미주석 배경 권위로 채택하지 않는다. `tt_augment`의 flip/labels 미변환 FIXME도 정적 관찰이며 실제 GT 오류 재현·수정이 아니다.

`train_seg.py`는 epoch별 TEST classification을 계산하며 segmentation 출력은 TEST에서 무시한다(258~276행). 293행 comment는 TEST 기준 저장이라 적었지만 실제 조건은 294행 `val_acc > best_val_acc`다. comment와 실제 조건을 구별하고 저자의 score/loss/optimizer/회차/모델 구조·threshold·평가를 SafeLog protocol/성능으로 채택하지 않았다. 원저자 코드를 실행하거나 수정을 시도하지 않았다.

## 판정

**검토한 이 연구 코드 경로는 현재 자연발생 산업 RGB TRAIN native 손상 pixel+명시 배경 우선순위에서 binary 취득 전에 제외한다.** 새 명확한 원저자 native 자연 TRAIN pixel GT/unknown/배경·원분리/촬영 대응 release 근거가 없으면 같은 source/README/API/코드·archive/합성 함수/모델을 반복 조사하지 않는다. archive 전체의 GT 부재 증명으로 확대하지 않는다.

archive262055692bytes와 MD5는 여전히 publisher 선언이며 archive/원사진/mask/annotation localMD5_SHA_CRC/실물 GT 검증0이다. 모든 container/가능한 사본 UNASSIGNED 보호, 사진별TEST/VALHELD/최종 FIT_VAL_CAL·물리그룹/전graph/최소/full예산 미검증을 유지한다. pixel/시각QA/raster/features/우리 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0·readyfalse다. 새 학습 실패나 성능 개선이 아니다. 모델/ZIP/앱·기존 원형부분실패epoch로그cache보고서·지문/graph/GT/역할을 보존한다. 각 data+run2MiB/예비1GiB와 최종 보호10 SHA/실제 장부/Git 증거는 최신 outputs 영수증으로 남긴다.
