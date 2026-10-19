# v162 — 430-byte 원저자 주석 규격 취득 전 검토

## 필요성과 범위

v161에서 보존한 API는 주석 규격 TXT의 정확 fileID·URL·430bytes·MD5를 선언한다. native GT 값과 unknown/배경 설명이 있는지 확인하는 작은 metadata 단계가 필요하다. 원사진·mask·ZIP을 받기 전 규격만 검토하며, 규격만으로 실제 GT/좌표·원역할·촬영group·독립성·fullgraph·고정 최소·학습준비가 확인됐다고 하지 않는다.

고정 주소는 `https://zenodo.org/api/records/4694695/files/Semantic%20Map%20Specification.txt/content`다. GET1/정확200/동일URL/identity/단일 Content-Length430/본문430/선언MD5 `045123a74ed25224737ec165328c5bb6`/strict UTF-8를 모두 확인하고 local SHA256을 새로 기록한다. MIME은 사전 고정한 text/plain 또는 application/octet-stream만 허용한다. header32KiB, 저장·임시·run2MiB+예비1GiB가 상한이다. v161 API 원형7143bytes/SHA d8c130887b487bd1c37a1d40b45de780db30963aa965bc44a4640174c0b21080를 먼저 로컬 대조하며 API 재요청0이다.

redirect/재시도/alternate/HEAD/Range/preview·다른 file 링크0이다. 불일치 시 원형 started/header 또는 상한 prefix+full-header SHA/부분body/failure/stdout/stderr를 보존하고1회 중단한다. 텍스트는 실행하지 않으며 값·좌표를 추정해 GT를 만들지 않는다. source family의 모든 가능 container/member/record/사본 UNASSIGNED 및 원TEST/publisherVAL 보호는 body 전 저장한다.

## 사전 공개할 정확 세 파일

`safelog-cpp/ai-training/` 아래 다음 세 파일만 별도 publication commit/diff로 준비한다.

- `scripts/inspect_wood_specification_v162.py`
- `reports/WOOD_V162_SPECIFICATION_PROTOCOL.json`
- `reports/WOOD_V162_SPECIFICATION_PREREGISTRATION_KO.md`

목적지는 **https://github.com/jungsahun43-commits/robo.git / feature/ai-engine**다. 기존 마지막 verified remote274567e8af7be2c6e507063a600713dce109b873를 부모로 세 경로만 추가한 별도 publication commit을 만든다. 새 exact 세 파일·목적지의 명시 승인과 비force 사전 push·원격 검증 전 HTTP0이다. 이전 exact3/exact6 승인은 완료·소진했으며 재사용하지 않는다. local HEAD 전체·v157 이후 미승인 보고서·보관2문서·source 원문/사진/GT/모델/weights는 이 전송 범위 밖이다. 실제 remote가 바뀌면 force/병합/다른ref로 우회하지 않는다.

helper는 새 publication 성공 영수증·300초 내 remote proof·정확 로컬/commit blob을 확인한다. 새120초 내 성공 CIM 명령줄/시작시각·process/operation.lock0·PID23948부재·v85 actual450/epoch30·실 여유를 확인하고 자신의 PID-token lock만 만들고 일치할 때 해제한다. 다른lock/deadtraining.lock23948/awakehelper·전원·GPU 변경0이다.

## 로컬 검사와 다음 gate

새 offline 합성 fixture는 본문 길이/MD5/UTF8, HTTP MIME/길이/encoding/redirect 중단과 header 보존, header상한/strictJSON/시각 guard를 확인한다. 합성 MD5는 테스트 안에서만 명시적으로 가정한 값이며 실제 publisher430bytes를 받았다고 세지 않는다. 기존 v161 helper·fixture·API는 실행하지 않는다. fixture 원형 로그·실행 당시 script/policy SHA를 별도 영수증으로 보존한다.

data CC BY4.0은 이미 보존 API의 선언이다. 출처저자·라이선스·버전변경·제삼자 조건을 유지하며 paper/code/mirror로 권리를 대체하지 않는다. 사진·mask·bbox·archive·PDF·authorcode·weights 취득/pixel해독/시각QA/raster/features/우리score/gradient/optimizer/CAL/TEST/모델API/CPU/GPU/CPP/ZIP/APK/추가삭제·압축0이다. metadata2MiB는 full156.6GB나 학습 예산이 아니다.

모든 이전 source·GTunknown/배경·원역할/물리그룹·전graph/고정최소/full예산·pixel 전 별도정책·공통encoder 단일모델/보호exact·VAL선택/CAL/고정TEST1회·FPFN/브래킷튜브/retention·새runtime/패키지 실검증·보관/새 exact 삭제 승인 조건은 유지한다. **ready=false**다. 실제 취득이 완료·중단되면 반복/상한완화/대체endpoint/full파일로 우회하지 않는다.
