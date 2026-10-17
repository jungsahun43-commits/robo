# v121 HRCDS 공개 metadata 전용 사전등록

원저자 Mendeley V1 DOI10.17632/6x4dzzrs2h.1, Pengwei Guo/Yi Bao, 2025-02-25의 공개 페이지는 pixel 주석과 data CC BY4.0을 명시한다. 실제 RGB·nativeGT/unknown/좌표·원역할·촬영그룹·정상지원은 미검증이며 아직 자료/학습 준비 통과가 아니다. concrete의 corrosion strain을 금속의 측정 부식량/구조 안전등급으로 재라벨하지 않는다.

## 허용 범위

`inspect_hrcds_metadata_v121.py`와 JSON 정책·이 문서를 일반 commit/push하고 원격 HEAD·로컬 blob SHA 일치를 확인한 뒤만 HTTP한다. 원저자 page 한 개와 이전에 확인한 공식 익명 route의 folders/root-files 두 endpoint만 받는다. HTML4MiB·각JSON2MiB·응답합8MiB·전체저장32MiB 상한과 예비1GiB를 고정한다. 추가 pagination·nested file 목록·사진/mask/archive/checksum 파일·PDF·code·thumbnail·download URL을 따라가지 않는다. redirect·로그인/동의·PII·자동연락·우회·재시도0이다.

HTML 내 단일 INITIAL_STATE 할당의 우변을 strict JSON으로만 읽는다. 중복 key·비유한값·뒤따르는 JavaScript·불명구조·datasetID/version/license 불일치·HTTP/형식/상한 오류에서 중단하고 원본/partial/header/실패를 보존한다. JavaScript·외부코드·loader를 실행하지 않는다. 합성 메타데이터 fixture 세 조건으로 순수JSON·trailing JS 거부·중복key 거부를 확인한다. 원사진 합성/QA나 학습 unit으로 세지 않는다.

## 역할·중단·후속

root에 열거된 모든 container와 포함 record를 binary 전 보호대상으로 기록하며 원TEST 개별 ID/최종분리는 아직 미정이다. 이름만으로 TRAIN/TEST/정상·손상을 추정하지 않고 이번 단계에서 분리·역할 이동·내용 QA를 하지 않는다. 전체 tree 목록/파일 CRC·배포 checksum 검증을 완료했다고 주장하지 않는다. source GT/raster/무결성·원분리·그룹과 oldheld 전이 연결은 이후 별도 자료 코드JSON의 사전 고정 사항이다.

자료 취득/해독 전에 원TEST/사본예약·고정역할·사진/group/validpixel 최소·실제좌표/unknown·오류중단·held제외·실제byte/임시 예산을 별도 사전 push한다. 자료 통과 뒤에만 한공통encoder·한분기 학습 조건을 별도 사전 push한다. 이번 source사진/mask/archive/PDF/code/weight·pixel/시각QA/특징/점수/gradient/optimizer/CAL_TEST0/모델protocol미정이다. 기존 GT·역할·graph·모델·모든 원본/실패/로그/cache/ZIP과 data 이용조건을 보존한다. GPU작업·새densefile·fullparent복제·awakehelper/전원 변경0이다.
