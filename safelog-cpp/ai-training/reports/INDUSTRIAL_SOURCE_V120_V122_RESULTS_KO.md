# v120–v122 WFDD 선별·HRCDS 공개 metadata

2026-10-17 UTC. 실제 repo의 WFDD/저자/제목/URL 및 HRCDS/DOI 검색은 오류 없이 기존 완료기록과 미일치했다. v8–v119 단계는 다시 실행하지 않았다.

## v120 WFDD: 정상-only TRAIN

[원저자 GLASS](https://github.com/cqylunlun/GLASS)와 [공식 논문 부록의 색인텍스트](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08382-supp.pdf)는 TRAIN3657정상/TEST203정상+241결함을 설명한다. 배포자 선언이며 확보·독립사진 검증이 아니다. 현재 자연 손상pixel TRAIN 우선순위에서 취득 전 제외하고 원TEST를 옮기지 않는다.

README는 code와 dataset MIT를 명시하며 생산현장3종과 제삼자 Cloth Flaw 출처1종을 구분한다. 실제 archive/LICENSE·제삼자 허가·좌표·분리·그룹·무결성은 미검사다. 별도 binarized foreground mask와 MAD-man/MAD-sys를 자연 손상GT로 승격하지 않는다. 모델/합성코드·외부 명령·download link·원사진/PDF를 실행/취득하지 않았다. 제삼자 mirror의 중복/softmask 주장도 원자료 검증으로 사용하지 않았다.

## v121 HRCDS: 사전등록 후 metadata 형식 중단

[원저자 Mendeley V1](https://data.mendeley.com/datasets/6x4dzzrs2h/1), Pengwei Guo/Yi Bao, DOI10.17632/6x4dzzrs2h.1/2025-02-25는 concrete damage pixel 주석과 data CC BY4.0을 명시한다. 실제 RGB·nativeGT/unknown·정상지원·원역할·촬영그룹은 미검사다. 원문 corrosion strain을 금속의 측정깊이·구조안전등급으로 바꾸지 않는다.

`a6da5a1` 사전push·원격 확인 후 공개 HTML124430bytes를 받았다. 로컬SHA7310b421ea61c380e7a8f89eae61c4d430715d1cdc6da8a858b7d1b2de548604다. 실제 INITIAL_STATE 뒤에 page의 `window.initialTime` 정수 할당이 추가돼 엄격한 단일할당 계약에서 중단했다. API0/sourcepixel0/optimizer0이다. 초기 코드/정책·HTML/header/receipt·실패JSON·stderr855bytes·metadata fixture3조건을 보존했다. 학습/GT 실패가 아니며 통과로 소급 변경하지 않는다.

## v122: 저장한 HTML 재사용·미시도 API만 완료

`49bd806`에 복구 코드/정책을 사전push한 뒤 HTML을 SHA로 재사용했다. strict JSON을 읽고 관찰한 십진 render-time suffix만 허용했으며 JS를 실행하지 않았다. 이 시간은 촬영날짜가 아니다. page 재요청0, 미시도 공식 folders/root-files API2개에서713bytes(2+711)를 받았다. 합125143bytes의 공개 metadata이며 사진·mask·archive 취득은0이다. 새 metadata fixture3조건과 실행오류0/stderr0을 확인했다.

root 목록은 `HRCDS.zip` 하나, 선언204919973bytes/배포자SHA256 `0582f706a8d3d419f76021ee83a58e80a91c2b56bb4681e43a9089f0756e1bf5`를 반환했고 폴더는0이었다. ZIP을 받지 않았으므로 이 값을 실제 ZIP의 localSHA/CRC 검증으로 부르지 않는다. archive 전체와 포함record를 binary 전 `UNASSIGNED_PROTECTED_BINARY_NOT_ACQUIRED`로 보호했다. 실제 사진별 원TEST/최종역할·최소·좌표·그룹·전이graph·준비통과는 아직 없다.

완료 JSON은 `data/industrial-hrcds-v122-metadata/metadata-complete.json`, SHA65d83d7e3b74821353c77c28d0581d4ae7f40d191e38bc5a5b08622a5cb9062a다. 보호 container 목록 SHA3d337ff81159b9cc82afa7eef84ddfbe5d9df38a5e0c0110063f05e141333518이다. v121 data/run127857/378bytes와 v122 data/run11045/113bytes를 보존했다(별도 WORK stdout/stderr와 최종 보고서 제외).

공식 [버전 비교의 색인텍스트](https://data.mendeley.com/datasets/compare/6x4dzzrs2h)에서 V2가 MDMCS, DOI10.17632/6x4dzzrs2h.2/2025-10-11로 이름이 바뀐 사실도 발견했다. V2 직접 page/file metadata는 아직 미취득이며 V1을 자동 교체하지 않는다. 같은 family를 새 독립사진으로 세지 않는다. `71a5de9` 이전 기록에서 MDMCS/동일DOI-family 검색은 미일치했다.

## 실제 범위와 다음 gate

`data/industrial-source-v120-v122-metadata/observations.json`의 웹관찰과, SHA로 고정한 실제 publisher HTML/API metadata를 구별한다. source사진/mask/archive/PDF/code/weight 취득·pixel/내용QA/특징/점수/gradient/optimizer/CAL_TEST/API/CPU/GPU/CPP/ZIP/APK 모두0, 새 모델protocol0이다. metadata 검사 완료를 학습/성능 개선으로 설명하지 않는다. 완료 optimizer 부분합60430은 그대로이며 전체 역사 총량이 아니다.

모든 기존 모델·GT/역할/graph·원본/실패/epoch/로그/cache/ZIP·부모 이용조건·앱 출력을 보존한다. 다음은 `INDUSTRIAL_SOURCE_AFTER_V122_NEXT_KO.md`를 따르며, 현재 자료는 ready=false다.
