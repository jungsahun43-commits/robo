# v123 MDMCS V2 공개 metadata와 V1 배포자 identity 비교

원저자 Mendeley V2 DOI10.17632/6x4dzzrs2h.2/2025-10-11/Pengwei Guo·Yi Bao의 공개 페이지는 1200이미지·4손상pixel 주석과 data CC BY4.0을 설명한다. 이는 실제 파일 검증·독립사진·원TRAIN/정상지원·좌표/촬영그룹 확인이 아니다. 원저자 논문 DOI10.1061/JBENF2.BEENG-7893의 공개 색인에서 동일 과제·자료 연결을 확인했으나 직접 ASCE403은 재시도/대체PDF/로그인/우회하지 않는다. article조건을data허가로 대체하지 않는다.

## 고정 범위

이 문서·script·JSON을 일반commit/push/원격 확인한 뒤 V2 page1개·공식 folders2/root-files version2 API2개만 HTTP한다. 기존 V1 page/API는 SHA로 재사용하며 재요청하지 않는다. HTML4MiB/각JSON2MiB/응답합8MiB/저장32MiB·예비1GiB, redirect·재시도·다른URL·nested/pagination·archive/checksum파일내용/thumbnail·PDF/code·로그인/동의/PII/연락0을 고정한다.

v122에서 검증한 strict JSON+정확한 decimal initialTime suffix 파서만 재사용한다. JS를 실행하지 않고 page render 시각을 촬영/분리/그룹으로 쓰지 않는다. id/version/title/license·형식/상한·예상 JSON 외 구조가 다르면 raw/partial/header/실패를 보존하고 중단한다. 기존 v121 실패/코드·v122 완료바이트를 변경하거나 재실행하지 않는다.

파일 id/filename/bytes/배포자SHA256 선언만 V1과 비교하며 expiring URL은 identity에서 제외한다. 세 metadata fixture로 expiry 변경 무시·SHA변경 구별·SHA누락 거부를 확인한다. 동일 선언이어도 실제 archive/사진이 로컬에서 exact 검증됐다는 뜻은 아니다. 같은 source family나 사본을 새 독립사진으로 더하지 않는다.

V2에 열거된 모든 container/포함record를 binary 전 UNASSIGNED 보호하고 V1 보호도 유지한다. 원사진별 TEST/최종역할·사진/group/validpixel 최소·정답/unknown·좌표·촬영/물리그룹·oldheld 전이 연결은 아직 미정이다. 새 source취득/픽셀 전에 별도 코드JSON을 사전push하고 원TEST 이동·역할충돌 완화·최소 완화 없이 확인해야 한다. 자료 통과 뒤 별도 단일 학습조건 사전push가 필요하다. 이번 source사진/mask/archive/PDF/code/weight·sourcepixel/QA/features/score/gradient/optimizer/CAL_TEST0/모델protocol미정이다.
