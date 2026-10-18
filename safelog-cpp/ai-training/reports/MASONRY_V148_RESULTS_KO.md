# Masonry v148 — 첫 HEAD200 뒤 필수 헤더 guard 중단

사전 코드·JSON·프로토콜3파일을 일반 push한 `e9ceab842b133f17f171fccdac4bd7baf11d7916`와 원격 feature/ai-engine 일치를 확인했다. 이번 새 helper의 가짜200/잘못된206에서 body읽기0, ZIP64 marker/record/entry·comment 중단 fixture3개가 통과했다. 기존 helper/fixture/HTTP 재실행은 0이다. 기존 로컬 삭제/보관 문서2개는 untracked로 유지하며 원격 전송하지 않았다.

정책SHA `3b1b3505126d874224ac090bd555cf869b4bd08d06df07876044fde3bd0480e2`, scriptSHA `fc0b254ac7068f73ca1f0b8e7217aae8d6ff0619f41592b45ac26eaa7e476162`다. fresh CIM 명령줄/시작시각·lock0/PID23948 없음·실제 v85 optimizer450/epoch30·16MiB 저장 예산+예비1GiB를 확인한 뒤 helper1회만 실행했다.

## 실제 요청과 중단 이유

고정 `masonry_crack_masks.zip` URL에 **HEAD1회/HTTP200/동일URL/선언 길이1052176bytes**를 관찰했다. 응답에 **Accept-Ranges와 ETag 헤더가 없었다**. 코드의 최초 중단 지점은 명시 Accept-Ranges bytes 요구였고, 보존 헤더를 정적으로 읽어 ETag도 없음을 기록했다. 헤더 부재는 이 서버의 모든 범위 요청이 영구 불가능하다는 증명이 아니다. 검사 순서를 바꾸거나 조건을 완화해 같은 요청을 다시 하지 않는다.

**GET0/HEAD body읽기0/실제 network body bytes0**이며 imagesZIP HEAD/Range0, EOCD/CD/ZIP64 원자료 취득0/실제 member목록0이다. 본문으로 Range 동작을 시험하거나 전체 ZIP·대체 endpoint·redirect로 우회하지 않았다. 저장한 HEAD에는 Last-Modified가 있으나 촬영일/자료 버전으로 쓰지 않았다. rate-limit/Retry-After 응답 필드도 재시도 허가로 쓰지 않았다.

helper exit1/stdout0/stderr716bytes는 **사전 정책 guard 중단1회**이며 학습 실패·서버 HTTP 실패가 아니다. 헤더·failure.json·원형 코드/정책·started·stdout/stderr·보호 기록을 보존한다. 자기 PID/token operation.lock만 해제했고 deadtraining.lock23948·다른 lock·awakehelper/전원은 변경하지 않았다.

## 보호·해석 범위

새 요청 전에 두 container/모든 가능한 member·record·사본을 UNASSIGNED 보호했다. 목록은 받지 못했으므로 사진별 TEST/VALHELD 예약·원역할·촬영그룹/전graph·고정 사진/group/validpixel 최소·실 nativeGT/unknown·RGB좌표/해독·full보존예산은 계속 미검증/readyfalse다. v147의 실제 publisher API5826bytes/CC BY4.0·파일 크기/MD5 선언과 v146 web 관찰은 그대로 유지한다. 전체 ZIP SHA/MD5·memberCRC는 검사하지 않았다. 원저자 Storey, Katherine A / Snowball, Andrew L·DOI10.5281/zenodo.18458458·라이선스/출처·버전/변경·제삼자 권리는 유지한다.

새 사진/mask/annotation/fullZIP/PDF/원저자code/weights 취득0, pixel·시각QA/raster/features/우리 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0, 새 pixel/split/model protocol0이다. HEAD 헤더 metadata 관찰을 전체 외부 HTTP0이나 이미 취득한 v147 원문0으로 설명하지 않는다. 이번 source 원문 body0은 이번 고정 HEAD 요청 범위에 한정된다.

자료 후보 이름 SBDD/Steel Bridge Defect Dataset의 로컬 report/script/output 텍스트 검색은 정상 exit1 미일치였지만 후속 단일 web 탐색으로 원저자 자료를 식별하지 못했다. 이름은 미확인 탐색 토큰이며 정식 datasetbrand·새 release·자료 부재 증명이 아니다. 무관 검색결과를 열거나 개인정보/서지 원문을 별도 source로 저장하지 않았다. 별도 v149 source 단계/취득은 0, web traffic 총량은 미측정이다. 같은 탐색을 반복하지 않는다.

사용자가 별도 승인한 A의 중복 사본6개/12GiB 삭제는 기존 완료 사실이며 이번 삭제0이다. B/C와 추가 폐기는 승인되지 않았다. 원본8개·상이한 part07~08·사진/GT/평가자료·모델/앱·역할/graph·원형/실패/epoch/로그/cache/보고서/ZIP을 유지한다. 보호10의 실제 마지막 SHA 확인은 이번 최종 영수증에 기록하며 전체 source 원본 SHA 재감사로 확대하지 않는다.
