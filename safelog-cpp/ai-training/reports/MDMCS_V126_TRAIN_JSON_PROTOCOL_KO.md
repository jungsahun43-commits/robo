# MDMCS v126 — 첫 원TRAIN JSON 한 파일의 구조 관찰 사전조건

Guo/Bao MendeleyV2 DOI10.17632/6x4dzzrs2h.2/data CC BY4.0·출처·버전·변경·제삼자권리와 모든 이전 보존조건을 유지한다. v124 STOP/v125 중앙목록 완료를 재실행하지 않는다. 이 코드·JSON·문서를 일반commit/push/원격 확인한 뒤에만 새로운 member Range를 실행한다.

## 내용 전에 고정한 대상과 보호

고정 CD의 경로순 첫 `HRCDS/train_annotations/train_0001.json` 하나만 허용한다. 선언 local offset15883587/압축16232/확장153629bytes/CRC32 `7f8c0a15`/deflate8/flags8이다. 선택에 사진·annotation내용·모델·TEST 점수를 사용하지 않았다. 전체 source·사본 지문/graph 검증은 아니다.

요청 전에 원목록의 명시 test_annotations/test_image/test_mark_color/test_mask400파일을 원TEST 보호, val_annotations/val_image/val_mask300파일을 publisherVAL HELD로 기록한다. 모든3711멤버/가능한 사본의 UNASSIGNED 보호도 유지한다. 원TRAIN 하나는 schema probe이며 FIT/VAL/CAL/TEST 최종역할 승격이 아니다. 숫자·basename·해상도·지문을 물리그룹 증명으로 쓰지 않는다.

## 한 번의 제한 취득

v125에서 이미 확인한 정확한 공개 S3 URL·strong ETag `"488999a12587cbb5e18a092ada0916ae-10"`·storage version·archive크기204919973와 CD/보호 SHA를 고정한다. HEAD/EOCD/CD/V1/V2 metadata 재요청0이다. 동일 URL에 다음 **GET3회만** 허용한다.

1. 선택 local header 고정30bytes.
2. 그 header의 filename+extra(이름길이 CD와 일치/extra≤4096bytes).
3. 해당 member의 정확 compressed16232bytes.

각 응답은 body 전에 exact206/Content-Range/Content-Length/If-Range/동일 strong ETag/identity/같은 storage version을 확인한다. 200full·redirect·변경·HTTP오류·timeout·상한/형식/원본명 불일치면 중단하고 원형·부분·헤더·로그를 보존한다. 재시도·다른 URL·전체 ZIP/다른 member/descriptor 취득0이다. body 합≤65536bytes, data/run≤16MiB, 예비1GiB다.

local header의 signature/version/암호화·flags/method/크기/CRC를 CD와 대조한다. flags8은 local 크기/CRC0 또는 같은 CD값만 허용하며 descriptor를 추가로 읽지 않는다. filename bytes exact/안전 extra/noZIP64/Unicode alternate를 확인하고 compressed 범위가 다음 member/CD를 침범하지 않아야 한다. bounded raw-deflate≤153630bytes/stream EOF·정확153629bytes/실제 CRC32를 확인한다. CRC는 이 JSON 한 member의 바이트 검사이며 전체 ZIP SHA/모든 member CRC/정답 검증이 아니다.

## JSON 진단 계약

원문 UTF8/strict JSON(중복키·비유한수 거절)을 데이터로만 읽고 원문 그대로 보존한다. object root/최대50000shapes 범위를 요구한다. 선언 schema의 필드형·version·imageWidth/Height·imagePath basename·shape_type/label별 개수·group_id의 **주석내 자료형**·좌표의 유한2수쌍/선언 image edge 범위만 기록한다. 관찰된 예상 schema 차이는 issue로 남기고 임의 기본값·좌표배율·GT 변환·보충사진을 쓰지 않는다. numeric closed-edge `[0,width]×[0,height]` 검사는 원저자 raster 계약이나 실제 native RGB/GT 좌표검증이 아니다. polygon·rectangle 등을 서로 바꾸지 않는다.

imagePath는 실제 경로로 열지 않는다. imageData가 있으면 type/encoded길이/stringSHA를 기록하고 payload가 취득됐음을 명시한다. base64 imageData decode/사진 표시·시각QA/PNG·RGB·mask 해독0이다. encoded payload를 사진 취득0이라는 말로 숨기지 않는다. root/shape flags·라벨의 존재는 unknown/배경 의미의 증명이 아니며 미주석을 무부식 정상으로 선언하지 않는다. 실제 mask 팔레트·JSON raster 대응·RGB/native 의미·원GT 권위·자연발생·물리 촬영그룹·전graph·최소/ready는 미검증이다.

새 합성 metadata fixture3은 local header/name guard, bounded deflate/CRC guard, strict JSON/imageData 미해독을 확인한다. source/학습자료가 아니며 완료 v121–v125 fixture를 다시 돌리지 않는다. 기존의 순수 helper만 hash확인해 import하고 old main을 호출하지 않는다.

source annotation member1개를 취득하는 단계이며 원사진/mask 파일·pixel QA·raster·encoder/score·gradient/optimizer·CAL_TEST·API_CPU_GPU_CPP_ZIP_APK0이다. 실제 source 픽셀 전 별도 TEST/사본·HELD 보호·고정 원TRAIN 개발역할/사진·group·validpixel 최소·coords/decode·unknown·오류제외·oldheld/source graph·예산 코드JSON 사전push가 필요하다. 자료통과 뒤 별도 단일모델 사전조건 원칙도 유지한다. 현재 학습 준비·현장독립·장비건강·성능개선을 주장하지 않는다.

새 Python 전 fresh hostCIM 명령줄/시작시각·잠금PID·실제 v85장부450/epoch30 대조/read-only process_exists/GPU0/중복0/dead training.lock23948 보존/awakehelper·전원 변경0이다. 모든 원본·부분·실패·로그·epoch/cache/ZIP·GT/역할/graph·부모·앱 출력은 보존한다. 일반 HEAD:feature/ai-engine push만, raw JSON/ZIP범위 Git제외/기준완화0/자동 앱교체0/단계마다 자동후속 pause0이다.
