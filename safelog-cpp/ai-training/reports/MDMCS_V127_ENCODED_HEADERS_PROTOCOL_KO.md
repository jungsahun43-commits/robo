# MDMCS v127 — 고정 첫 TRAIN encoded JPG·PNG 헤더 사전조건

원저자 Guo/Bao MendeleyV2 DOI10.17632/6x4dzzrs2h.2/data CC BY4.0·출처·버전·변경·제삼자권리와 모든 이전 보호조건을 유지한다. v8–v126 완료단계를 다시 실행하지 않는다. 새 코드JSON/문서를 일반commit/push/원격 확인한 뒤만 아래 파일2개를 취득한다.

## 고정 대상과 보호

- `HRCDS/train_image/train_0001.jpg`: 압축327755/원 encoded343999bytes/CRC0d530a25/localoffset22596979.
- `HRCDS/train_mask/train_0001_mask.png`: 압축8327/원 encoded8550bytes/CRC2f6a46a3/localoffset185964664.

이는 내용/모델 점수 전부터 고정한 첫 TRAIN의 경로상 후보대응이다. basename을 실제 사진동일성·GT·물리그룹 증명으로 쓰지 않고 `_mask` 이름을 그대로 보존한다. v126 보호기록 SHA와 원TEST400/publisherVAL300HELD/전체3711멤버·가능한 사본 UNASSIGNED 보호를 내용 전에 새기록으로 유지한다. 최종 FIT_VAL_CAL/사본 지문·graph/최소를 통과시킨 것이 아니다.

## 취득과 바이트 검사

고정 CD/기존 URL·quoted strong ETag·storageversion·총204919973bytes의 SHA 기록을 재사용한다. HEAD/EOCD/CD/JSON/V1V2 metadata를 재요청하지 않는다. 각 member의 localheader30 → name/extra(≤4096) → 정확 compressed body만 GET한다. 최대6회/network≤1MiB/data-run≤16MiB/예비1GiB다. exact206/Content-Range/Content-Length/If-Range/같은 ETag·storageversion/identity를 body 전에 검사한다. 200full/redirect/변경/오류·timeout·상한/형식/크기 불일치면 부분·원형·헤더·실패를 보존하고 중단한다. retry/다른 URL/다른 사진/전체ZIP 우회0이다.

v126의 순수 localheader/name/extra/extent/bounded raw-deflate·CRC 함수를 그대로 import하고 old main/fixture는 실행하지 않는다. ZIP해제는 원 encoded JPG/PNG bytes의 복원이다. 각 member의 정확 길이·CRC를 검증해 원형을 저장하며 source image파일1/mask파일1을 실제 취득했음을 명시한다. 전체 ZIP SHA/모든 멤버 CRC 검사로 부르지 않는다.

## 헤더 관찰 계약

JPEG는 SOI/EOI 경계와 첫 scan 이전 marker·SOF·SOS 구조를 한정 파싱하고 width/height/precision/components를 기록한다. entropy scan은 해석·해독하지 않는다. format 필드는 [ITU-T T.81 Annex B.2.2](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)의 공개 규격을 참고했으며 이는 data허가·원저자 GT 계약이 아니다.

PNG는 signature·IHDR·PLTE·tRNS·IEND 구조와 encoded chunk CRC만 검사한다. IDAT byte수/CRC를 기록하고 **IDAT를 inflate하지 않는다**. ancillary payload도 실행/해독하지 않는다. IHDR·palette·CRC 범위는 [W3C PNG3 규격](https://www.w3.org/TR/2025/REC-png-3-20250624/)을 참고한다. palette entry는 모두 쓰인다는 뜻이 아니며 GT class mapping으로 변환하지 않는다.

정확 magic으로 JPEG/PNG를 구별한 뒤, JPG의 기대 JPEG·mask의 기대 PNG·두 헤더의1080×720이 고정 v126 JSON 선언과 일치해야 한다. 불일치면 관찰을 보존해 STOP하고 resize/rename/좌표배율·다른사진 보충0이다. 성공도 실제 RGB 색의미·원GT/팔레트 class/배경·unknown·JSON raster 대응·자연발생·물리그룹·독립성 통과가 아니다. 사진/마스크 픽셀 배열·Pillow/NumPy·시각QA/히스토그램/raster·EXIF group 추정·모델/encoder/score·gradient/optimizer·CAL_TEST0이다.

새 header fixture3은 PNG header/palette/CRC와 IDAT 미해독, JPEG SOF/SOS와 scan 미해독, truncated/unknown 거절을 검사한다. fixture의 비해독 sentinel은 사진·GT/학습자료가 아니다. 원저자 코드·loader/저장모델 실행0이다.

실제 pixel 전에 별도 TEST/사본·HELD 보호·하나의 원TRAIN 개발역할/사진_group_validpixel 최소·GT/unknown·coords/decode·오류제외·전체source/oldheld graph·예산 코드JSON을 먼저 사전push한다. 자료통과 뒤에만 별도 단일모델 조건을 사전push하며 기존 v16/승인3head/원v62 exact·한encoder/원forward 계약·delta보호/epoch0제외·TEST한번/TEST탐색0/기준완화0/자동 앱교체0을 유지한다.

새 Python 전 freshhostCIM 명령줄/시작시각·lockPID·실제v85장부450/epoch30 확인/read-onlyprocess_exists/GPU0/중복0/deadtraining.lock23948보존/awakehelper·전원변경0이다. 원본·부분·실패·epoch/log/cache/ZIP 삭제0/예비1GiB/fullparent·densefile·전체확장0/일반 HEAD:feature/ai-engine push만/rawencoded Git제외/단계마다 자동후속 pause0이다.
