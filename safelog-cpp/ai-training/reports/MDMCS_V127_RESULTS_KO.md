# MDMCS v127 — 고정 TRAIN encoded JPG·mask PNG 헤더 완료

코드/조건을 `fce27b3e07ae2a72df83f6bb3a4d2b5e4dddcd01`로 사전 일반push/원격 확인한 뒤 고정한 두 member만 취득했다. 정책 SHA256 `81867d8a056f3309a67526a821b3f6c0609e618175edc3611773ef5ffbb99121`다. v126 이전 JSON/HEAD/EOCD/CD/metadata·완료 helper/fixture 재실행0이다.

## 실제 취득·보호

TEST400/publisherVAL300HELD/전체3711멤버·가능한 사본 UNASSIGNED 보호를 내용 전에 유지했다. 이번 보호기록 SHA256은 `6e15ae2ada5c12eb3bbcc8f54e043e2fdab754d4064ad59bbd4cab949f92eef9`다. 실제 사진동일성·원GT/최종 FIT_VAL_CAL·물리그룹/전graph 검증은 아니다.

| 고정 member | header+name/extra | deflate body | 수신 합 | 원 encoded bytes | 실제 member CRC32 |
|---|---:|---:|---:|---:|---|
| train_image/train_0001.jpg | 30+64 | 327755 | 327849 | 343999 | 0d530a25 |
| train_mask/train_0001_mask.png | 30+68 | 8327 | 8425 | 8550 | 2f6a46a3 |
| 합계 | | | **336274** | **352549** | |

GET6회는 body 전에 exact206/Content-Range/길이·If-Range/기존 strong ETag·같은 storageversion/identity를 확인했다. 각 local header/name/extra와 다음 member/CD 경계를 검사하고 bounded ZIP-deflate 해제로 원 encoded JPEG/PNG 바이트를 복원해 정확 길이·CRC를 검증했다. JPEG/PNG 픽셀을 복원한 것은 아니다. 전체 ZIP SHA/모든 member CRC는 미검사다.

원 encoded JPEG localSHA256 `b932d504e9568b79491a0a4da30bae88459520c6995609acddc52227f622e9b8`, PNG localSHA256 `0f15965b50da7f7c1637a328cbba8b7d569d51e32f4e443f347e7db2786c14dd`다. source image파일1+mask파일1을 실제 취득했으며 **취득0이라고 설명하지 않는다**. 고유 자연사진/독립부품2개가 아니고 mask는 추가 사진이 아니다. 모든 원형·압축범위·헤더·영수증·보호·관찰·로그를 보존했다.

## 헤더 관찰

- JPEG: SOF0/8bit/3components,1080×720. 첫 SOS 이후343376bytes는 entropy decode 없이 그대로 보존했다. SOF 구성요소 선언을 실제 RGB 색의미·현장 촬영/그룹 증명으로 확대하지 않는다. 헤더 범위의 기술 근거는 [ITU-T T.81 AnnexB.2.2](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)다.
- PNG: IHDR1080×720/8bit/color_type0/grayscale/noninterlaced, IHDR/IDAT/IEND뿐이다. **PLTE와 tRNS는 null/없음**이다. IDAT1개8493encodedbytes의 chunk CRC를 검사했으나 inflate/픽셀값/히스토그램0이다. type0의 의미와 header/CRC 범위는 [W3C PNG3](https://www.w3.org/TR/2025/REC-png-3-20250624/)을 참고했다.
- 두 header의 크기는 v126 JSON 선언1080×720과 일치했고 기대 JPEG/PNG와도 일치했다. 이것은 실제 JSON polygon과 mask raster 대응·damage class code·배경/unknown/정답 권위·자연발생/물리그룹/전graph/최소 통과가 아니다. resize/rename/좌표배율/GT변경0이다.

새 header fixture3/helperexit0/오류0/stderr0이다. 결과 직전 data/run896078bytes로16MiB 이내이며 예비1GiB를 유지했다. fixture의 scan/IDAT sentinel은 해독하지 않았고 사진·GT/학습자료가 아니다. 완료 fixture를 다시 돌리지 않았다.

## 원GT 의미의 남은 조건

새 grayscale 관찰에 맞춰 **이미 보존한** V2 배포자 설명을 특정 질문(수치 class code·background/unknown·촬영그룹)에 대해 읽었다. 해당 description은1200이미지/4손상 pixel주석을 선언하지만 검사한 이 description에는 수치 코드표·unknown/미주석 범위·물리 촬영대응표가 없었다. 이는 모든 저자 자료에 그런 근거가 없다는 증명이 아니다. 새 HTTP·ASCE403 재시도0이다. 범위를 한정한 `preserved-description-contract-review.json`을 별도 보존했다.

class0/검은색을 무부식/wholephoto 정상으로 추정하거나 v126의 spalling2/crack11·corrosion 부재를 corrosion 음성으로 바꾸지 않는다. PNG 수치값을 해독해 관찰하더라도 원저자 class/배경 권위가 자동 해결되지는 않는다. 다음은 새 공개 원저자 class/unknown/배경·export 계약의 텍스트 근거 검토가 우선이며, 아직 pixel 프로토콜은 없다.

이번 source 픽셀 배열해독·시각QA/히스토그램/raster·Pillow/NumPy·원저자 loader/code/model·encoder/score·gradient/optimizer·CAL_TEST·API_CPU_GPU_CPP_ZIP_APK0이다. source JSON1개는 v126 완료 이력, 이번 encoded image1+mask1은 새 취득 이력으로 구별한다. 새 모델/성능개선·현장독립·장비건강 주장0/readyfalse다. v85 실제450/epoch30·실패포함 완료추적 일부60430은 불변이며 후자는 전체 역사 총량이 아니다. 부모·실패·GT/역할/graph·앱 출력 exact와 Guo/Bao·MendeleyV2 DOI10.17632/6x4dzzrs2h.2/data CC BY4.0·출처/버전/변경·제삼자권리·모든 부모 조건을 유지했다.
