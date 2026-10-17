# v126 이후 — 고정 TRAIN encoded 파일의 header 검토 제안

v8–v126 각 완료단계를 다시 실행하지 않는다. v126 원TRAIN JSON1개의 CRC/구조 관찰이 완료됐으며 원문·범위·보호·실패/로그를 보존한다. 같은 JSON/HEAD/EOCD/CD/V1V2 metadata·완료 helper/fixture·ASCE403을 재요청하지 않는다. 모든 이전 보호조건과 data CC BY4.0/GuoBao/DOI10.17632/6x4dzzrs2h.2·출처/버전/변경·제삼자권리 조건을 유지한다.

v126 JSON의 version5.5.0/1080×720/polygon13/spalling2+crack11/imageData null/유한좌표1432는 관찰값이다. 실제 RGB/mask/native/GT/배경/unknown/물리그룹·전graph·최소/ready 통과가 아니다. root flags{}·group_id null은 전체 정상지원이나 독립부품의 증거가 아니며 corrosion 라벨 부재를 corrosion 음성으로 바꾸지 않는다.

## v127은 아직 정식 코드·취득·pixel/분리·학습조건이 없는 제안

고정 첫 TRAIN을 그대로 유지하고 보존 CD의 다음 **정확 이름 두 member**만 검토하는 코드JSON/문서를 먼저 일반commit/push한다. mask 이름을 basename으로 추정해 수정하지 않고 실제 CD의 `_mask` suffix를 유지한다.

- `HRCDS/train_image/train_0001.jpg`: 선언 압축327755/확장343999bytes/CRC0d530a25/localoffset22596979.
- `HRCDS/train_mask/train_0001_mask.png`: 선언 압축8327/확장8550bytes/CRC2f6a46a3/localoffset185964664.

이것은 경로상 후보대응이며 실제 동일사진/GT 대응 증명이 아니다. 첫 JSON 내용에서 좋은 라벨/영역을 골라 다른 사진으로 바꾸지 않는다. TEST400/모든사본 가능성·publisherVAL300HELD·전체3711UNASSIGNED 보호/제한 원TRAIN 검토를 먼저 기록한다.

기존 S3 URL/ETag/storageversion/CD SHA를 고정하고 새 local header30+name/extra+각compressed body만 exact206/Content-Range/길이/If-Range/sameETag/version으로 제한한다. HEAD/목록 재요청·retry/redirect/전체ZIP/다른사진0, network예산≤1MiB/data-run≤16MiB/예비1GiB와 오류중단·부분보존을 사전 고정한다. ZIP member를 bounded 해제해 **원 encoded JPG/PNG bytes**의 CRC/길이를 확인하는 것이고, JPG·PNG source 파일2개를 취득했음을 명시한다. source 픽셀 배열은 해독하지 않는다.

그 뒤 표준 parser의 정적 header만으로 JPEG SOF의 width/height/bitdepth/components, PNG IHDR/PLTE/tRNS 등 header 선언과 chunk CRC를 제한 관찰하는 한 조건을 고정한다. IDAT는 decompress하지 않고 JPEG scan도 해독하지 않는다. Pillow/NumPy/renderer/원저자 loader·모델 실행/사진출력/시각QA/픽셀 히스토그램/마스크 raster/GT수정0이다. JSON1080×720과 encoded header 크기의 일치 여부만 기록하고, mismatch는 추정 resize/좌표배율로 복구하지 않는다. RGB/팔레트 component 선언을 실제 색의미·GT class mapping·배경/unknown 권위로 확정하지 않는다. 이것은 아직 정식 v127 정책이 아니다.

실제 pixel 단계 전에는 별도 코드JSON으로 원TEST/사본·publisherVAL HELD 보호·원TRAIN의 하나의 고정 개발역할·사진/group/validpixel 최소·원GT/unknown·coords/decode·오류제외·전체 source/oldheld bridge·실제예산을 먼저 고정한다. old8995TEST/InsPLAD6417평가·6481TEST/v57graph94/Ostrava105/VTv66v68/Riosv78의 완료지문·graph·잠긴 모든 역할/현재Ostrava15/VT44/Rios7TEST를 재사용하고 새source만 계산한다. HELD/TEST/not_used 전이연결TRAIN·역할충돌 제외/사본동일group/이동·재분리·최소완화0/고정첫FIT만QA/VAL_CAL_TEST 선택·GT수정0/선택밖bridge 미검사 한계를 지킨다. 실제 물리그룹 근거가 없으면 파일이름·해상도·지문을 독립성 증명으로 쓰지 않는다.

자료통과 뒤에만 별도 단일모델 조건을 사전push하고 features/gradient한다. v16+승인3head+원v62 exact/한encoder/새namespace·구별method/원forward 계약·작은delta SHA/restoreexact/epoch0제외/CAL고정 후 TEST한번/TESTcut_stride_계수_구조 반복금지/기준완화0/자동 앱·alias·경고·primary·ZIP 교체0이다. 새Python 전 freshCIM·잠금·실제장부/read-onlyprocess_exists/GPU한개/중복0/deadtraining.lock23948/awakehelper·전원 변경0, 원본·부분·실패·epoch/log/cache/ZIP 보존/예비1GiB/fullparent·densefile·전체확장0/boundedRAM을 유지한다. 일반 HEAD:feature/ai-engine push만, 계속승인에 따라 자동후속은 단계마다 pause하지 않는다.
