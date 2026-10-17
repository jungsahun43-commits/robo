# MDMCS v126 — 고정 첫 TRAIN JSON 구조 관찰 완료

## 사전등록·보호·수신량

코드/정책/문서를 `4c644816c6141141890c6645c966d082e3871d37`로 사전 일반push/원격 확인한 뒤 실행했다. 정책 SHA256 `20aabcc3e933eb29b68d524c9b9b25323582c4a1b70ff6cee76c637c584c7712`다. 원 V1/V2 metadata·HEAD·EOCD/CD·완료 helper/fixture를 재실행하지 않았다.

원목록의 명시 TEST400파일과 publisherVAL300파일을 HELD 보호하고, 모든3711멤버/가능한 사본의 UNASSIGNED 보호를 유지한 기록을 내용 전에 저장했다. 보호기록 SHA256 `33dcb2abe4c58cbc4844d9e846e73065f5fe184384b307bb00f16f060a83974b`다. 사진별 사본동일성/물리그룹·최종 FIT_VAL_CAL/전graph 검증은 아니다. 경로순 첫 `HRCDS/train_annotations/train_0001.json` 하나만 조회했다.

| 실제206 Range body | bytes |
|---|---:|
| local header | 30 |
| local filename+extra | 71 |
| annotation deflate | 16232 |
| 합계 | **16333** |

각 응답은 body 전에 같은 URL/정확206·Content-Range·길이·If-Range·기존 strong ETag·storage version/identity를 확인했다. local offset15883587, payload15883688–15899919이며 다음 member/CD를 침범하지 않았다. flags8의 local 크기/CRC0은 선언된 descriptor mode로 허용했고 descriptor를 추가로 받지 않았다. bounded deflate가 정확153629bytes를 반환했으며 이 member의 **실제 CRC32 `7f8c0a15`**를 확인했다. 전체 archive SHA나 다른 멤버 CRC 검증은 아니다.

원문 JSON localSHA256은 `5a7ec7c42e895126b335b114dd35d5bc901a77963b1d10add99188b2535a4a46`이다. 원형 JSON·압축/헤더 범위·각 헤더/영수증·보호·관찰·로그를 보존했다. 새 metadata fixture3통과/helperexit0/오류0/stderr0, 결과 직전 data/run369052bytes로16MiB 이내였다. 예비1GiB를 유지했다.

## 실제 JSON 내용에서 관찰한 범위

- root는 version/flags/shapes/imagePath/imageData/imageHeight/imageWidth다. 선언 version은 `5.5.0`이며 설치된 renderer/원저자 mask exporter revision을 검증한 값이 아니다.
- 선언 크기1080×720, imagePath basename은 `train_0001.jpg`와 일치했다. 그 경로를 실제로 열거나 파일대응을 검증하지 않았다.
- imageData는 **null**이다. 이 JSON에는 취득한 embedded RGB payload가 없고 imageData decode/표시0이다.
- shapes13개는 모두 명시 polygon이며 literal label은 spalling2/crack11이다. 이를 corrosion 음성·wholephoto 정상 라벨로 바꾸지 않는다.
- 좌표1432개는 유한2수쌍이고 선언 edge 범위 밖0, min(0,0)/max(1079,719)다. 이 numeric 검사는 실제 RGB/native 좌표·mask raster 대응 계약의 통과가 아니다. 좌표/GT를 수정하거나 rasterize하지 않았다.
- root flags는 비었고 관찰된 root key에 background/unknown/ignore 명시가 없었다. 이것을 모든 미주석이 무부식/정상이라는 의미로 해석하지 않는다. group_id는13개 모두 null이며 촬영/물리부품 그룹의 증거가 아니다.

source annotation JSON **1개를 실제 취득**했으므로 모든 source내용 취득0이라고 설명하지 않는다. source RGB/mask 파일·전체ZIP·PDF/code/weight 취득0, source pixel decode/시각QA·raster·encoder/score·gradient/optimizer·CAL_TEST·API_CPU_GPU_CPP_ZIP_APK0이다. v85 실제450/epoch30·완료추적 일부60430은 그대로이며 후자는 실패포함 부분합으로 전체 역사 총량이 아니다. 기존 보호모델/실패·역할·GT/graph·출력을 보존했다. 새 pixel/분리/모델 protocol은 미정/readyfalse다.

## 다음에 필요한 근거

고정 JSON 구조에서 polygon/선언 크기를 확인했으나 실제 JPEG/PNG의 크기·RGB/팔레트·좌표대응, 원저자 GT/배경/unknown, 자연발생·원분리/물리그룹·전체 graph·최소가 미검증이다. 다음 문서는 같은 고정 TRAIN의 encoded JPG/명시 `_mask.png` 두 member에 대한 **별도 사전등록 header-only 검토 제안**이다. 이 단계에서 그 파일들을 받거나 해독하지 않았다. 원 TEST/VAL의 내용을 선택·열람하지 않고 GT를 고치지 않는다.

Guo/Bao·MendeleyV2 DOI10.17632/6x4dzzrs2h.2/data CC BY4.0의 출처/버전/변경·제삼자 권리와 모든 부모 조건을 유지한다. v124 정책STOP/v125 목록 완료/v126 annotation 관찰을 새 학습 실패·모델 성능개선·현장독립 증명으로 부르지 않는다.
