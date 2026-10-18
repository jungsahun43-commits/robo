# v135 — LICENSE 인용 DOI와 현재 record의 자료군 확인

사전 일반 push `a56bb1c543cd10e603a72e6fba9c387f106b082b`와 원격 일치, fresh CIM/lock/실제 optimizer 장부 확인 뒤 새 metadata helper를 한 번 실행했다. 정책 SHA256은 `ec43ae92a9bc03d9f8deef98a72e2dd592cdec2b6c85cc43284b0d2bc3873d14`, script SHA256은 `5efa80ef93e17ce9c331f954c8cfad8da10c98499ef9ea95580b176c04ee2383`이다. v134 원문·완료 helper/fixture를 다시 받거나 실행하지 않았다.

## 새로 받은 publisher metadata

보존 LICENSE의 요청 인용 DOI에서 연결한 [record15414238 API](https://zenodo.org/api/records/15414238) **6318bytes**와 보존 record17154852 API의 `links.versions`가 안내한 [버전 목록](https://zenodo.org/api/records/17154852/versions) **12567bytes**만 받았다. GET2/총 **18885bodybytes**이며 exact200·동일 URL·JSON MIME·identity·길이·상한을 통과했다. redirect/retry/latest/pagination/nested link/file content 취득0이다.

| 보존 원형 | 실제 bytes | local SHA256 |
|---|---:|---|
| publisher-cited-record.json | 6318 | 92772992ca35b0548afe700c1247e38059d25ef144a17c3cb3b100a3a9e5ca3e |
| publisher-version-list.json | 12567 | 6cfbe6c128fb75a41a5ee81de54a71f1e3dde648742b97c49400a20550787448 |

이는 실제 로컬 고정 publisher JSON의 SHA이며 저자 제공 archive SHA가 아니다. raw/headers/receipts/보호기록/started/stdout/stderr를 보존했다. helper exit0/stderr0/오류0/새 unit fixture0이다. 완료 직후 data26234/run205bytes로 고정16MiB 이내이며 1GiB 예비공간을 유지했다.

## 확인된 관계와 한계

인용 record15414238의 `doi`는 `10.5281/zenodo.15414238`이고 `conceptrecid=15414237`, `conceptdoi=10.5281/zenodo.15414237`이다. 보존 current record17154852도 같은 conceptrecid/DOI를 가진다. 버전 목록은 total2와 두 record15414238/17154852를 반환했다. **LICENSE 인용과 현재 record가 같은 publisher 자료군이라는 관계를 확인했다.** LICENSE의 요청 인용을 임의 수정하지 않는다.

반환된 두 항목에는 명시 `versions.index`/`metadata.version` 값이 없으므로 순서·날짜로15414238을 V1이라고 확정하지 않았다.15414238은 concept DOI 자체도 아니다. 보존 웹의 V2 표시와 각 API의 service revision은 구별하며 service revision을 dataset 버전 번호나 촬영일로 바꾸지 않는다. creator 중복을 포함한 원형 서지값도 덮어쓰지 않는다. cited API의 CC BY4.0 metadata와 v134 실제 data LICENSE 조건을 유지하고 논문 CC BY-NC-ND4.0과 구별한다.

두 record에서 같은 이름인 네 파일은 모두 **배포자 선언 크기·MD5가 일치**했다: CSV9371, README2459, LICENSE931, ZIP4504525071bytes. 새 CSV/ZIP/README/LICENSE 내용은 받지 않았으며 실제 local 파일·archive·사진의 동일성/SHA/CRC 검증은 아니다. 같은 자료군을 새 독립 사진으로 더하지 않는다. cited 목록 전체 container/가능한 record도 file content 전에 UNASSIGNED 보호했고 SHA는 `3d40460a36b0e2c51867d00f93d39cacff79e6ee9163d3932cad5b14e30edb70`이다. 사진별 TEST87 예약·최종 역할·물리 그룹 검증은 아니다.

`data/industrial-footpath-v135-citation/citation-complete.json` SHA256은 `aa2a477188fc45a4c80add66ae7e5b226bd27034242978905402e562bebba7dd`다. 이것은 인용 관계/파일 선언 비교 결과이며 source ready 판정이 아니다.

## 학습 상태

source 사진/mask/annotation/CSV/archive/PDF/code/weight 취득0, pixel/시각QA/raster/features/우리 모델 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0이다. 새 JSON metadata2개의 실제 취득을 전체 source 수신0으로 숨기지 않는다. data REST metadata와 모델 API 검사를 구별한다. 새 정식 pixel/split/model protocol0이며 readyfalse다.

원GT/unknown·실제 RGB/좌표/원분리·촬영 물리그룹·source/oldheld 전 graph·고정 최소/실제 보존 예산 gate는 미통과다. 4.5GB archive의 전체 취득·확장이나 부분 선택으로 전체 graph/독립성 ready를 주장하지 않는다. 모든 원형/부분/실패/epoch/log/cache/보고서/ZIP, 기존 v127 JPG1+mask1/v126 JSON1과 GT/역할/graph·모델/앱을 보존했다. 자기 PID/token의 임시 operation.lock만 해제했고 dead training.lock23948·다른 lock/awakehelper/전원 변경0이다. 학습 실패나 성능 개선이 아니다.
