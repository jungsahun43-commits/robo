# v91 부분 진단 보존 및 v92 헤더 조사 조건

v91은 `48032b02651d25af59f952c1e8821b223bbe2c2e`의 사전 push/원격 확인 후 기존 파일만 읽었다. COCO ID52/107/178의 원 JPEG 3개, BMP15개, PNG3개의 수치 결과와 지문을 각각 저장했다. 다음 ID1의 JPEG 형식·모드·크기·orientation 조건 assertion에서 중단했으며 실패/로그/세 완료 결과를 보존한다. 아직 나머지 세 사진의 픽셀은 해독하지 않았다. 형식 불일치의 구체 원인은 아직 확정하지 않는다. 원본/GT 수정, 새 다운로드·시각 QA·특징·모델점수·gradient·optimizer는 모두 0이다.

확인된 제공 corrosion BMP의 흰색 영역은 ID52=12,317/ID178=58,974픽셀이며 각각 독립 Pillow COCO raster와 exact였다. ID107은 COCO corrosion polygon2개/Pillow29,076픽셀이지만 corrosion BMP 흰색은0이었다. 같은 ID107의 hull은1,159,017픽셀로 reference와 exact인 반면 marine_growth는COCO0개인데 BMP4,654픽셀이다. 단순한 전체 반전으로 해결된다고 판단하지 않는다. 세 샘플의 결과로 전체 데이터 정답 오류를 선언하거나 다른 mask를 추정 복구하지 않는다. 제공 BMP/COCO/PNG를 각각 원형 보존한다.

이 값과 v90 코드 순서로 보면 v90은 ID52의5쌍을 통과한 후 ID107 corrosion에서 존재 검사를 실패했다. v90 JPEG6회/BMP6회/PNG1회·고유RGB2장이며 v91은RGB3회/BMP15회/PNG3회·추가고유RGB1장이다. 중복 해독을 새 자연사진으로 더하지 않는다. 전체 고유 source RGB 해독은3장이며 v90의 원본 취득6장과 구별한다.

v92는 `scripts/inspect_liaci_remaining_headers.py`/`industrial-liaci-v92-header-policy.json`을 먼저 일반 push한 뒤 ID1/3/8의 이미 받은21파일 헤더만 읽는다. 정책SHA256 `d91d251ff782a134a58684a57945d4245549a5b8b232aea085b0be9588a577cd`. Pillow open/getexif 메타데이터만 사용하며 load/copy/convert/array/show는 호출하지 않는다. 실제 format/mode/size/nframes/orientation을 기록하며 COCO size와 비교한다. 알려지지 않은 형식은 보존 후 중단하고 추정 resize/flip/색·주석 변경을 하지 않는다. 새 픽셀·다운로드·시각 열람·특징·학습은0, 저장상한1MiB/예비1GiB이다. 세 완료 수치 결과/파일SHA를 재사용하며 v90/v91을 다시 실행하지 않는다.

원TEST191/사본/not_used332 제외와 v90 역할은 보존하고 항상 ready=false이다. 전체 source graph/oldheld 제외·최종 좌표 의미·최소 표본/학습 프로토콜은 아직 미완료다. LIACi CC BY-NC-SA4.0/논문 별도조건/모든 기존 부모 이용조건을 유지한다.
