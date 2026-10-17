# v125 이후 — 원 TRAIN annotation의 별도 schema 검토 제안

v8–v125 완료단계를 다시 실행하지 않는다. v124 HEAD302/body0 STOP과 v125 exact Location의EOCD22+중앙목록390566bytes 완료를 구별한다. 같은 원 URL/Location HEAD·목록 Range·V1/V2 metadata·ASCE403을 재요청하지 않는다. 원형·부분·실패·모든 epoch/log/cache/ZIP·GT/역할/graph/부모·앱 출력을 보존한다. v123 이후 모든 보호조건은 계속 적용한다.

## v126은 아직 정식 코드·정책·새 내용 취득이 없는 제안

먼저 보존된 CD/EOCD/ETag·publisher metadata와 보호목록 SHA를 고정한다. 원문 경로 `HRCDS/test_annotations/test_image/test_mark_color/test_mask`400파일 및 모든 사본 가능성을 내용 전에 계속 보호하고 publisher val_*300파일도 HELD로 고정한다. 원 train_*만 향후 개발 검토 대상이며, 현재 모든 멤버 UNASSIGNED 보호를 해제해 최종 학습역할로 배정한 것은 아니다. 숫자나 같은 basename을 물리그룹·실제 사진동일성·정답 검증으로 쓰지 않는다.

검토할 한 조건은 사전에 이름으로 고정한 `HRCDS/train_annotations/train_0001.json` **한 파일만**이다. CD선언 압축16232/확장153629bytes이며 내용 선택 전의 경로순 첫 TRAIN JSON이다. local header(고정30+name/extra)와 해당 compressed body만 exact206/Content-Range/length/If-Range/기존 ETag/실제 CRC·길이로 검증하는 새 코드JSON·byte/저장/예비1GiB·오류중단·TEST예약을 먼저 일반commit/push한다. 실패 시 부분 보존/재시도·전체 ZIP 우회0이다. 이 문서는 아직 정식 정책이 아니다.

JSON은 duplicate key/비유한수·예상 schema·shape_type·labels·좌표범위·imagePath/width/height·flags와 명시 unknown/배경의 범위만 검토한다. annotation을 Python/JS/loader로 실행하지 않는다. `imageData`가 있으면 encoded payload 존재·길이를 기록하되 decode/사진출력/모델입력0을 유지하고, encoded 사진 bytes 취득을 metadata만의 무사진 취득으로 숨기지 않는다. 원형 JSON을 변환·수정하거나 모든 미주석을 정상 GT로 확정하지 않는다. 원 mask 팔레트/좌표와 JSON raster 대응·RGB/native 의미·실제 정답 권위는 이 단계만으로 통과시키지 않는다. source QA·VAL/CAL/TEST 내용 열람·모델 점수·gradient0이다.

실제 RGB/mask 픽셀을 검토하기 전에는 별도 코드JSON으로 원TEST/모든사본·publisherVAL HELD 보호/고정 원TRAIN 개발역할·사진/group/validpixel 최소·원GT/unknown·coords/decode·오류제외·전체source bridge/oldheld 연결·저장예산을 사전 고정한다. old8995TEST/InsPLAD6417평가·6481TEST/v57graph94/Ostrava105/VTv66v68/Riosv78의 완료지문·graph·역할과 현재Ostrava15/VT44/Rios7TEST를 재사용해 새source만 계산한다. TEST/not_used/HELD 연결TRAIN·역할충돌 제외/사본동일group/역할이동·재분리·최소완화금지/고정 첫FIT만QA/선택밖bridge 미검사 한계를 지킨다. 실제 물리그룹을 제공하지 않으면 이름/해상도/지문을 물리독립성 증거로 바꾸지 않는다.

자료통과 뒤에만 별도 단일모델 사전조건으로 features/gradient를 허용한다. v16+승인3head+원v62 exact/한encoder/새namespace·구별method/원forward 계약·작은delta SHA/restoreexact/epoch0 제외/CAL고정 후 TEST한번/TESTcut_stride_계수_구조 반복금지/기준완화0/자동 앱·alias·경고·primary·ZIP 교체0이다. 새Python 전 fresh hostCIM·잠금·실제 장부/read-onlyprocess_exists/GPU한개/중복0/deadtraining.lock23948보존/awakehelper·전원 변경0, 예비1GiB/작은저장/boundedRAM/전체확장·fullparent·densefile0을 유지한다. 일반 HEAD:feature/ai-engine push만, 계속승인에 따라 자동후속은 단계마다 pause하지 않는다.
