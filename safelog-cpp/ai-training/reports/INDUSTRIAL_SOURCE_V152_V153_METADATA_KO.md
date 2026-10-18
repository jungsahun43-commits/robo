# v152/v153 — 관련 세라믹 record 확인과 별도 Ceramic Cracks 원저자 검토

v151 원문/관찰/목록·UNASSIGNED 보호·최신 보관 확인을 먼저 읽었다. A6파일12GiB 삭제는 이전 완료/승인소진이며 B_C와 추가 삭제는 미승인이다. v8~v151 완료·실패중단 helper/fixture/HTTP를 재실행하지 않았다. 두 기존 로컬 보관 검토문서는 미추적/원격전송0으로 유지한다.

## v152 — 아직 읽지 않았던 관련 record

`wmptdnw356` 이력검색은 정상 exit0으로 **v150/v151의 링크·미검토 메모만** 확인했다. 이를 새 source 정상 미일치로 보고하지 않는다. 새 정확DOI `10.17632/wmptdnw356.1` 검색은 정상 exit1이었다. 그 뒤 원배포가 연결한 [Surface defect dataset for sanitary ceramics](https://data.mendeley.com/datasets/wmptdnw356/1)의 공개 webtext 한 개만 읽었다.

표기 Version1/2025-07-28·contributor **Murat Çöpoğlu 한 명**·data CC BY-NC3.0을 보존한다. v150의 세 contributor와 섞지 않는다. 원사진106/결함67/정상39·green weakpaint/thresholdmask·회전/512가공/18560patch·TRAIN_VAL_TEST 설명은 v150에서 관찰한 선언과 일치한다. 실제 file/archive/사진 동일 SHA 비교나 동일 원본·버전 순서·새 독립사진 증명은 아니다. 날짜·제목V2 유무로 두 record를 V1/V2로 확정하지 않는다.

검토 text에서는 원본↔paint/mask↔회전/patch의 파일별 대응·원TEST/VALHELD·촬영물리그룹·nativeGT_unknown/미주석·명시배경/coords를 해소할 새 계약을 확보하지 못했다. 전체 저자 자료에 없다는 증명은 아니다. 두 record/가능한 사본의 UNASSIGNED 보호와 취득 전 보류를 유지한다. 같은 page/API/download/검색을 새 명확한 원저자 계약 없이 반복하지 않는다. 이 related page는 재조회가 아니라 이번 최초 검토였고 API/root-list/관련 링크·archive 취득은 0이다.

## v153 — 별도 Ceramic Cracks

정확 논문제목·DOI·원repo와 확인 저자 이력은 실제 report/script/config/dataset/source/outputs 텍스트에서 각각 정상 exit1이었다. v150의 일반 탐색에 노출됐던 후보를 새로 선별했으며 같은 데이터셋으로 세지 않는다.

[원repo](https://github.com/gerivansantos/ceramic-cracks-dataset)의 master 공개 root/README는 제목과 train/test 디렉터리를 보여준다. [공식 논문 색인](https://www.mdpi.com/2076-3417/11/13/6017) **Ceramic Cracks Segmentation with Deep Learning**, DOI `10.3390/app11136017`, 2021-06-28, Gerivan Santos Junior / Janderson Ferreira / Cristian Millán-Arias / Ramiro Daniel / Alberto Casado Junior / Bruno J. T. Fernandes를 확인했다. direct publisher body는 요청하지 않았다.

원저자는 건물 외벽 ceramic균열 RGB167/256×256, 수작업 GT의 균열 white·나머지 black, 연구의 random70% TRAIN/30% TEST와 관련 crack분할 후속 연구의 public use를 설명한다. 이는 유용한 원저자 선언이며 mask부재/bbox-only로 오인하지 않는다. repo의 실제 파일별 split/GT/unknown·독립 건물/촬영group·checksum/바이트/immutable revision은 미검증이다. GT 설명과 모델 출력 threshold를 구별하며 전처리/예측/loss·모델점수를 원GT나 SafeLog protocol로 채택하지 않는다. 256선언·그림 caption의 resize와 모델입력 크기로 실제 원촬영 해상도·좌표 배율을 추정하지 않는다.

public-use 안내의 연구 범위를 보존하되 검토한 root/색인만으로 data고유 상세허가·제삼자권리·상업앱 사용을 확정하지 않았다. root목록에 LICENSE가 안 보이는 것을 모든 자료의 허가 부재 증명으로 쓰지 않는다. article/code 권리를 dataset 조건으로 바꾸지 않고 별도 source의 CC BY-NC3.0도 전용하지 않는다. master/commit수/논문 날짜를 datarevision·촬영일로 쓰지 않는다. train/test 이름은 실사진수·그룹·최종FIT_VAL_CAL 증명이 아니다. 알려진 test 경로/가능한 사본은 보호하고 나머지도 UNASSIGNED 유지한다. **취득 전 보류/readyfalse**이며 전체 GT/정상자료 부재 주장은 아니다. 외벽 metadata를 산업기계 전체건강·새 건물검사 과제로 자동 확대하지 않는다.

## 이번 실행 범위

원저자 공개 webtext만 읽었으며 publisher 고정 rawHTML/API/PDF/code/weights·사진/mask/annotation/archive 로컬 취득0, web 전체 traffic bytes 미측정/전체HTTP0 주장0이다. 새 직접 HTTP 오류0/재시도·대체endpoint/mirror0, train/test 폴더·history/raw/API·그림/PDF/reference/관련논문·model 링크 추적0, 자동연락/login/동의/PII/우회0이다. sourcepixel/시각QA/raster/features/우리score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0, 새 dataset fetch/decode/training helper·unit·fixture·정식 취득/pixel/split/model protocol0이다. 관찰 메모는 `data/industrial-source-v152-v153-metadata/primary-web-observations.json`에 별도 보관하며 publisher 원문/자료checksum이 아니다. 기록·host/Git PowerShell은 datasetjob이 아니다.

이전 v151 rawmetadata736bytes와 v127 JPG1mask1/v126 JSON1/v141ref/v142v143text2 등의 실제 취득, 모든 원형·부분·실패·epoch/log/cache·GT/역할/graph·모델/앱은 보존한다. 보호10 최종 SHA·freshhost·Git/재개 영수증은 outputs에 기록한다. 이번 두 검토는 새 학습 실패·성능 개선이 아니며 사용자에게 필요한 조치나 학습준비 상태 변화가 없어 정기 알림은 조용히 유지한다.
