# v154 — Severstal 공식 공개 안내 검토, 동의·접근 조건으로 취득 전 보류

2026-10-19 한국시간. 직전 v152/v153 보고서 2개는 사용자의 구체적 승인으로 `8595595dd8fe32b289992ab051417420b3608310`까지 일반 push 및 원격 확인이 완료됐다. 그 승인을 이번 보고서나 후속 자료의 전송 허가로 재사용하지 않는다. 이번 결과는 로컬에 보관한다.

## 최초 검토 범위

기존 실제 report/script/config/dataset/output 텍스트에서 `Severstal: Steel Defect Detection`, URL 식별자 `severstal-steel-defect-detection`, `Severstal`을 정상 exit1로 확인한 뒤 공식 Kaggle 검색 색인 text만 검토했다. 확인된 인용 이름 Alexey Grishin / BorisV / iBardintsev도 별도 정상 exit1이었다. 공식 citation의 나머지 표기 inversion / Oleg를 그대로 보존한다. host Severstal과 citation 표기를 구별하며 개인별 원사진 저작권자를 추정하지 않는다. 자료 DOI나 immutable dataset revision은 확인하지 않았다.

- [공식 데이터 안내](https://www.kaggle.com/c/severstal-steel-defect-detection/data): 철강 제조 결함 위치·종류, 결함이 없는 이미지 및 단일/복수 클래스 가능성, ClassId 1~4의 분할 주석과 `train_images/`, `train.csv`, 별도 `test_images/`를 안내한다. 여러 떨어진 결함 영역도 클래스별 한 행으로 표현한다. 이 안내는 유용한 원배포 선언이며 bbox-only나 GT 부재로 해석하지 않는다.
- [공식 개요](https://www.kaggle.com/c/severstal-steel-defect-detection/overview): 평판 강재 생산·고주파 카메라의 검사 맥락과 2019 대회 인용을 안내한다. 대회 날짜는 촬영일·datarevision이 아니다. 대회 Dice 평가나 제출 실행 조건을 SafeLog 수용 기준으로 채택하지 않는다.

공식 data 색인은 자료 접근에 로그인·대회 규칙 동의를 요구하고, 라이선스를 대회 규칙에 종속시킨다. **취득 전 보류 / readyfalse**다. 로그인·규칙 동의·참가·연락·PII 제출·자격정보 사용을 하지 않았다. rules 페이지나 다운로드·API·Kernels·모델·제삼자 mirror/변환본도 열지 않았다. 공개 색인으로 모든 접근의 영구 불가를 주장하지 않는다.

실제 data 고유 허가와 제삼자 권리, 자연발생 RGB 여부·native GT 값/unknown/미주석/명시 배경·좌표 해독, 원파일 역할·물리 강판/촬영 묶음·source-oldheld 전체 graph, 파일별 bytes/checksum·작은 익명 접근·고정 최소·full 저장/임시/run 예산은 통과하지 않았다. 안내의 무결함 가능성을 파일별 검증 정상으로, train/test 폴더를 최종 FIT/VAL/CAL이나 물리 독립으로 승격하지 않는다. 제삼자 검색 결과의 수량·회색조/해상도·허가·코드/성능 설명은 계약으로 채택하지 않았다.

## 실행·보존

새 dataset fetch/decode/training helper·unit·fixture·정식 취득/pixel/split/model protocol은 0이다. 원사진·mask·annotation CSV·archive·PDF·API/rawHTML·외부 code/weights의 로컬 취득과 source pixel/시각 QA/raster/features/우리 score/gradient/optimizer/CAL_TEST/모델 API_CPU_GPU_CPP_ZIP_APK는 0이다. web 검색의 traffic bytes는 미측정이며 전체 HTTP 0이라고 하지 않는다. 새 직접 publisher body 요청·HTTP 오류·재시도·alternate 접근은 0이다. 관찰 메모는 web 요약이며 publisher 원문이나 dataset checksum이 아니다.

초기 행정 확인은 제한된 CIM 조회 오류 후 inline 명령이 계속되고 검색 roots도 비어 있었으므로 그 snapshot과 exit1을 gate 통과나 미일치 근거로 쓰지 않았다. 원형을 보존하고, 엄격한 중단 처리·존재하는 경로·허용된 읽기 전용 CIM으로 새 정상 snapshot/search만 완료했다. dataset HTTP/helper/학습 재실행이나 모델 오류가 아니다.

기존 source 원형·부분·실패·epoch/log/cache·GT/역할/graph·보호모델/패키지를 보존한다. A 6개 사본 12GiB 삭제는 이전 완료/승인 소진이며 추가 삭제 0이다. 이번 공개 안내 검토는 학습 수행이나 성능 개선이 아니다. 새 명확한 원저자 공개 release·독립적인 접근/이용 근거 없이 같은 Severstal 페이지·검색·rules·API·mirror·코드·모델·다운로드를 반복하지 않는다.
