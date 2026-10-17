# v111–v112 원저자 자료 접근 범위 검토

2026-10-17 UTC. 실제 repository report/doc/source에서 정확 이름·제목·저자의 기존 기록을 검색한 뒤 두 후보를 순서대로 확인했다. v8–v110 완료단계는 재실행하지 않았다.

## v111 SpotRust

[원저자 저장소 README](https://github.com/StuvX/SpotRust)는 산업 파트너 제한으로 논문에 사용한 dataset을 제공할 수 없다고 명시한다. 공개 코드나 pretrained model이 실제 TRAIN 사진·GT 공개를 뜻하지 않는다. software 허락을 data 권리로 대체하지 않고 **취득 전 보류**했다. 모델 link를 열거나 코드/weight를 취득·실행하지 않았다. 실제 주석·원분리·촬영/부품묶음·파일목록·checksum은 미검사다.

## v112 화물 컨테이너 부식 자료

David Ornelas·Daniel Canedo·António J. R. Neves의 [원논문](https://www.mdpi.com/2071-1050/17/14/6480), DOI10.3390/su17146480의 공식 색인텍스트는 컨테이너 사진500장(부식250/비부식250)과 CVAT pixel 주석을 설명한다. 이는 확보·검증 사진 수가 아니다. Data Availability는 교신저자 요청형이며 privacy 사유로 비공개라고 명시한다. 원data 권리·익명 공개 file·native GT·원분리·물리그룹·무결성 검증 전 **취득 전 보류**한다.

직접 publisher 조회는429였으며 재시도/대체접근은 하지 않았다. 이 결과를 전체 익명접근 영구불가 증명으로 확대하지 않는다. 근거는 공식 publisher 색인텍스트이며 로컬 고정 원문이 아니다. 자동 연락·로그인·PII 대리제출·mirror 취득·접근우회는0이다.

## 실제 수행 범위

`data/industrial-source-v111-v112-metadata/primary-web-screens.json`은 웹 관찰메모이며 publisher 원문 SHA나 archive 검증이 아니다. 새source 사진/mask/archive/PDF/code/weight 취득·pixel/내용QA·encoder/점수·gradient/optimizer·CAL/TEST·API/CPU/GPU/CPP/ZIP/APK 모두0이다. 취득·분리·학습 protocol도 새로 등록하지 않았다. 실제학습 실패나 성능 개선으로 설명하지 않는다.

기존 모델·원GT/TEST·역할·graph·원본·실패·epoch·로그/cache/ZIP·앱 출력을 보존한다. 추적 완료 연구 optimizer 부분합60430은 그대로이며 전체 역사 총량이 아니다. 후속은 `INDUSTRIAL_SOURCE_AFTER_V112_NEXT_KO.md`를 따른다.
