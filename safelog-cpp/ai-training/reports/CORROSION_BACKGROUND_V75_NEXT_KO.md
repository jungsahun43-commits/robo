# v75: 배경을 직접 주석한 자료 확보, 다음은 좌표·분리 검증

## 원저자 출처와 확보 범위

[Mendeley Data V1](https://data.mendeley.com/datasets/kcyn4nhv2c/1), DOI 10.17632/kcyn4nhv2c.1. Marina Rios, Paulo Ivson Netto, Deane Roehl, 2023. 공식 데이터 조건은 CC BY 4.0이며 원저자·출처·버전·변경을 표시한다. 별도 제삼자 권리가 표시된 내용은 추가 확인한다. 기존 자료 이용조건은 바꾸지 않는다.

원저자 설명은 전체 1,024사진 중 35사진에 부식/배경 각각 다섯 5×5 patch를 주석했다는 것이다. 전체 binary mask나 전체 정상 사진으로 확대하지 않는다. 실제 patch 형식·좌표·의미와 물리부품/촬영 독립성은 아직 미검사다.

`1560dc8`에 취득 조건을 먼저 일반 push한 뒤 **Labeled images.zip만** 받았다. 262,967,506bytes, 배포자 SHA256 `6fbd29524a81f4d8250a3cf978da09ad09d48f65522677792156d0bdf454fcce`와 로컬 값이 일치한다. 안전 경로·전체 outer ZIP 멤버 CRC/encoded SHA를 검사했다. 151멤버/선언 확장 434,641,796bytes지만 전체 확장은 하지 않았다. 배포자의 Complete database.zip 2,211,263,435bytes는 받지 않았다.

원본 사진·주석의 실제 픽셀/좌표 내용 해독, 시각 열람, 특징/모델 점수/gradient/optimizer는 모두 0이다. README/외부 원저자 코드를 실행하지 않았다.

## 파일 이름으로 확인한 주의점

- Original images에는 **34 TIFF**가 있다.
- Background 및 Corrosion에는 각각 **35개의 `.tif.zip`**이 있다. 내부 ZIP의 목록/형식/주석 좌표는 아직 읽지 않았다.
- Label visualization에는 **36 TIFF**가 있으며 `selectionImag203-1.tif` 추가 사본이 있다. 이들은 원본 입력 또는 새 자연 사진으로 세지 않는다.
- `Imag83`의 양쪽 주석 ZIP과 visualization은 있으나 **Original images/Imag83.tif는 없다.** 다른 파일을 그 원본으로 간주하거나 visualization에서 표시를 지워 clean GT를 만들지 않는다. 숫자 파일명만으로 물리부품을 추정하지 않는다.
- desktop.ini 6개는 데이터/정답이 아니다. 이 파일 수준 불일치는 35개 정답 전체가 틀렸다는 증명이 아니다.

## v76에서 먼저 고정할 내용

아직 v76 코드·정식 자료 정책·TEST 예약·최종 FIT/VAL/CAL·학습 조건은 없다. 다음 순서에서 픽셀/주석 해독 전에 코드 JSON을 고정·commit/push한다.

1. 이름/SHA로 원본·양쪽 주석·visualization·사본 관계와 원저자 분리 유무를 기록한다. 원TEST가 없으면 하나의 결정적 TEST-ID 예약 규칙을 먼저 선언한다. 새 연결 후 TEST와 연결된 TRAIN은 제외하며 원TEST를 옮기지 않는다.
2. 누락83은 제외 대상으로 고정하거나, 원저자 전체 배포본의 **공개 선택 접근이 가능한지 먼저 검증**한다. 실제 URL은 보존된 공식 파일 목록에 있다. Range/무결성/임시량 조건을 먼저 고정하고 가능하다고 가정하지 않는다. 현재 여유에서 2.2GB 전체 다운로드·기존자료 삭제는 하지 않는다. 확보 불가면 누락 상태를 보존하고 조건을 완화하지 않는다.
3. nested ZIP은 안전경로/개별CRC/encodedSHA/상한부터 검사한다. 실제 형식이 ImageJ ROI 등인지 확인한 뒤 해당 공식 좌표/사각형 경계/버전 의미를 확인한다. 확실하지 않은 좌표·라벨은 추정 복구하지 않는다.
4. TIFF 단일/복수 frame, RGB 보기, EXIF/크기/좌표 대응·에러 중단 조건을 고정한다. 실제 주석 지원만 valid positive/negative로 쓰고 나머지는 unknown255다. 작은 patch를 전체 녹 없음/장비 정상 정답으로 만들지 않는다. 주석 시각화는 학습 입력에서 제외한다.
5. 기존 완료된 old8995/InsPLAD6417/새6481/전이 graph와 Ostrava105/v66/v68 지문·역할을 재사용한다. 새 원본 보기만 exact RGB/8방향 pHash2로 연결하고 모든 현재 TEST 연결·기존 역할 충돌을 제외한다. 기존 전체 해독/거리 감사를 다시 돌리지 않는다.
6. 남은 원저자 원본과 연결 그룹에서만 한 가지 배정·최소 사진/그룹/양성·음성 patch와 512 유효 픽셀 조건을 **결과 전에** 고정한다. 표본/좌표 미달이면 학습 전 중단한다. 픽셀 QA는 고정 첫 FIT만, VAL/CAL/TEST 내용 선택/원 GT 수정은 하지 않는다.
7. 이 자료가 준비되더라도 새 학습을 자동 시작하지 않는다. 작은 patch가 512 전체 축소에서 희석될 수 있으므로 입력 좌표/샘플링 가설 하나를 먼저 구체화한다. 추론 crop을 TEST 정답으로 고르지 않는다. 새 encoder/source/category embedding 없이 같은 모델의 분기 하나, 기존 v16/승인 세 head/v62 기준 전체 exact 보존, loss/회차/lr/VAL/CAL/수용·retention/장치/delta/저장량을 별도 사전 push한다. 이전 작은 mask 전이·동일 혼합비·TEST cut 재탐색을 반복하지 않는다.

v73은 실제 1,920회 후 전체 제외, v74는 VAL-only 진단 완료다. VT unknown을 음성으로 바꾸지 않고 기존 승인 ZIP을 유지한다. source의 양성 recall만으로 앱 전체사진 FP/FN 20%나 현장 독립성을 주장하지 않는다. 모든 원본·실패·각 epoch·로그·캐시·ZIP을 보존하고 ≥1GiB reserve, bounded RAM/online features, 작은 parent-bound delta 원칙을 지킨다.
