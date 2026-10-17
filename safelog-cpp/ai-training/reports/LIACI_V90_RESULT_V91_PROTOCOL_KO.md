# LIACi v90 중단과 v91 기존 파일 수치 진단

v90은 `574dda50027c96955aba68974fbbdfffde9eeb39`에 사전 push/원격 확인한 뒤 실행했다. 정책 SHA256은 `4b89a9505f5491afb17afc311eb0be8a7c5d9c2ed9b5d9ebb0562087897d3954`이다. 고정 presetFIT 6장에 대응하는 42멤버를 공개 206 Range로 받았고 각 파일의 크기/CRC/encoded SHA를 검사했다. Range 합계 3,059,018 bytes, 원본 encoded 파일 합계 7,523,829 bytes이다. 압축 Range/헤더와 encoded 원본을 모두 보존했다. 전체 ZIP SHA나 모든 멤버 CRC를 검사한 것은 아니다.

수치 해독 중 `Provided mask vs COCO category presence differs`로 중단했다. `runs/industrial-liaci-coordinate-v90/failure-20261017T072000Z.json`, stdout/stderr와 원본을 보존한다. v90을 통과로 소급 변경하거나 동일 다운로드/완료 작업을 재실행하지 않는다. 픽셀 읽기 전에 수행한 합성 unit3 통과와 최초 JPEG 속성 fixture 오류는 별도 보존이다. GT를 수정하거나 학습·모델 점수·gradient·optimizer를 실행하지 않았다. source 시각 QA도 0이다.

## v91 사전 조건

`scripts/diagnose_liaci_bitmap.py`와 `industrial-liaci-v91-bitmap-policy.json`을 사전 일반 push한 뒤만 실행한다. 정책 SHA256은 `87a819a098f3286d2a457e8997e8293a800e4ab3b66dcaf549ee5d08d67d329e`이다. 기존 42파일 SHA와 v90 정책/실패/원 COCO SHA를 묶는다. 새 네트워크 취득은 0이며 같은 presetFIT 6장만 읽는다. 6JPEG/30BMP/6PNG의 원본 크기·프레임·EXIF 상한을 확인하고 native 배열·BMP DIB/depth/팔레트·색별 수량·원 COCO의 독립 Pillow reference 면적을 기록한다. 원본 RGB의 정확 SHA와 8방향 pHash는 향후 재사용한다. 최대 새 저장 8MiB와 1GiB reserve를 지킨다.

정확한 검정/흰색일 때 양쪽의 reference 차이를 기록하지만, 잘 맞는 polarity나 일부 샘플을 골라 정답으로 전환하지 않는다. Pillow가 원저자 exporter라는 증거도 없다. segmentation PNG 색을 임의의 클래스 표로 추정하지 않는다. 불명확한 크기·좌표·형식은 중단하며 resize/flip/색반전/GT 복구는 하지 않는다. v90 최초 실패 위치·그 전 decode 횟수는 기록된 layer 순서와 동일한 존재 검사로 재구성해 반복 해독/새 고유 사진을 구분한다.

모든 결과는 형식 진단이며 `ready_for_training=false`를 유지한다. 최종 좌표/의미 계약, 전체 1893원본의 전이적 연결·oldheld 제외·역할 충돌·최소 조건은 아직 미완료다. 원TEST191/14사본과 not_used332 제외, preset 역할, 기존 모든 GT/모델/출력/이용조건은 유지한다. 기존 전역 해독/거리감사, VAL/CAL/TEST 내용/특징/점수, gradient/optimizer, API/장치/C++/ZIP 검사는 모두 0이다. LIACi 데이터 CC BY-NC-SA4.0과 논문 CC BY4.0은 구별한다.
