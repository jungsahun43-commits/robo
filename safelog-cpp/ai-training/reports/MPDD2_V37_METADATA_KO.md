# v37 MPDD2 공개 목록 메타데이터 점검

원저자 README가 직접 연결한 공개 Google Drive 폴더에서 회색 브래킷과 rosette의 하위 목록만 읽었다. 로그인·브라우저 인증정보 사용·개인정보 제출·사진 다운로드·사진 해독·모델 실행은0이다. HTML은 자료로 해석하고 JavaScript/원배포 코드를 실행하지 않았다.

확인한 파일 목록은 다음과 같다. **파일 목록 수이며 완전한 원본 사진 수·주석 검증 수가 아니다.**

|공개 경로|현재 목록 수|
|---|---:|
|bracket_grey/train/normal|50|
|bracket_grey/test/normal|19|
|bracket_grey/test/missing_label|27|
|bracket_grey/test/parts_mismatch|38|
|rosette/train/normal|50|
|rosette/test/normal|13|
|rosette/test/anomalous|27|

TRAIN 목록이 각50개이므로 공개 페이지의 페이지 처리/전체 목록 여부를 확인하기 전에는 수량을 확정하지 않는다. 현재 확인된 정상 TEST19/13개만으로는 유형별20개 최소조건을 충족했다고 말할 수 없다. 사후에 publisher TRAIN을 TEST로 옮기거나 두 유형의 정상 수를 합쳐 각 유형의 통과로 주장하지 않는다. 정확한 전체 목록·파일 크기·해상도·라벨 일치·현재 TEST와 중복·촬영 묶음·최종분리는 아직 미검사다.

회색 브래킷을 기존 검정 브래킷 라벨과 합치지 않는다. missing_label/parts_mismatch라는 디렉터리명을 위치 mask나 내부 기능 고장의 정답으로 확대하지 않는다. 이전 brown/white와 겹치지 않는다고 아직 가정하지 않는다. 실제 학습 조건도 미확정이다.

G-Links v35는 보수적 근접 그룹 때문에 학습 전 중단한 상태를 유지한다. v36 TRAIN만의 진단도 물리 부품 독립성이나 확정 중복을 증명하지 못했다. 두 자료 모두 현재 모델 성능 개선으로 반영하지 않았다. 기존 v16-r1과 모든 원본·실패·위치는 그대로다.

다음은 원저자 자료의 촬영/부품 ID 설명과 공개 목록 완전성을 확인해 **학습에 안전하게 쓸 실제 자료가 있는지 먼저 판단**한다. 원저자에게 연락하거나 접근 동의/개인정보를 제출하는 작업은 하지 않았다. 근거가 없으면 기존 TEST/근접 제외 규칙이나 최소표본을 낮춰 학습시키지 않는다.

출처: [MPDD2 원저자 저장소](https://github.com/stepanje/MPDD2), [원저자 LICENSE](https://github.com/stepanje/MPDD2/blob/main/LICENSE), DOI10.1109/ICUMT57764.2022.9943437. CC BY-NC-SA4.0과 인용/동일조건, 다른 기존 출처의 개별 조건을 유지한다.
