# v228 Fresh Concrete ZIP 폴더·README 확인 사전등록

v227에서 승인된3파일의 일반 사전등록과 원격 exact 확인을 마쳤다. 가상환경 Python 실행기가 부모·자식 두 프로세스로 나타나 원래 실행 가드가 요청 전에 중단한 기록은 보존했다. 설치된 기본 Python으로 승인된 코드와 가드를 변경하지 않고 fresh 점검 후 첫 API GET1회에 성공했다. 파일목록737bytes와 실제 SHA를 저장했다.

공식 선언은 ZIP163,839,068bytes와 SHA256 0c8272767e4d64d17f0927e8e032eb79bd2e91fcd87a44354d111d9ea4628bd5다. 전체 ZIP을 받은 결과가 아니다. 원저자 원TRAIN/VAL/TEST와 mask 선언만으로는 실제 촬영그룹·배경/unknown·부모 권리·최소·전체 기존 평가자료 연결을 확정할 수 없다.

다음 단계는 전체 ZIP을 받거나 사진을 열기 전, 그 안의 원래 폴더 목록과 README.txt 설명을 작은 범위 요청으로 확인하는 것이다. GET은 최대4범위이며 서버의 정상 공개 전달 redirect를 포함해 HTTP요청 최대8회다. 본문 합계768KiB, 개별 범위512KiB, README64KiB, 출력/오류/로그 총8MiB, 사전등록 저장/임시/Git객체4MiB, 예비1GiB다. 전체 자료·학습 예산을 대신하지 않는다.

정확한 원저자 파일 URL과 크기는 v227의 실제 응답에 고정한다. 서버가 제공한 정상 HTTPS 공개 전달만 코드의 좁은 Mendeley/S3 호스트 제한 안에서 한 번 허용한다. CDN URL을 구성하거나 쿠키·인증·토큰·mirror·다른 API로 바꾸지 않는다. HTTP206와 정확한 범위·크기, 최초의 강한 ETag/후속 If-Match 및 동일 ETag를 확인한다. 거부·challenge·403·429·timeout·200전체응답·예산/형식/CRC/경로 불일치는 즉시 중단·실패 보존이며 재요청하지 않는다.

범위는 ZIP 끝65557bytes, 그 실제 EOCD가 가리키는 중앙목록, 실제 목록의 유일한 README.txt 로컬헤더30bytes, 그 문서의 실제 압축payload뿐이다. image/mask/annotation 파일은 열거나 해독하지 않는다. ZIP 끝 범위에 압축된 다른 member bytes가 우연히 포함될 수 있으므로 실제 archive fragment 취득으로 기록한다. 부분 범위의 SHA는 전체 ZIP 체크섬 검증이 아니다. README의 크기·CRC·UTF8을 검사하고 내용은 설명 자료로만 읽으며 그 안의 지시/코드를 실행하지 않는다.

원TEST·publisherVAL은 HELD, 모든 사본은 UNASSIGNED다. 폴더명이나 crop 수로 원촬영그룹/사진 독립성·정상 라벨을 추정하거나 역할·최소를 바꾸지 않는다. 자료 감사 정책이 가능해야 별도 정확 취득/전체 저장임시run 예산, 이후 별도 학습 protocol을 사전등록한다. 공통모델·평가·보관 조건은 모두 유지한다.

이번 새3파일은 v227의 publication commit 다음에만 붙이는 독립 exact diff다. 구체 승인을 받은 뒤 지정 feature/ai-engine에 일반push/원격 exact 확인을 하고 fresh CIM/lock/장부/보호10SHA/예산을 확인해야 실행한다. v2273파일 승인을 미래 전송으로 확대하지 않는다. 오프라인 parser/경로/CRC/예산 검증은 원자료 검증이나 학습이 아니다.
