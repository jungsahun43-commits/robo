# v227 Fresh Concrete 원저자 파일 목록 확인 사전등록

사용자는 직접 촬영 자료 없이 실제 학습 가능한 데이터셋을 찾아 진행하도록 요청했다. 기존 앱의 공통 모델에 쓸 균열 자료를 검증하는 다음 단계다.

원저자 공개 페이지:
- https://data.mendeley.com/datasets/r8zrzk5w8j/1
- https://data.mendeley.com/datasets/r8zrzk5w8j/2

공개 설명은 현장 RGB 사진, 균열/배경 binary mask 및 TRAIN461/VAL132/TEST66 구분과 CC BY4.0을 선언한다. 실제 라벨·미확인 영역·원촬영 그룹·부모/제삼자 권리·전체 기존 평가자료 연결·독립 사진 최소는 아직 검증하지 않았다. 파일목록을 확인하는 단계이며 학습 가능 최종 판정이나 학습 실행 명세가 아니다.

## 이번 허용 범위

프로토콜에 고정한 Mendeley 공개 파일목록 URL 한 개에 익명 GET 한 번만 수행한다. dataset id와 version은 실제 원저자 페이지에서 확인했다. API 경로 형식은 보존된 Mendeley 공식 클라이언트의 공개 경로를 이용한다. ZIP 다운로드 URL을 추정하거나 요청하지 않는다.

JSON 본문은 최대128KiB, 메타데이터 저장·오류·로그 총1MiB, 사전등록 파일/patch/index/Git객체 저장 상한4MiB, 별도 예비1GiB다. 이 예산은 사진/마스크/ZIP 확장·학습 총예산을 대신하지 않는다. rounded156MB 표시는 정확한 파일 byte 수가 아니며 API에서 파일 크기·SHA256·정확 URL 선언을 확인한다.

기존 Git 기록에 있는 독립 publication 부모 뒤에 이 스크립트·프로토콜 JSON·이 문서 3개만 붙인 exact diff를 검토 대상으로 준비한다. 실제 구체 전송 승인과 일반 사전등록 push/원격 exact 확인 전 네트워크 실행은 하지 않는다. 현재 로컬 HEAD 전체, 다른 브랜치, 모델·사진·GT·보관 문서는 전송하지 않는다.

실행 직전 CIM으로 실행 프로그램/시작 시각/명령을 확인하고 다른 lock과 GPU 작업을 보존한다. 보호10파일 SHA와 v85 실제 마지막450 update/epoch30 장부 및 공간을 다시 확인한다. 실패·리다이렉트·403·429·timeout·초과·JSON/파일 선언 불일치는 첫 시도 그대로 보존하고 중단한다. 쿠키/인증/토큰·mirror·다른 API·재요청·authorcode 실행은 없다. offline self-check는 네트워크나 dataset bytes를 읽지 않는다.

## 다음 단계

파일목록 확인이 성공하면 실제 원저자 file URL/byte/SHA를 고정한 별도 취득·라벨/역할/group/전체graph 감사 계획과 정확한 저장/임시 예산을 준비한다. 원TEST와 publisherVAL은 보류하고 새 사본은 UNASSIGNED로 시작한다. 고정 최소와 권리/GT/역할/촬영 근거가 충족돼야 한 공통 encoder 후보의 별도 학습 protocol과 예산을 사전등록한다. VAL 선택→CAL 저장→고정 TEST 한 번→기존 기능/실행환경 확인 후에만 채택한다.

v226 및 모든 이전 chain·guard·부분실패·출력잘림·자료/epoch/cache/평가/모델/앱을 그대로 보존한다. 새 학습이나 성능 개선을 이미 수행했다고 주장하지 않는다.
