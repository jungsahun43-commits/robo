# v136 — Motor Commutator Surface Defect Dataset 취득 전 보류

기존 repo의 정확명칭/확인 저자·논문제목·DOI/URL 텍스트 검색은 정상 exit1 미일치였다. 초기 MCSD 약어 검색에서 나온 언어모델/의료/현저성·micro-capacitor 자료는 이 motor-commutator 자료와 연결하지 않았다. SteelSeg/CorroNet/Rust3D/RustNet은 이름 이력 검색만 했고 외부검토/새 source 단계로 세지 않았다. 모든 과거 commit/ignored source 원문을 감사한 것은 아니다. 초기 AGENTS 탐색의 존재하지 않는 `.agents` 경로 오류를 source 이력 미일치로 쓰지 않았고 실제 ancestor/repo AGENTS 탐색은 별도 완료했다.

원저자 Yuanhong Qiu, Shuanlong Niu, Tongzhi Niu, Weifeng Li, Bin Li의 [Joint-Prior-Based Uneven Illumination Image Enhancement for Surface Defect Detection](https://www.mdpi.com/2073-8994/14/7/1473), Symmetry14(7)/1473, DOI10.3390/sym14071473/2022-07-19의 공식 publisher 검색 색인을 검토했다. 직접 같은 URL의 body 조회 한 번은 HTTP429였고 재시도·다른 endpoint/PDF/mirror0이다. 이후 새로운 정확제목 검색은 공식 도메인 색인의 서지/공개 text만 읽었으며 실패 body를 취득한 것이 아니다.

해당 저자 text는 실제 생산라인에서 수집한 MCSD 이미지1420/256×256, LabelMe로 만든 분할 GT, TRAIN994/TEST426을 선언한다. 이 수량은 실제 검증 사진·독립 부품/촬영그룹·고유 원TRAIN 수가 아니다. mask 부재/bbox-only로 오인하지 않는다. 원자료의 RGB/nativeGT 값·unknown/미주석·명시 정상배경·좌표/export revision·파일별 원역할/물리 촬영묶음·data 고유 허가/제삼자 권리·release/목록/bytes/checksum·작은 익명 접근/보존 예산은 미검증이다.

공식 색인의 Data Availability Statement는 실험 자료에 관해 제1저자에게 문의하도록 안내한다. 공개 원자료 취득의 필수 gate를 충족하지 못해 **취득 전 보류**했다. 이것은 모든 자료/유효 GT/허가의 부재나 영구 익명 접근 불가 증명이 아니다. 논문/기관 조건을 dataset 허가로 쓰지 않는다. 알고리즘의 background prior를 원GT의 음성/unknown 계약으로 쓰거나 enhancement·Gaussian noise/모델 출력·성능/회차를 SafeLog data/protocol로 채택하지 않았다. motor-commutator 표면 metadata를 볼스크루/일반나사·산업설비 전체건강이나 새 검사과제로 자동 추가하지 않는다.

PDF/그림·원사진/GT/영상/archive/API/rawHTML/code/model/weights 로컬 취득·원저자 코드 실행0, 자동 연락·로그인/동의·PII 제출·접근 우회0이다. 같은 검색/논문/publisher429·다른 endpoint/PDF/저장소를 새 명확한 원저자 public data release 근거 없이 반복하지 않는다. 기존 RSDDs source를 이 논문의 증강/분리로 다시 평가하지 않는다.

관찰기록 `data/industrial-source-v136-metadata/primary-web-observations.json`은 웹 관찰 메모이며 publisher 고정 원문/자료 checksum이 아니다. 웹 traffic bytes는 미측정이며 전체 HTTP0으로 설명하지 않는다. source pixel/시각QA/raster/features/우리 모델 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0, dataset fetch/decode/training helper0/새 unit0/정식 취득·pixel·split·model protocol0/readyfalse다. HTTP429 한 건은 학습 오류가 아니다. 기록 작성·host/Git PowerShell은 dataset job 수에 넣지 않는다. 모든 기존 원형/부분/실패/epoch/log/cache/보고서/ZIP·GT/역할/graph/보호 모델/앱 출력과 v127 JPG1+mask1/v126 JSON1 취득을 보존한다.
