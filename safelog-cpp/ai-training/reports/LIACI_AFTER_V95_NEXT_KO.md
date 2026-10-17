# v95 다음: 불확실한 LIACi 학습은 보류, 새 근거만 검토

v8–v95 각각의 완료 단계는 재실행하지 않는다. v94/v95는 공식 관련 저장소의 고정 commit과8개 정적 텍스트 검토까지 완료했지만 현재LIACi COCO/BMP/PNG의 GT권위·생성 revision은 확정하지 못했다. 관련 ONNX inference/mosaic→COCO 코드는 원저자 주석 exporter라는 증거가 아니다. 같은 repo/tree/README/code/원논문/6사진·mask를 다시 취득하거나 이전수치·거리·학습을 반복하지 않는다.

## 재개 가능한 조건

새로 제시되거나 공개된 원저자의 명확한 주석 생산·수정 버전/좌표·라벨·unknown/사진대응 근거, 또는 저자 수정GT가 있으면 보존한SHA/원래실패와 별도 계약으로 검토한다. 기존 BMP를 자동수정/색반전/배율추정하거나 COCO를 근거 없이 정답으로 우선하지 않는다. author와의 자동연락/PII/동의 대리제출은 금지한다. 문서의 예시 inspection 이름/날짜·모델embedding·미취득모델을 사진-촬영 mapping이나 자연GT로 쓰지 않는다.

디스크 실제 여유를 재확인한다. v93 기본 전체지문 준비2,135,089,959bytes/256MiB run 포함2,403,525,415bytes가 필요하고 mask 추가 비용 전이다. 예비1GiB/모든원본보존을 유지하며 전체 취득이나 새 학습을 시작하지 않는다. 공간만 늘어도 GT가 자동 해결되지는 않는다. 새전체취득/지문/최종역할검증/모델은 별도 사전 코드JSON과 일반push 전 실행하지 않는다.

## 독립 가능한 다른 자료 우선순위

현재LIACi의 GT·공간 조건이 계속 미충족이면 동일 검색/문서조사를 되풀이하지 않는다. 아직 연구하지 않은 **원저자 실제산업손상 TRAIN 위치주석과 명시적 정상/배경지원** 자료를 한 후보씩 공개 메타데이터부터 검토할 수 있다. 기존 research/source 보고서를 먼저 검색해 v8–v95 완료·실패·이용조건 미해결 자료를 재탐색하지 않는다. 새 자료는 현재 확보사진0/학습0이며 미리 통과나 선택접근 가능성을 주장하지 않는다.

공식 데이터조건·저자/DOI/버전·원분리·물리촬영 묶음·원본/GT 대응·파일목록/체크섬·작은공개선택접근·임시량을 확인한다. 코드/논문 조건으로 data 조건을 바꾸거나 mirror를 원저자 배포로 꾸미지 않는다. 정상전체 사진·pixel배경·부분unknown의 GT 범위를 구분하고 분류/상자GT를 pixelmask로 바꾸지 않는다. 합성·제작손상·QA·자동제안·현재TEST를 자연TRAIN으로 승격하지 않는다. 새source 픽셀 전에 고정TEST예약/사진·group·validpixel최소/좌표·오류·exclusion/하나의역할/저장량 정책을 코드JSON에 사전push한다. 접근우회/로그인/연락처 공유를 하지 않는다.

v90 preset역할과 원TEST191/14사본/not_used332, 기존Ostrava15/VT44/Rios7/old8995/InsPLAD6417평가·6481TEST와 모든 완료지문·graph·역할은 보존한다. 새source의 현재TEST 연결TRAIN·기존역할충돌은 제외하고 미달시완화/이동하지 않는다. 기존전역감사·완료지문 해독/거리계산을 반복하지 않는다. 물리/촬영 정보가 없으면 proxy를독립증명으로부르지 않는다.

모든새학습은 한공통encoder/새namespace와 기존v16+승인3head+원v62 전체tensor-buffer-meta/기준·primary위치 exact보존, 두기존forward계약보존이다. GT준비뒤 한구조/초기값/학습대상·loss·sampling·회차/lr/VAL/CAL/수용·retention·장치/delta/예산을 사전push한다. epoch0승격/TEST cut·구조·계수반복/앱·alias·경고·지도자동교체는 금지한다. 과거검사수를 새CPU_GPU_CPP_ZIP검사로 세지 않는다.

사용자 계속승인에 따라 automation은ACTIVE를 유지한다. 필수조건 미충족으로 변화가 없을 때는 조용히 유지하고 동일실패/공간부족을 매번알리지 않는다. 필수 사람조치나 명시중단 없이 단순단계완료로 자동일시중지하지 않는다. 새실행전CIM 명령줄/시작시각·lockPID 대조/GPU하나,read-only process_exists/Windows os.kill금지,training.lock23948/awakehelper해제상태/모든원본·실패·로그·cache·ZIP을 보존한다. main/forcepush금지/HEAD:feature/ai-engine 일반push만,raw/weights/images/ZIP Git제외와 모든부모data조건을 유지한다.
