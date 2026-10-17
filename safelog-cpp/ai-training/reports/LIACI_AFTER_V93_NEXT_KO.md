# v93 다음 우선: 공식 주석 표현 근거와 저장량

v8–v93의 각 완료 연구/취득/수치 조사를 다시 실행하지 않는다. v90 직접BMP↔COCO 존재 계약과 v91 JPEG 한정 계약은 실패한 채 보존했고 v92 헤더/v93 나머지 수치 조사는 끝났다. v93은 학습 준비나 개선이 아니다. 새 학습/encoder/gradient/optimizer0이며 다음 v94는 제안뿐이다.

## 독립적으로 가능한 다음 확인

이미 보존한 원논문·공식 app metadata·COCO/각 파일 SHA를 재사용한다. 기존 6장과 모든원본/실패를 다시 다운로드/픽셀decode하지 않는다. 원저자 공개 문서/고정 revision의 변환 코드/공개 데이터 갱신 기록에 COCO와 categoryBMP/segmentationPNG의 생성·클래스·layer/overlap 처리 근거가 있는지 텍스트로만 확인한다. 원저자 코드를 실행하거나 notebook 저장outputs/TEST 예시를 열지 않는다. 제삼자 변환을 저자 수정 GT로 주장하거나 저자에게 자동 메시지/연락처/PII를 보내지 않는다.

ID107 COCO/BMP의 실제 의미 불일치와 원본1/3의 PNG signature는 구별한다. PNG를 원형 그대로 읽는 것은 좌표·GT 수정이 아니지만 mask 누락을 추정 복원할 수는 없다. 제공BMP 자체를 고치거나 좋은layer만 골라 준비통과로 바꾸지 않는다. 독립적인 원 COCO polygon 계약을 검토하려면 실제 공식 raster·좌표·겹침·미주석/unknown 의미와 사진 대응의 근거를 먼저 확보하고 별도 코드/JSON을 사전 push한다. Pillow reference가 저자 exporter라는 주장은 금지한다. GT 권위 관계가 확인되지 않으면 현 자료 학습을 시작하지 않는다.

공개 검색에서 SINTEF `LIACI_context`는 관련 contextualization 코드로 발견했지만 현재 dataset exporter라는 증거는 없다. 임의의 제삼자 mirror/모델출력·분류라벨·embedding을 pixelGT나 물리촬영ID로 대체하지 않는다. author similarity CSV/embedding이 있다고 물리 선박 독립성 증명으로 쓰지 않는다. 원 ZIP의 다른 비이미지 멤버는 similarity CSV와 ResNet embedding이며 미취득이다. 기존 CSV/COCO/중앙목록은 재취득하지 않는다.

## 저장량과 원분리 gate

v93 저장량 보고서상 준비만93,700,903bytes,256MiB run 포함362,136,359bytes 부족이며 mask 추가 비용 전이다. 실제 호스트 여유를 먼저 다시 계산한다. 예비1GiB를 침범하거나 기존 원본/실패/epoch/로그/캐시/ZIP을 삭제하지 않는다. 최소 약0.4GB, 실제로는1GB 이상의 추가 여유가 필요하다. fullZIP/전체확장/fullparent 복제/새dense파일을 시작하지 않는다. 공간 부족 중에는 공개 텍스트 근거·작은 metadata 조사만 가능하다. 사용자가 계속을 요청했으므로 단순 단계 완료 때문에 automation을 일시중지하지 않는다. 필수 조건을 기다리는 동안 변화가 없으면 조용히 유지한다.

전체sourceRGB/모든사본의 transitive fingerprint closure를 완료하기 전 미선택원본 bridge를 숨겨 readyFIT/독립성으로 부르지 않는다. old8995/InsPLAD6417평가·6481TEST/v57graph94/Ostrava105/VTv66·v68/Riosv78 완료 지문·graph·잠긴 역할을 재사용한다. 원TEST191/14사본,not_used332,기존Ostrava15/VT44/Rios7 TEST는 유지한다. 현재TEST/not_used연결TRAIN과 개발역할충돌은 제외하고 사전 v90 최소/역할을 이동·완화하지 않는다. 새 전체decode/거리/최종분리는 별도 사전 정책이 필요하다. 물리 촬영 mapping은 여전히 없다.

## 모델 및 자동후속 제한

새 모델 조건은 미확정이다. 기존 v16/승인3head/원v62 전 tensor-buffer-meta/정규화/기준/기본위치, 두 forward 계약을 exact보존하고 한공통encoder/새namespace 원칙을 유지한다. 자료 준비 후 하나의 구조/초기값/실제GT·unknown/입력/손실/sampling/회차/lr/VAL/CAL/수용·retention·장치/delta/저장량을 코드JSON에 사전 일반 push한 뒤만 특징/gradient를 실행한다. 기존 TEST에 맞춘 cut/구조/계수 반복, epoch0승격, 자동앱·alias·경고·지도교체는 금지한다. 과거 패키지 검사를 새 검사로 세지 않는다.

새 작업 전 CIM 명령줄/시작시각·lock PID 확인/GPU 하나씩, Windows os.kill(pid,0) 금지/read-only process_exists, 죽은 training.lock23948 보존, awakehelper 재시작·연장·전원계획 변경 금지다. 일반push는 HEAD:feature/ai-engine만/main·force금지/raw·weights·ZIP·images Git제외다. 모든 부모 데이터 조건과 LIACi CC BY-NC-SA4.0를 유지한다.
