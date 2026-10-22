# v176 KSDD2 고정 CAL·TEST 평가 사전 계획

상태: PREPARED / 새 exact3 전송 승인 전 HOLD. 기존 v175에서 선택한 delta 한 개를 평가한다. 준비 중 모델·사진·CAL/TEST·gradient·optimizer는0이다.

## 고정 후보와 기존 역할

- v175 완료: 기존 FIT1,400 / 내부 VAL441, 12epoch·실제768회, 선택6epoch(당시384회).
- runs/industrial-ksdd2-stream-v175/development-delta.pt: 368,717bytes / SHA256 444f8fbbd94bafa80f5ae777d51c86a6c84a490715f67fe0f67249143e8e06eb.
- v16 exact 부모·공통encoder·승인 v51 세 head·원v62·보호 sidecar를 유지한다. 새로운 학습·loss·구조·epoch·source·조합계수 후보가 없다.
- 기존 CAL490(442/48, proxy49그룹)과 원 publisher TEST1,004(894/110, proxy101그룹)를 고정한다. 원TRAIN 내부 CAL이며 기존 FIT/VAL/CAL/TEST 역할·동일 사본 group을 옮기지 않는다.

고정 manifest SHA256은 9d2f60b4bb9826c7a2211e69e2ff3056a3c19d83ea02173cdb71fb102df1235f다. 파일 SHA·원크기·원mask 양성 pixel 수·label·고정 사진/group/원좌표 pixel 총량이 다르면 중단한다. 오류 기록 제외로 수량을 맞추지 않는다. label당 사진20 이상·proxy그룹3 이상을 유지한다.

mask0은 기존 publisher 검사 대상 component의 명시 배경, >0은 기존 원손상 정답이다. padding은 무효, 기계 내부 건강은 unknown이다. 일반 부식·다른 부품 정상으로 확대하지 않는다. 기존 고정 mirror 취득/CC BY-NC-SA4 연구·인용·변경표시·동일조건·부모제삼자 권리 제한과 촬영 proxy 한계를 유지한다. 공식 원본 동일성·publisherchecksum·물리 독립·제삼자 권리 전체 통과나 새 source 수용을 주장하지 않는다. source-oldheld graph·GTunknown·원역할·지문을 바꾸지 않는다.

## 추론과 부모 비교

v175 그대로 RGB bicubic 종횡유지256×704·padding127·원 ImageNet정규화·low40/high960 FP16 roundtrip→FP32다. augmentation은 없다. 모든 모듈 eval / requires_grad false / inference_mode이며 optimizer·gradient0이다. 한 사진씩 처리하고 RGB/feature/map 전체 cache를 만들지 않는다.

후보 사진 점수는 유효 view sigmoid 상위32개 평균이다. view 확률을 고정 bilinear로 올리고 letterbox를 제거해 원좌표로 bilinear 복원한다. GT는 원mask 좌표 그대로 비교하며 보간·수정·내용 선택하지 않는다.

현재 부모 사진은 같은 encoder feature와 원 ksdd_head·기존 CAL threshold로 계산한다. 부모 primary pixel은 원 RGB256 square·원 forward_equipment 재구성 지도·기존 pixel threshold를 원좌표로 복원한다. 부모의 forward_surface(original,view,category) 및 RuntimeRealMaskMain.forward_real_mask 계약을 유지한다. 과거 v10/v11 성능 수치를 이번 측정으로 재사용하지 않고 같은 TEST pass에서 부모와 후보를 새로 비교한다.

## CAL에서만 정하는 threshold

기존 v10의 같은 규칙을 사전에 고정한다.

1. CAL490 사진 점수 고유값·최솟값 바로 아래1값·0.5를 threshold 후보로 삼는다. 판정은 score > threshold다. FPR/FNR 큰 값, 두 오류 합,0.5와 거리, threshold 값 순으로 최소를 택한다.
2. pixel threshold는0.1~0.9(0.1간격)다. 원좌표 CAL FNR≤0.15가 있으면 IoU 큰 순·FNR 작은 순·threshold 큰 순, 없으면 FNR 작은 순·IoU 큰 순·threshold 큰 순이다.
3. 선택 threshold·protocol/delta/부모stateSHA·CAL490/TEST0을 calibration.json에 exclusive write·flush·fsync·실제 readback한다.
4. TEST-start.json에 저장 CAL SHA를 exclusive 기록한 뒤 원TEST를 시작한다. 매 TEST 사진 전 같은 CAL SHA를 확인한다. TEST 후 구조·stride·계수·threshold·epoch를 재선택하지 않는다.

## 원TEST 한 번의 metric gate

고정된 후보 하나로 원TEST1,004를 id 순서로 한 번 처리한다. 같은 pass에서 부모 사진 점수와 원좌표 primary pixel도 비교한다.

- 사진 label당20 이상/group3 이상, FPR·FNR 각각20% 이하.
- 사진 FPR·FNR이 현재 부모보다 각각 악화0.
- 원좌표 pixel IoU는 부모보다 엄격히 커야 한다.
- 원좌표 pixel FNR·FPR이 부모보다 각각 악화0.
- 모든 부모 tensor/buffer/file/meta·보호 sidecar·추가 state SHA와 CAL 저장 순서가 정확해야 한다.

하나라도 실패하면 부모를 유지하고 이 후보를 닫는다. TEST를 본 뒤 photo-only로 바꾸거나 threshold·구조·수량·최소 조건을 완화해 다시 평가하지 않는다. 새 결과는 KSDD2 metric gate에 한정되며 브래킷/튜브 개선이나 독립 공장 검증이 아니다.

## 통과 뒤 별도 runtime·retention

이번 코드는 CAL→저장→고정TEST·부모 비교까지다. runtime 구현·패키지 생성은 포함하지 않는다. 결과는 항상 ready=false/deployed=false다. metric을 모두 통과한 후보만 다음 별도 구현·코드JSON·사전 조건·예산의 준비 대상으로 남긴다.

- v16+승인3head+원v62의 모든 tensor/buffer/meta/정규화/운영점/호출계약을 exact로 보존한다.
- 기존 고정 old TEST/VAL/CAL·지문/graph를 재사용해 모든 old task/alias 출력과 oldPNG를 실제 비교한다. 원TEST/publisherVAL HELD, 사본 동일group·개발역할 하나·held-notused 전이/역할충돌 제외·이동재분리/최소완화0을 유지한다.
- 실제 CPU/GPU 판정/원좌표지도·C++ operation·모든 oldalias/oldPNG·새 고정 ZIP을 새로 확인한다. 과거 검사 수를 이번 수로 재사용하지 않는다.
- 사진 label20/group3/FPFN각20%, 브래킷·튜브 label20/FN5%p감소/FP증가5%p이내·FP20%이하/n≥10유형 FN회귀5%p이내·모든 retention을 유지한다.
- metric만으로 전체 평가 완료·앱 채택·지도/primaryZIP 교체를 선언하지 않는다. 이 delta는 v16 부모가 필요한 개발 부가 state이며 단독 앱 모델이 아니다.

## 자원과 중단

새 run은 runs/industrial-ksdd2-stream-v176-evaluation이다. 출력32MiB·임시16MiB·예비1GiB·CPU peak working set4GiB·GPU allocated tensor4GiB·원사진 최대40,000,000pixel·최대7,200초다. 부모 파일·cache·source archive 복제/확장/다운로드0이다. 이 소규모 예산으로 새 source full취득/학습 예산을 대신하지 않는다.

실행 전 fresh CIM/PID/명령줄/시작/실제 부모launcher·lock·RAM/디스크·읽기 전용 GPU inventory를 확인한다. 승인 v175의 실제 launcher 식별 함수 원문을 그대로 복사했다. 확인 불가·다른 관련 작업/lock은 거부한다. Windows os.kill(pid,0), signal·프로세스 중지·외부 lock 삭제·awakehelper·전원·driverTDR·게임 설정 변경0이다. WDDM desktop context로 학습 중복을 단정하거나 전체 GPU/C++ idle 감사라고 주장하지 않는다.

기존 run이 있으면 시작하지 않는다. start/CAL/TEST journal·실패 수량과 원문을 보존하며 실패·중단·완료를 자동 resume/replay하지 않는다. 소유 pid/token lock만 종료 시 해제한다. 모든 과거 실취득·학습/부분실패·epoch·GT/역할graph·모델·앱·cache·ZIP을 보존한다. 새 삭제·압축0, NTFS5완료 반복0/A6삭제승인 소진/B_C 새exact삭제는 새 명시승인 필요다.

## 새 exact3 전송과 준비 검증

전송 대상은 scripts/evaluate_ksdd2_stream_v176.py, reports/industrial-ksdd2-stream-v176-evaluation-protocol.json, 이 계획서3개다. 기존 별도 publication lastverified d69c56271d87178e019bcff2509e24fb7b1644d2 부모에 exact commit/diff를 준비한다. 사진·GT·모델·raw·보관2문서는 전송하지 않는다. v173/v174/v175 exact3 승인은 각각 소진됐다.

기존 ‘실행 전 코드/조건 사전 push’와 ‘새 전송은 exact payload별도 승인’ 규칙 때문에 이 새3파일의 구체승인·일반 push·원격 exact 확인 후에만 모델·새 pixel·CAL/TEST를 실행한다. 전체 localHEAD/main/force/다른ref 우회0이다.

준비 검사는 syntax AST·실제3경로/script/reportSHA·고정 manifest 수량/역할/pixel metadata와 승인 v175 process guard 함수 동일성, 이미 쓰던6건 비관련 launcher 거부만 다룬다. 새 데이터 fixture·사진 해독·raster QA·모델/feature/score·CAL/TEST·gradient·optimizer0이다. 준비 검사는 성능 통과가 아니다. 모든 기존 원형 guard·실패·영수증·행정 초안·terminalLF·참조된 전체 chain을 보존한다.
