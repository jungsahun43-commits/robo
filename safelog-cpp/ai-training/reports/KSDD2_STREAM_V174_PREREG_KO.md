# v174 KSDD2 기존 TRAIN 스트리밍 학습 사전 계획

## v173 실행 전 중단과 별도 수정안

v173 정확3파일은 승인 후 `122cf556ce4ab46eb366fe36f15af3a532ee3f92`로 일반 push/원격 exact 확인을 완료했다(2026-10-22 UTC13:15:40). 실행은 중복 프로세스 검사에서 중단됐다. Windows venv 실행기와 실제 base Python이 두 프로세스로 나타나는 경우를 구분하지 못했다. torch/모델 로드·데이터 픽셀·gradient·optimizer·CAL/TEST는0이며 run 디렉터리와 operation.lock도 만들기 전이었다. stdout은 출력이 없어 파일이 생성되지 않았고 실제 stderr 원문은 보존했다. 없는 stdout을 stat하려던 최초 행정 기록 준비도 쓰기 전 실패한 원형 tool 출력 그대로 남겼으며 이를 학습 실패 횟수나 소급 통과로 바꾸지 않는다.

v173 코드/JSON/계획서/실패 로그는 수정하거나 재실행하지 않는다. v174는 이 실행 전 제어 부분을 바로잡은 별도 namespace/run이다. 학습 함수 본문은 v173 문자열을 v174 식별자로 바꾼 것 외에는 같음을 검사했다. 같은 고정자료/seed/손실/optimizer/12epoch/768회 조건이며 TEST에 근거한 모델 조정이 아니다.

자기 venv 실행기는 실제 직계 부모 PID, 현재 프로세스의 부모 PID 연결, 정확한 venv/base 실행 경로, Windows 방식으로 해석한 전체 인자와 실제 sys.orig_argv의 일치, 부모가0~5초 먼저 시작했다는 조건을 모두 확인할 때만 같은 호출로 인정한다. 하나라도 불명확하면 거부한다. Python 이름만 보고 제외하거나 다른 job/lock을 무시하지 않는다. 실제 CIM 부모/자식 확인과 관계없는 경우를 거부하는6건의 행정 제어 검사가 통과했다. 이 검사는 데이터 fixture·사진·torch·gradient가 아니다.

새 publication의 부모는 마지막 원격 exact `122cf556ce4ab46eb366fe36f15af3a532ee3f92`이며, 이번 v174 Python·JSON·계획서3파일은 별도 exact 전송 승인을 받아야 한다. 이전 v173 승인·원형 guard·실패를 재사용해 전송하거나 실행하지 않는다. 아래 학습 조건과 모든 공통 gate는 그대로 유지한다.


사용자가 2026-10-22 한국시간에 다른 자료로 계속 학습해 달라고 요청했다. 새 부식 자료 검색 대신, 이미 확보하고 역할을 고정한 KSDD2를 재사용하는 새 학습 후보 한 개를 준비한다. 이 문서는 학습 완료나 성능 개선 보고가 아니다.

## 자료와 목적

FIT 정상1,254/실제 결함146장, 내부 VAL 정상389/결함52장과 고정 그룹140/45를 그대로 쓴다. 원래 TRAIN에서 정해 둔 개발 역할만 사용한다. CAL490장과 원TEST1,004장은 읽거나 점수를 내지 않는다. 기존 GT·unknown/배경 해석·사진 및 그룹 분리·기존 모델을 바꾸지 않는다. 정상은 출판자가 검사한 부품 표면 범위이며 기계 전체의 건강을 뜻하지 않는다.

과거 취득은 고정 LFS mirror의 바이트였다. 공식 원본 archive와 동일하다는 검증 및 출판자 제공 checksum 검증은 하지 못했다. CC BY-NC-SA4.0 비상업 연구·원저자/출처 인용·변경 표시·동일조건과 제삼자 권리를 유지한다. 새 data release나 권리를 통과했다고 기록하지 않는다. 과거 고정 감사/이용 범위의 재사용이며, 물리 촬영 그룹·독립 현장 검증도 여전히 미확정이다. 원본 지문/역할·전체 기존 held graph를 재배정하거나 구제하지 않는다.

## 한 후보

기존 v16 공통 encoder와 모든 부모 분기는 고정한다. 기존 KSDD 판별부를 새 `ksdd_stream_v174` 이름으로 복사하고 fusion만 학습한다. high projection도 고정한다. v51 승인3head·원v62·보호 모델/ZIP은 원형 그대로 둔다. 별도 encoder와 가짜 source/category/date embedding을 추가하지 않으며 `forward_surface(original,view,category)`와 `RuntimeRealMaskMain.forward_real_mask` 계약을 유지한다.

기존 전체 유효영역 BCE 대신 각 사진의 실제 양성/명시 배경 픽셀 BCE 평균을0.5/0.5로 균형화하고 기존 Dice 및0.5 사진 BCE를 더한다. 정상 사진은 배경 BCE만 쓴다. 원래 좌표의 비율 유지256×704 입력·nearest GT·padding 제외·FP16 특징 경계는 유지한다. FIT RGB와 GT를 함께 H/V 반전해 실제 공통 encoder에서 차례로 계산한다. 기존 v10/v17의 스크립트나 완료 run을 다시 돌리지 않는다.

12epoch×64step=예정 실제 optimizer768회, effective batch8(4장씩2microbatch), seed173173, AdamW lr1e-4/wd1e-4/cosine최소1e-6/clip5/FP32/TF32off 한 가지 조건이다. 정상/결함별 그룹 크기 역수로 추출한다. 반복 입력/변형은 새 사진으로 세지 않는다. 고정 VAL의 정상/결함 loss를 균등 가중하여 완료 epoch1~12 중 가장 이른 최솟값을 고른다. epoch0는 비교용이며 선택에서 제외한다. 초기 VAL보다 개선되지 않으면 개발 산출물로만 보관한다.

## 저장·오류·평가

사진을 microbatch 단위로 읽고 특징은 메모리에서 처리한 뒤 버린다. 전체 feature cache·부모 파일 복제·새 archive 확장은 없다. 예비1GiB 외 새 run128MiB/임시32MiB/delta4MiB/CPU working set4GiB/GPU tensor4GiB/최대2시간으로 제한한다. 매 epoch 가중치·optimizer·RNG와 step별 실제 갱신 장부를 남긴다. 실패·부분 상태를 보존하고 자동 재실행/삭제를 하지 않는다. 새 run만 만들고 기존 lock을 보존하며 자기 token의 operation.lock만 종료 시 해제한다.

출력은 v16에 묶인 작은 추가 분기 delta다. parent_file/file/state SHA, added/combined state SHA와 직렬화 복원을 확인한다. 단독 앱 모델이나 primary/alias/지도/ZIP 교체가 아니다. 이 단계의 CAL/TEST/성능 수치는0이다. 후속 채택은 원래 고정 CAL/TEST1회·retention·사진별 FP/FN·표본/그룹 최소·CPU/GPU/C++/oldalias/oldPNG gate를 모두 유지한 별도 사전 계획 후에만 한다. 기존 공개 TEST를 이미 본 개발 이력이 있으므로 독립 산업현장 검증이라고 주장하지 않는다.

## 실행 전 전송 범위

이번 Python·JSON·이 문서 세 파일만 마지막 verifiedremote122cf556 부모의 별도 일반 commit으로 준비한다. 목적지는 `https://github.com/jungsahun43-commits/robo.git`의 `feature/ai-engine`이다. 현재 local HEAD 전체를 보내지 않는다. 사진·GT·가중치·개인 보관2문서·GPU 진단 내용은 전송 payload에서 제외한다. 이전 전송 승인은 소진되었으므로 exact diff 검토 가능한 새 승인 후에만 전송/원격 exact 확인을 하고 학습을 실행한다. 준비 중 compile/metadata 검사는 사진 해독·torch 모델 실행·gradient·optimizer가 아니다.

## 이전 공통 조건 원문

실제 rawmetadata/외부 statictextcode는 별도 exact URL/version/길이SHA상한/오류중단실패보존/실저장임시run 예산 codeJSON 일반 사전push와 원격 확인 전 취득0이다. 새 외부전송은 exact payload/diff/목적지를 검토 가능하게 준비한 뒤 필요한 구체승인을 한 번 요청한다. v162/v161 exact3와 v154~156 exact6 승인은 각각 완료/소진이다. 별도 publication 이력 verifiedremote18aa342는 현재 localHEAD ancestor가 아니다. 현재HEAD 전체push·단순ahead/현재원격일치 주장0이다. 후속 exact 승인 payload만 verifiedremote 부모 별도 commit으로 준비하며 raw/사진/GT/weights/모델/보관2문서는 Git에서 제외한다. main/force/다른ref/간접 우회0이다.

source 제삼자권리·원출처revision·자연RGB nativeGTunknown/배경/coords/export·원역할·물리촬영group·source-oldheld 전체graph·고정 사진/group/validpixel 최소·checksum·작은 익명접근·full 예산 미달 중 취득학습0이다. paper/code/mirror 권리대체, class/bbox/등급/objectmask/합성/제작손상/QA/모델출력/현재TEST의 자연pixelTRAIN 승격, 새과제·authorcode 실행·자동 연락/login/동의/PII 우회0이다.

새pixel 전 원TEST사본/publisherVAL HELD/한 원TRAIN 개발역할/고정 최소/GTunknowncoordsdecode 오류제외/source-oldheld graph/full 예산 별도 사전정책부터 준비한다. 기존8995TEST/InsPLAD6417평가6481TEST/v57graph94/Ostrava105/VTv66v68/Riosv78 잠긴 지문graph역할·Ostrava15/VT44/Rios7TEST를 재사용한다. 기존 전체 RGBpHash 재감사0, held-notused 전이TRAIN 역할충돌 제외, 사본 같은group, 개발역할 하나, 이동재분리·최소완화0, 첫FIT만 QA, held내용선택/GT수정0, 미검사bridge로 전체물리독립ready 주장0이다.

자료 통과 뒤 공통encoder 단일모델의 구조/초기값/GTunknown/대상/loss_sampling/회차lr/VAL선택/CAL/retention/장치/delta예산을 별도 사전push한다. v16+승인3head+원v62 모든 tensor/buffer/meta·정규화기준 primary exact와 forward_surface(original,view,category)·RuntimeRealMaskMain.forward_real_mask 계약을 보존한다. 별도encoder·가짜 source_date_category embedding0, v16 bound 작은 delta parent_file/state/added/combined SHA·serializedrestoreexact이며 단독 앱모델이 아니다. epoch0 제외, CAL저장 후 고정TEST1회, TEST계수/stride/구조/threshold 반복탐색·완화0이다. 사진 각 TESTlabel20/group3/FPFN각20%, 브래킷·튜브 각label20/FN5%p감소/FP증가5%p이내·FP20%이하/n>=10유형 FN회귀5%p이내/retention을 유지한다. 통과후보만 새고정ZIP CPU/GPU·실CPP·alloldalias·oldPNG를 실제 검증한다. 과거 검사수 재사용·앱alias 경고/primaryZIP 자동교체0이다.

새job 전 fresh CIM 명령줄/시작시각/lockPID/실제장부/예비1GiB+정확 단계 저장/임시/run 예산을 확인한다. read-only process_exists만 사용하며 Windows os.kill(pid,0) 금지, 다른 lock 보존, GPU중복0, awakehelper 재시작/연장·전원변경0이다. 작은metadata/bounded검색을 유지하고 불필요full다운로드/확장·435MB부모복제·기존featurecache복제·dense전체feature0, 후속 bounded RAM online streaming을 유지한다.

모든 과거 실취득·원형 부분실패·epoch 로그·cache·보고서·ZIP·GT 역할graph·모델·앱을 보존한다. NTFS5 내용보존1.548GiB 최적화는 완료했으며 반복재압축·보호모델/고정평가 확장0이다. A6파일12GiB 영구삭제 완료·승인 소진, B_C 등 추가 exact 영구삭제는 새 명시승인 필수다. 모든 실제 수용guard 통과 뒤 TRAIN/중복압축/cache exact목록 승인 후 폐기하며 고정TEST/VAL/CAL·보호모델/최신패키지·출처/license·설정/평가오류·역할/group/지문은 기본 보관한다. 새삭제는 WORK경로/reparse/보호중복/CIM/lock/size/mtime/SHA/보존사본/LiteralPath/영수증 필수이며 폴더재귀/와일드카드0이다.

계속승인은 유지한다. 필수gate 외 단계별 질문·같은 공간/완료승인/push/기존 GPU오류 알림 반복0, 변화 없거나 비실행 DONT_NOTIFY, 의미 있는 완료/실패/필수 사람조치만 알린다.
