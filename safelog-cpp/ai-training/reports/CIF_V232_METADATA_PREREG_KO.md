# CiF 원저자 파일 정보 사전등록 v232

PAGG v230 정보 조회는 성공했지만 v231 실제 파일 첫 Range 요청은 HTTP406/본문·조각0으로 중단됐다. 등록된 3파일과 실패·처음 실행 감싸기 방식의 preflight 중단 기록은 보존한다. 같은 PAGG 파일/부모/API/헤더 변경·reader·mirror 재시도나 전체 ZIP fallback은 하지 않는다.

이번 후보는 Cracks in the Foundation(CiF)의 원저자 IBM Research/ETH 프로젝트가 직접 연결한 ibm-research/cif-dataset이다. v188에서 검색 색인만 본 nicolaaaaa/cif_dataset_toy와 같은 가족임을 유지한다. 새로 확인한 원저자 전체 배포의 데이터 이용 조건·사진/주석 배포 허용 acknowledgment·COCO 다각형·원본/타일·원TRAIN/VAL/TEST 선언이 이번 첫 원본 배포 정보 검토의 근거다. 과거 toy를 독립 자료로 승격하거나 다시 읽지 않는다.

원본 프로젝트에는 약15만 원사진, 25만 손상 주석과 5년 수집·시설 파트너가 제시된다. HF card에는 6개 손상 범주, 원본/full과 1024타일의 부모 파일명·좌표 필드, 원역할별 구분, CDLA Permissive2.0이 명시돼 있다. root Files의 짧은 revision은75057c6이다. 약15만/300708행/default162k 표시를 사진/group 총량으로 합산하거나 조용히 통일하지 않는다. card의 ‘split마다 자체 shard’ 설명과 실제 TEST 여러 shard 파일목록도 구별한다.

이번 코드는 이 관찰된 revision의 공용 파일 정보 JSON을 1회만 최대1MiB로 읽는다. 공개/ungated·라이선스 선언·완전한40자 revision·실제 파일명/크기/Git blob/LFS SHA256 선언과 6원역할 파일목록을 확인하고 최대1MiB 목록 장부를 저장한다. 기존 로컬 HF 정보 조회 코드의 공개 route 형식과 공식 HfApi의 revision/files_metadata 의미를 근거로 하며, 옛 helper는 실행하지 않는다. 서버가 반환하는 체크섬은 실제 parquet를 받은 검증값이 아니다. 파일명/크기/버전은 추정하지 않는다.

이 단계는 parquet·사진·GT·CSV·code·모델을 받거나 디코딩하지 않는다. 인증·cookie·proxy·redirect·login·연락 없이 exact200 JSON/identity 응답만 허용한다. 첫 요청/형식/버전/크기/권한/분리목록 오류에서 원형과 partial을 남기고 중단한다. 페이지 추가 조회·재요청·다른 API/버전/헤더 우회는 없다.

저장·임시·실행 예산4MiB, 별도 등록 Git 임시4MiB와 디스크 예비1GiB, RAM 예산64MiB와 예비1GiB다. 실제 전체123GB 자료/해제/학습 예산이 아니다. 실행 직전 성공한 CIM의 명령줄/시작시각·scope lock·보호10 bytes/SHA·v85 실제450update/30epoch 장부·공간/RAM을 확인한다. base Python을 PowerShell에서 직접 실행한다. v227 고정 guard SHA를 확인해 재사용하며 다른 lock/모델/앱/평가/캐시를 변경하지 않는다.

공개 대상은 이 코드·JSON·설명서의 정확3파일이다. 별도 publication 부모 f412c5b8f93d892305d54777e7fdb2e1f47da0f1의 새 commit/diff를 준비하고 구체승인 뒤 fresh remote부모→일반 feature/ai-engine push→remote exact 검증 후 실행한다. 앞선4건 승인은 각각 완료·소진되어 재사용하지 않는다. 전체 local HEAD/main/force/다른ref 전송과 raw/사진/GT/weights/모델/비공개 보관2문서 전송은 없다.

자료는 아직 수용되지 않았다. 모든 사본은 UNASSIGNED, 원TEST/publisherVAL은 HELD다. 타일을 독립사진으로 세지 않는다. 실제 RGB·원TRAIN-native 현재손상 GT·명시 배경/unknown/좌표·권리 chain·원역할/물리촬영group/전체 source-oldheld graph·고정 최소·실bytes/full 예산을 확인한 후에만 별도 취득·자료감사·공통encoder 학습 protocol로 진행한다. 기존 부모 exact/VAL→저장CAL→고정TEST1회/retention/runtime/보존 조건은 유지한다. 이번에는 학습이나 앱 교체를 하지 않는다.
