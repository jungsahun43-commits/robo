# PAGG-Net 압축파일 내부 정보 사전등록 v231

PAGG-Net 원저자 기록 조회(v230)는 성공했다. 실제 Zenodo JSON에 CC BY 4.0/open, 버전 1.0.0, Zenodo.zip 30,643,774,745바이트, MD5 5f83eb3edab301920bb8cc2989fb5c6c 및 정확한 파일 주소가 있다. 이는 전체 ZIP 무결성·사진 권리 chain·원TRAIN 손상 GT·촬영 독립성 검증이나 학습 완료가 아니다.

이번에는 ZIP 전체를 받지 않는다. ZIP/ZIP64의 끝·내부 목록과 작은 README/LICENSE만 최대 17MiB 압축 조각 범위에서 확인한다. 논리 Range GET 최대16, 실제 HTTP 요청은 전달 포함 최대32다. 최초 HTTP206, 정확한 범위·전체길이·binary MIME·강한 ETag와 이후 If-Match가 맞아야 한다. 지원되지 않거나 첫 오류가 나면 중단하고 원형 실패를 남기며 재요청·전체 ZIP fallback·다른 경로 우회를 하지 않는다. 서버가 제공한 zenodo.org 내부 HTTPS 전달만 1회 허용한다.

내부 목록은 최대100,000개·중앙 목록16MiB·이름 합계8Mi 문자까지 제한한다. 실제 파일을 추출하지 않고 목록 JSONL만 최대64MiB로 저장한다. TEST/VAL/CAL 경로를 제외한 README/LICENSE 중 사전 고정한 사전순 최대3개만 각각64KiB 이내로 읽고 CRC·UTF-8을 확인한다. CSV·사진·mask·annotation·저자 코드는 열지 않는다. 목록/헤더 경계에 우연히 포함된 압축 사진 바이트가 있을 수 있으므로 전체 사진 bytes0이라고 주장하지 않는다. 이미지는 디코딩하지 않는다.

이 단계 저장·임시·실행 상한100MiB, 등록 Git 임시공간4MiB와 별도 디스크 예비1GiB를 확인한다. RAM 단계 예산256MiB와 예비1GiB를 fresh CIM으로 확인한다. 중앙 목록 count/size는 Python ZIP 파서의 eager 할당 전에 검사한다. 실제 측정 peak나 전체30.6GB 자료·학습 예산을 대신하지 않는다.

v227의 고정된 공통 보호 코드를 SHA로 검증해 재사용한다. 실행 직전 실제 명령줄/시작시각 CIM, 제한 scope lock, v85 실제450update/30epoch 장부, 보호10 bytes/SHA를 새로 확인한다. base Python을 직접 사용하며 v227의 venv 자기 process 중단 원형은 보존한다. 다른 lock·모델·평가·캐시는 변경하지 않는다.

이번 공개 대상은 코드·JSON 명세·이 설명서의 정확3개다. 이전 등록 승인3건은 각각 완료·소진됐으며 이번 전송에 재사용하지 않는다. 마지막 실제 검증 publication cec1094b616a11e1beb77976d0850cb30447536d의 별도 부모 commit으로 준비하고, 필요한 구체승인 후 fresh remote부모 확인→일반 feature/ai-engine push→원격 exact 확인 뒤에만 실행한다. local HEAD 전체·main·force·다른 ref 전송은 없다. raw 자료·사진·GT·weights·모델·비공개 보관2문서는 포함하지 않는다.

모든 가능한 사본은 UNASSIGNED, 원TEST/publisherVAL은 HELD다. TRAIN-native RGB 현재손상 GT의 배경/unknown 좌표, 원역할·물리 촬영group·전체 source-oldheld graph·고정 최소·권리/실bytes/full 예산을 별도로 통과해야 한다. 타일을 독립 원사진으로 세지 않는다. 기존 공통 encoder·부모 exact/평가/retention/runtime 조건과 닫힌 source/실패 보존 조건은 그대로다. 자료 수용·학습·앱 교체는 이번 단계에 포함되지 않는다.
