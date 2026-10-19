# v155 — CrackTree 원논문 공개 색인 검토, 현재 RGB TRAIN 면적 GT 조건 미통과

2026-10-19 한국시간. `CrackTree`라는 이름의 기존 검색 일치는 v154의 미검토 후보 메모 2개뿐이었다. 이를 정상 미일치로 보고하지 않는다. `CrackTree200`, 정확 논문 제목, 원repo 추정 토큰의 검색은 정상 exit1이었다. 원repo 추정 토큰은 실제 URL로 요청하지 않았다. 발견 후 확인한 DOI/PII 및 Qin Zou/Qingzhou Mao 이력도 실제 report/script/config/dataset/output 텍스트에서 정상 exit1이었다.

선별한 primary text는 [공식 publisher 색인](https://www.sciencedirect.com/science/article/pii/S0167865511003795)의 **CrackTree: Automatic crack detection from pavement images**, Qin Zou / Yu Cao / Qingquan Li / Qingzhou Mao / Song Wang, DOI `10.1016/j.patrec.2011.11.004`, Pattern Recognition Letters 33(3), 227–238, 2012-02-01이다. DOI의 연도나 논문일을 사진 촬영일·dataset revision으로 쓰지 않는다. 직접 publisher body/PDF 요청은 하지 않았다.

원저자는 실제 포장도로 이미지 206개, 800×600, 수작업 crack curve 정답을 평가에 사용한다고 설명한다. 알고리즘은 단색 pavement image에 초점을 두고 위치와 곡선 모양을 찾으며 균열 폭을 대상으로 하지 않는다고 명시한다. 이는 GT 부재나 bbox-only라는 뜻이 아니다. 다만 현재 우선순위인 자연산업 RGB TRAIN 손상 면적 native mask+명시 배경/unknown 계약을 입증하지 못한다. **취득 전 보류/readyfalse**다. 모든 배포 사진이 회색조라는 단정, 평가용 206개를 원TRAIN·독립부품으로 승격, curve를 두껍게 채워 면적 mask로 전환은 하지 않는다.

탐색 이름 CrackTree200과 논문의 206개를 동일 파일 release/새 독립 사진 수로 확정하지 않았다. 실제 data 고유 허가·제삼자 권리·공개 원배포/revision·RGB/nativeGTunknown/정상 배경/좌표·원역할/물리 촬영 묶음·file/bytes/checksum·작은 익명 접근·전체 graph/최소/full 예산은 미검증이다. 논문 TDM/저작권, 다른 논문/미러의 수량·split·변환 mask·license를 data 계약으로 쓰지 않는다. 그림·저자 PDF·프로필·repo·reference/related·download/code/model·API를 추적하거나 저자 연락/login/동의/PII/우회를 하지 않았다.

그림자 제거·tensor voting·확률지도·MST/pruning은 논문의 방법이며 원자료의 GT 값/배경 규약·SafeLog 구조/성능·source-oldheld 보호 graph가 아니다. C++ 구현/PC/시간·모델 평가 수치를 SafeLog runtime 검사나 학습 protocol로 채택하지 않았다. 도로 source를 설비 부식·전체 건강·별도 도로 검사 과제로 자동 확대하지 않는다.

이번에는 공개 검색 색인 text만 검토했다. rawHTML/API/PDF/code/weights/사진/mask/annotation/archive 로컬 취득0, sourcepixel/QA/raster/features/우리score/gradient/optimizer/CAL_TEST/모델API_CPU_GPU_CPP_ZIP_APK0, 새 dataset helper/unit/fixture/정식 취득/pixel/split/model protocol0이다. 직접 HTTP 오류·재시도0이며 전체 web traffic bytes는 미측정/전체HTTP0 주장은 하지 않는다. 관찰 메모는 web 요약이며 publisher 고정 원문이나 data checksum이 아니다.

기존 모든 원형·부분·실패·epoch/log/cache·GT/역할/graph·모델/앱을 보존하고 A12GiB 이전 완료 외 추가 삭제0이다. 결과 보고서 2개는 로컬만 보관하며 기존 v152/v153 두 파일 push 승인을 재사용하지 않는다. 새 명확한 원저자 자연 RGB TRAIN native 면적 GT/배경_unknown·권리/공개 release 근거 없이 같은 CrackTree 논문·검색·페이지·PDF·repo·배포/변환/code/model 경로를 반복하지 않는다. 아직 안 읽은 파일을 이용한 조건 우회도 하지 않는다.
