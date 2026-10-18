# v144 — CrackForest 원저자 공개 metadata 검토, 취득 전 보류

실제 repo의 report/script/doc 텍스트에서 CrackForest 정확명칭·확인 저자/제목·원저자 URL 및 DOI 검색 토큰을 먼저 검색했고 정상 exit1/미일치였다. 파일/권한 오류를 미일치 증거로 쓰지 않았다. 이 범위는 현재 repo 텍스트이며 모든 과거 commit/ignored 원형 전체 감사가 아니다. DeepCrack 기존 언급과 PECCD의 deepcrack class 문자열은 다른 이력이므로 재조사/새 source로 세지 않았다.

## 확인한 primary 범위

[Limeng Cui의 원저자 repo](https://github.com/cuilimeng/CrackForest-dataset)의 master 공개 README는 도시 도로 균열 주석 자료를 소개하고 비상업 연구 용도로 제공한다고 명시한다. 저자의 요청 인용은 Shi, Yong / Cui, Limeng / Qi, Zhiquan / Meng, Fan / Chen, Zhensong의 *Automatic road crack detection using random structured forests* (IEEE TITS17(12),3434–3445,2016)와 Cui/Qi/Chen/Meng/Shi의 *Pavement Distress Detection Using Random Decision Forests* (2015,95–102)이다. 원형 인용을 임의로 바꾸지 않는다. README의 Version1.0/2015-09-29는 배포 선언이고 immutable commit/file checksum·촬영 날짜가 아니다.

이번에 읽은 README/루트 목록 범위에서 data의 비상업 연구 제한은 확인했지만, 실제 RGB·자연발생/원TRAIN native GT 값·unknown/미주석·정상 배경/좌표·export/원분리·물리 촬영 묶음·file bytes/checksum·작은 익명 선택 접근/임시량·실제 보존 예산은 확인하지 못했다. groundTruth/image/seg 디렉터리 이름을 실제 파일 수·GT 권위/배경 코드·서로 독립인 촬영 그룹으로 쓰지 않는다. mask 부재/bbox-only·모든 자료 허가 부재나 영구 접근 불가라는 증명도 아니다. **필수 gate 전 취득 전 보류/readyfalse**다.

비상업 연구 조건을 상업 앱 허가·CC0/CC BY로 바꾸지 않고 source/저자/버전·요청 인용/변경·제삼자 권리 조건을 유지한다. 도로 자료를 일반 설비/부식·전체 건강·현장 독립 성능이나 새로운 도로 과제로 자동 추가하지 않는다. 저자 모델/논문의 점수·설계·분리를 SafeLog protocol로 채택하지 않았다.

## 미채택·미취득 범위

최초 검색에 함께 나온 issue의 제삼자 MAT→PNG 코드/label 값·annotation 도구 조언, catalog/mirror의 수량·변환/다운로드, 별도 demo의 155/118/37 및 미주석 정상 해석은 원저자 계약으로 채택하지 않았다. 그 코드/자료/링크를 별도로 열거나 실행하지 않았다. 원저자 repo의 groundTruth/image/seg 폴더·raw/ZIP/API·commit history/다른 endpoint/모델 링크도 추적하지 않았다. 정확 논문/DOI 토큰의 IEEE-domain 색인 검색은 결과0이었으며, 이 검색 결과를 논문/DOI/유효 GT 부재로 쓰지 않는다. 검색 토큰 `10.1109/TITS.2016.2552248`는 이번 primary 원문에서 검증한 dataset DOI가 아니다. 직접 publisher body/PDF 요청·새 HTTP 오류0이다.

웹 도구 공개 text 관찰만 수행했으며 전체 web traffic bytes는 미측정이다. 전체 HTTP0이라고 주장하지 않는다. raw dataset metadata/HTML/API/PDF/code/weights·원사진/mask/annotation/archive의 로컬 취득0, dataset fetch/decode/training helper0/새 unit0이다. 관찰 메모는 publisher 고정 원문·자료 checksum이 아니다. 원저자 코드 실행·자동연락/로그인·동의/PII대리제출·접근 우회0이다.

새 pixel/시각QA/raster/features/우리 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0·새 취득/pixel/split/model protocol0·readyfalse다. 기록·host/Git PowerShell은 dataset job으로 세지 않으며 새 학습 실패/성능 개선이 아니다. v127 JPG1+mask1/v126 JSON1와 v141 ref/v142~v143 author text2의 실제 완료 취득 이력 및 원형·부분·실패·epoch·로그·cache·보고서·ZIP·GT/역할/graph/보호 모델/앱을 보존한다.

새 명확한 원저자 data 허가/native 자연 TRAIN GT·배경_unknown/원분리·촬영 대응 release 없이 같은 README/논문/검색/issue·mirror/raw/archive/code를 반복 검토하지 않는다. 모든 이전 제한은 최신 resume와 필수 invariant/history 파일을 함께 적용한다.
