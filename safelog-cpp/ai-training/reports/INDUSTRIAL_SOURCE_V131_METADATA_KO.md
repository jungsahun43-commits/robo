# v131 — 하수관 pixel 분할 연구 공개 metadata 검토

최신 v130 재개/Git/보호10파일/host 기록과 다음 문서를 읽었다. 현재 Git 상태와 fresh host CIM 명령줄/시작시각·잠금 범위·실제 v85 장부를 확인했다. v8–v130 완료 취득/감사/학습/평가/fixture를 재실행하지 않았다. 이번은 한 연구의 공개 metadata 선별이며 새 학습 실패나 성능 개선이 아니다.

## 새 후보와 확인한 원출처

먼저 실제 `ai-training/reports`, `ai-training/scripts`, `safelog-cpp/docs`를 검색했다. SewerSeg 이름은 v130의 미조사 메모에만 있었고 완료 조사 근거가 아니었다. 이후 확인된 정확 논문제목·DOI/URL 및 저자 검색은 정상 exit1(미일치)이었다. 경로/권한 실패를 미일치로 바꾸지 않았다. SewerSeg라는 정식 dataset 이름/별도 release를 확인한 것은 아니며, 다음 정확한 논문 한 편을 v131 대상으로 삼았다.

[원저자 기관 Sejong University 기록](https://sejong.elsevierpure.com/en/publications/lightweight-pixel-level-semantic-segmentation-and-analysis-for-se/)은 *Lightweight pixel-level semantic segmentation and analysis for sewer defects using deep learning*, DOI10.1016/j.conbuildmat.2023.130792, Construction and Building Materials371/130792, 2023-03-31을 확인한다. 저자는 L. Minh Dang, Hanxiang Wang, Yanfen Li, Le Quan Nguyen, Tan N. Nguyen, Hyoung Kyu Song, Hyeonjoon Moon이다.

[공식 publisher 색인 텍스트](https://www.sciencedirect.com/science/article/pii/S0950061823005032)는 CCTV 하수관 10종 결함의 pixel 분할 연구와 수작업 수집/주석을 설명한다. 하이라이트의 11,124이미지와 dataset 절의 60영상/결함포함 3,699프레임은 서로 다른 서술 단위다. 이들을 더하거나 각각 검증 자연사진/독립부품/고유 TRAIN 수로 세지 않는다. 실제 관계·증강/파생 여부·중복/물리촬영 묶음·원역할은 미검증이다. classifier/bbox-only 또는 손상mask 부재로 오인해 제외하지 않았다.

## 취득 전 보류와 조회 한계

확인한 기관 metadata와 publisher 색인은 공개 source 파일의 data 고유 허가·제삼자 권리·release/revision·실제 RGB/native TRAIN GT·class/unknown/미주석·명시적 정상/배경 지원·좌표·원분리/그룹·file/bytes/checksum·작은 익명 선택접근/임시량·실제 예산을 통과시키는 계약이 아니다. 기관의 논문 저작권 표시를 data 허가로 쓰지 않았다. 논문 속 원저자 모델 점수/전처리·학습/중증도 기준을 SafeLog 성능이나 새 protocol로 채택하지 않았다. 하수관 사진/논문을 설비 전체 건강·독립 현장 성능으로 확대하거나 새 하수관 과제를 자동 추가하지 않는다.

직접 publisher page 조회 한 번은 HTTP403으로 실패했다. 같은 page/다른 endpoint/PDF/mirror 재시도0이다. 후속 정확제목/저자 검색으로 발견한 원저자 기관의 서지/abstract 텍스트만 한 번 열었으며, 논문 PDF·개인홈페이지 PDF·source code/model·이미지/영상·자료 archive/API는 열거나 로컬 취득하지 않았다. 기관 기록은 접근 우회나 비공개 자료 권한 증거가 아니다. 검색결과에 섞인 다른 논문의 corrected-GT GitHub 안내, Sewer-ML 분류자료, Co-UDlabs/다른 하수관 논문/모델을 이 연구의 source release/GT로 연결하지 않았다. 자동연락·로그인/동의/PII 제출·우회0이다.

필수 data/GT/역할/접근·예산 gate가 미확인이라 **취득 전 보류**다. 403을 영구 접근 불가나 자료/허가 부재의 증명으로 쓰지 않고, 원저자 자료 어디에도 유효 GT가 없다고 주장하지 않는다. 새 명확한 원저자 public release/허가/실제 계약 근거 없이 같은 논문/기관 page·403·검색/대체 PDF를 반복하지 않는다.

## 실제 수행량과 보존

`data/industrial-source-v131-metadata/primary-web-observations.json`은 웹 관찰 메모다. publisher 고정 원문 SHA/자료 checksum이 아니며 웹 traffic bytes는 미측정이다. 새 dataset rawHTML/API/PDF/code/weight·원사진/mask/annotation/archive 로컬 취득0, source 픽셀/시각QA/raster/features/우리 모델 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0이다. 새 dataset fetch/decode/training helper·unit fixture0, 새 정식 취득/pixel/split/model protocol0/readyfalse다. 관찰기록 작성과 host/Git 조회 PowerShell은 dataset job 수에서 제외한다. 이번 web page 실패1을 학습오류로 세지 않는다.

v85 실제 actual450/epoch30과 완료추적 일부 optimizer60430을 유지한다. 후자는 실패포함 부분합이며 전체 역사 총량이 아니다. 보호10 SHA·모든 원본/부분/실패/epoch/log/cache/보고서/ZIP·GT/graph/잠긴 역할·모델/앱 출력을 보존한다. dead training.lock23948·awakehelper/전원설정 변경0이다. lock 확인 범위는 AI root/runs와 WORK outputs이며 source data 전체 tree 열거 주장이 아니다. 사용자의 계속승인과 자동후속을 유지한다.
