# v145 — AeBAD 원저자 metadata 검토, 현재 자연 손상 pixel TRAIN 경로 제외

실제 repo의 report/doc/script/source 텍스트에서 AeBAD 이름·제목을 검색해 정상 exit1/미일치를 확인했다. 새 공개 색인에서 확인한 저자·DOI/arXiv·정확 원저자 repo URL도 다시 이력 검색해 정상 exit1이었다. 검색/권한 오류를 미일치 증거로 쓰지 않았다. 초기 URL 검색 토큰 `zhangzilongcmm/AeBAD`는 확인된 배포 URL이 아니며 요청하지 않았다. 확인된 원저자 연결은 [zhangzilongc/MMR](https://github.com/zhangzilongc/MMR)이다. 현재 텍스트 검색 범위이며 과거 모든 commit/ignored 원형 전체 감사를 뜻하지 않는다.

## primary 관찰과 판정

Zilong Zhang / Zhibin Zhao / Xingwu Zhang / Chuang Sun / Xuefeng Chen의 [원 arXiv abstract](https://arxiv.org/abs/2304.02216)는 *Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction*, arXiv2304.02216v2(2023-09-01)를 확인한다. [공식 publisher 색인](https://www.sciencedirect.com/science/article/pii/S0166361523001409)의 출판 식별자는 DOI10.1016/j.compind.2023.103990, Computers in Industry151/103990,2023-10이다. 논문 v2/날짜를 dataset revision·촬영 날짜로 쓰지 않는다.

원저자 master README의 두 하위 자료 설명과 디렉터리 안내는 AeBAD-S의 train/good, test/ablation, ground_truth 및 AeBAD-V의 train/good와 test/video anomaly를 구분한다. 조명·시점·background 조건의 TRAIN/TEST 차이 설명은 실제 물리 촬영 그룹/사진 독립성 감사가 아니다. background 폴더는 domain 조건이며 GT 배경/무부식 코드가 아니다. **이 정상 TRAIN 안내 경로는 현재 자연발생 산업 RGB TRAIN native 손상 pixel+명시 배경 우선순위에서 취득 전에 제외한다.** 실제 전체 파일별 분리·모든 TRAIN record를 감사한 것이 아니며 TEST 이동·정상분포 학습 실패 반복0이다.

README는 data CC BY4.0을 명시한다. source/저자·라이선스·요청 인용·버전/변경·제삼자 권리를 유지하고 paper/code 권리로 대체하지 않는다. 별도 LICENSE와 LICENSE-DATASET 파일명 및 GitHub license 탐지 표시는 보였지만 파일 원형·모든 제삼자 권리/실 archive 허가 전체를 검증하지 않았다. master/commit 수를 immutable dataset revision/checksum으로 쓰지 않는다.

AeBAD-V에 sample-level metric만 평가하고 pixel metric0이라고 적은 것은 평가 범위다. 이를 모든 video GT 부재로 단정하지 않는다. README의 원영상/모델 결과 영상과 abstract의 masked reconstruction 방법·모델점수는 원 native GT·SafeLog protocol/성능이 아니다. 결과 영상/그림·원사진/mask를 열거나 author code/모델을 실행하지 않았다. blade metadata를 전체 엔진 건강/현장 독립·0미탐 또는 새로운 영상 과제로 자동 확대하지 않는다.

## 취득·실행·검증 경계

새 primary 공개 web text만 읽었다. 직접 publisher body/PDF/experimental HTML/TeX·별도 LICENSE-DATASET/raw/API·Drive/Tian Yi Yun Pan·archive·MAE/model/code/config/run-script·영상 링크는 추적하지 않았다. 최초 검색에 나타난 제삼자 변환본의 train/test/수량·mask·용량·license 설명은 채택하지 않았다. raw dataset metadata/HTML/API/PDF/code/weights·원사진/mask/annotation/archive 로컬 취득0이며 web 전체 traffic bytes 미측정/전체 HTTP0 주장0이다. 새 web HTTP 오류0이다.

실제 data의 자연발생·RGB/nativeGT_unknown·미주석/배경/coords/export·파일별 원분리/물리 촬영 묶음·file bytes/checksum·작은 익명 선택접근/임시량·고정 사진_group_validpixel 최소·source-oldheld 전체graph·full 보존 예산은 미검증/readyfalse다. mask 부재·모든 손상이 합성이라는 주장0이다. source 사진/pixel·시각QA/raster/features/우리 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0, dataset fetch/decode/training helper0/새 unit0/새 취득_pixel_split_model protocol0이다. host/Git/관찰 기록 PowerShell은 dataset job이 아니며 새 학습 실패·성능 개선도 아니다.

모든 기존 원형·부분·실패·epoch·로그·cache·보고서·ZIP·GT/역할/graph·모델/앱과 v127 JPG1mask1/v126 JSON1/v141 ref/v142~v143 author text2의 실제 취득 이력은 보존한다. 자동연락/로그인·동의/PII 대리 제출·접근 우회0이다. 새 명확한 원저자 자연 손상 TRAIN native pixel+배경_unknown release 없이 동일 AeBAD/MMR 논문·README·검색·download/mirror/loader/license/code/모델을 반복하지 않는다. 이전 완료·보류 source도 재조사하지 않는다.
