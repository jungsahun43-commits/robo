# v100 ReinAD 원저자 자료 선별

기존 보고서/코드/문서에 정확 이름·원URL의 완료 기록이 없는 것을 확인하고 [원저자 프로젝트](https://tocmac.github.io/ReinAD/)가 직접 연결한 [ReinAD 배포본](https://huggingface.co/datasets/Tocmac/ReinAD)을 선별했다. IVD-SEG의 두 새 원자료를 명시하는 추가 문서는 발견하지 못해 그 조사는 더 반복하지 않는다.

[NeurIPS2025 원논문](https://papers.nips.cc/paper_files/paper/2025/file/1bdcb065d40203a00bd39831153338bb-Paper-Datasets_and_Benchmarks_Track.pdf)은 실제 생산 중 발생한 이상, 원저자 모델의 초벌 주석을 사람이 보정하는 과정,56제품/87,084사진을 설명한다. 표준 train/test는 서로 다른 제품 범주다. 이는 원저자 주장이고 개별 RGB·GT·부품/촬영 독립성·우리 성능의 검증이 아니다. 초벌 자동주석만을 정답으로 사용하지 않는다. PDF 텍스트만 확인했고 예시 사진은 열람하지 않았다.

직접 연결된 data card는 **CC BY-NC-SA4.0**와 HDF5의 normal/anomaly/대응mask를 설명한다. 비상업 연구·저자/출처/버전/변경·동일조건을 유지하며 웹페이지·논문·코드 조건과 구별한다. 공개 파일목록은 전체 약124GB, 가장 작은 TRAIN `cable_2.h5` 약8.34MB다. 작은 파일 하나가 준비된 독립 학습자료라는 뜻은 아니다. 원논문의 범주 간 평가와 같은 범주 감독학습 실험을 혼동하거나 원TEST를 재배정하지 않는다.

## 다음 v101의 제한

한정된 공개 metadata만 취득한다. HF info→고정 revision→같은 tree/README의 경로·길이·gitblob/LFS SHA를 교차 확인하고 **모든 원TEST HDF5의 바이트 식별자**를 후속 binary 접근 전에 예약한다. 이것은 아직 사진별 ID나 실제 파일checksum 검증이 아니다. HDF5 파일/사진/mask/코드/모델은 받거나 실행하지 않는다.

정식 조건은 `industrial-reinad-v101-metadata-policy.json`이다. 단일응답1MiB/전체수신4MiB/저장8MiB/예비1GiB, 익명HTTPS·무재시도·무리다이렉트·무인증, 지원밖 schema/페이지네이션/경로/불일치는 실패 보존 후 중단한다. 가장 작은 TRAIN은 미래 구조 확인 후보로만 기록한다. 좌표/dtype/표본/원분리·모든 held 연결/전체 전이 graph/별도 예산이 통과하기 전에는 ready=false와 특징·학습0을 유지한다. 아직 source 취득/학습 프로토콜은 없다.
