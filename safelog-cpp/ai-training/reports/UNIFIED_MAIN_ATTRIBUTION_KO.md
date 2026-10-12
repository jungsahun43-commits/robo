# 통합 메인 · 자료 출처와 사용 조건

이 전달 파일은 비상업 학술 연구·교육용 모델이다. 모델에는 다음5출처의 학습 이력이 있다. 원본 데이터 사진·주석은 ZIP에 포함하지 않는다. 기존 교사 모델도 배포하지 않는다.

| 출처 | 원본·조건 | 이번 사용·변경 |
|---|---|---|
| DACL10k | Johannes Flotzinger; Philipp J. Rösch; Thomas Braml. [원논문·자료 안내](https://arxiv.org/abs/2309.00460), CC BY-NC4.0. [배포 LICENSE 원문](LICENSE_DACL.md) | 시설 사진·격자 정답7종, 출판자19종 보조 태그. 다른 시설의 안전 정답으로 확대하지 않음 |
| DamSegment | [Mendeley V1](https://data.mendeley.com/datasets/z5z6gtt5t4/1), CC BY4.0 | 균열·박락과 출판자 intact 정의. 분류 Non-Crack의 두 항목 음성은 저자 정의에서 변환한 의미이며 별도 native spalling 음성 mask라고 표현하지 않음 |
| CODEBRIM | [저자 원본 프로젝트](https://github.com/MrtnMndt/meta-learning-CODEBRIM), [원문 사용 조건](LICENSE_CODEBRIM.md) | 작성자 patch·5종 정답. 젖은 표면·표면 공동은 unknown. 조건5의 데이터 자체를 포함하지 않는 학습 모델 공유 범위에 따라 비상업 연구·교육용으로 전달 |
| BeltCrack | Huang Jianghong; Ji Luping; Ma Xin; Ye Mao. [ScienceDB DOI10.57760/sciencedb.31181](https://doi.org/10.57760/sciencedb.31181), [CC BY-SA4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 작성자 embedded 사진과 crack polygon/rectangle, 640 letterbox·padding 제외. 사진 없는 통계 그림만 전달. 데이터 라이선스는 작성자 코드의 Apache2와 별도 |
| 컨베이어 이물질 | Frederico Martins Sousa; Thiago Eustaquio Alves de Oliveira; Bruno Nazário Coelho; Saul Emanuel Delabrida Silva. [Mendeley V1](https://data.mendeley.com/datasets/s25x2bnshz/1), DOI10.17632/s25x2bnshz.1, [CC BY4.0](https://creativecommons.org/licenses/by/4.0/) | normal-op/anomaly 사진 정답. 철광석 흐름의 이물질 부재이며 벨트 기계 정상 정답이 아님 |

CODEBRIM 인용: Martin Mundt, Sagnik Majumder, Sreenivas Murali, Panagiotis Panetsos, Visvanathan Ramesh. *Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with the COncrete DEfect BRidge IMage Dataset*. CVPR2019.

이번 학습은 출판자 정답을 사용하며 새 AI 정답을 생성하지 않았다. 모르는 항목을 음성으로 만들지 않는다. 학습 파일에는 이전 시설 메인과 벨트 출력의 가중치 이력도 포함되며 동일 자료 조건을 유지한다. IDEA ND 보충자료와 PCB는 이번 메인에 넣지 않았다. 모델·출처·변경 설명과 원문 조건을 함께 유지한다.
