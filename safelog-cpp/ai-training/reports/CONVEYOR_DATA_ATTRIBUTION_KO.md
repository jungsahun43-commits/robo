# 컨베이어 자료 출처와 표시 변경

## BeltCrack

- 작성자: Huang, Jianghong; Ji, Luping; Ma, Xin; Ye, Mao.
- 데이터: *BeltCrack*, ScienceDB V1, [DOI10.57760/sciencedb.31181](https://doi.org/10.57760/sciencedb.31181).
- 데이터 라이선스: [CC BY-SA4.0](https://creativecommons.org/licenses/by-sa/4.0/). 작성자 코드의 Apache2.0과 별도다.
- 작성자 연구: *BeltCrack: the First Sequential-image Industrial Conveyor Belt Crack Detection Dataset and Its Baseline with Triple-domain Feature Learning*, Pattern Recognition,2026,113598, DOI10.1016/j.patcog.2026.113598.
- 실제 사용: 배포 V1 파일에서 작성자 embedded `imageData`와 `crack` polygon/rectangle를 읽었다. 발표의14,087장과 실제 파일의9,645장을 동일하다고 주장하지 않는다.

`beltcrack-expansion-examples.png`는 원본을640 letterbox 좌표로 축소하고 패딩을 제거해 표시한 뒤, 작성자 다각형 정답 또는 모델0.5 예측 영역을 주황색으로 겹친 파생 그림이다. `beltcrack-expansion-comparison.png`는 같은 작성자 검증 사진에서 측정한 수치를 표시한다. 이 두 파생 그림은 **CC BY-SA4.0**으로 제공하며 본 출처·변경 설명을 함께 유지한다. 사진 자체를 생성하거나 정상·손상 정답을 새로 만들지 않았다.

## 컨베이어 이물질 자료

- 작성자: Frederico Martins Sousa; Thiago Eustaquio Alves de Oliveira; Bruno Nazário Coelho; Saul Emanuel Delabrida Silva.
- 데이터: [작성자 Mendeley V1](https://data.mendeley.com/datasets/s25x2bnshz/1), DOI10.17632/s25x2bnshz.1,2025.
- 데이터 라이선스: [CC BY4.0](https://creativecommons.org/licenses/by/4.0/).
- 작성자의 정상 의미는 광석만 흐르는 장면이며, 이상 의미는 도입한 이물질이다. 정상 벨트의 기계적 상태나 균열 없음의 정답과 다르다.

두 데이터의 클래스·사진·정상 의미를 섞어 전체 설비 고장·안전 가동 성능으로 해석하지 않는다. 원본·체크포인트·보고서 수치는 각 실험의 별도 SHA와 기록을 따른다.
