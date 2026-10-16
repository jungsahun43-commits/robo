# v72 부분주석 정답 자료 검증

- 사전 코드·정책: `68cc0a6` 일반 push 후 주석 계산. 새 RGB 해독·중복 거리 재계산·특징·optimizer는 0회다.
- 기존 Ostrava 105장과 VT 440장의 역할을 유지했다. FIT 60+276, VAL 15+60, CAL 15+60, 원 TEST 15+44이며 새 자연 사진을 확보한 수가 아니다.
- VT Fair/Poor/Severe만 양성 1로 투영했다. Good/배경 및 원 ignore는 unknown 255다. VT 전체 음성 픽셀은 0개이고 padding도 손실 제외다. Ostrava는 원 L>0 양성 및 알려진 배경을 보존했다.
- 512 FIT에서 Ostrava 양성 4,559,995 / 음성 11,168,645 픽셀, VT 양성 13,365,587 / unknown 40,630,445 / padding 18,355,712 픽셀이다. 최소 표본·픽셀 조건 및 고정 역할을 통과했다. 새로운 정답 이미지 파일은 만들지 않았다.
- 새 unit 3개는 VT 음성 혼입 금지, 실제 169 mask 양성, 비율 유지 좌표·padding을 검사했다.
- 넓은 수동 polygon은 모든 미세 녹 경계를 전문가가 검수한 mask가 아니다. VT unknown 영역의 이진 FP는 측정할 수 없다. 전체 정상 TEST 부족으로 앱 사진 FP/FN 20% 통과를 선언하지 않는다.

VT: Bianchi/Hebdon, Virginia Tech V2, data CC0, DOI 10.7294/16624663.v2 및 10.1061/(ASCE)CP.1943-5487.0001045. Ostrava: Matěj Frič, Zenodo V2, CC BY 4.0, DOI 10.5281/zenodo.11235637. 변경은 위 부분주석 투영이며 원파일을 수정하지 않았다. 기존 InsPLAD 및 부모 자료 이용조건은 유지한다.
