# v82 FIT 입력·분기 진단

`48ad3cb`에 코드·조건을 먼저 push하고 FIT만의 고정2×2 비교를 했다. Rios15와 Ostrava60 전체 FIT이며 VAL/CAL/TEST 새 점수·gradient·optimizer·시각 열람은0이다. 기존 Ostrava FIT whole512 특징 캐시를 SHA/ID/raw/mask/encoder/protocol로 검증해 재사용했고 새 native1700+Rioswhole15=1715 encoder 입력만 계산했다. 같은 두 입력 보기의 특징을 원v62/선택실패v81 분기에 공유했다. 원본75장이 새로 추가된 것은 아니다.

| FIT 사진별 알려진 class 균형 BCE 평균 | v62 whole512 | v62 native | v81 whole512 | v81 native |
|---|---:|---:|---:|---:|
| Rios15 제공 ROI | 1.54795 | 1.67845 | 0.38048 | 0.06715 |
| Ostrava60 실제 fullmask | 0.05226 | 0.25518 | 0.16099 | 0.33220 |

임계값을 새로 고르지 않았다. unknown255는 음성이나 BCE 분모에 넣지 않았고 실제 fullnegative 사진은 있는 음성 class만 계산했다. 같은 기존 분기에서도 native 입력의 Ostrava FIT BCE가 커졌고, 같은 입력에서 새 분기로 바꿔도 달라졌다. 이를 미관측 TEST 실패 원인의 인과 증명·독립 성능·사진 정확도로 확대하지 않는다.

기존 모델 전체 tensor/buffer 전후 SHA `91180ead165861bb4856e8e251843c766c2ca6890cf3067d623cb7e07724afc5`는 exact이며 기존 FIT 캐시와 묶은 코드·자료 SHA도 유지됐다. 새 합성 unit2개가 unknown/전체음성 계약을 통과했다. v81의 전체 제외 판정과 앱·위치·ZIP은 그대로다.

다음 한 가설은 **전체512 기존 응답을 native 학습 분기에 보존하는 증류**다. 기존 FIT에 대한 원v62 soft 응답은 모델 보존용 정규화 목표이며 검증된 자연 정답을 추가한 것이 아니다. 정확한 구조/초기값/샘플링/손실/회차/VAL/CAL/gates를 별도 코드·JSON에 고정하고 push한 후 실제 gradient를 실행한다. 여러 계수/구조/TEST 임계값 조합을 돌리지 않는다.

Rios/Netto/Roehl MendeleyV1 CC BY4.0 DOI10.17632/kcyn4nhv2c.1, Ostrava CC BY4.0 DOI10.5281/zenodo.11235637, InsPLAD CC BY-NC3.0/VT CC0 및 모든 부모 이용조건을 유지한다.
