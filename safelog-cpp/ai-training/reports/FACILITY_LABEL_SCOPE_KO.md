# 시설 AI 학습 정답의 범위

## 확인한 정답과 표현

이번 검수는 기존 TRAIN의 전체 사진·출판자 patch·파생 crop 메타데이터와 DACL TRAIN 원본 annotation을 검증한다. DACL shared split index에는 TRAIN/val/test의 boxes·ignored_labels 요약이 함께 있으며 전체 JSON을 로드하지만 TRAIN 정보만 집계·검증에 사용한다. Native VAL/TEST annotation과 사진은 열지 않고, heldout 클래스 요약·사진·모델 예측은 학습 표본 선별에 사용하지 않는다. 결과 수와 입력 SHA는 `facility-target-negative-data-audit.json`에 기록한다.

`targets=0`은 출판자 주석 또는 의미 정의에 근거해 변환한 해당 항목의 음성이다. `targets=-1`은 확인하지 않은 항목이다. 둘을 합쳐 정상으로 처리하지 않는다. 균열·박락이 모두 0인 사진에도 녹 얼룩, 백화, 낙서, 풍화, 설비 부품이나 확인하지 않은 손상이 있을 수 있다. 사진의 구조적 안전성이나 법적 적합성을 판정한 정답은 없다.

기존 TRAIN base 14,248행 가운데 균열·박락 모두 음성인 행은 5,915개다. DACL 2,459개, DamSegment 641개, CODEBRIM 2,815개다. 파생 crop에도 두 항목 모두 음성인 6,399행이 있다. 이 crop은 기존 TRAIN 부모에서 생성한 자료이며 독립 사진 수가 늘어난 것은 아니다. Base의 `full`이라는 구현 명칭에는 CODEBRIM 출판자 crop/patch도 포함된다.

## 소스별 known과 unknown

| 소스 | known 사진 정답 | 항상 unknown인 항목 | 음성 해석 |
|---|---|---|---|
| DACL 원본 TRAIN | 기존 7종과 출판자 19종 사진 태그 | 추가로 정의하지 않은 도장·미장 유형 | 지정한 태그의 부재. 전체 정상·안전 정답이 아님 |
| DamSegment TRAIN | 탐지·분할의 균열·박락 annotation, 분류 Non-Crack의 저자 intact 정의에 따른 두 항목 음성 변환 | 나머지 5종 | 직접 분할 정답과 분류의 의미 변환을 구분. 일반 정상·안전 정답이 아님 |
| CODEBRIM TRAIN | 균열·박락·녹 얼룩·철근 노출·백화 | 젖은 표면·표면 공동 | 출판자 Background도 5종 부재만 known |
| 기존 S2DS TRAIN supplement | 균열·박락·녹 얼룩·백화 | 철근 노출·젖은 표면·표면 공동 | 원본 semantic mask가 지정한 4종의 부재만 known |
| ConViD supplement | 출판자 crack/Spalling 폴더의 해당 양성 1종 | 나머지 6종 전부 | 음성 정답 없음. 폴더가 다른 결함의 부재를 증명하지 않음 |

CODEBRIM 기존 TRAIN에는 출판자가 Background로 분류한 2,185 patch가 있으며 437 원본 parent ID에서 왔다. 이 자료도 젖은 표면·표면 공동을 0으로 승격하지 않는다. 모호한 모든 XML 항목이 0인 기존 원본 record는 준비 단계에서 배제되어 있다.

### DamSegment Non-Crack 음성의 근거

공식 데이터 소개의 `crack vs. non-crack` 폴더 이름만으로 박락 음성을 확정할 수는 없다. [저자 원논문 4.4](https://pmc.ncbi.nlm.nih.gov/articles/PMC13247583/#sec0016)는 분류 Crack에 균열·박락 같은 눈에 보이는 표면 손상이 포함되고 Non-Crack은 intact concrete라고 정의하며, 분류 예측을 저자들이 수동 확인했다고 설명한다. 기존 준비 코드는 이 의미를 근거로 Non-Crack을 두 항목 음성으로 변환했다. Classification Crack의 두 항목 양성을 분리할 수 없으므로 그 1,000개는 기존 준비 단계에서 제외했다.

현재 Dam TRAIN base 1,585행은 직접 탐지·분할 정답이 있는 946행과 Classification/Non-Crack 639행이다. 기존 두 항목 모두 음성인 641행은 탐지·분할 2행과 분류 Non-Crack 639행으로 구성된다. 분류의 박락 0은 **별도로 배포된 Spalling=0 태그나 native classification mask가 아니라 저자 intact 서술에 근거한 변환**이다. 이 분류 사진의 공간 negative도 사진의 항목 부재에서 유도한 격자 정답이며 native 분할 mask를 읽은 결과라고 표현하지 않는다.

근거는 DOI `10.1016/j.dib.2026.112671`, PMCID `PMC13247583`, 기존 JATS XML SHA `ca6c883d943c1cc2c00f1d16b61e49efc48525da584d61a0576af1cd189a7258`에서 확인했다. [공식 데이터 기록](https://data.mendeley.com/datasets/z5z6gtt5t4/1)의 데이터 라이선스는 CC BY 4.0, 원논문 본문의 라이선스는 CC BY-NC 4.0으로 구분한다.

논문 자체도 범위를 RGB에서 확인 가능한 균열·박락 두 항목으로 제한하고 수동 정답의 경계·심각도 해석 차이를 한계로 설명한다. intact라는 이름을 다른 모든 손상·숨은 결함·구조적 안전의 음성으로 확장하지 않는다. 현재 고정 대조 실험의 TRAIN/VAL/TEST와 역사적 benchmark는 변경하지 않았다. 이 저자 정의가 확인되므로 단지 폴더 이름 때문에 TRAIN 박락을 unknown으로 바꾸어야 한다고 결론 내리지는 않는다. 추후 별도의 불확실한 사진이나 반대 근거가 확인되면 새 TRAIN 버전에서 해당 클래스의 photo/pixel unknown 처리와 고정 대조를 검토하고 기존 결과와 구분한다.

## DACL 19종 정보로 가능한 일

출판자 vocabulary는 `ACrack, Bearing, Cavity, Crack, Drainage, EJoint, Efflorescence, ExposedRebars, Graffiti, Hollowareas, JTape, PEquipment, Restformwork, Rockpocket, Rust, Spalling, WConccor, Weathering, Wetspot`이다. 원본 TRAIN 6,225개 annotation의 SHA·태그를 auxiliary manifest 및 기존 7종과 대조한다.

`paint` 또는 `plaster`라는 원본 정답은 없다. Graffiti를 일반 도장으로, Hollowareas를 미장으로, Weathering을 정상 거친 콘크리트로 바꾸지 않는다. 다음 집계는 모양 혼동을 줄이는 실험 후보의 proxy이며 재분류 정답이 아니다.

| 기존 native 태그 조건 | 균열·박락 모두 음성인 DACL 전체 사진 수 |
|---|---:|
| EJoint 또는 JTape | 481 |
| Weathering/Hollowareas/Rockpocket/WConccor 중 하나 | 1,030 |
| Graffiti | 341 |
| 위 조건 전체 합집합 | 1,458 |

기존 auxiliary head는 이러한 19종 정답을 이미 사용한다. 적용 범위는 원본 DACL TRAIN 전체 사진뿐이며 다른 소스나 crop에는 auxiliary unknown을 유지한다. 부모가 가진 태그를 부분 crop에 그대로 복사하면 실제 crop에 없는 손상이 있다고 가르칠 수 있다.

DACL의 기존 7종이 모두 0인 전체 사진은 960개지만, 이 안에도 Weathering 310개·Graffiti 189개 등 다른 native 태그가 있다. 원본 19종 annotation이 비어 있는 53개도 전문가의 현장 정상·안전 확인 사진이라고 표현하지 않는다.

## 앱의 7종과 source vocabulary가 다른 부분

| 기존 내부 출력 | source와 연결 | 해석의 한계 |
|---|---|---|
| concrete_crack | DACL Crack+ACrack, Dam crack, CODEBRIM Crack | 역사적 내부 키이며 API는 surface_crack으로 연결한다. 사진마다 콘크리트 재질을 검증했다는 뜻은 아님 |
| concrete_spalling | DACL Spalling, Dam spall, CODEBRIM Spallation | 일반 도장 박락·미장 탈락을 별도로 판정하는 정답 없음. Scaling을 자동으로 이 항목으로 바꾸지 않음 |
| rust_stain | DACL Rust, CODEBRIM CorrosionStain | 별도 YOLO 산업 금속 metal_corrosion 정답과 동일한 범위라고 단정하지 않음 |
| exposed_rebar | DACL ExposedRebars, CODEBRIM ExposedBars | 일반 노출 배선·부품 결함으로 확장하지 않음 |
| wet_surface | DACL Wetspot | 젖은 표면이며 배관 누수 원인·누수 위치를 증명하지 않음 |
| efflorescence | DACL/CODEBRIM Efflorescence | 흰 도장·먼지·석고를 전문가 정답 없이 교정하지 않음 |
| surface_cavity | DACL Cavity | Hollowareas/Rockpocket 및 임의 구멍을 모두 이 항목으로 합치지 않음 |

학습 자료의 상당수는 교량·콘크리트 표면이다. 공장 설비 전체, 실제 산업 현장의 오류율, 7종 외의 위험을 검증한 자료라고 표현하지 않는다. AI는 점검 의견을 보조하고 최종 현장 판단은 담당자가 확인한다.

### 2026-10-05 원본 분류 정의 재확인

공식 논문은 `Crack`과 `ACrack`(Alligator Crack)을 콘크리트 결함으로 분류한다.
ACrack은 여러 균열이 가지 친 망상 형태다. 현재 두 태그를 합친 출력은 두 형태를
따로 판별한 결과가 아니며, API의 `surface_crack`도 개별 사진의 재질 판정 결과는 아니다.
[DACL10k 원논문 A.3·A.5](https://arxiv.org/html/2309.00460)

논문은 `Spalling`, `Rockpocket`, `WConccor`의 외관이 비슷해도 원인과 평가가 다르다고
설명한다. `Hollowareas`는 타격 검사 후 남긴 분필 경계의 주석이며 사진만으로 숨은
공동을 직접 확인한 정답이 아니다. 현재 준비 코드는 이 태그들을 `Spalling`이나
`Cavity`에 합치지 않는다. [DACL10k 원논문 3.2·A.5](https://arxiv.org/html/2309.00460)

공식 toolkit의 v2 vocabulary는 19종이며 첫 arXiv 버전의 18종과 구분한다.
현재 native 19종 보조 정답과 7종 출력 매핑을 유지했다.
[저자 toolkit의 버전·클래스 설명](https://github.com/phiyodr/dacl10k-toolkit#labelsclasses)
이 확인으로 원본 사진의 정답을 교정하거나 이번 고정 실험의 평가 기준을 바꾸지는 않았다.
후속 분석에서는 기존 공통 지표와 함께 원본 태그별 오류를 별도 진단할 수 있다.

## 이번 손상 구분 실험의 설계

아래 고정 설계로 대조군·보강군 각각 6epoch를 실제 완료했다. 최대 검증 오류는 초기 모델 22.28% / 대조군 23.06% / 보강군 22.28%로 초기 모델보다 낮아지지 않았다. 연구 후보로 채택하지 않았으며 다음 우선순위와 항목별 관측 오차 범위는 [완료 결과](FACILITY_TARGET_DISCRIMINATION_RESULTS_KO.md)를 따른다. 아래 내용은 실행한 계획의 기록이다.

새 데이터를 대량 추가하기 전에 기존 auxiliary 모델과 기존 TRAIN을 유지한 대조 실험을 권한다. Treatment에서만 전체 사진의 같은 domain 안에 있는 known 양성과 known 음성의 점수 순서를 학습하는 작은 ranking loss를 추가한다. 예를 들어 균열의 양성/음성 pair는 균열 정답만 보고 만들며 박락가 unknown이라고 정상으로 간주하지 않는다.

- 기존 7종·pixel loss·auxiliary loss·domain 비율·full/crop 표본 추출·초기 모델·seed·학습 횟수를 대조군과 같게 둔다.
- 새 ranking 항은 후보 고정값 0.25 한 개로 먼저 비교한다. ConViD 사진과 파생 crop을 ranking pair에 포함하지 않는다. class별 known 양/음성이 같은 domain에 모두 있을 때만 적용한다.
- 기존 sampler의 균열 기대 양성/음성 비율은 약 36.96%/63.04%, 박락은 약 42.02%/57.97%다. 음성 표본 자체가 없는 상태는 아니다. 분포·정답 범위·혼동 유형을 따로 확인하고, 어려운 구분을 돕는지 비교한다.
- Batch size 8에서 Dam/Code의 같은 domain pair가 드물 수 있으므로 실제 유효 pair 수와 ranking loss를 domain·class별로 기록한다. 유효 pair가 없으면 분모를 1로 보호하고 0을 반환하며 unknown에는 gradient를 주지 않는다.
- 모델 점수로 어려운 표본을 찾는다면 고정 초기 모델의 TRAIN 점수만 사용한다. 이미 시도된 일반 hard mining과 새 실험을 섞지 않는다. native proxy 조건만으로 가중치를 바꾸는 실험도 ranking과 동시에 넣지 않는다.

기존 고정 validation은 대조군과 후보를 비교하는 용도로만 사용한다. 개별 validation/test 오답을 보고 TRAIN 사진을 선정하거나 원본 labels를 수정하지 않는다. 반복 실험으로 validation에 맞춰졌을 가능성은 별도 한계로 남는다. 개선을 확인하기 전 성능 향상이나 5% 미만 오류를 약속하지 않는다.

## 공식 출처와 조건

- [DACL 저자 toolkit](https://github.com/phiyodr/dacl10k-toolkit): 교량의 19종 semantic annotation과 CC BY-NC 4.0. 기존 `datasets/facility_sources.json`의 원본 SHA·citation과 연결한다.
- [CODEBRIM 공식 공개 기록](https://zenodo.org/records/2620293): 로컬 원본 이용 조건 SHA와 기존 preparation audit를 재사용한다. 비상업 연구·교육 조건과 원본·변형 데이터 재배포 제한을 유지한다. 이번 공개 보고서에는 사진·개별 annotation 경로를 넣지 않는다.
- [ConViD V4 공식 기록](https://data.mendeley.com/datasets/fx3rthfjhy/4): CC BY 4.0. 원본 폴더 정보만으로 전문가가 미장·도장 라벨을 교정했다고 주장하지 않는다.
- 외부 후보인 [SDNET2018 공식 대학 기록](https://digitalcommons.usu.edu/all_datasets/48/)과 [Concrete Crack Images for Classification V2](https://data.mendeley.com/datasets/5y9wdsg2zt/2)는 균열 음성과 CC BY 4.0을 제공한다. 둘 다 많은 사진이 소수의 원본에서 나온 작은 patch여서 일반 산업 정상·박락 음성을 증명하지 못한다. 이번에는 받거나 학습하지 않았다. 기존 접근 차단에 대한 우회·재시도도 하지 않았다.
