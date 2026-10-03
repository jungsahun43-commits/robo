# 기존 산업 현장 데이터 재사용 검토

확인일: 2026-10-03. 기존 파일·학습 기록을 읽고 공식 출처를 확인한 감사 문서다. 이 검토에서 원본·파생 데이터 복제, adapter 변경, 추론 또는 학습을 실행하지 않았다.

## 결론

**Ostrava는 이미 확보한 자료 중 산업 시설의 금속 표면에 가장 가까운 재사용 후보다.** 다만 현재 사진 전체 7종 분류기의 `rust_stain`과 원본의 `metal_corrosion`을 같은 정답으로 자동 처리하지 않는다. 가장 작은 유효한 확장은 원본 금속 부식 mask를 사용하는 **학습 전용 보조 공간 감독**이며, 앱의 7종 출력은 유지한다. 이 확장의 효과는 아직 측정하지 않았다.

SH17은 제조업 환경의 사람·보호구 사진으로, 시설 7종의 정답이 없다. 해당 사진을 시설 전체 정상이나 균열·박락 음성으로 자동 추가할 수 없다. 두 출처에는 균열·박락의 감독 정답이 없으므로 현재 균열·박락 오차 목표를 달성할 근거로 삼을 수 없다.

## 실제 디스크와 학습 기록

| 출처 | 실제 분할·정답 | 기존 사용 | 현재 7종 분류기 직접 입력 |
|---|---|---|---|
| Ostrava v2 | 원본 사진 105개. 공식 TRAIN 75 / validation 15 / test 15. 금속 부식 RGB mask | `facility-corrosion` YOLO 120epoch, `facility-corrosion-refined` 13epoch. 앱에 `metal_corrosion` 별도 모델 연결 기록 있음 | core와 작은 영역 후보 manifest 모두 미포함 |
| SH17 v1 | 원본 목록 TRAIN 6,479 / val 1,620. 17종 보호구·사람·신체·도구 bbox. 원본 크기를 줄인 `sh17-1280` 복사본 8,099개 | `sh17-ppe` YOLO 100epoch, 이를 초기 모델로 한 `ppe-sh17-transfer` 70epoch | core와 작은 영역 후보 manifest 모두 미포함 |

기존 두 facility 분류기 manifest의 도메인은 DACL 16,811행, DamSegment 3,040행, CODEBRIM 6,438행이다. 이 수에는 TRAIN crop이 포함되므로 독립 원본 사진 수가 아니다.

SH17의 공식 클래스 ID는 저자의 YAML 순서로 처리했다. 현재 `prepare_sh17_training_copy.py`는 각 사진의 YOLO 클래스별 수를 원본 VOC XML 객체 수와 대조한 후 크기를 줄인다. 이 검증은 보호구 bbox 검증이며 시설 손상의 정답 검증은 아니다.

## Ostrava 원본 mask 확인

전체 105개 mask를 읽었다. 모두 2,048×2,048이고 원본 양성 색은 RGB `(61,245,61)`, 배경은 `(0,0,0)`이다. 기존 `convert('L') > 0` 판독에서 양성 값은 169다.

| 공식 분할 | 사진 | mask 양성 사진 | 빈 mask 사진 |
|---|---:|---:|---:|
| TRAIN | 75 | 72 | 3 |
| validation | 15 | 15 | 0 |
| test | 15 | 15 | 0 |

TRAIN의 73·74·75는 빈 mask다. 기존 YOLO records도 이 세 사진의 박스를 비워 두었다. 따라서 자료 전체가 엄밀히 양성 사진만으로 구성되었다고 쓰면 부정확하다. 반면 **validation/test에는 사진 단위 부식 음성이 없어 해당 분할의 FPR을 정의할 수 없다.** TRAIN의 빈 mask 3개를 검증 음성으로 이동하지 않는다.

TRAIN 73·74와 전체 양성 mask를 가진 TRAIN 04 사진을 시각 확인했다. 빈 mask 두 장은 서로 다른 표면 질감으로 보이며, 이것만으로 정상 금속 또는 시설 전체 정상이라는 의미를 부여할 수 없다. 원본의 mask는 금속 부식 과제의 주석이다. 다른 시설 종류의 상태를 설명하지 않는다. 재질이나 부식 정도를 새 전문가 정답으로 확정한 검토는 아니다.

기존 YOLO 변환은 `mask > 0`의 8방향 연결 영역을 박스로 만들고 원본 64pixel 미만 조각을 제외했다. 생성 박스는 TRAIN 482 / val 87 / test 138개이며 제외 조각은 전체 138개다. 이것은 새로운 사진 707개나 원본 의미 분할 성능을 뜻하지 않는다. 새 공간 감독에서는 원본 mask와 기존 YOLO 박스의 정답 범위를 구분해야 한다.

## 7종·unknown 처리

| 현재 출력 | Ostrava에서 확인되는 감독 | SH17에서 확인되는 감독 |
|---|---|---|
| 표면 균열·콘크리트 박락 | 없음: `-1` | 없음: `-1` |
| 녹 얼룩 (`rust_stain`) | 원본은 금속 부식 (`metal_corrosion`). 정답 정의 대조 없이 `1/0`으로 전달하지 않음 | 없음: `-1` |
| 철근 노출·젖은 표면·백화·표면 공동 | 없음: `-1` | 없음: `-1` |

SH17 전체 사진의 main 7종이 모두 unknown이면 기존 masked 감독 손실은 0이다. 사진 수만 늘려 넣어도 시설 정답을 학습하는 효과는 생기지 않는다. 자체 모델의 예측을 검증 정답으로 다시 저장하지 않는다.

Ostrava의 native 과제를 별도 보조 head로 학습할 경우 main 7종은 unknown이고, `metal_corrosion`의 공간 감독만 알려진 것으로 처리한다. 보조 head의 추론 결과를 현재 `rust_stain` API로 자동 변환하지 않는다. 현재 앱의 별도 금속 부식 탐지는 기존 7종 사진 분류기와 구분해서 성능을 기록한다.

## 가장 효율적인 확장 1안

**기존 Ostrava TRAIN의 binary 부식 mask를 학습 전용 보조 공간 과제로 재사용하는 제한된 비교 실험**을 권고한다.

1. native `metal_corrosion` 정의를 보존한 1채널 공간 head를 공유 backbone에 추가한다. 추론 계약은 기존 7종으로 유지한다. 금속 부식 사진이 대다수 양성이므로 사진 전체 양성 태그만 반복하는 새 보조 분류 head보다 mask 내부의 양성·배경 정보를 이용할 수 있다.
2. 공식 TRAIN 75개만 후보로 한다. 3개 빈 mask도 원본 주석 그대로 기록하고, 다른 7종의 음성 정답으로 사용하지 않는다. source image/mask SHA, 공식 분할, 양성 면적, geometry, unknown을 adapter audit에 남긴다.
3. 아래 중복 검수를 통과한 TRAIN만 사용하고, 원래 시설 학습량과 새 출처의 노출량을 기록한다. 반복 crop이나 oversampling은 새로운 독립 사진으로 계산하지 않는다.
4. 기존 초기 모델·학습 예산·평가 방식과 비교하는 control을 둔다. 7종별 기존 세 출처의 미탐·오탐 및 AP 변화를 함께 비교해 균열·박락의 성능 하락도 확인한다. 금속 보조 감독의 개선이 7종 향상으로 이어진다고 미리 단정하지 않는다.

새 공장 금속 부식 사진을 가져온 확장이 아니라 **이미 YOLO에 사용했던 원본의 재사용**이다. 산업 현장 전체의 장면 다양성을 확보한 작업도 아니다. 핵심 목표가 균열·박락 오차 감소라면 별도의 정확한 균열·박락 정답을 가진 산업 시설 출처를 먼저 확보하는 편이 직접적이다. SH17 시설 전체 재주석이나 self-supervised 학습은 이 작은 확장보다 변경·검수 범위가 크다.

## holdout·근접 중복 관리 계획

- Ostrava 공식 val/test 15+15는 TRAIN으로 이동하거나 threshold·recipe 반복 선택에 추가 사용하지 않는다. 이미 기존 YOLO와 최적화에서 평가한 분할이므로 새 독립 공장 평가라고 표현하지 않는다.
- 신규 adapter를 만들기 전에 105개 전체의 파일 SHA·EXIF 보정 후 decoded RGB SHA·perceptual hash를 계산한다. source 내부와 기존 DACL/DamSegment/CODEBRIM/S2DS 원본 TRAIN/val/test에 대조한다. 기존 detail은 부모 ID로 연결하고, 타일끼리 다른 분할로 나누지 않는다.
- exact 또는 사전 지정한 near-duplicate 규칙으로 묶인 그룹은 분할을 넘지 않게 한다. 새 TRAIN이 기존 holdout과 겹치면 TRAIN 후보에서 제외하며 holdout을 학습으로 이동시키지 않는다. 필요하면 crop 관계를 ORB 등으로 별도 확인한다.
- 중복 선별은 사진과 원본 분할 메타데이터로 고정하고 모델의 오답 여부로 사진을 제거하지 않는다. 제외 수·이유·규칙·입력 SHA를 기록한다.
- 현장/설비/촬영 세션 ID가 없는 경우 perceptual/crop 검수가 장면 독립성을 증명하지 못한다. 학습 이후 실제 대상 시설의 양성·정상 사진을 별도 확보해야 사진 단위 오탐과 일반화 성능을 확인할 수 있다.

## 공식 출처와 이용 조건 확인

- [Ostrava v2 공식 Zenodo 기록](https://zenodo.org/records/11235637)은 산업 단지 부식 사진 105개와 mask·공식 분할을 설명한다. Matěj Frič, 2024, DOI `10.5281/zenodo.11235637`.
- 라이선스는 [DataCite 공식 DOI 등록 메타데이터](https://api.datacite.org/dois/10.5281/zenodo.11235637)의 `rightsList`에서 `Creative Commons Attribution 4.0 International`, SPDX `cc-by-4.0`, [공식 라이선스 링크](https://creativecommons.org/licenses/by/4.0/legalcode)를 확인했다. 로컬 `SOURCE.json`/출처 registry 기록과 일치한다. 이번에 Zenodo 웹 추출의 Rights 값이 비었고 직접 HTML 요청은 403이어서 공식 DOI 메타데이터를 읽었다. 원본 ZIP을 다시 받지 않았다.
- [SH17 저자 저장소](https://github.com/ahmadmughees/SH17dataset)는 제조업 환경의 사람·보호구 데이터와 CC BY-NC-SA 4.0 공개 조건을 설명하며 Pexels 원본 이용 조건도 함께 안내한다. 교육·연구용 사용과 출처·라이선스 기록을 유지한다. 원본·파생 사진은 현재 ignored data 디렉터리에 보관하며 본 문서에 배포하지 않는다.

## 읽은 로컬 근거

`datasets/facility_sources.json`, `datasets/sources.json`, `reports/preparation-ostrava-corrosion.json`, `reports/dataset-audit-ostrava-corrosion-yolo.json`, `reports/dataset-audit-sh17-1280.json`, `data/ostrava-corrosion/SOURCE.json`, Ostrava 원본 mask 105개와 YOLO records, `data/sh17-1280/PREPARATION.json`, 관련 preparation scripts, 기존 학습 `results.csv` 및 시설/PPE 결과 문서를 대조했다.

원본 `data/sh17/SOURCE.json`은 접근이 거부되어 읽지 못했다. SH17 분할·클래스·사용 기록은 접근 가능한 1,280px 사본의 preparation/audit와 기존 학습 기록으로 확인했다. 접근 실패를 원본 자료 재검증 완료로 계산하지 않았다.
