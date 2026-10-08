# RC2119 추가 자료 검사와 제한된 학습 범위

검사일: 2026-10-08. [공식 RC2119 V1](https://data.mendeley.com/datasets/2vkm6k4cfg/1), DOI10.17632/2vkm6k4cfg.1, CC BY4.0.

## 실제로 확보·검사한 자료

사진·JSON·마스크 ZIP3개, 총160,207,406바이트를 공식 배포 URL에서 받았다. 배포자 전체 SHA256·파일 크기·모든 ZIP CRC를 확인하고 새 데이터 폴더에만 해제했다. 세 파일 묶음 각각2,119개 파일이 같은 basename으로 대응한다. 원본 자료는 Git에 올리지 않는다.

| 저자 항목 | 실제 JSON 양성 사진 | 실제 polygon instance | 원본 mask ID | 학습 사용 |
|---|---:|---:|---:|---|
| Crack | 1,315 | 1,833 | 1 | 명시적 균열 양성만 |
| Concrete spalling | 184 | 282 | 2 | 명시적 박락 양성만 |
| Rebar exposure | 82 | 89 | 3 | 이번 추가 학습에서는 미확인 |
| Rebar corrosion | 662 | 832 | 4 | 녹 흔적으로 변경하지 않음 |
| Concrete crushing | 81 | 88 | 5 | 박락·공동으로 변경하지 않음 |

같은 사진에 여러 항목이 있을 수 있어 양성 사진 수를 더해 전체 사진 수로 해석하면 안 된다. 공식 설명의 균열 instance1,883개와 실제 ZIP의1,833개는50개 차이가 있다. 이번 기록과 정답은 실제 원본 파일을 기준으로 한다.

설명 페이지는 색상 마스크를 소개하지만 다운로드한 PNG는 단일 채널 `L`, background0 및 ID1–5다. 단일 클래스 JSON과 마스크가 짝을 이루는 사진에서 모든5종 ID의 일관된 대응을 확인했다. 설명의 빨강·초록 RGB 값을 회색 PNG에 대입하지 않았다.

## 정렬·중복 검사

- 사진29장은 EXIF 촬영 방향을 적용하면 JSON·마스크 크기와 맞는다. 방향을 추측하거나 임의로 마스크를 회전하지 않았다.
- 사진6장의 polygon 좌표는 프레임 밖 등 검토가 필요하다. 이들의 전체 사진 near 그룹까지8장을 격리했다.
- 원래 공개 자료 모든 분할과 기존 ConViD 원본을 합친19,135개 사진 기록에 대해 native decoded RGB SHA와 전체 사진 dHash를 비교했다. 전체 사진 중복 의심59장과 위 좌표 그룹8장을 제외했다.
- OpenCV SIFT로 기존 사진에서1,577,652개 국소 특징을 추출하고 새2,119장을 비교했다. homography 조건에 맞는 shared-view 후보25장을 찾았으며, 앞선 제외와 겹치지 않는24장을 추가로 제외했다.
- 최종적으로2,028장, 전체 사진 near 그룹 대표1,592개가 남는다. 남은 박락 양성 사진은179장이다. “검사를 통과한 후보”이며 독립적인179개 현장이나 부모 사진으로 확인된 수가 아니다.

해시·국소 특징 검사는 발견한 중복 후보를 보수적으로 제외한다. 검출되지 않은 crop·회전·같은 현장 사진까지 없다는 증명은 아니다. 상세 사진 경로·개별 매칭·원본 주석은 제외된 로컬 파일에 보관한다. 보류 사진의 정답이나 모델 점수를 자료 선정에 사용하지 않았다.

## 새 학습에 넣을 정답

전체 사진 near 그룹을 중복 선택하지 않도록 박락100·균열100개 그룹 대표, 합계200장을 선택했다. 이 사진들에 균열 양성117개·박락 양성100개의 항목·사진 사례가 있다. 새 사진의 표시되지 않은 항목과 배경을 음성으로 채우지 않는다.

저자가 표시한 foreground 픽셀이 하나라도 있는80×80 셀을 양성 위치로 추출한다. 셀 값0은 미확인이다. 새 사진의 기존 spatial loss known mask는 전부0으로 두고, 별도의 양성 위치 손실만 명시된 셀을 학습한다. 새 음성 사진·배경 음성 픽셀·19종 보조 태그 정답은 모두0개다.

원래 학습 자료·정답은 보존한다. 매 epoch14,248번 표본 추출 중704번(4.94%)을 새 자료로 바꾸고, 후보6epoch를 계획했다. 원래7종/19종/80 지도 구조·BN 통계 동결·기존 항목 교사·학습률을 유지한다. 과거6epoch 대조군은 재사용한다. 데이터와 추가 양성 위치 손실을 동시에 바꾸므로 데이터만의 인과 효과를 입증하는 비교는 아니다.

본 문서는 데이터 확보·정답 검사 기록이다. 실제 학습 완료와 오탐·미탐 결과는 별도 `FACILITY_RC_POSITIVE_STUDY_RESULTS_KO.md` 및 검증 JSON이 생성된 뒤 확인한다. 계획한6epoch, 사전 점검 업데이트 또는 데이터 검사 자체를 완료 학습량으로 세지 않는다.

## 재현 순서

기존 원본 데이터와 원래·과거 대조군 가중치/로컬 증거가 갖춰진 새 workspace에서 실행한다. 이미 완료된 검사·학습 폴더를 덮어쓰지 않는다.

```powershell
.venv/Scripts/python.exe scripts/fetch_rc2119.py --download --extract
.venv/Scripts/python.exe scripts/audit_rc2119.py
.venv/Scripts/python.exe scripts/screen_rc2119_crops.py
.venv/Scripts/python.exe scripts/facility_rc_positive.py prepare
```

학습 전 조건은 Git의 `facility-rc-positive-study-protocol.json`에 고정되어 있다. 원래 소스112개와 이번 소스11개, 총123개의 바이트를 보존해야 한다. TEST 추론과 앱 기본 프로필 교체는 이 작업에 포함하지 않는다. 같은 source-VAL의 반복 선택 결과를 독립 공장 현장 성능으로 해석하지 않는다.

## 저자가 요청한 인용

Wang, J., & Ueda, T. (2025). Automatic damage detection and segmentation using deep learning algorithms in reinforced concrete structure inspections. Structural Concrete26(5),5511–5534.

Wang, J., Wang, Z., Wang, Y., & Li, Z. (2025). Automated multi-type damage detection framework in reinforced concrete structures via data augmentation and deep segmentation networks. Journal of Civil Structural Health Monitoring15(8),3861–3884.

이번 사용에서는 선언된 방향 보정·이미지 축소·foreground 셀 추출을 적용한다. CC BY4.0 출처와 변경을 표시하며 저자의 승인이나 안전 인증을 뜻하지 않는다.
