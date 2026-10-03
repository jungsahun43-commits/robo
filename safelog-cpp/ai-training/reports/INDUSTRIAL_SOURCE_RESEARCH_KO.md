# 산업체 환경에 가까운 공개 자료 후보 조사

조회일: 2026-10-03, 한국 시간 기준. 이 조사에서는 공식 데이터 페이지·저자 저장소·공개 파일 메타데이터를 읽고 HEAD 접근만 확인했다. 조사 자체에서는 사진·아카이브를 다운로드하거나 학습하지 않았다. 이후 주 작업의 실제 취득 상태는 아래 추가 확인에 따로 기록한다. 용량은 공식 메타데이터 값이며, 원본 파일 수를 독립 장면 수로 해석하지 않는다.

## 후보와 순서

| 환경 적합성 순서 | 공식 후보 | 확인한 환경·주석 | 공식 용량 | 취득 상태 |
|---|---|---|---:|---|
| 1 | [IHRCD-Det](https://github.com/eggnog1307/IHRCD-Det) | 중국 산업유산 철근콘크리트 보·슬래브·기둥·벽 등. 578개 원본에서 6,388개 1024² 조각, 8종 손상 탐지 | 공개 README에 바이트 용량 없음 | Baidu 배포 접근이 브라우저 보안 정책으로 차단돼 취득 중단. raw downloaded=false, trained=false |
| 2 | [PECCD V1](https://data.mendeley.com/datasets/w7549ryvx2/1) | 2023년 지진 후 시리아 해안 지역의 손상 건물. 여러 균열 유형·박락·scaling·holes | RAR 3,767,446,444바이트 | 이후 주 작업에서 공식 SHA 검증 완료. 클래스 숫자 ID 미확인으로 학습 보류 |
| 3 | [ConViD V4](https://data.mendeley.com/datasets/fx3rthfjhy/4) | 인도 Pune 야외 콘크리트 스마트폰 사진. crack·spalling·honeycomb·void 폴더 분류 | 1,672파일 합계 5,001,479,789바이트 | 이후 주 작업에서 고정200사진만574,082,757B 취득·개별SHA 확인. 중복 선별 후TRAIN192장 준비 |
| 4 | [Concrete Crack Images for Classification V2](https://data.mendeley.com/datasets/5y9wdsg2zt/2) | METU 캠퍼스 건물 콘크리트의 균열 유무. 458개 고해상도 사진에서 40,000개 227² 조각 | RAR 241,363,336바이트 | 공식 익명 파일 URL의 HEAD200, 본문 미다운로드 |

환경 적합성 순서와 즉시 실행 순서는 다르다. IHRCD는 배포 접근이 차단돼 현재 사용할 수 없으며 우회 취득을 시도하지 않는다. PECCD는 파일 해시를 확인했으나 숫자 ID를 클래스 이름으로 연결할 근거가 없어 학습을 보류한다. ConViD의 공식 명명 폴더는 제한된 양성 사진 보강 후보이며 아래 조건을 검토한다. CCIC는 작은 대안이지만 균열 한 항목 보강에 한정된다.

## 1. IHRCD-Det: 환경은 가장 가깝지만 취득 조건 확인 필요

Shougang·Wuhan Iron and Steel 등 산업유산 현장에서 촬영했으며, 구조 부재는 보·슬래브·기둥·벽·지붕/트러스·설비 기초로 설명된다. 원본8종은 Crack, Spalling, Exposed reinforcement, Rust stain, Chemical corrosion, Hole, Repair mark, Anthropogenic mark이다. [저자 설명](https://github.com/eggnog1307/IHRCD-Det), [CC BY4.0 명시 LICENSE](https://github.com/eggnog1307/IHRCD-Det/blob/main/LICENSE).

현재7항목에 대한 보수적 매핑 제안은 Crack→균열, Spalling→박락, Exposed reinforcement→철근 노출, Rust stain→녹 흔적이다. Hole은 기존 Cavity 정의와 실제 주석을 확인한 후 판단한다. Chemical corrosion을 백화·젖은 표면으로 바꾸거나 Repair mark를 정상으로 자동 지정하지 않는다. 자료에 주석되지 않은 항목은 미확인으로 유지한다.

산업유산은 가동 중인 모든 공장 환경을 대표하지 않는다. 6,388개 조각을 6,388개 독립 현장으로 세지 않고 원본578개 부모와 장소별 분할을 확인해야 한다. README의 저자 벤치마크를 우리 모델의 정확도로 표시하지 않는다. 현재 Baidu 배포 접근은 브라우저 보안 정책으로 명시 차단됐으며, 자동 승인 검토의 거부와 구분한다. 다른 브라우저·API·우회 다운로드를 시도하지 않는다. 공식 GitHub 설명과 라이선스 확인만 증거로 유지한다.

## 2. PECCD: 건물 균열·박락 후보

공식 출처는 지진 피해 건물이며 공장이라고 명시되지 않았다. 손상 양상이 일반 산업체 일상 점검과 다를 수 있다. 균열 유형은 현재 균열 항목, spalling은 박락으로 검토할 수 있으나 scaling·holes는 원본 정의와 정답 파일을 먼저 확인한다. 공식 페이지의 분류 이름만으로 다른7항목이 없다고 판단하거나 위치 마스크를 만들지 않는다. CC BY4.0. [공식 출처](https://data.mendeley.com/datasets/w7549ryvx2/1).

- 파일: `SyrianPostEarthquakeCrackDataset.rar`
- 공식 SHA256: `9acc37537c2a618c22bbda55d1353823d4e2dd82d98ae729d0c30315c72388ec`
- [익명 다운로드 주소](https://data.mendeley.com/public-files/datasets/w7549ryvx2/files/9765aed0-e9bb-4af2-9129-130d664d02d5/file_downloaded)
- 메타데이터 용량: 3,767,446,444바이트. HEAD200 및 최종 응답 Content-Length 일치.

추가 확인: 주 작업에서 원본 다운로드와 공식 SHA256 검증을 완료했다. 압축파일 일부 항목의 파일명 읽기 문제와 원본 클래스 숫자 ID 정의 미확인이 남아 있다. 공식 [저자 SSRN 초록](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5758816)은6종 손상 이름과 LabelImg 사용을 설명하지만 숫자 ID→이름 대응을 명시하지 않는다. 논문 PDF 조회는403으로 접근되지 않았다. 웹페이지·초록의 클래스 나열 순서를 숫자 ID 순서로 가정하지 않는다. **명시 클래스 매핑이 확인되기 전에는 이 자료로 학습하지 않는다.**

## 3. ConViD: 스마트폰 콘크리트 사진, 시설 종류 불명

최신 V4는 2026-09-12 공개됐으며 소비자 스마트폰으로 Pune 야외 콘크리트를 촬영했다고 설명한다. 공장 벽·바닥·기둥이라는 시설 종류는 확인되지 않았으므로 **스마트폰 콘크리트 사진 후보**로 기록한다. CC BY4.0. [공식 출처·촬영 방법](https://data.mendeley.com/datasets/fx3rthfjhy/4).

| 폴더 | 공식 파일 수 | 메타데이터 용량 |
|---|---:|---:|
| crack |365 |1,125,767,328바이트 |
| Spalling |576 |1,662,697,710바이트 |
| Honeycomb |511 |1,998,559,220바이트 |
| VOID |220 |214,455,531바이트 |

파일 메타데이터는 각 파일의 SHA256과 직접 URL을 제공한다. 클래스별 위치 주석은 이 조사에서 확인되지 않았다. 균열·박락 폴더는 해당 양성만 보수적으로 사용하고 다른 항목을 미확인으로 두는 검토가 필요하다. Honeycomb을 박락으로 자동 합치지 않으며 VOID도 현재 공동 클래스와의 정의 일치를 확인한다. 한 손상의 여러 각도 촬영을 서로 독립된 장면으로 나누면 안 된다. 파일 크기가 다양한 만큼 품질·중복·출처 일관성을 실제 취득 후 점검한다.

공개 목록 API 형식: `https://data.mendeley.com/public-api/datasets/fx3rthfjhy/files?folder_id=FOLDER_ID&version=4&$start=0&$limit=1000`. 폴더 ID는 [공식 폴더 메타데이터](https://data.mendeley.com/public-api/datasets/fx3rthfjhy/folders/4)에서 조회한다. 이 형식으로 각 폴더를 한 번씩 읽었으며 모든 폴더가1000파일 미만이었다.

### 제한된 양성 보강의 조건

공식 `crack` 폴더는 균열1, `Spalling` 폴더는 박락1로 사용할 수 있는 후보이다. 해당 사진의 나머지6항목은 `-1` 미확인으로 두며, 상대 손상을0으로 만들지 않는다. 폴더 분류 정답으로 위치 마스크를 생성하지 않는다. Honeycomb·VOID는 이 제한 보강에서 제외한다.

전체5GB를 모두 취득하기 전에 주 작업에서 보강 사진 수와 선정 규칙을 고정한다. 동일 손상의 연속 촬영·원본 그룹, 중복/근접/부분 잘림 및 기존 VAL/TEST와의 겹침을 확인하고 겹친 부모를 학습에서 제외한다. 사진에 표시된 양성이 실제로 보이는지도 검수한다. 명확하지 않은 경우는 정답 변경과 구분된 검수 의견으로 남긴다.

양성만 추가하면 모델이 손상 있다고 제안하는 비율과 오탐률이 증가할 수 있다. 기존 정상·혼동 TRAIN 자료를 유지하고 제한된 추출 비중으로 비교하며, 기존 세 검증 자료의 균열·박락 미탐률과 오탐률을 모두 확인한다. 이 보강을 산업체 현장 정확도 검증이나5% 목표 달성으로 설명하지 않는다. 이 조사 단계에서 ConViD raw downloaded=false, trained=false이며 실제 제한 취득·학습은 주 작업의 별도 결정과 기록을 따른다.

## 4. CCIC: 작은 건물 콘크리트 균열 자료

공식 페이지는 METU 캠퍼스 건물 촬영과 균열 양성/음성 각20,000개, 458개 원본 사진으로부터의 파생을 명시한다. CC BY4.0. 원본458개 그룹의 식별 가능 여부를 확인하고, 40,000개 조각의 임의 분할 정확도를 새로운 현장 성능으로 발표하지 않는다. 균열 유무만 주석됐으므로 박락·녹·철근·젖음·백화·공동은 미확인이다. 공장이나 벽/바닥/기둥별 검증 자료로 이름을 바꾸지 않는다. [공식 출처](https://data.mendeley.com/datasets/5y9wdsg2zt/2).

- 파일: `Concrete Crack Images for Classification.rar`
- 공식 SHA256: `08d7dc505a4f5a0330cee2fa2a1ae4b5b4f98bcffd0549b774caf7959bb1f02f`
- [익명 다운로드 주소](https://data.mendeley.com/public-files/datasets/5y9wdsg2zt/files/8a70d8a5-bce9-4291-bab9-b48cfb3e87c3/file_downloaded)
- 메타데이터 용량: 241,363,336바이트. HEAD200 및 최종 응답 Content-Length 일치.

## 보류한 후보와 조사 한계

- [CUBIT-Seg 저자 저장소](https://github.com/CUHK-USR-Group/Defect-Dataset): 건물 외벽 균열5,462개·박락1,160개와 픽셀 주석은 유용해 보인다. 이 조사에서는 데이터에 적용되는 명시 라이선스와 실제 익명 취득 조건을 확인하지 못해 즉시 학습 후보에서 보류한다.
- [HU Infrastructure Cracks](https://hu-infrastructure-cracks.org/): 캠퍼스의 벽·포장 균열과 부모 crack_id를 설명한다. 페이지 최신136기록과 [Zenodo V1의48기록](https://zenodo.org/records/20829348)이 다르므로 버전 고정·라벨·다운로드 실물을 추가 확인해야 한다. 공장 자료라고 부르지 않는다.
- SDNET·DeepCrack·MVTec나 금속 제조품의 표면 결함 자료를 산업체 시설 사진으로 바꾸어 설명하지 않는다. 기존 교량·댐 자료도 원래 출처를 유지한다.

추가 학습 전에 파일 해시·라벨 정의·원본 부모·동일/유사/잘린 사진 중복을 검사한다. 기존 VAL/TEST와 겹치는 새 자료를 학습에 넣지 않고, 원래 정답과 분모를 바꿔5% 목표를 통과시키지 않는다. 취득한 자료가 공장 현장의 성능을 검증하는 평가 자료인지, 보조 학습 자료인지를 따로 기록한다.

ConViD 실제 준비 기록은 [공개 감사](facility-convid-data-audit.json)와 [대조 실험 계획](FACILITY_BUILDING_SUPPLEMENT_PLAN_KO.md)에 있다. 공식 출처 인용은 Gayakwad, Milind (2026), “ConViD — Concrete Visual Defect Dataset”, Mendeley Data, v4, DOI10.17632/fx3rthfjhy.4이다. 동일 손상의 여러 각도 사진이 포함되므로192장의 준비 사진을192개의 독립 현장이라고 표시하지 않는다. 샘플8장의 AI 육안 확인은 전문가 클래스 정의 검증이나 정확도 측정이 아니다.
