# LIACi v90–v93 원본·주석 표현 조사 결과

## 완료 범위

v90 고정 presetFIT6장(52/107/178/1/3/8)에 대응한 원본6개/categoryBMP30개/segmentationPNG6개만 취득했다. 각42멤버의 CRC·크기·encoded SHA를 검사했다. 새 Range 합3,059,018bytes/원래encoded 합7,523,829bytes이며 압축 Range·헤더·원본을 모두 보존했다. 전체1,044,750,953byte ZIP을 받거나 전체 SHA/모든멤버 CRC를 검사한 것은 아니다. 기존 v87–v89 ZIP metadata Range4,448,724bytes와 합하면7,507,742bytes이며 원논문 PDF/웹 텍스트 취득은 별도다.

원논문은 공식 NVA OpenFile의 공개 다운로드로5597231bytes를 보존했다. 로컬 SHA256 `cf27b8bce088effcec4b6d45c5fa61f20382fc14d49a15de6f641f14490767da`는 배포자 제공 SHA가 아니다. 본문 텍스트와 원논문의 클래스 정의 표 한 쪽만 확인했고 source 예시 사진/새source 시각QA는0이다. 논문은 물 배경 미주석, 검사 기준/물리부품 layer 중첩, 선박 이름 비공개를 설명한다. 누락 corrosion을 전체 무부식 사진 정답으로 바꾸지 않았다. 공식 앱의 데이터 CC BY-NC-SA4.0은 논문 CC BY4.0과 구별한다.

## 결과와 중단 이력

| 단계 | 사전 push | 실행 결과 |
|---|---|---|
|v90|574dda50027c96955aba68974fbbdfffde9eeb39|42멤버 취득/CRC 완료. ID107 corrosion BMP와 COCO 존재 불일치에서 중단|
|v91|48032b02651d25af59f952c1e8821b223bbe2c2e|기존3사진/15BMP/3PNG 수치·지문 저장. 다음 ID1의 JPEG 한정 assertion에서 중단|
|v92|cf2c2c099207a7dafd7efba99c2ca91f54ad4bd2|남은21헤더 완료. ID1/3은 .jpg 이름이지만 실제 PNG_RGB, native크기는COCO와 같음|
|v93|48a3435c90546397a1ea8c5107bfe5bdd37651e9|이미 관찰한 PNG1/3·JPEG8의 남은 수치 조사 완료. 기존3완료 결과는SHA 재사용|

최종 source RGB6개/categoryBMP30개/segmentationPNG6개 모두 native 크기가 COCO와 같았다. BMP는1비트이다. 이 6파일은 실제JPEG4/PNG2이며 extension만으로 decoder를 가정하지 않는다. 원본을 변환/덮어쓰지 않았다. sourceRGB 고유6개의 지문을 저장했지만 전체source/oldheld graph에는 아직 연결하지 않았다.

Pillow 원float polygon 표현과 제공 BMP의 **30개 layer 비교 중27개 exact/3개 비동일**이다. Pillow가 원저자 exporter라는 뜻은 아니다.

- ID52/178 corrosion은 각12,317/58,974픽셀로 reference와 exact였다.
- ID107 corrosion은 COCO polygon2개/reference29,076픽셀인데 제공BMP 흰색0이다. marine_growth는COCO0개인데 제공BMP4,654픽셀이다. hull은1,159,017픽셀로 exact이므로 전체 좌표 배율/반전으로 고칠 근거가 없다.
- ID3 paint_peel은 제공25,450/reference25,451픽셀로1픽셀 차이였다. 원본 경계를 수정하지 않았다.

이 세 차이를 전체 원주석이 틀렸다는 증명, 일부mask 교체, 잘 맞는 사진만 선택하는 근거로 사용하지 않는다. segmentation PNG는 원래 색별 수량/팔레트만 기록하고 classmap을 추정하지 않았다. COCO·BMP·PNG의 권위/생성·수정 시점 관계는 미확인이다. 직접 BMP↔COCO 계약은 **실패 상태 유지**, 진단 완료는 준비 통과가 아니다.

## 실제 해독·실패와 보존

v90 RGB6/BMP6/segPNG1 호출(고유RGB2), v91 RGB3/BMP15/segPNG3 호출(추가고유1), v93 RGB3/BMP15/segPNG3 호출(추가고유3)이다. 합계RGB12/BMP36/segPNG7 호출이며 고유 자연source는6장이다. v92는21헤더만 읽었고 픽셀 decode0이다. v91 ID1 헤더 assertion은 아직 RGB load 전이었다.

최초 합성unit3 중 JPEG `n_frames` 속성 오류1개는 source 읽기 전에 원형을 보존하고 `getattr(...,1)`로 수정한 뒤3개 통과했다. v90/v91의 source 계약 실패, 원본/부분Range/로그/원코드를 모두 보존했다. 결과를 읽는 PowerShell pipe 문법 오류1개는 명령을 고쳐 읽기만 재개했으며 학습/원자료 취득/완료감사는 재실행하지 않았다. 새 모델 학습 runtime 오류로 세지 않는다.

**새 encoder 특징/모델점수/gradient/optimizer/VAL_CAL_TEST 모델평가/API/장치/C++/ZIP/APK는 모두0**이다. 완료 일부연구 optimizer60,430은 이전까지의 실패 포함 부분합 그대로이며 전체 역사 총량이 아니다. 기존10개 보호 모델·ZIP·실패delta의 SHA를 이번 끝에 다시 확인해 모두 이전과 같았다. 모든 원본 전체 SHA를 재계산한 것은 아니다. 기존 기본앱/승인 세head/원v62/기본위치/경고/alias는 바꾸지 않았다.

## 준비 상태와 저장량

원TEST191/각14사본 예약, not_used332 제외를 유지한다. preset 역할 FIT956/VAL207/CAL207/TEST191/excluded332는 최종 근접제외 전 잠금이며 readyFIT가 아니다. 첫 고정6장도 역할 충돌·held 연결이 확인되면 제외하며 재배정/보충하지 않는다. 전체1893원본의 전이적 연결·oldheld 제외·연결그룹/유효pixel 최소는 아직 미검사이고 `ready_for_training=false`이다. 물리 선박·촬영 독립성도 미검증이다.

v93 종료 여유2,041,389,056bytes에서 남은1887RGB의 압축범위+헤더 상한943,907,623bytes, graph/임시/활동112MiB 및 예비1GiB를 보존하면 자료준비 필요량2,135,089,959bytes이다. 준비만도93,700,903bytes가 부족하다. 별도256MiB run까지 보존하면2,403,525,415bytes가 필요해 **362,136,359bytes 부족**하다. 추가mask/새GT 계약 비용은 미포함이다. 최소 약0.4GB 추가 여유, 실제 진행에는1GB 이상 추가 여유가 적절하다. 전체 ZIP/435MB부모 복사/새densecache/기존 자료 삭제로 해결하지 않는다. 새 취득/전체decode/학습을 시작하지 않았다.

다음은 `LIACI_AFTER_V93_NEXT_KO.md`의 공개 원저자 주석 표현 근거 확인과 저장량 gate이다. 자료 준비·학습 프로토콜은 아직 없다. 새 모델 성능 개선/앱 통과를 주장하지 않는다.

출처: [SINTEF LIACi](https://www.sintef.no/en/projects/2021/liaci/), [원자료 앱](https://liaci.sintef.cloud/), [원논문](https://doi.org/10.1109/JOE.2022.3219129), [공개 원논문 보존본](https://nva.sikt.no/registration/0198cc6adfe2-9b2a60a8-4a59-4f57-b098-67c36b2e957c). 모든 부모 데이터의 이용조건/저자·출처·버전·변경 표시를 유지한다.
