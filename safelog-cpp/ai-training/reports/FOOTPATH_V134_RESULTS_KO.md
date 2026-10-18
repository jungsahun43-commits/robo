# v134 — 작은 publisher metadata/LICENSE/README 취득 완료

사전 일반push `df35ec11c94fe79ed673b5d9887b42229be19f90`와 원격 일치·fresh host gate 뒤 새 helper를 한 번 실행했다. 정책 SHA256 `17b13b662f3eb2cb2530a4b5e7151f56c39742aa437d040e09848aaabd7f1a69`, script SHA256 `75cb4106590af09b9cc87a112f86054dd8dec79df7ae6aea8620e4523b8fcd5e`다. 이전 metadata/helper/fixture·학습/평가를 재실행하지 않았다.

## 실제 수신과 무결성

Zenodo record17154852 API6110bytes + LICENSE931 + README2459 = **GET3/9500bodybytes**를 실제 수신했다. exact200·동일URL·identity·MIME/길이/cap을 통과했고 두 텍스트는 [원배포 목록](https://zenodo.org/records/17154852)의 MD5와 일치했다. publisher MD5를 저자 제공 SHA256으로 바꾸지 않는다.

| 정적 파일 | 배포자 MD5 | 실제 local SHA256 |
|---|---|---|
| LICENSE.txt | f1df8d37c907cf13773275b8d4f29c17 | 18364a33110cd111f47b01990aaec78de2cf8cfa6b39452467a408a32ff4653e |
| README.md 원명 Indian Footpath Damage Segmentation Dataset.md | 2dd7d4b3666cca70bab23cbd4be93ec7 | aef785a1256bd59dc0aa43bf8ff4245764b1d077a3668565079f443c8f7315ab |

API는 네 파일을 선언한다. CSV9371·README2459·LICENSE931·ZIP**4504525071bytes**다. CSV와 ZIP은 받지 않았다. ZIP 배포자 MD5 `edafaba9d33ba8d9c3e989a33e0affb4`는 선언일 뿐 실제 archive SHA/CRC/member 검증이 아니다. 목록 전체 container/가능한 record는 내용 전에 UNASSIGNED 보호했으며 사진별 TEST87/최종역할을 확정한 것은 아니다. 보호기록 SHA `e9fa5cd18144fceff62124a8c6b94e22e663599af9a840d0c06433ad86451c16`이다.

`data/industrial-footpath-v134-metadata/metadata-complete.json` SHA `eba1e91648c2227db49d02ef180b9aabacdfa1a61387a545b860dede166d742a`와 API 원문 SHA `377e82824c82ac0437bf588cc7f434cbfa6fe6280106918c4dc6c9dee4a93ce4`를 고정했다. raw/headers/receipts/보호/started·stdout/stderr를 보존한다. helper exit0/stderr0/새fixture0이다. 완료직후 data18793/run205bytes로16MiB 이내였고 이후 별도 관찰기록도 작은 텍스트뿐이다.

## 읽은 계약과 남은 한계

실제 LICENSE는 dataset·이미지·주석·문서에 CC BY4.0을 명시한다. Priti Chakurkar/2025·출처/라이선스/변경 표시 조건을 유지하며 제삼자 권리까지 감사했다고 주장하지 않는다. [논문](https://www.nature.com/articles/s41597-025-06207-x)의 CC BY-NC-ND4.0과 data 조건을 구별한다. README는 원TRAIN411/VAL85/TEST87·JPG/PNG pair와 annotation 설명을 제공하지만 파일 대응·GT 값/unknown·물리촬영 그룹의 실제 검증은 아니다.

LICENSE의 요청 인용 DOI는 **10.5281/zenodo.15414238**인데 현재 record DOI는 **10.5281/zenodo.17154852**, 보존 API concept DOI는 **10.5281/zenodo.15414237**이다. 이를 조용히 수정하거나 15414238을 concept/V1으로 확정하지 않았다. 다른 DOI/versions/latest 페이지는 열지 않았다. 웹은 V2를 표시했으나 API `metadata.version`은 null이고 service `revision=4`는 dataset V4 증명이 아니다. 생성/수정 timestamp는 촬영일이 아니며 PowerShell의 +09:00 표시는 원시 timestamp의 같은 instant를 해석해야 한다.

권리 명시·원역할 텍스트가 확보된 것은 진전이지만 source 이미지·mask·좌표/unknown·native palette/actualGT·촬영/oldheld 전체 연결·최소/저장량은 통과하지 않았다. data0/background를 wholephoto 정상/무부식으로 바꾸거나 LabelMe 일반코드·원논문 전처리·dualencoder·중증도/점수를 가져오지 않는다. 보도/보행위험 자료를 산업설비/부식/전체건강으로 확대하거나 새 과제를 자동 추가하지 않는다. full4.5GB ZIP은 현재 공간 예산 밖이고 원형 삭제/예비1GiB 침범·부분을 선택해 전체graph/독립성 ready로 꾸미지 않는다.

## 실제 수행량

publisher rawAPI1개와 정적 text2개를 실제 취득했으므로 전체 source 수신0이라고 설명하지 않는다. source 원사진/mask/annotation/CSV/archive/PDF/code/weight 취득0, pixel/QA/raster/features/우리 모델 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0이다. metadata REST GET와 모델 API 검사를 구별한다. 새 dataset metadata helper1·오류0이며 학습실패가 아니다. 현재 PID/token 임시 operation.lock만 해제했고 dead training.lock23948·다른 lock/awakehelper/전원0이다. 기존 모든 GT/역할/graph·모델/앱 출력과 원형/실패/epoch/log/cache/ZIP을 보존한다. readyfalse다.
