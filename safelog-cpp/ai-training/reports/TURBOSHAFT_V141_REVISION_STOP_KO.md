# Turboshaft v141 — 내부 URL equality guard 중단

사전 일반 push `c6972e439d9036867d9be2d0460398f220658e37`와 원격 일치·fresh CIM/locks/실제 v85 장부 확인 뒤 새 helper를 한 번만 실행했다. 정책 SHA256 `0a9b8ec0afbd568a6528be6cecf0394e81eb45ec45ec59a97e7e0f5ce6ec4454`, script SHA256 `40371b9cfc824da2d321670d8106ebff83fd90cdeeaf54eb535502ab5bb015bf`다.

새 [단일 ref API](https://api.github.com/repos/matousc89/synthetic-corrosion-segmentation/git/ref/heads/main)는 HTTP200/동일 최종URL/application-json/identity/Content-Length382로 382바이트를 실제 반환했다. JSON 내부 `url`은 `git/refs/heads/main`이고 요청은 `git/ref/heads/main`이므로, 내부 URL도 요청과 같아야 한다는 고정 guard에서 중단했다. helper exit1/stdout0/stderr487bytes다. HTTP 실패·학습 실패로 세지 않는다. code/정책·raw JSON·headers·started·failure·stdout/stderr/부분을 보존했으며 원문·guard를 수정하거나 v141을 소급 통과시키지 않았다. 같은 endpoint/alternate URL 재요청0이다.

보존 raw JSON의 local SHA256 `e1f3e88a1e8fdfeeb7653e54351f311dce1d0b8cdcf2c72a2dd13227f77a5c0b`, headers SHA256 `2f706c6ab9b2e1fa282023d6e7927c6ba367970c871c15e27e7ef9ef599bed8b`, failure SHA256 `dde879c3b3bfdcecdf6210e6ebeac8dfbd45b04f387ff160a30c5ab97b17470c`다. 로컬 정적 관찰에서 `ref=refs/heads/main`, object type=commit, SHA `7a63ebc763b94b918b23ab79e3eb3abef4e1b01a`를 읽었다. 이 관찰은 v142 별도 사전 정책에서 정확한 보존 바이트/헤더/실패와 함께 고정했다. repo commit은 dataset revision·촬영일·저자 파일 checksum·GT 권위가 아니다. 새 파일 취득은 v142의 별도 사전 push 뒤만 진행했다.

v141 실제 새 raw metadata1/GET1/382bytes 취득을 source 전체 수신0으로 숨기지 않는다. 사진/mask/annotation/archive/PDF/author code/weight 취득0·pixel/QA/raster/features/우리 score/gradient/optimizer/CAL_TEST/delivery0·readyfalse다. 기존 Zenodo/README/API/helper/fixture 재실행0, 모든 container/가능한 사본 UNASSIGNED 보호·기존 GT/roles/graph/model/app/원형·부분·실패를 보존한다.
