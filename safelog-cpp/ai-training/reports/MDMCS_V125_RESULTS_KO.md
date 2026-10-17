# MDMCS v125 — 공개 Location의 중앙목록 소량 취득 완료

## 사전등록과 실제 요청

원배포 URL HEAD302/body0에서 중단한 v124는 그대로 보존했다. 그 응답이 직접 지정한 정확 공개 S3 URL만 별도 코드/JSON/문서로 고정하여 `0dc676e039b199b76256ea47b6577c2f60af9afd`에 사전 일반push/원격 확인 후 실행했다. 정책 SHA256은 `383e9c0cb92f439c9becfe6aa904ba96c4623d1694c65121ef15b642158435c5`다. 원배포 endpoint·V1/V2 page/API·v124 main/완료 fixture 재실행0이다.

2026-10-17T22:23:48–51Z 실제 HEAD200/204919973bytes/Accept-Ranges bytes/strong ETag `"488999a12587cbb5e18a092ada0916ae-10"`를 확인했다. 이 multipart ETag를 archive SHA256이나 MD5 검증으로 부르지 않는다. Last-Modified는2025-10-11이며 촬영날짜·GT 생성revision으로 쓰지 않는다. 정확206/Content-Range/Content-Length/동일 ETag/If-Range를 body 전에 확인한 GET2회만 수행했다.

| 수신 내용 | 실제 body bytes |
|---|---:|
| 마지막 EOCD22 | 22 |
| 중앙목록 | 390566 |
| 합계 | **390588** |

전체204919973bytes ZIP·local header·member body는 취득하지 않았다. EOCD는 단일disk/noZIP64/comment0이고 중앙목록은offset204529385에서EOCD직전까지다. 목록SHA256 `60f880aff1620766b449e73c2eddce4826feec77541db6fb39536d6cc05d9489`다. 3711멤버/선언 확장244856676bytes를 안전경로·대소문자 중복·암호화·symlink·형식/상한에 대해 검사했다. 이는 실제 archive SHA/멤버 CRC/GT 검증이 아니다. 원배포자 선언SHA0582f706…와 로컬 전체 ZIP을 비교하지 않았다.

## 목록 관찰과 보호

| 원 ZIP의 경로 | JPG 이름 수 | JSON 이름 수 | mask PNG 이름 수 | 추가 color PNG |
|---|---:|---:|---:|---:|
| train_* | 1000 | 1000 | 1000 | 0 |
| val_* | 100 | 100 | 100 | 0 |
| test_* | 100 | 100 | 100 | 100 |

파일3700+디렉터리11이다. 이 표는 경로·확장자 선언 수량이며 실제 RGB/정답·서로 대응하는 고유사진·정상지원·독립부품 검증 수량이 아니다. 원문 README/분리 CSV 등은 이 목록에 없고 JSON 내용은 읽지 않았다.

모든3711멤버/가능한 사본은 UNASSIGNED 보호 상태다. 원 v125 parser가 허용한 정확한 소문자 `test` 토큰은 `test_image` 등을 일치시키지 않아 좁은 토큰 목록은0이었다. 전체 멤버 보호는 계속 적용됐다. 이후 **이미 받은 목록만** 읽어 실제 명시 `test_annotations/test_image/test_mark_color/test_mask` 경로의400파일을 내용 전 추가 보호한 별도 관찰기록을 저장했다. 숫자/무늬로 역할·그룹을 추정하거나 TRAIN으로 배정하지 않았다. 사진별 최종 TEST/사본 identity/개발역할·물리 촬영그룹·전source/oldheld graph·GT/unknown·최소는 여전히 미검증이다.

원형 EOCD/CD·모든 헤더/영수증/목록·`all-members-protected-before-content.json`·`directory-path-observations-and-test-protection.json`·로그를 보존했다. 후자 SHA256은 `656e5cf7b17059457740445d6e94cc073b5362cafbc38a4c81068c8c26a81c9f`다. 새 Location provenance fixture1통과/helper오류0/stderr0, v124 guard fixture3과 metadata STOP/exit1은 그대로 보존한다. v125 결과 직전 data/run 저장2973552bytes로32MiB 상한 이내였다.

source 사진/mask/member annotation/PDF/code/weight 취득·pixel/시각QA·encoder/score·gradient/optimizer·CAL_TEST·API_CPU_GPU_CPP_ZIP_APK0이다. ZIP metadata 바이트를 취득했으므로 source 수신량0으로 설명하지 않는다. v85 실제450/epoch30과 완료 추적 일부60430은 그대로이며 후자는 실패포함 부분합으로 전체 역사 총량이 아니다. 새 모델 protocol/readyfalse다. 기존 부모/실패·모든 역할/GT/graph/앱 출력은 보존했다.

## 다음 단계의 범위

이제 원 TRAIN/VAL/TEST 이름과 annotation/mask 경로가 존재한다는 새 근거가 생겼다. 다음은 별도 코드JSON 사전push 후, 고정 첫 TRAIN JSON 하나의 원형·schema/좌표·imageData 존재 여부만 검토하는 제안이다. RGB/mask 해독·최종분리·학습 승인이 아니다. 원 GT/unknown/정상지원·물리그룹과 전체 graph/최소를 충족해야 별도 pixel/학습 조건으로 갈 수 있다. v124 STOP을 소급 통과하거나 좋은 일부만으로 계약을 만들지 않는다. data CC BY4.0·Guo/Bao·DOI10.17632/6x4dzzrs2h.2 및 모든 부모 출처/버전/변경/제삼자 권리를 유지한다.
