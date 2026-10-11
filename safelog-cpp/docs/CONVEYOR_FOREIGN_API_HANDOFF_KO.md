# 역할3 → 역할4·역할2: 컨베이어 이물질 모델 연결

## 각자 연결할 내용

- **역할3:** 실제 학습 모델, 로컬 CPU 서버, 고정 결과 형식, 성능·오류 사례를 전달한다.
- **역할4:** 컨베이어 점검 화면에서 ‘벨트 표면 균열’과 ‘철광석 흐름의 이물질’을 선택하게 하고, 사진 전송 → AI 제안 → 점검자 확인 → 조치·재점검·보고서 흐름을 연결한다.
- **역할1:** 근접 촬영·벨트 영역 자르기 안내, 선택한 사진 표시, 이물질 의심/미검출·사람 확인 UI를 만든다.
- **역할2:** 설비 ID, 원본/전송 사진, 분석 종류·모델 SHA·사진 SHA, AI 점수·사람의 최종 확인, 조치 이력을 저장한다.

완성 앱의 실제 연결은 역할4가 담당한다. 이번 전달물은 실행 가능한 모델·API이며 모바일 APK 제작 완료를 뜻하지 않는다. Python은 모델 서버에서 사용하고 앱 화면·네트워크 연결은 기존 Qt/C++로 구현한다.

## 먼저 확인할 모델 범위

이물질 모델은 철광석 컨베이어 실험 자료의 **벨트 영역 근접 사진**으로 학습했다. 택배·식품 운반 벨트나 모터·롤러 내부 고장의 정답은 학습하지 않았다. 새로운 현장 사진은 사람 검토용 연구 후보로 표시한다.

`conveyor_foreign_object`는 사진 전체의 이물질 유무 분류다. 위치 정답이 없어 경계 상자·이물질 heatmap을 그리지 않는다. 표면 균열 위치 표시가 필요하면 기존 `belt_crack` API를 선택한다. 이물질 미검출을 벨트 정상·안전으로 바꾸지 않는다.

## 1. 서버 실행

저장소에서는 `safelog-cpp/ai-training` 폴더에서 실행한다. Git에는 모델이 없으므로 전달 ZIP의 모델을 별도로 받는다.

```powershell
.venv/Scripts/python.exe -m safelog_ai.conveyor_foreign_server --checkpoint runs/conveyor-foreign-grouped-v1/final.pt --sha256 9d646df61e488bd90d6f00aeaba8bd67c184f9470526139636f335443d1825fe --port 8772
```

새 통합 ZIP을 압축 해제한 폴더에서는 다음 명령을 쓴다.

```powershell
python -m safelog_ai.conveyor_foreign_server --checkpoint conveyor-foreign-grouped-v1/final.pt --sha256 9d646df61e488bd90d6f00aeaba8bd67c184f9470526139636f335443d1825fe --port 8772
```

- `GET http://127.0.0.1:8772/health`
- API 설명: `http://127.0.0.1:8772/docs`
- `POST http://127.0.0.1:8772/v1/equipment/conveyor/foreign/analyze`
- Android USB 연결: `adb reverse tcp:8772 tcp:8772`

서버는 로컬 CPU로 실행한다. Android 앱 자체에서 Python 모델이 실행되는 구성은 아니다. 개발 PC에서 먼저 연결하며, 원격 서버 배포는 팀의 후속 통합 단계다.

## 2. 요청

```json
{
  "equipmentCategory": "conveyor_belt",
  "sceneType": "iron_ore_closeup",
  "promptVersion": "conveyor-foreign-v1",
  "image": "<JPEG/PNG 파일 바이트의 base64>"
}
```

`sceneType`을 명시해야 한다. 설비 등록/점검 종류에 이 촬영 범위가 맞을 때만 호출한다. 다른 종류는422이며, 모델 미설정은503이다. 이미지8MB 초과는413, 잘못된 base64·읽을 수 없는 이미지·12MP 초과는422다. 이 오류들은 정상/이물질 미검출 결과로 저장하지 않는다.

EXIF 방향을 반영한 뒤 종횡비를 유지해384 letterbox로 분석한다. `promptVersion`은 호출 형식 버전이며 모델에 텍스트를 전달하는 프롬프트가 아니다.

## 3. 응답과 화면 표시

| 필드 | 앱의 처리 |
|---|---|
| `schema` | `safelog.equipment.conveyor-foreign.v1`인지 확인 |
| `equipmentCategory`, `sceneType`, `className` | `conveyor_belt`, `iron_ore_closeup`, `conveyor_foreign_object`에 맞는 화면으로 연결 |
| `candidateDetected` | true는 ‘이물질 의심’, false는 ‘이번 사진에서 이물질 미검출’ |
| `presenceScore`, `threshold` | 원래 sigmoid 점수와 고정0.5 기준을 기록. `presenceScore > 0.5`일 때 후보 있음 |
| `scoreIsCalibratedProbability` | false. 점수를 실제 현장의 정확도나 고장 확률로 표시하지 않음 |
| `localizationAvailable` | false. 이물질 위치가 추정됐다고 표시하지 않음 |
| `mechanicalFailureState` | 항상 `not_assessed`. 가동/고장 확인은 사람이 별도 입력 |
| `humanConfirmationRequired` | true. 채택·수정·거절과 확인 의견을 받음 |
| `fieldPerformanceValidated` | false. 공개 실험 자료의 수치와 신규 현장 성능을 구분 |
| `modelName`, `modelSha256`, `inputSha256`, `promptVersion` | DB·보고서에 버전과 출처로 저장 |
| `inputWidth`, `inputHeight` | EXIF 방향 적용 후 전송한 사진의 크기 |

## 4. Qt/C++ 최소 전송 예제

이 코드는 기존 화면의 장수명 `QNetworkAccessManager manager`와 선택한 사진 `QString photoPath`에 연결하는 예시다. 화면의 촬영 범위 확인·DB 저장·사용자 메시지는 역할4/2가 연결한다.

```cpp
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>
#include <QDebug>

QFile imageFile(photoPath);
if (!imageFile.open(QIODevice::ReadOnly)) return;
const QByteArray bytes = imageFile.readAll();
if (bytes.isEmpty() || bytes.size() > 8 * 1024 * 1024) return;
const QJsonObject body{
    {"equipmentCategory", "conveyor_belt"},
    {"sceneType", "iron_ore_closeup"},
    {"promptVersion", "conveyor-foreign-v1"},
    {"image", QString::fromLatin1(bytes.toBase64())}
};
QNetworkRequest request{QUrl{"http://127.0.0.1:8772/v1/equipment/conveyor/foreign/analyze"}};
request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
QNetworkReply* reply = manager.post(request, QJsonDocument(body).toJson(QJsonDocument::Compact));
QObject::connect(reply, &QNetworkReply::finished, &manager, [reply] {
    const QByteArray responseBytes = reply->readAll();
    if (reply->error() != QNetworkReply::NoError) {
        qWarning() << reply->errorString();  // UI: 분석 실패, 재시도/수동 검토
        reply->deleteLater();
        return;
    }
    QJsonParseError error;
    const auto doc = QJsonDocument::fromJson(responseBytes, &error);
    const auto value = doc.object();
    if (error.error != QJsonParseError::NoError || !doc.isObject()
        || value.value("schema").toString() != "safelog.equipment.conveyor-foreign.v1"
        || value.value("className").toString() != "conveyor_foreign_object"
        || !value.value("candidateDetected").isBool()) {
        qWarning() << "Unexpected AI response";
        reply->deleteLater();
        return;
    }
    // value를 화면 상태와 DB 계층에 전달한다.
    // candidateDetected가 false여도 정상 가동 판정은 만들지 않는다.
    qDebug() << value.value("candidateDetected").toBool()
             << value.value("modelSha256").toString();
    reply->deleteLater();
});
```

## 5. DB·조치·보고서

점검 사진을 자르면 `originalPhotoId`와 실제 전송한 `analysisPhotoId`를 모두 저장한다. 반환 `inputSha256`은 전송한 파일 바이트의 해시다. 원본과 혼동하지 않는다.

`analysisType`, `sceneType`, `className`, `candidateDetected`, `presenceScore`, `threshold`, 모델/사진 SHA, `promptVersion`, 분석 시각을 보존한다. 사람 확인은 별도 `reviewDecision`(accept/edit/reject), `reviewerId`, `reviewComment`, `reviewedAt`으로 기록한다. AI 결과를 사람의 최종 확인으로 덮어쓰지 않는다.

보고서는 설비 ID·사진·AI 제안·사람 확인·이물질 제거/점검 조치·재점검 사진을 함께 보여준다. 통신 실패 시 분석 실패로 남기고 수동 확인을 이어 간다.

## 6. 전달한 모델의 실제 성능

실제30epoch·1,170회 업데이트, 학습1,230장/12촬영명이다. 최종 TEST448장/4촬영명에서 사진 오탐1/256=0.39%, 미탐1/192=0.52%다. VAL320장/3촬영명의 미탐은8/128=6.25%이며 `canga2`에서8/64=12.50%다.

따라서 이물질 시험 묶음에서5% 미만이지만, 모든 자료·현장·설비에서5% 미만인 모델이라고 소개하지 않는다. 원본은 같은 실험 설비의 연속 사진이며 다른 공장 검증이 아니다. [실제 결과·오류 사진](../ai-training/reports/CONVEYOR_FOREIGN_RESULTS_KO.md)을 함께 전달한다.

기존 균열 API는 [벨트 연결 문서](EQUIPMENT_API_HANDOFF_KO.md)의8771번 포트를 사용한다. 두 모델의 정상 의미·오차율·출력 형식은 각각 유지한다.
