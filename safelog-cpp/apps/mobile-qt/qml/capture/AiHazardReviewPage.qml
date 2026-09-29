import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../common"
Page {
    id: page
    property bool manual: aiController.state === "ManualFallback"
    property bool success: aiController.state === "Success"
    function populate() {
        description.text = success ? aiController.result.description || "" : captureController.draft.memo || ""
        action.text = success ? aiController.result.action || "" : captureController.draft.action || ""
    }
    Component.onCompleted: populate()
    Connections { target: aiController; function onStateChanged() { page.populate() } }
    ScrollView {
        anchors.fill: parent; contentWidth: availableWidth
        ColumnLayout {
            width: parent.width; spacing: 12
            Label { text: "AI 위험 검토"; font.pixelSize: 24; font.bold: true; Layout.margins: 16 }
            Label { text: "AI 분석 결과는 참고 제안입니다.\n최종 안전 판단과 확인은 사람이 수행합니다."; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16; color: "#126b58" }
            LoadingOverlay { visible: aiController.loading; Layout.fillWidth: true; Layout.margins: 16 }
            Label { visible: page.success; text: (aiController.result.category || "") + " · 위험 등급 " + (aiController.result.riskLevel || "") + " / 5\n신뢰도 " + (aiController.result.confidence || 0) + " · " + (aiController.result.modelName || ""); wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            ErrorPanel { visible: !page.success && !page.manual && !aiController.loading; text: "AI 분석에 실패했습니다.\n" + aiController.state + " · " + aiController.error; Layout.fillWidth: true; Layout.margins: 16 }
            Button { visible: !page.success && !page.manual && !aiController.loading; text: "다시 시도"; Layout.fillWidth: true; Layout.margins: 16; onClicked: captureController.retry() }
            Button { visible: !page.success && !page.manual && !aiController.loading; text: "정상 Mock으로 다시 시도"; Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16; onClicked: { aiController.demoFailure = ""; captureController.retry() } }
            Button { visible: !page.manual; text: aiController.loading ? "분석 취소 · 수동 입력" : "수동 입력"; Layout.fillWidth: true; Layout.margins: 16; onClicked: aiController.manualFallback() }
            Label { visible: page.manual; text: "수동 입력 · 사람이 작성한 위험 기록"; Layout.margins: 16 }
            TextArea { id: description; visible: page.success || page.manual; placeholderText: "위험 설명"; Accessible.name: "위험 설명"; wrapMode: TextEdit.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            TextArea { id: action; visible: page.success || page.manual; placeholderText: "권장 조치"; Accessible.name: "권장 조치"; wrapMode: TextEdit.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            ErrorPanel { text: captureController.error; Layout.fillWidth: true; Layout.margins: 16 }
            Button { visible: page.success; text: "제안 채택 / 수정 내용 저장"; Layout.fillWidth: true; Layout.margins: 16; onClicked: captureController.saveReview("Accepted", description.text, action.text) }
            Button { visible: page.success; text: "AI 제안 거절 · 최초 입력 유지"; Layout.fillWidth: true; Layout.margins: 16; onClicked: captureController.saveReview("Rejected", captureController.draft.memo, captureController.draft.action) }
            Button { visible: page.manual; text: "수동 내용 저장"; Layout.fillWidth: true; Layout.margins: 16; onClicked: captureController.saveReview("Manual", description.text, action.text) }
        }
    }
}
