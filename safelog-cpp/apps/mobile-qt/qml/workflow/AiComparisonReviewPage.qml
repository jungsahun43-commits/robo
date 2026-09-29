import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../common"
Page {
    id: page
    signal reportRequested(string inspectionId)
    property bool inspector: authController.session.role === "Inspector"
    property bool success: aiController.state === "Success"
    property bool verified: workflowController.selected.status === "verified"
    ScrollView {
        anchors.fill: parent; contentWidth: availableWidth
        ColumnLayout {
            width: parent.width; spacing: 12
            Label { text: "조치 전후 비교"; font.pixelSize: 24; font.bold: true; Layout.margins: 16 }
            Label { text: "조치 전 사진"; Layout.leftMargin: 16 }
            PhotoPreview { source: workflowController.selected.beforePhotoPath || ""; Layout.fillWidth: true; Layout.margins: 16 }
            Label { text: "조치 후 사진"; Layout.leftMargin: 16 }
            PhotoPreview { source: workflowController.selected.afterPhotoPath || ""; Layout.fillWidth: true; Layout.margins: 16 }
            Label { text: workflowController.selected.actionNote || ""; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            Label { text: "AI 제안 · 최종 판단은 사람이 수행합니다."; font.bold: true; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16; color: "#1e58a5" }
            BusyIndicator { visible: aiController.loading; running: visible; Layout.alignment: Qt.AlignHCenter }
            Label { visible: aiController.loading; text: "AI가 조치 전후 사진을 비교하고 있습니다."; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            Label { visible: page.success; text: "개선 가능성: " + (aiController.result.likelyResolved ? "높음 (AI 제안)" : "잔여 위험 확인 필요") + "\n잔여 위험: " + (aiController.result.remainingRisks || []).join("\n") + "\n추가 확인 의견: " + (aiController.result.assessment || "") + "\n신뢰도: " + (aiController.result.confidence || 0); wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            ErrorPanel { text: aiController.error ? "AI 비교 분석 실패: " + aiController.state + "\n" + aiController.error : ""; Layout.fillWidth: true; Layout.margins: 16 }
            Button { visible: !page.verified && !aiController.loading; text: "다시 비교"; Layout.fillWidth: true; Layout.margins: 16; onClicked: { confirmation.checked = false; workflowController.compare() } }
            Button { visible: !page.verified && aiController.error.length > 0; text: "정상 Mock으로 다시 시도"; Layout.fillWidth: true; Layout.margins: 16; onClicked: { aiController.demoFailure = ""; workflowController.compare() } }
            Button { visible: !page.verified; text: "AI 없이 직접 확인 (분석 취소)"; Layout.fillWidth: true; Layout.margins: 16; onClicked: { workflowController.directReview(); decision.currentIndex = 2 } }
            Label { visible: aiController.state === "ManualFallback"; text: "AI 없이 전후 사진과 조치 내용을 직접 확인합니다."; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            Label { text: "사람의 최종 판단"; font.bold: true; font.pixelSize: 20; Layout.margins: 16; color: "#126b58" }
            Label { visible: !page.inspector; text: "원 점검자가 로그인하여 최종 확인해야 합니다."; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            ComboBox { id: decision; visible: page.inspector && !page.verified; model: ["AI 의견 채택", "AI 의견 거절", "직접 확인"]; Layout.fillWidth: true; Layout.margins: 16 }
            TextArea { id: note; visible: page.inspector && !page.verified; placeholderText: "직접 확인한 내용 또는 추가 조치 요청 사유"; wrapMode: TextEdit.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            CheckBox { id: confirmation; visible: page.inspector && !page.verified; text: "전후 사진과 조치 내용을 직접 확인했습니다"; Layout.fillWidth: true; Layout.margins: 16 }
            ErrorPanel { text: workflowController.error; Layout.fillWidth: true; Layout.margins: 16 }
            Button { visible: page.inspector && !page.verified; text: "최종 확인"; enabled: !aiController.loading && workflowController.selected.status === "pending_review"; Layout.fillWidth: true; Layout.margins: 16; onClicked: workflowController.verify(["Accepted", "Rejected", "Manual"][decision.currentIndex], note.text, confirmation.checked) }
            Button { visible: page.inspector && workflowController.selected.status === "pending_review"; text: "추가 조치 요청"; Layout.fillWidth: true; Layout.margins: 16; onClicked: workflowController.requestChanges(note.text) }
            Label { visible: workflowController.selected.status === "in_progress"; text: "추가 조치를 요청했습니다. 담당자가 다시 조치 후 사진을 제출해야 합니다."; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            Label { visible: page.verified; text: "Verified · 점검자 최종 확인 완료"; Layout.margins: 16 }
            Button { visible: page.verified && page.inspector; text: "보고서 보기"; Layout.fillWidth: true; Layout.margins: 16; onClicked: page.reportRequested(workflowController.selected.inspectionId) }
        }
    }
}
