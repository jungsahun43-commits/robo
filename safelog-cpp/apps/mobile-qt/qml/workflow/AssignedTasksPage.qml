import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../common"
Page {
    id: page
    property string menuTitle: "위험 및 조치 목록"
    signal actionRequested()
    signal reviewRequested()
    signal reportRequested(string inspectionId)
    signal hazardRequested(string findingId)
    ScrollView {
        anchors.fill: parent; contentWidth: availableWidth
        ColumnLayout {
            width: parent.width; spacing: 14
            Label { text: page.menuTitle; font.pixelSize: 24; font.bold: true; Layout.margins: 16 }
            Label { text: "현재 역할: " + (authController.session.role || ""); Layout.leftMargin: 16 }
            ErrorPanel { text: workflowController.error; Layout.fillWidth: true; Layout.margins: 16 }
            Label { visible: workflowController.findings.length === 0; text: "표시할 작업이 없습니다.\nInspector가 점검을 등록하고 Manager가 담당자를 지정하세요."; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            Repeater {
                model: workflowController.findings
                delegate: Frame {
                    required property var modelData
                    Layout.fillWidth: true; Layout.margins: 12
                    ColumnLayout {
                        width: parent.width
                        Label { text: modelData.location; font.bold: true; wrapMode: Text.Wrap; Layout.fillWidth: true }
                        Label { text: "위험 등급: " + (modelData.riskLevel || "미지정") + " / 5 · " + modelData.status; wrapMode: Text.Wrap; Layout.fillWidth: true }
                        Label { text: modelData.description; wrapMode: Text.Wrap; Layout.fillWidth: true }
                        Label { text: modelData.assigneeName; wrapMode: Text.Wrap; Layout.fillWidth: true }
                        Button {
                            visible: authController.session.role === "Manager" && modelData.status === "open"
                            text: "이조치(assignee-1)에게 배정"; Layout.fillWidth: true
                            onClicked: if (workflowController.selectFinding(modelData.findingId)) workflowController.assign("assignee-1")
                        }
                        Button {
                            visible: authController.session.role === "Assignee" && (modelData.status === "open" || modelData.status === "in_progress")
                            text: modelData.status === "open" ? "조치 시작" : "조치 내용 제출"; Layout.fillWidth: true
                            onClicked: {
                                if (!workflowController.selectFinding(modelData.findingId)) return
                                if (modelData.status === "open" && !workflowController.beginWork()) return
                                page.actionRequested()
                            }
                        }
                        Button {
                            visible: modelData.status === "pending_review"
                            text: authController.session.role === "Inspector" ? "전후 비교 · 최종 확인" : "조치 전후 비교 보기"
                            Layout.fillWidth: true
                            onClicked: {
                                if (!workflowController.selectFinding(modelData.findingId)) return
                                page.reviewRequested()
                                workflowController.compare()
                            }
                        }
                        Button { visible: authController.session.role === "Inspector" && modelData.status === "open"; text: "AI 위험 검토"; Layout.fillWidth: true; onClicked: page.hazardRequested(modelData.findingId) }
                        Button { visible: authController.session.role !== "Assignee"; text: modelData.status === "verified" ? "완료 보고서" : "진행 보고서"; Layout.fillWidth: true; onClicked: page.reportRequested(modelData.inspectionId) }
                    }
                }
            }
        }
    }
}
