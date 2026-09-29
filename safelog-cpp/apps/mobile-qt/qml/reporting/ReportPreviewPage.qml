import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../common"
Page {
    property string inspectionId: ""
    ScrollView {
        anchors.fill: parent; contentWidth: availableWidth
        ColumnLayout {
            width: parent.width; spacing: 12
            Label { text: "SafeLog AI 안전점검 보고서"; font.pixelSize: 24; font.bold: true; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            BusyIndicator { visible: reportingController.state === "Loading"; running: visible; Layout.alignment: Qt.AlignHCenter }
            ErrorPanel { text: reportingController.error; Layout.fillWidth: true; Layout.margins: 16 }
            Button { text: "보고서 다시 생성"; enabled: reportingController.state !== "Loading"; Layout.margins: 16; onClicked: reportingController.generate(inspectionId) }
            Label { text: (reportingController.report.site || "") + "\n" + (reportingController.report.address || "") + "\n점검자: " + (reportingController.report.inspector || "") + "\n점검 일시: " + (reportingController.report.inspectedAt || ""); wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            Repeater {
                model: reportingController.report.findings || []
                delegate: Frame {
                    required property var modelData
                    Layout.fillWidth: true; Layout.margins: 12
                    ColumnLayout {
                        width: parent.width
                        Label { text: modelData.location + "\n" + modelData.description + "\n위험 등급: " + (modelData.riskLevel || "미지정") + " / 5\n개선 의견: " + modelData.actionOpinion + "\n담당자: " + modelData.assignee + "\n최종 상태: " + modelData.status; wrapMode: Text.Wrap; Layout.fillWidth: true }
                        Repeater { model: modelData.photos; delegate: ColumnLayout { required property var modelData; Layout.fillWidth: true; Label { text: modelData.kind === "before" ? "조치 전" : "조치 후" } PhotoPreview { source: modelData.path; Layout.fillWidth: true } } }
                        Label { text: "AI 제안 및 사용자 검토 결정"; color: "#1e58a5"; font.bold: true }
                        Repeater {
                            model: modelData.analyses
                            delegate: Label { required property var modelData; text: modelData.type + " · " + modelData.model + "\n프롬프트: " + modelData.prompt + " · 신뢰도: " + modelData.confidence + "\n검토: " + modelData.decision + " · " + modelData.reviewer + "\n" + modelData.json; wrapMode: Text.Wrap; Layout.fillWidth: true }
                        }
                        Label { text: "사람의 조치 / 최종 판단과 AI 참고 기록"; color: "#126b58"; font.bold: true; wrapMode: Text.Wrap; Layout.fillWidth: true }
                        Repeater {
                            model: modelData.logs
                            delegate: Label { required property var modelData; text: (modelData.isAi ? "[AI 참고 제안] " : "[사람의 기록] ") + modelData.action + " · " + modelData.actor + "\n" + modelData.note; color: modelData.isAi ? "#1e58a5" : "#126b58"; wrapMode: Text.Wrap; Layout.fillWidth: true }
                        }
                    }
                }
            }
            Label { visible: reportingController.state === "Success"; text: "사진을 포함한 HTML 보고서 생성 완료\n" + reportingController.outputPath; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            Label { text: "공유는 개발용 Mock입니다. 실제 Android 공유창은 후속 어댑터 연결이 필요합니다."; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            ComboBox { id: shareOutcome; model: ["ShareSucceeded", "ShareCancelled", "ShareFailed"]; Layout.fillWidth: true; Layout.margins: 16 }
            Button { text: "Mock 공유 실행"; enabled: reportingController.state === "Success"; Layout.fillWidth: true; Layout.margins: 16; onClicked: reportingController.share(shareOutcome.currentText) }
            Label { text: reportingController.shareState === "ShareCancelled" ? "공유를 취소했습니다." : reportingController.shareState === "ShareSucceeded" ? "Mock 공유 성공 (실제 전송 없음)" : reportingController.shareState; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
        }
    }
}
