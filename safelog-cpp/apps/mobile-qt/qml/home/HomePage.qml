import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
Page {
    signal navigate(string menu)
    property var session: authController.session
    property var menus: session.role === "Inspector" ? ["점검 등록", "AI 검토", "최종 확인", "완료 보고서"] :
                        session.role === "Manager" ? ["담당자 지정", "전체 진행 상황", "검토 상태", "보고서"] :
                        session.role === "Assignee" ? ["배정된 작업", "조치 제출", "진행 중 작업"] : []
    ScrollView {
        anchors.fill: parent; contentWidth: availableWidth
        ColumnLayout {
            width: parent.width; spacing: 12
            Label { text: (session.userName || "") + "님"; font.pixelSize: 26; font.bold: true; Layout.margins: 16 }
            Label { text: (session.userId || "") + " · " + (session.role || ""); Layout.margins: 16 }
            Label { text: session.siteName || ""; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            Label { text: "대시보드 · 현재 실행 중 등록한 기록"; font.bold: true; Layout.margins: 16 }
            Repeater {
                model: ["미처리 위험", "진행 중 조치", "검토 대기", "완료된 점검"]
                delegate: Label {
                    required property string modelData
                    required property int index
                    text: modelData + ": " + (appController.dashboard[index] || 0)
                    Layout.leftMargin: 16
                }
            }
            Repeater {
                model: menus
                delegate: Button { required property string modelData; text: modelData; Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16; onClicked: navigate(modelData) }
            }
            Label { text: "Development / Mock 역할 전환"; font.bold: true; Layout.margins: 16 }
            Repeater {
                model: ["inspector-1", "manager-1", "assignee-1"]
                delegate: Button {
                    required property string modelData
                    text: "Mock 로그인: " + modelData; Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16
                    onClicked: { const user = modelData; authController.logout(); authController.login(user, "demo1234") }
                }
            }
            Button { text: "로그아웃"; Layout.margins: 16; onClicked: authController.logout() }
        }
    }
}
