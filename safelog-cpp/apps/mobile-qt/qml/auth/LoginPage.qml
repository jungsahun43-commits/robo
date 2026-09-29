import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../common"
Page {
    ScrollView {
        anchors.fill: parent; contentWidth: availableWidth
        ColumnLayout {
            width: parent.width; spacing: 16
            Label { text: "SafeLog AI 로그인"; font.pixelSize: 26; font.bold: true; Layout.margins: 16 }
            Label { text: "Mock 개발 모드 · 기록은 앱 종료 시 초기화됩니다."; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            TextField { id: userId; objectName: "loginUserId"; placeholderText: "사용자 ID"; Accessible.name: "사용자 ID"; Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16; enabled: !authController.loading }
            TextField { id: password; objectName: "loginPassword"; placeholderText: "비밀번호"; Accessible.name: "비밀번호"; echoMode: TextInput.Password; Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16; enabled: !authController.loading; onAccepted: login.clicked() }
            Button { id: login; text: authController.loading ? "로그인 중…" : "로그인"; Layout.fillWidth: true; Layout.margins: 16; enabled: !authController.loading && userId.text.length > 0 && password.text.length > 0; onClicked: authController.login(userId.text, password.text) }
            BusyIndicator { running: authController.loading; visible: running; Layout.alignment: Qt.AlignHCenter }
            ErrorPanel { text: authController.error; Layout.fillWidth: true; Layout.margins: 16 }
            Label { text: "시연 계정\n김안전 · inspector-1\n박관리 · manager-1\n이조치 · assignee-1\n공통 비밀번호: demo1234"; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
        }
    }
}
