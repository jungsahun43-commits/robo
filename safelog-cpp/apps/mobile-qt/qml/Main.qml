import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    width: 390
    height: 760
    visible: true
    title: "SafeLog"
    color: "#f4f7f6"

    header: ToolBar {
        contentHeight: 62
        background: Rectangle { color: "#126b58" }
        Label {
            anchors.centerIn: parent
            text: "세이프로그"
            color: "white"
            font.pixelSize: 21
            font.bold: true
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 16

        Label { text: "현장 안전점검"; font.pixelSize: 26; font.bold: true; color: "#17212b" }
        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: "사진 등록부터 조치 확인, 보고서 생성까지 한 흐름으로 관리합니다."
            color: "#526069"
        }
        Frame {
            Layout.fillWidth: true
            ColumnLayout {
                anchors.fill: parent
                Label { text: "세이프 금속 가공공장"; font.bold: true; font.pixelSize: 18 }
                Label { text: appController.statusMessage; color: "#126b58" }
            }
        }
        Button {
            Layout.fillWidth: true
            text: "통합 시나리오 실행"
            onClicked: appController.runDemoScenario()
        }
        Label { text: "개발 메뉴"; font.bold: true; font.pixelSize: 18 }
        Repeater {
            model: ["A · 새 점검/사진 등록", "B · 로컬 저장소 상태", "C · 담당 업무/조치 확인", "D · 보고서 미리보기"]
            delegate: Button { required property string modelData; Layout.fillWidth: true; text: modelData }
        }
        Item { Layout.fillHeight: true }
        Label {
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignHCenter
            text: "학술제 프로토타입 · 법적 적합성은 별도 검토 필요"
            color: "#7b878d"
            font.pixelSize: 11
        }
    }
}
