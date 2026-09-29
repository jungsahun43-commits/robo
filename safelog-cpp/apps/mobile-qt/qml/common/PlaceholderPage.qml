import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
Page {
    property string menuTitle: "준비 중"
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 20
        Label { text: menuTitle; font.pixelSize: 24; font.bold: true; wrapMode: Text.Wrap; Layout.fillWidth: true }
        Label { text: "2단계에서 연결할 화면입니다.\n현재는 사진 입력과 AI 위험 검토를 시연할 수 있습니다."; wrapMode: Text.Wrap; Layout.fillWidth: true }
        Item { Layout.fillHeight: true }
    }
}
