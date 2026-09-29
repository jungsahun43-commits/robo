import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
Pane {
    ColumnLayout {
        anchors.fill: parent
        BusyIndicator { running: true; Layout.alignment: Qt.AlignHCenter }
        Label { text: "AI가 현장 사진을 분석하고 있습니다."; wrapMode: Text.Wrap; Layout.fillWidth: true }
    }
}
