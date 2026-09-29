import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
Page {
    signal homeRequested()
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 20
        Label { text: "위험 기록 저장 완료"; font.pixelSize: 24; font.bold: true; wrapMode: Text.Wrap; Layout.fillWidth: true }
        Label { text: "기록 ID: " + (captureController.draft.findingId || ""); wrapMode: Text.Wrap; Layout.fillWidth: true }
        Label { text: "상태: Open (미처리)\n최종 안전 확인은 아직 수행되지 않았습니다.\n개발용 메모리 저장소이므로 앱 종료 시 기록이 초기화됩니다."; wrapMode: Text.Wrap; Layout.fillWidth: true }
        Button { text: "홈으로"; Layout.fillWidth: true; onClicked: homeRequested() }
        Item { Layout.fillHeight: true }
    }
}
