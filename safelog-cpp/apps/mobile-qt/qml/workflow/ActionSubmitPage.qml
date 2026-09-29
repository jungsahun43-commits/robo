import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../common"
Page {
    FileDialog { id: picker; title: "조치 후 사진"; nameFilters: ["사진 (*.png *.jpg *.jpeg)"]; onAccepted: photo.text = selectedFile.toString() }
    ScrollView {
        anchors.fill: parent; contentWidth: availableWidth
        ColumnLayout {
            width: parent.width; spacing: 12
            Label { text: "조치 제출"; font.pixelSize: 24; font.bold: true; Layout.margins: 16 }
            Label { text: workflowController.selected.location || ""; Layout.margins: 16 }
            Label { text: "수행한 조치"; Layout.leftMargin: 16 }
            TextArea { id: action; text: "적치물을 지정 보관구역으로 이동하고 통로를 확보함"; wrapMode: TextEdit.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            PhotoPreview { source: photo.text; Layout.fillWidth: true; Layout.margins: 16 }
            Button { text: "조치 후 사진 선택"; Layout.fillWidth: true; Layout.margins: 16; onClicked: picker.open() }
            Button { text: "Mock 조치 후 사진 사용"; Layout.fillWidth: true; Layout.margins: 16; onClicked: photo.text = captureController.demoPhoto(true) }
            TextField { id: photo; placeholderText: "조치 후 사진 경로"; Layout.fillWidth: true; Layout.margins: 16 }
            TextArea { id: memo; placeholderText: "추가 메모 (선택)"; wrapMode: TextEdit.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            Label { text: "개발용 AI 비교 응답"; Layout.leftMargin: 16 }
            ComboBox { model: ["정상 Mock", "Failure", "Timeout", "InvalidJson", "ConnectionError"]; Layout.fillWidth: true; Layout.margins: 16; Component.onCompleted: aiController.demoFailure = ""; onActivated: aiController.demoFailure = currentIndex === 0 ? "" : currentText }
            ErrorPanel { text: workflowController.error; Layout.fillWidth: true; Layout.margins: 16 }
            Button { text: "조치 제출"; enabled: workflowController.selected.status === "in_progress"; Layout.fillWidth: true; Layout.margins: 16; onClicked: workflowController.submitAction(action.text, photo.text, memo.text) }
        }
    }
}
