// Temporary role-1 screen. Replace behind the same controller boundary.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../common"
Page {
    signal captured()
    FileDialog {
        id: picker
        title: "조치 전 사진 선택"
        nameFilters: ["사진 (*.png *.jpg *.jpeg)"]
        onAccepted: photo.text = selectedFile.toString()
    }
    ScrollView {
        anchors.fill: parent; contentWidth: availableWidth
        ColumnLayout {
            width: parent.width; spacing: 12
            Label { text: "새 점검 · 사진 입력"; font.pixelSize: 24; font.bold: true; Layout.margins: 16 }
            PhotoPreview { source: photo.text; Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16 }
            Button { text: "사진 선택"; Layout.fillWidth: true; Layout.margins: 16; onClicked: picker.open() }
            Button { text: "Mock 시연 이미지 사용"; Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16; onClicked: photo.text = captureController.demoPhoto() }
            TextField { id: photo; placeholderText: "로컬 사진 경로 또는 file URL"; Accessible.name: "사진 경로"; Layout.fillWidth: true; Layout.margins: 16 }
            TextField { id: location; text: "2층 가공라인 통로"; placeholderText: "장소"; Accessible.name: "장소"; Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16 }
            TextArea { id: memo; text: "통로에 자재가 적치되어 있음"; placeholderText: "발견 메모"; Accessible.name: "발견 메모"; wrapMode: TextEdit.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
            TextArea { id: action; text: "적치물을 지정 보관구역으로 이동"; placeholderText: "수동 권장 조치"; Accessible.name: "수동 권장 조치"; wrapMode: TextEdit.Wrap; Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16 }
            Label { text: "개발용 AI 응답 시연"; Layout.leftMargin: 16 }
            ComboBox {
                Layout.fillWidth: true; Layout.margins: 16
                model: ["정상 Mock", "Failure", "Timeout", "InvalidJson", "ConnectionError"]
                onActivated: aiController.demoFailure = currentIndex === 0 ? "" : currentText
                Component.onCompleted: { currentIndex = 0; aiController.demoFailure = "" }
            }
            ErrorPanel { text: captureController.error; Layout.fillWidth: true; Layout.margins: 16 }
            Button {
                text: "등록 후 AI 분석"; Layout.fillWidth: true; Layout.margins: 16; enabled: !aiController.loading
                onClicked: if (captureController.createFinding(photo.text, location.text, memo.text, action.text)) captured()
            }
            Label { text: "사진과 입력 내용은 먼저 Open 상태로 저장됩니다. AI 결과는 별도로 검토합니다."; wrapMode: Text.Wrap; Layout.fillWidth: true; Layout.margins: 16 }
        }
    }
}
