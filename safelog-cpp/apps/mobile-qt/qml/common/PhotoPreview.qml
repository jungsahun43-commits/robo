import QtQuick
import QtQuick.Controls
Rectangle {
    id: preview
    property string source: ""
    implicitHeight: 180
    color: "#e1e8ec"
    radius: 8
    Image { id: photo; source: preview.source.startsWith("/") ? "file://" + preview.source : preview.source; anchors.fill: parent; fillMode: Image.PreserveAspectFit; asynchronous: true }
    Label { anchors.centerIn: parent; visible: photo.status !== Image.Ready; text: photo.status === Image.Error ? "사진을 읽을 수 없습니다" : "사진을 선택하세요" }
}
