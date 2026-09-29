import QtQuick
import QtQuick.Controls
Label {
    padding: 12
    wrapMode: Text.Wrap
    color: "#a22222"
    visible: text.length > 0
    background: Rectangle { color: "#ffebeb"; radius: 8 }
    Accessible.role: Accessible.AlertMessage
    Accessible.name: text
}
