"""Live FPV display window.

Subscribes to Frame, renders the grayscale image, shows a connection
status line. Keyboard events are forwarded to a ManualControl instance —
the window itself does no control logic.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from src.control.manual import ManualControl
from src.messages import Frame


class FpvWindow(QtWidgets.QWidget):
    def __init__(self, manual: ManualControl, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Crazyflie FPV")
        self._manual = manual

        self.image_label = QtWidgets.QLabel("Waiting for video...")
        self.status_label = QtWidgets.QLabel("Initialising...")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)

    @QtCore.pyqtSlot(str)
    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    @QtCore.pyqtSlot(object)
    def on_frame(self, frame: Frame) -> None:
        img = frame.image
        h, w = img.shape[:2]
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg.scaled(w * 2, h * 2)))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        self._manual.handle_key_press(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        self._manual.handle_key_release(event)
