"""Live FPV display window.

Subscribes to Frame and GateDetection2D, renders each frame paired with
its matching detection (green polyline corners), and shows a connection
status line. The detector runs on its own thread (~65 ms behind camera),
so we buffer recent frames and only paint when the matching detection
arrives — display rate ends up at detection rate, but every overlay
lines up with the image underneath. Keyboard events are forwarded to a
ManualControl instance — the window itself does no control logic.
"""

from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from src.control.manual import ManualControl
from src.messages import Frame, GateDetection2D


class FpvWindow(QtWidgets.QWidget):
    key_pressed = QtCore.pyqtSignal()

    SCALE = 2
    FRAME_BUFFER = 16  # frames retained while we wait for matching detections
    DETECTION_BUFFER = 16

    def __init__(self, manual: ManualControl, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Crazyflie FPV")
        self._manual = manual
        self._frames: OrderedDict[int, Frame] = OrderedDict()
        self._detections: OrderedDict[int, GateDetection2D] = OrderedDict()
        self._rgb_buf: np.ndarray | None = None

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
        self._frames[frame.seq] = frame
        while len(self._frames) > self.FRAME_BUFFER:
            self._frames.popitem(last=False)
        det = self._detections.pop(frame.seq, None)
        self._paint(frame, det)

    @QtCore.pyqtSlot(object)
    def on_detection(self, det: GateDetection2D) -> None:
        frame = self._frames.pop(det.frame_seq, None)
        if frame is None:
            self._detections[det.frame_seq] = det
            while len(self._detections) > self.DETECTION_BUFFER:
                self._detections.popitem(last=False)
            return
        # Drop any older buffered frames — they'll never get a detection now.
        for seq in [s for s in self._frames if s < det.frame_seq]:
            del self._frames[seq]
        self._paint(frame, det)

    def _paint(self, frame: Frame, det: GateDetection2D | None = None) -> None:
        img = frame.image
        h, w = img.shape[:2]
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        if det is not None:
            for q in det.corners_px:
                cv2.polylines(
                    rgb, [q.astype(np.int32)],
                    isClosed=True, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA,
                )
        # Keep a reference so QImage's data isn't freed before paint.
        self._rgb_buf = np.ascontiguousarray(rgb)
        qimg = QtGui.QImage(
            self._rgb_buf.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888,
        )
        self.image_label.setPixmap(
            QtGui.QPixmap.fromImage(qimg.scaled(w * self.SCALE, h * self.SCALE))
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not event.isAutoRepeat():
            self.key_pressed.emit()
        self._manual.handle_key_press(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        self._manual.handle_key_release(event)
