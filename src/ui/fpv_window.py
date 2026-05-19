"""Live FPV display window.

Subscribes to Frame and GateDetection2D, renders each frame paired with
its matching detection (green polyline corners), and shows a connection
status line. Frames are undistorted with the calibration intrinsics
before display, and detection corners are remapped through the same
distortion model so the overlay still lines up with the gate. The
detector runs on its own thread (~65 ms behind camera), so we buffer
recent frames and only paint when the matching detection arrives —
display rate ends up at detection rate, but every overlay lines up
with the image underneath. Keyboard events are forwarded to a
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
    SCALE = 2
    FRAME_BUFFER = 16  # frames retained while we wait for matching detections

    def __init__(
        self,
        manual: ManualControl,
        *,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Crazyflie FPV")
        self._manual = manual
        self._frames: OrderedDict[int, Frame] = OrderedDict()
        self._rgb_buf: np.ndarray | None = None
        self._K = np.asarray(camera_matrix, dtype=np.float64)
        self._dist = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        # Remap LUTs are built lazily on the first frame, once we know the
        # image size. cv2.remap with fixed-point maps is much faster than
        # calling cv2.undistort per frame.
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None
        self._map_size: tuple[int, int] | None = None  # (w, h)

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

    @QtCore.pyqtSlot(object)
    def on_detection(self, det: GateDetection2D) -> None:
        frame = self._frames.pop(det.frame_seq, None)
        if frame is None:
            return  # detection arrived after its frame fell out of the buffer
        # Drop any older buffered frames — they'll never get a detection now.
        for seq in [s for s in self._frames if s < det.frame_seq]:
            del self._frames[seq]
        self._paint(frame, det)

    def _paint(self, frame: Frame, det: GateDetection2D) -> None:
        img = self._undistort(frame.image)
        h, w = img.shape[:2]
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        for q in det.corners_px:
            q_und = self._undistort_points(q)
            cv2.polylines(
                rgb, [q_und.astype(np.int32)],
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

    def _ensure_maps(self, size_wh: tuple[int, int]) -> None:
        if self._map_size == size_wh and self._map1 is not None:
            return
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._K, self._dist, R=None, newCameraMatrix=self._K,
            size=size_wh, m1type=cv2.CV_16SC2,
        )
        self._map_size = size_wh

    def _undistort(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        self._ensure_maps((w, h))
        return cv2.remap(img, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

    def _undistort_points(self, pts: np.ndarray) -> np.ndarray:
        src = pts.astype(np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(src, self._K, self._dist, P=self._K)
        return und.reshape(-1, 2)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        self._manual.handle_key_press(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        self._manual.handle_key_release(event)
