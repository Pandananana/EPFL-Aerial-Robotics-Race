"""Live gate detector: Frame in, GateDetection2D out.

Thin adapter over the offline-trained models in `models/`. The choice of
model is config-driven; each backend exposes the same
`predict_gates(image) -> list[(4,2) np.ndarray]` contract documented in
the project CLAUDE.md, so swapping is a one-line config change.

Inference is currently synchronous on whatever thread the `on_frame` slot
runs on. If YOLO inference becomes the bottleneck and starts backing up
the Qt event queue, moveToThread() this object onto its own QThread —
the public signal/slot interface stays identical.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6 import QtCore

from src.messages import Frame, GateDetection2D


class GateDetector(QtCore.QObject):
    detection_ready = QtCore.pyqtSignal(object)  # GateDetection2D

    def __init__(self, *, model_name: str = "yolo_pose", parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._predict = _load_predictor(model_name)

    @QtCore.pyqtSlot(object)
    def on_frame(self, frame: Frame) -> None:
        quads = self._predict(frame.image)
        self.detection_ready.emit(GateDetection2D(
            timestamp=frame.timestamp,
            frame_seq=frame.seq,
            corners_px=[np.asarray(q, dtype=np.float32) for q in quads],
        ))


def _load_predictor(name: str) -> Callable[[np.ndarray], list[np.ndarray]]:
    if name == "yolo_pose":
        from models.yolo_pose.detector import predict_gates
        return predict_gates
    if name == "yolo_seg":
        from models.yolo_seg.detector import predict_gates
        return predict_gates
    if name == "yolo_obb":
        from models.yolo_obb.detector import predict_gates
        return predict_gates
    if name == "hough":
        from models.hough_detector import predict_gates
        return predict_gates
    raise ValueError(f"unknown detector: {name!r}")
