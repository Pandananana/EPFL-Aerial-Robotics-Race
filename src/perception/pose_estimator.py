"""3D gate pose estimation.

Subscribes to GateDetection2D, lifts each 2D quad into 3D using camera
intrinsics, the known gate height (0.48 m inner LED frame), and a
width grid search that minimises reprojection error. Emits Gate3D with
corners in OpenCV camera frame (x right, y down, z forward), metres.

World-frame transformation (using DronePose) is intentionally left for
the planning layer — it owns the body/world conventions and the gate-
selection logic that needs them.

The solver is the same as the previous detect_3d_gates.py; see comments
there for the gate-width grid search rationale.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6 import QtCore

from src.messages import Gate3D, GateDetection2D


class PoseEstimator(QtCore.QObject):
    gate_ready = QtCore.pyqtSignal(object)  # Gate3D
    EDGE_MARGIN_PX = 6.0

    def __init__(
        self,
        *,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        gate_height_m: float,
        width_search: tuple[float, float, float],
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._K = np.asarray(camera_matrix, dtype=np.float64)
        self._dist = np.asarray(dist_coeffs, dtype=np.float64)
        self._h = gate_height_m
        start, stop, step = width_search
        self._widths = np.arange(start, stop, step)

    @QtCore.pyqtSlot(object)
    def on_detection(self, det: GateDetection2D) -> None:
        corners_cam: list[np.ndarray] = []
        widths: list[float] = []
        errors: list[float] = []
        near_edges: list[bool] = []
        for q in det.corners_px:
            if q.shape != (4, 2):
                continue
            result = self._solve_gate_3d(q)
            if result is None:
                continue
            c, w, e = result
            corners_cam.append(c)
            widths.append(w)
            errors.append(e)
            near_edges.append(self._near_image_edge(q, det.image_shape_hw))

        self.gate_ready.emit(Gate3D(
            timestamp=det.timestamp,
            frame_seq=det.frame_seq,
            corners_cam_m=corners_cam,
            widths_m=widths,
            reprojection_errors_px=errors,
            near_image_edge=near_edges,
        ))

    def _near_image_edge(
        self,
        corners_px: np.ndarray,
        image_shape_hw: tuple[int, int] | None,
    ) -> bool:
        if image_shape_hw is None:
            return False
        h, w = image_shape_hw
        pts = np.asarray(corners_px, dtype=np.float64)
        return bool(
            np.any(pts[:, 0] <= self.EDGE_MARGIN_PX)
            or np.any(pts[:, 0] >= (w - 1 - self.EDGE_MARGIN_PX))
            or np.any(pts[:, 1] <= self.EDGE_MARGIN_PX)
            or np.any(pts[:, 1] >= (h - 1 - self.EDGE_MARGIN_PX))
        )

    def _gate_model(self, width: float) -> np.ndarray:
        h = self._h / 2.0
        w = width / 2.0
        return np.array(
            [
                [-w, -h, 0.0],  # TL
                [+w, -h, 0.0],  # TR
                [+w, +h, 0.0],  # BR
                [-w, +h, 0.0],  # BL
            ],
            dtype=np.float64,
        )

    def _solve_gate_3d(
        self, image_points: np.ndarray
    ) -> tuple[np.ndarray, float, float] | None:
        img = image_points.astype(np.float64).reshape(-1, 1, 2)
        best: tuple[float, np.ndarray, np.ndarray, float] | None = None
        for w in self._widths:
            model_pts = self._gate_model(float(w))
            try:
                ok, rvec, tvec, errs = cv2.solvePnPGeneric(
                    model_pts, img, self._K, self._dist,
                    flags=cv2.SOLVEPNP_IPPE,
                )
            except cv2.error:
                continue
            if not ok or not len(rvec):
                continue
            for r, t, e in zip(rvec, tvec, errs):
                err_px = float(np.asarray(e).reshape(-1)[0])
                if best is None or err_px < best[0]:
                    best = (err_px, r, t, float(w))
        if best is None:
            return None
        err_px, rvec, tvec, width = best
        R, _ = cv2.Rodrigues(rvec)
        corners_cam = (R @ self._gate_model(width).T + tvec.reshape(3, 1)).T
        return corners_cam, width, err_px
