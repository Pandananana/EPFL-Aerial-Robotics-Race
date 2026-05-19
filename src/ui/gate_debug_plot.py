"""Matplotlib 3D gate debug plot for Webots runs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PyQt6 import QtCore

from src.control.states.gate_tracker import camera_corners_to_world
from src.messages import DronePose, Gate3D


def _true_gate_corners(row: dict[str, str]) -> np.ndarray:
    center = np.array(
        [float(row["x"]), float(row["y"]), float(row["z"])],
        dtype=np.float64,
    )
    theta = float(row["theta"])
    width = float(row["width"])
    height = float(row["height"])

    width_axis = np.array([math.cos(theta), math.sin(theta), 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    half_w = 0.5 * width * width_axis
    half_h = 0.5 * height * z_axis
    return np.array(
        [
            center - half_w + half_h,
            center + half_w + half_h,
            center + half_w - half_h,
            center - half_w - half_h,
        ],
        dtype=np.float64,
    )


def _plot_gate(ax, corners: np.ndarray, *, color: str, label: str) -> None:
    closed = np.vstack([corners, corners[0]])
    ax.plot(
        closed[:, 1],
        closed[:, 0],
        closed[:, 2],
        color=color,
        linewidth=2,
        label=label,
    )
    center = corners.mean(axis=0)
    ax.scatter(center[1], center[0], center[2], color=color, s=45)
    ax.text(center[1], center[0], center[2], label, color=color)


class GateDebugPlotter(QtCore.QObject):
    """Shows true Webots gates and current 3D detections in one 3D plot."""

    def __init__(
        self,
        *,
        truth_csv: Path,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._truth = self._load_truth(truth_csv)
        self._pose: DronePose | None = None
        self._poses_by_timestamp: dict[float, DronePose] = {}
        self._raw_estimates: list[np.ndarray] = []
        self._kalman_estimate: np.ndarray | None = None
        self._closed = False

        import matplotlib.pyplot as plt

        self._plt = plt
        self._plt.ion()
        self._fig = self._plt.figure("Webots gate debug", figsize=(8, 7))
        self._ax = self._fig.add_subplot(1, 1, 1, projection="3d")
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    @staticmethod
    def _load_truth(path: Path) -> dict[int, np.ndarray]:
        with open(path, newline="") as f:
            rows = csv.DictReader(f)
            return {int(row["Gate"]): _true_gate_corners(row) for row in rows}

    def _on_key(self, event) -> None:
        if event.key == "x":
            self._closed = True
            self._plt.close(self._fig)

    @QtCore.pyqtSlot(object)
    def on_pose(self, pose: DronePose) -> None:
        self._pose = pose
        self._poses_by_timestamp[pose.timestamp] = pose
        if len(self._poses_by_timestamp) > 512:
            oldest = sorted(self._poses_by_timestamp)[:128]
            for timestamp in oldest:
                del self._poses_by_timestamp[timestamp]

    @QtCore.pyqtSlot(object)
    def on_gate(self, gate: Gate3D) -> None:
        pose = self._poses_by_timestamp.get(gate.timestamp, self._pose)
        if self._closed or pose is None:
            return
        estimates = [
            np.asarray(camera_corners_to_world(corners, pose), dtype=np.float64)
            for corners in gate.corners_cam_m
        ]
        self._raw_estimates = estimates
        self._draw()

    @QtCore.pyqtSlot(object)
    def on_gate_estimate(self, corners: object) -> None:
        if self._closed:
            return
        estimate = np.asarray(corners, dtype=np.float64)
        if estimate.shape != (4, 3):
            return
        self._kalman_estimate = estimate
        self._draw()

    def _draw(self) -> None:
        self._ax.clear()

        for gate_id, corners in self._truth.items():
            _plot_gate(self._ax, corners, color="lime", label=f"T{gate_id}")

        truth_centers = {
            gate_id: corners.mean(axis=0) for gate_id, corners in self._truth.items()
        }
        for idx, corners in enumerate(self._raw_estimates):
            center = corners.mean(axis=0)
            if truth_centers:
                nearest_id = min(
                    truth_centers,
                    key=lambda gate_id: float(np.linalg.norm(center - truth_centers[gate_id])),
                )
                label = f"E{nearest_id}"
                true_center = truth_centers[nearest_id]
                self._ax.plot(
                    [true_center[1], center[1]],
                    [true_center[0], center[0]],
                    [true_center[2], center[2]],
                    "k--",
                    linewidth=1,
                    alpha=0.5,
                )
            else:
                label = f"E{idx}"
            _plot_gate(self._ax, corners, color="red", label=label)

        if self._kalman_estimate is not None:
            center = self._kalman_estimate.mean(axis=0)
            if truth_centers:
                nearest_id = min(
                    truth_centers,
                    key=lambda gate_id: float(np.linalg.norm(center - truth_centers[gate_id])),
                )
                label = f"K{nearest_id}"
            else:
                label = "K"
            _plot_gate(self._ax, self._kalman_estimate, color="blue", label=label)

        self._ax.set_xlabel("Y")
        self._ax.set_ylabel("X")
        self._ax.set_zlabel("Z")
        self._ax.set_title("True gates vs current estimates")
        self._ax.set_xlim(2, -2)
        self._ax.set_ylim(-2, 2)
        self._ax.set_zlim(0, 2.5)
        self._ax.view_init(elev=40, azim=-80)
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)
