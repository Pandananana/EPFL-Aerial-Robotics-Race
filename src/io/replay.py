"""Replay a recording directory as if it were a live source.

Reads data/recordings/<id>/measurements.csv and re-emits Frame + DronePose
messages at the rate they were originally captured (scaled by `speed`).

ReplayThread implements both the VideoSource and DroneLink protocols
(see src/io/sources.py) so it can be wired in wherever UdpVideoThread +
CrazyflieLink would go. set_setpoint and send_stop are no-ops: in replay
there is no drone to command, so the controller / manual control's
output is dropped on the floor — only live or Webots can actually drive.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
from PyQt6 import QtCore

from src.messages import DronePose, Frame, Setpoint


class ReplayThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)  # Frame
    pose_ready = QtCore.pyqtSignal(object)  # DronePose
    connected = QtCore.pyqtSignal(str)

    def __init__(
        self,
        recording_dir: Path,
        *,
        speed: float = 1.0,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._dir = Path(recording_dir)
        self._speed = speed

    def open(self) -> None:
        """DroneLink lifecycle. The thread is started separately as the
        video source; here we just fire `connected` so the UI updates."""
        self.connected.emit(f"replay:{self._dir.name}")

    def close(self) -> None:
        self.requestInterruption()
        self.wait()

    @QtCore.pyqtSlot(object)
    def set_setpoint(self, sp: Setpoint) -> None:  # noqa: ARG002
        """No-op: replay cannot command a drone."""

    @QtCore.pyqtSlot()
    def send_stop(self) -> None:
        """No-op: replay cannot command a drone."""

    def run(self) -> None:
        csv_path = self._dir / "measurements.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return

        t0_rec = float(rows[0]["timestamp"])
        t0_wall = time.monotonic()
        seq = 0

        for r in rows:
            if self.isInterruptionRequested():
                return
            t_rec = float(r["timestamp"])
            target = t0_wall + (t_rec - t0_rec) / self._speed
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            self.pose_ready.emit(DronePose(
                timestamp=t_rec,
                x=float(r["x"]), y=float(r["y"]), z=float(r["z"]),
                roll=float(r["roll"]), pitch=float(r["pitch"]), yaw=float(r["yaw"]),
            ))

            img_path = self._dir / r["image"]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            seq += 1
            self.frame_ready.emit(Frame(timestamp=t_rec, seq=seq, image=img))
