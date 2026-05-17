"""Records live data to disk in the same format the perception pipeline
already consumes.

For each Frame, writes img_NNNNNN.png and appends a measurements.csv row
pairing the image with the most recent DronePose. This matches the
recording layout described in CLAUDE.md so existing labeling/eval tools
keep working.

Recorder is a QObject; subscribe to Frame and DronePose signals with
auto-connection and slots will dispatch onto the thread Recorder lives
on (typically main), so on_frame and on_pose are serialised — no extra
locking needed.
"""

from __future__ import annotations

import csv
import datetime
import os

import cv2
from PyQt6 import QtCore

from src.bus import Latest
from src.messages import DronePose, Frame


class Recorder(QtCore.QObject):
    def __init__(self, *, base_dir: str, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.save_dir, exist_ok=True)
        self._csv_file = open(
            os.path.join(self.save_dir, "measurements.csv"), "w", newline=""
        )
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(
            ["timestamp", "image", "x", "y", "z", "roll", "pitch", "yaw"]
        )
        self._pose: Latest[DronePose] = Latest()
        self._count = 0

    @QtCore.pyqtSlot(object)
    def on_pose(self, pose: DronePose) -> None:
        self._pose.set(pose)

    @QtCore.pyqtSlot(object)
    def on_frame(self, frame: Frame) -> None:
        self._count += 1
        filename = f"img_{self._count:06d}.png"
        cv2.imwrite(os.path.join(self.save_dir, filename), frame.image)
        p = self._pose.get()
        if p is None:
            row = [frame.timestamp, filename, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            row = [
                frame.timestamp, filename,
                p.x, p.y, p.z, p.roll, p.pitch, p.yaw,
            ]
        self._csv.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        self._csv_file.close()
        print(f"Saved {self._count} frames to {self.save_dir}")

    @property
    def frame_count(self) -> int:
        return self._count
