"""Records live data to disk in the same format the perception pipeline
already consumes.

For each Frame, writes img_NNNNNN.png and appends a measurements.csv row
pairing the image with the most recent DronePose. This matches the
recording layout described in CLAUDE.md so existing labeling/eval tools
keep working.

Also writes run_log.csv with the high-rate pose stream, FSM state, frame
names, perception summaries, waypoints, and setpoints for post-run analysis.

Recorder is a QObject; subscribe to Frame and DronePose signals with
auto-connection and slots will dispatch onto the thread Recorder lives
on (typically main), so on_frame and on_pose are serialised — no extra
locking needed.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import time

import cv2
import numpy as np
from PyQt6 import QtCore

from src.bus import Latest
from src.control.states.gate_tracker import camera_corners_to_world
from src.messages import (
    DronePose,
    Frame,
    Gate3D,
    GateDetection2D,
    GateEstimate,
    Setpoint,
    Waypoint,
)


class Recorder(QtCore.QObject):
    def __init__(
        self,
        *,
        base_dir: str,
        pose_log_every_n: int = 10,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.save_dir, exist_ok=True)
        self._csv_file = open(
            os.path.join(self.save_dir, "measurements.csv"), "w", newline=""
        )
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(
            [
                "timestamp", "image", "x", "y", "z", "roll", "pitch", "yaw",
                "lighthouse_bs_visible",
            ]
        )

        self._log_file = open(
            os.path.join(self.save_dir, "run_log.csv"), "w", newline=""
        )
        self._log_fields = [
            "timestamp",
            "wall_time",
            "event",
            "state",
            "frame_seq",
            "image",
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "lighthouse_bs_visible",
            "waypoint_x",
            "waypoint_y",
            "waypoint_z",
            "waypoint_yaw",
            "waypoint_max_speed_mps",
            "detected_gates",
            "gate_widths_m",
            "gate_reprojection_errors_px",
            "gate_world_centers_m",
            "message",
        ]
        self._log = csv.DictWriter(self._log_file, fieldnames=self._log_fields)
        self._log.writeheader()

        self._pose: Latest[DronePose] = Latest()
        self._gate_estimates: list[GateEstimate] = []
        self._state = "IDLE"
        self._count = 0
        self._log_every_n = max(1, int(pose_log_every_n))
        self._pose_count = 0
        self._waypoint_count = 0
        self._setpoint_count = 0
        self._write_log(event="run_start", message=f"save_dir={self.save_dir}")

    @QtCore.pyqtSlot(object)
    def on_pose(self, pose: DronePose) -> None:
        self._pose.set(pose)
        self._pose_count += 1
        if self._pose_count % self._log_every_n != 0:
            return
        self._write_log(
            timestamp=pose.timestamp,
            event="pose",
            x=pose.x,
            y=pose.y,
            z=pose.z,
            roll=pose.roll,
            pitch=pose.pitch,
            yaw=pose.yaw,
            lighthouse_bs_visible=pose.lighthouse_bs_visible,
        )

    @QtCore.pyqtSlot(object)
    def on_frame(self, frame: Frame) -> None:
        self._count += 1
        filename = f"img_{self._count:06d}.png"
        cv2.imwrite(os.path.join(self.save_dir, filename), frame.image)
        p = self._pose.get()
        if p is None:
            row = [
                frame.timestamp, filename, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "",
            ]
        else:
            row = [
                frame.timestamp, filename,
                p.x, p.y, p.z, p.roll, p.pitch, p.yaw,
                "" if p.lighthouse_bs_visible is None else p.lighthouse_bs_visible,
            ]
        self._csv.writerow(row)
        self._csv_file.flush()
        self._write_log(
            timestamp=frame.timestamp,
            event="frame",
            frame_seq=frame.seq,
            image=filename,
        )

    @QtCore.pyqtSlot(str)
    def on_state_changed(self, state: str) -> None:
        self._state = state
        self._write_log(event="state_changed", message=state)

    @QtCore.pyqtSlot(object)
    def on_detection(self, detection: GateDetection2D) -> None:
        self._write_log(
            timestamp=detection.timestamp,
            event="detection_2d",
            frame_seq=detection.frame_seq,
            detected_gates=len(detection.corners_px),
        )

    @QtCore.pyqtSlot(object)
    def on_gate(self, gate: Gate3D) -> None:
        pose = self._pose.get()
        world_centers = None
        if pose is not None and gate.corners_cam_m:
            world_centers = json.dumps([
                np.mean(camera_corners_to_world(c, pose), axis=0).tolist()
                for c in gate.corners_cam_m
            ])
        self._write_log(
            timestamp=gate.timestamp,
            event="gate_3d",
            frame_seq=gate.frame_seq,
            detected_gates=len(gate.corners_cam_m),
            gate_widths_m=json.dumps(gate.widths_m),
            gate_reprojection_errors_px=json.dumps(gate.reprojection_errors_px),
            gate_world_centers_m=world_centers or "",
            lighthouse_bs_visible=(
                "" if pose is None or pose.lighthouse_bs_visible is None
                else pose.lighthouse_bs_visible
            ),
        )

    @QtCore.pyqtSlot(object)
    def on_waypoint(self, waypoint: Waypoint) -> None:
        self._waypoint_count += 1
        if self._waypoint_count % self._log_every_n != 0:
            return
        self._write_log(
            timestamp=waypoint.timestamp,
            event="waypoint",
            waypoint_x=waypoint.x,
            waypoint_y=waypoint.y,
            waypoint_z=waypoint.z,
            waypoint_yaw=waypoint.yaw,
            waypoint_max_speed_mps=waypoint.max_speed_mps,
        )

    @QtCore.pyqtSlot(object)
    def on_setpoint(self, setpoint: Setpoint) -> None:
        self._setpoint_count += 1
        if self._setpoint_count % self._log_every_n != 0:
            return
        self._write_log(event="setpoint")

    @QtCore.pyqtSlot(object)
    def on_gate_estimated(self, est: GateEstimate) -> None:
        self._gate_estimates.append(est)
        self._write_log(
            event="gate_estimated",
            message=f"gate={est.gate_num} x={est.x:.3f} y={est.y:.3f} z={est.z:.3f} "
                    f"theta={est.theta_rad:.4f} w={est.width_m:.3f} h={est.height_m:.3f}",
        )
        self._save_gates_csv()

    @QtCore.pyqtSlot(str)
    def on_connected(self, status: str) -> None:
        self._write_log(event="connected", message=status)

    def close(self) -> None:
        self._write_log(event="run_end", message=f"frames={self._count}")
        self._csv_file.close()
        self._log_file.close()
        print(f"Saved {self._count} frames to {self.save_dir}")

    @property
    def frame_count(self) -> int:
        return self._count

    def _save_gates_csv(self) -> None:
        path = os.path.join(self.save_dir, "gates_estimates.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Gate", "x", "y", "z", "theta", "width", "height"])
            for est in self._gate_estimates:
                w.writerow([
                    est.gate_num,
                    round(est.x, 4),
                    round(est.y, 4),
                    round(est.z, 4),
                    round(est.theta_rad, 4),
                    round(est.width_m, 4),
                    round(est.height_m, 4),
                ])
        print(f"Saved gate estimates to {path}", flush=True)

    def _write_log(self, **values: object) -> None:
        row = {field: "" for field in self._log_fields}
        row["timestamp"] = values.pop("timestamp", time.time())
        row["wall_time"] = time.time()
        row["state"] = self._state
        for key, value in values.items():
            if key in row:
                row[key] = value
        self._log.writerow(row)
        self._log_file.flush()
