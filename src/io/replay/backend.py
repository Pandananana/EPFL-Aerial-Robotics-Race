"""Replay a recording directory as if it were a live source.

Reads data/recordings/<id>/measurements.csv and re-emits Frame + DronePose
messages at the rate they were originally captured (scaled by `speed`).
With step mode enabled, each keypress advances one recorded row instead.

If run_log.csv is present in the same directory, `state_changed` and
`gate_estimated` signals are emitted in timestamp order alongside the
frame/pose stream so the UI shows the same FSM behaviour as the original run.

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

from src.messages import DronePose, Frame, GateEstimate, Setpoint


class ReplayThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)   # Frame
    pose_ready = QtCore.pyqtSignal(object)    # DronePose
    connected = QtCore.pyqtSignal(str)
    state_changed = QtCore.pyqtSignal(str)    # FSM state name from run_log
    gate_estimated = QtCore.pyqtSignal(object)  # GateEstimate from run_log

    def __init__(
        self,
        recording_dir: Path,
        *,
        speed: float = 1.0,
        step: bool = False,
        start_frame: int = 1,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._dir = Path(recording_dir)
        self._speed = speed
        self._step = step
        self._start_frame = max(1, int(start_frame))
        self._mutex = QtCore.QMutex()
        self._step_ready = QtCore.QWaitCondition()
        self._pending_steps = 0

    def open(self) -> None:
        """DroneLink lifecycle. The thread is started separately as the
        video source; here we just fire `connected` so the UI updates."""
        self.connected.emit(f"replay:{self._dir.name}")

    def close(self) -> None:
        self.requestInterruption()
        self._wake()
        self.wait()

    @QtCore.pyqtSlot(object)
    def set_setpoint(self, sp: Setpoint) -> None:  # noqa: ARG002
        """No-op: replay cannot command a drone."""

    @QtCore.pyqtSlot()
    def send_stop(self) -> None:
        """No-op: replay cannot command a drone."""

    @QtCore.pyqtSlot()
    def advance(self) -> None:
        """Advance one recorded row in step mode."""
        if not self._step:
            return
        self._mutex.lock()
        try:
            self._pending_steps += 1
            self._step_ready.wakeOne()
        finally:
            self._mutex.unlock()

    def run(self) -> None:
        csv_path = self._dir / "measurements.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        rows = rows[self._start_frame - 1:]
        if not rows:
            return

        log_events = self._load_log_events()

        t0_rec = float(rows[0]["timestamp"])
        t0_wall = time.monotonic()
        seq = self._start_frame - 1
        log_idx = 0

        for r in rows:
            if self.isInterruptionRequested():
                return
            t_rec = float(r["timestamp"])
            if self._step:
                self._wait_for_step()
                if self.isInterruptionRequested():
                    return
            else:
                target = t0_wall + (t_rec - t0_rec) / self._speed
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

            self.pose_ready.emit(DronePose(
                timestamp=t_rec,
                x=float(r["x"]), y=float(r["y"]), z=float(r["z"]),
                roll=float(r["roll"]), pitch=float(r["pitch"]), yaw=float(r["yaw"]),
                lighthouse_bs_visible=(
                    int(r["lighthouse_bs_visible"])
                    if r.get("lighthouse_bs_visible") else None
                ),
            ))

            img_path = self._dir / r["image"]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                seq += 1
                self.frame_ready.emit(Frame(timestamp=t_rec, seq=seq, image=img))

            # Emit all run_log events whose timestamp falls up to this row.
            while log_idx < len(log_events) and log_events[log_idx][0] <= t_rec:
                _, kind, payload = log_events[log_idx]
                if kind == "state_changed":
                    self.state_changed.emit(payload)
                elif kind == "gate_estimated":
                    self.gate_estimated.emit(payload)
                log_idx += 1

    # ------------------------------------------------------------------
    # Run-log helpers
    # ------------------------------------------------------------------

    def _load_log_events(self) -> list[tuple[float, str, object]]:
        """Return a timestamp-sorted list of (timestamp, kind, payload) tuples.

        Only `state_changed` and `gate_estimated` events are extracted.
        Returns an empty list when run_log.csv is absent.
        """
        log_path = self._dir / "run_log.csv"
        if not log_path.exists():
            return []

        events: list[tuple[float, str, object]] = []
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                event = row.get("event", "")
                try:
                    ts = float(row["timestamp"])
                except (KeyError, ValueError):
                    continue

                if event == "state_changed":
                    state_name = row.get("message", "").strip()
                    if state_name:
                        events.append((ts, "state_changed", state_name))

                elif event == "gate_estimated":
                    est = _parse_gate_estimate(row.get("message", ""))
                    if est is not None:
                        events.append((ts, "gate_estimated", est))

        events.sort(key=lambda e: e[0])
        return events

    def _wait_for_step(self) -> None:
        self._mutex.lock()
        try:
            while self._pending_steps <= 0 and not self.isInterruptionRequested():
                self._step_ready.wait(self._mutex, 100)
            if self._pending_steps > 0:
                self._pending_steps -= 1
        finally:
            self._mutex.unlock()

    def _wake(self) -> None:
        self._mutex.lock()
        try:
            self._step_ready.wakeAll()
        finally:
            self._mutex.unlock()


def _parse_gate_estimate(message: str) -> GateEstimate | None:
    """Parse the recorder's gate_estimated message string back to a GateEstimate.

    Expected format: "gate=1 x=0.096 y=0.746 z=1.545 theta=-0.0944 w=0.768 h=0.393"
    """
    try:
        fields: dict[str, float] = {}
        for token in message.split():
            k, _, v = token.partition("=")
            fields[k] = float(v)
        return GateEstimate(
            gate_num=int(fields["gate"]),
            x=fields["x"],
            y=fields["y"],
            z=fields["z"],
            theta_rad=fields["theta"],
            width_m=fields["w"],
            height_m=fields["h"],
        )
    except (KeyError, ValueError):
        return None
