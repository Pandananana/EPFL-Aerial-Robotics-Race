"""Selects which gate to fly to next and emits a target waypoint.

STUB — owned by the controls team. The current implementation just
caches the most recent Gate3D and DronePose; it does not emit any
waypoints. Fill in target-selection logic to make it real.

Subscribes to Gate3D (perception) and DronePose (cflib).
Publishes Waypoint via `waypoint_ready` (define the dataclass in
src/messages.py when ready).
"""

from __future__ import annotations

from PyQt6 import QtCore

from src.bus import Latest
from src.messages import DronePose, Gate3D


class Planner(QtCore.QObject):
    waypoint_ready = QtCore.pyqtSignal(object)  # Waypoint (TBD)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._gate: Latest[Gate3D] = Latest()
        self._pose: Latest[DronePose] = Latest()

    @QtCore.pyqtSlot(object)
    def on_gate(self, gate: Gate3D) -> None:
        self._gate.set(gate)

    @QtCore.pyqtSlot(object)
    def on_pose(self, pose: DronePose) -> None:
        self._pose.set(pose)
