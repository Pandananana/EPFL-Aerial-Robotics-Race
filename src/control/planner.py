"""Mission FSM dispatcher.

Owns the gate tracker and the FSM `Context`, forwards pose updates and
gate detections to the current state, and swaps to whatever state the
state returns. The states themselves live under `src/control/states/` so
this file stays small — see `states/base.py` for the State contract.

The mission today is:

    TAKEOFF -> N x (SEARCH -> APPROACH -> MEASURE -> PASS_THROUGH)
            -> RETURN_HOME -> LAND -> DONE

Racing isn't implemented yet; it will slot in between the final
pass-through and the return-home transition.
"""

from __future__ import annotations

import logging
import math

from PyQt6 import QtCore

from src.bus import Latest
from src.control.states.base import Context, State
from src.control.states.gate_tracker import GateTracker
from src.control.states.takeoff import TakeoffState
from src.messages import DronePose, Gate3D

logger = logging.getLogger(__name__)


class Planner(QtCore.QObject):
    waypoint_ready = QtCore.pyqtSignal(object)   # Waypoint
    mission_done = QtCore.pyqtSignal()           # fires once after landing
    state_changed = QtCore.pyqtSignal(str)       # state class name
    gate_estimate_ready = QtCore.pyqtSignal(object)  # current Kalman gate corners

    DEFAULT_GATE_COUNT = 5

    def __init__(
        self,
        *,
        default_height_m: float,
        n_gates: int = DEFAULT_GATE_COUNT,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._pose: Latest[DronePose] = Latest()
        self._takeoff_height_m = default_height_m
        self._n_gates = n_gates

        self._state: State | None = None
        self._tracker = GateTracker()
        self._gates_done = 0
        self._start_x: float | None = None
        self._start_y: float | None = None
        self._start_yaw_rad: float | None = None

    @QtCore.pyqtSlot()
    def start(self) -> None:
        """Kick off the mission. Idempotent while already running."""
        if self._state is not None:
            return
        self._tracker.reset()
        self._gates_done = 0
        self._start_x = None
        self._start_y = None
        self._start_yaw_rad = None
        self._state = TakeoffState()
        logger.info("Mission start (target %d gates)", self._n_gates)
        self.state_changed.emit(type(self._state).__name__)

    @QtCore.pyqtSlot(object)
    def on_pose(self, pose: DronePose) -> None:
        self._pose.set(pose)
        if self._state is None:
            return
        if self._start_x is None:
            self._start_x = pose.x
            self._start_y = pose.y
            self._start_yaw_rad = math.radians(pose.yaw)

        ctx = self._make_context(pose)
        next_state = self._state.tick(ctx)
        # `gates_done` is a plain int on the context; states bump it but the
        # planner owns the source of truth, so pull it back every tick.
        self._gates_done = ctx.gates_done
        if next_state is not None:
            self._state = next_state
            self.state_changed.emit(type(self._state).__name__)

    @QtCore.pyqtSlot(object)
    def on_gate(self, gate: Gate3D) -> None:
        pose = self._pose.get()
        if self._state is None or pose is None:
            return
        ctx = self._make_context(pose)
        self._state.on_gate(ctx, gate)
        if self._tracker.kalman is not None:
            self.gate_estimate_ready.emit(self._tracker.kalman.corners())

    def _make_context(self, pose: DronePose) -> Context:
        return Context(
            pose=pose,
            start_x=float(self._start_x if self._start_x is not None else pose.x),
            start_y=float(self._start_y if self._start_y is not None else pose.y),
            start_yaw_rad=float(
                self._start_yaw_rad if self._start_yaw_rad is not None
                else math.radians(pose.yaw)
            ),
            tracker=self._tracker,
            gates_done=self._gates_done,
            n_gates=self._n_gates,
            takeoff_height_m=self._takeoff_height_m,
            emit_waypoint=self.waypoint_ready.emit,
            notify_mission_done=self.mission_done.emit,
        )
