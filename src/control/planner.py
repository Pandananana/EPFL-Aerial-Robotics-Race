"""Mission FSM dispatcher.

Owns the gate tracker and the FSM `Context`, forwards pose updates and
gate detections to the current state, and swaps to whatever state the
state returns. The states themselves live under `src/control/states/` so
this file stays small — see `states/base.py` for the State contract.

Two mission shapes are supported, picked by whether a preloaded gate list
is supplied to the planner:

- **Full** (no preloaded gates):
    TAKEOFF -> N x (SEARCH -> APPROACH -> MEASURE -> PASS_THROUGH)
            -> RETURN_HOME -> LAND (motors stay armed)
            -> SAVE_GATES (writes gates.csv) -> TAKEOFF
            -> RACE -> RETURN_HOME -> LAND (terminal) -> DONE

- **Race-only** (preloaded gates, e.g. from a prior recon lap's CSV):
    TAKEOFF -> RACE -> RETURN_HOME -> LAND (terminal) -> DONE
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from PyQt6 import QtCore

from src.bus import Latest
from src.control.gates_csv import RecordedGate
from src.control.states.base import Context, State
from src.control.states.gate_tracker import GateTracker
from src.control.states.race import RaceState
from src.control.states.takeoff import TakeoffState
from src.messages import DronePose, Gate3D

logger = logging.getLogger(__name__)


class Planner(QtCore.QObject):
    waypoint_ready = QtCore.pyqtSignal(object)   # Waypoint
    mission_done = QtCore.pyqtSignal()           # fires once after landing
    state_changed = QtCore.pyqtSignal(str)       # state class name
    gate_estimate_ready = QtCore.pyqtSignal(object)  # current Kalman gate corners

    DEFAULT_GATE_COUNT = 5
    DEFAULT_NUM_RACE_LAPS = 2

    def __init__(
        self,
        *,
        default_height_m: float,
        n_gates: int = DEFAULT_GATE_COUNT,
        preloaded_gates: list[RecordedGate] | None = None,
        gates_save_path: Path | None = None,
        num_race_laps: int = DEFAULT_NUM_RACE_LAPS,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._pose: Latest[DronePose] = Latest()
        self._takeoff_height_m = default_height_m
        self._n_gates = n_gates
        self._preloaded_gates = list(preloaded_gates) if preloaded_gates else None
        self._gates_save_path = Path(gates_save_path) if gates_save_path is not None else None
        self._num_race_laps = int(num_race_laps)

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
        self._tracker.recorded_gates = []
        self._gates_done = 0
        self._start_x = None
        self._start_y = None
        self._start_yaw_rad = None
        if self._preloaded_gates:
            race = RaceState(self._preloaded_gates, num_laps=self._num_race_laps)
            self._state = TakeoffState(then=race)
            logger.info(
                "Mission start (race-only, %d preloaded gates, %d laps)",
                len(self._preloaded_gates), self._num_race_laps,
            )
        else:
            self._state = TakeoffState()
            logger.info(
                "Mission start (full: recon %d gates -> race %d laps)",
                self._n_gates, self._num_race_laps,
            )
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
            gates_save_path=self._gates_save_path,
            num_race_laps=self._num_race_laps,
        )
