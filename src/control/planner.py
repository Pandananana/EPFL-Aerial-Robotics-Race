"""Mission state machine + gate selection.

Runs the three-phase race plan:

    IDLE -> TAKEOFF -> RECON_LAP -> RETURN_LAND -> RACE_LAP (x2) -> FINAL_LAND

The FSM is ticked from `on_pose` (the pose stream is the fastest input we
get, ~100 Hz on live, and is the right cadence for closed-loop decisions).
Every tick emits a `Waypoint` for the controller to track, except in IDLE
where the planner stays silent so manual control can still drive the link.

Gate inputs (`on_gate`) feed a world-frame gate map built up during recon
and reused as a prior during race laps. Only the TAKEOFF phase is wired
up right now; the rest of the FSM is stubbed (it holds the takeoff
waypoint) and the gate map is unused.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from PyQt6 import QtCore

from src.bus import Latest
from src.messages import DronePose, Gate3D, Waypoint


class Phase(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    RECON_LAP = auto()      # Needs Implementation
    RETURN_LAND = auto()    # Needs Implementation
    RACE_LAP = auto()       # Needs Implementation
    FINAL_LAND = auto()     # Needs Implementation


@dataclass
class GateEstimate:
    """World-frame gate pose, accumulated across observations."""
    x: float
    y: float
    z: float
    yaw: float
    n_observations: int


class Planner(QtCore.QObject):
    waypoint_ready = QtCore.pyqtSignal(object)  # Waypoint
    phase_changed = QtCore.pyqtSignal(object)   # Phase

    # Takeoff is "complete" once z is within this band of the target height
    # for SETTLE_TIME_S consecutive seconds.
    TAKEOFF_SETTLE_TOLERANCE_M = 0.05
    TAKEOFF_SETTLE_TIME_S = 1.0
    # We're not moving laterally during takeoff, so this is just a label.
    TAKEOFF_SPEED_MPS = 0.0

    def __init__(
        self,
        *,
        default_height_m: float,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._gate: Latest[Gate3D] = Latest()
        self._pose: Latest[DronePose] = Latest()

        self._target_height = default_height_m
        self._phase: Phase = Phase.IDLE

        # Captured on the first pose after start() — anchors takeoff and
        # is the "return to start" target for RETURN_LAND.
        self._start_x: float | None = None
        self._start_y: float | None = None
        self._start_yaw: float | None = None

        # When z first entered the takeoff settle band on the current ascent.
        self._takeoff_settled_at: float | None = None

        # Hold target — the last waypoint we emitted. Stubbed phases keep
        # emitting this so the drone hovers safely while we work on them.
        self._hold: Waypoint | None = None

        # World-frame gate map built during RECON_LAP, used as prior during
        # RACE_LAP. Keys are stable gate indices assigned by nearest-neighbor
        # association. Unused until RECON_LAP is implemented.
        self._gate_map: dict[int, GateEstimate] = {}
        self._visit_order: list[int] = []
        self._race_lap_count: int = 0

    @QtCore.pyqtSlot()
    def start(self) -> None:
        """Kick off the mission. The drone must be on the ground and armed.

        Idempotent — calling twice while already running is a no-op.
        """
        if self._phase is not Phase.IDLE:
            return
        self._start_x = None  # captured on first pose after this
        self._start_y = None
        self._start_yaw = None
        self._takeoff_settled_at = None
        self._set_phase(Phase.TAKEOFF)

    @QtCore.pyqtSlot(object)
    def on_gate(self, gate: Gate3D) -> None:
        self._gate.set(gate)
        # TODO: in RECON_LAP, transform corners to world frame using the
        # matched pose and update self._gate_map via nearest-neighbor.

    @QtCore.pyqtSlot(object)
    def on_pose(self, pose: DronePose) -> None:
        self._pose.set(pose)
        if self._phase is Phase.IDLE:
            return
        if self._phase is Phase.TAKEOFF:
            self._tick_takeoff(pose)
        else:
            # Stubbed phases: hold the last waypoint so the drone hovers
            # in place instead of crashing into NotImplementedError.
            if self._hold is not None:
                self.waypoint_ready.emit(self._hold)

    def _tick_takeoff(self, pose: DronePose) -> None:
        if self._start_x is None:
            self._start_x = pose.x
            self._start_y = pose.y
            self._start_yaw = pose.yaw

        wp = Waypoint(
            timestamp=pose.timestamp,
            x=self._start_x,
            y=self._start_y,
            z=self._target_height,
            yaw=self._start_yaw or 0.0,
            max_speed_mps=self.TAKEOFF_SPEED_MPS,
        )
        self._hold = wp
        self.waypoint_ready.emit(wp)

        if abs(pose.z - self._target_height) <= self.TAKEOFF_SETTLE_TOLERANCE_M:
            if self._takeoff_settled_at is None:
                self._takeoff_settled_at = pose.timestamp
            elif pose.timestamp - self._takeoff_settled_at >= self.TAKEOFF_SETTLE_TIME_S:
                self._set_phase(Phase.RECON_LAP)
        else:
            self._takeoff_settled_at = None

    def _set_phase(self, phase: Phase) -> None:
        if phase is self._phase:
            return
        self._phase = phase
        self.phase_changed.emit(phase)
