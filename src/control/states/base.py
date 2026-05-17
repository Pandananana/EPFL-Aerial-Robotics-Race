"""Base abstractions for the recon-lap FSM.

States are small and pure: `tick(ctx)` returns either `None` (stay) or a
new State (transition). Per-state scratch (timers, sub-targets) lives on
the instance; mission-wide scratch (start pose, gate tracker, gate count)
lives on `Context`, which is rebuilt fresh from the planner each tick.

The `tracker` field on the Context is a shared, long-lived `GateTracker` so
mutations from one state are visible to the next. Detections arrive on
their own signal — the planner forwards each one to the current state's
`on_gate` hook, which defaults to a no-op so non-perception states don't
have to care.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.messages import DronePose, Gate3D, Waypoint

if TYPE_CHECKING:
    from src.control.states.gate_tracker import GateTracker


@dataclass
class Context:
    """Mutable shared state surfaced to each state every tick."""
    pose: DronePose
    start_x: float
    start_y: float
    start_yaw_rad: float
    tracker: "GateTracker"
    gates_done: int
    n_gates: int
    takeoff_height_m: float
    emit_waypoint: Callable[[Waypoint], None]
    notify_mission_done: Callable[[], None]

    def emit(
        self,
        x: float,
        y: float,
        z: float,
        yaw_rad: float,
        max_speed_mps: float,
    ) -> None:
        self.emit_waypoint(Waypoint(
            timestamp=self.pose.timestamp,
            x=float(x), y=float(y), z=float(z),
            yaw=math.degrees(yaw_rad),
            max_speed_mps=float(max_speed_mps),
        ))


class State(ABC):
    """Abstract FSM state. Subclass and override `tick` (and optionally `on_gate`)."""

    @abstractmethod
    def tick(self, ctx: Context) -> "State | None": ...

    def on_gate(self, ctx: Context, gate: Gate3D) -> None:
        """Hook for detection-consuming states. Default: ignore."""
        return None
