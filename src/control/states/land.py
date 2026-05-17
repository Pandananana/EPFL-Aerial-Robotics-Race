"""Ramp the height target down to the ground, then signal mission done.

We track a wall-clock-style ramp on `pose.timestamp` so the descent rate
is independent of how often the planner ticks. When `pose.z` reaches the
ground band, we notify the planner (which forwards to the link's stop) and
transition to the terminal `DoneState`.
"""

from __future__ import annotations

import logging

from src.control.states.base import Context, State

logger = logging.getLogger(__name__)


class LandState(State):
    DESCENT_RATE_MPS = 0.3
    GROUND_Z_M = 0.12
    LAND_SPEED_MPS = 0.3

    def __init__(self) -> None:
        self._start_t: float | None = None
        self._start_z: float | None = None

    def tick(self, ctx: Context) -> State | None:
        if self._start_t is None:
            self._start_t = ctx.pose.timestamp
            self._start_z = ctx.pose.z

        elapsed = ctx.pose.timestamp - self._start_t
        target_z = max(0.0, (self._start_z or 0.0) - self.DESCENT_RATE_MPS * elapsed)

        ctx.emit(
            ctx.start_x, ctx.start_y, target_z,
            ctx.start_yaw_rad, self.LAND_SPEED_MPS,
        )

        if ctx.pose.z <= self.GROUND_Z_M:
            logger.info("Landed; mission complete")
            ctx.notify_mission_done()
            from src.control.states.done import DoneState
            return DoneState()
        return None
