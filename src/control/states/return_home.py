"""Fly back to the takeoff XY at current altitude, then descend.

Stays at whatever height the drone is at when this state is entered so
there is no altitude jump after the final gate. Once both XY position and
yaw are within tolerance, hand off to the landing ramp.
"""

from __future__ import annotations

import logging
import math

from src.control.states.base import Context, State

logger = logging.getLogger(__name__)


class ReturnHomeState(State):
    REACHED_M = 0.20
    YAW_REACHED_RAD = math.radians(10.0)
    RETURN_SPEED_MPS = 0.2

    def __init__(self, then_after_land: State | None = None) -> None:
        """`then_after_land` is forwarded to LandState. None means the landing
        is terminal (mission_done fires)."""
        self._then_after_land = then_after_land
        self._cruise_z: float | None = None

    def tick(self, ctx: Context) -> State | None:
        if self._cruise_z is None:
            self._cruise_z = ctx.pose.z

        ctx.emit(
            ctx.start_x, ctx.start_y, self._cruise_z,
            ctx.start_yaw_rad, self.RETURN_SPEED_MPS,
        )
        dx = ctx.pose.x - ctx.start_x
        dy = ctx.pose.y - ctx.start_y
        dist = math.hypot(dx, dy)
        yaw_err = abs(((ctx.start_yaw_rad - math.radians(ctx.pose.yaw) + math.pi) % (2 * math.pi)) - math.pi)
        if dist < self.REACHED_M and yaw_err < self.YAW_REACHED_RAD:
            logger.info("Returned to start at z=%.2f; landing", self._cruise_z)
            from src.control.states.land import LandState
            return LandState(then=self._then_after_land)
        return None
