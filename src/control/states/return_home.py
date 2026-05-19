"""Fly back to the takeoff XY at recon altitude, facing the original yaw.

Stays at the recon height so the controller is just tracking a horizontal
waypoint and a yaw target — no descent yet. When both are within
tolerance, hand off to the landing ramp.
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

    def tick(self, ctx: Context) -> State | None:
        ctx.emit(
            ctx.start_x, ctx.start_y, ctx.takeoff_height_m,
            ctx.start_yaw_rad, self.RETURN_SPEED_MPS,
        )
        dx = ctx.pose.x - ctx.start_x
        dy = ctx.pose.y - ctx.start_y
        dist = math.hypot(dx, dy)
        yaw_err = abs(((ctx.start_yaw_rad - math.radians(ctx.pose.yaw) + math.pi) % (2 * math.pi)) - math.pi)
        if dist < self.REACHED_M and yaw_err < self.YAW_REACHED_RAD:
            logger.info("Returned to start; landing")
            from src.control.states.land import LandState
            return LandState()
        return None
