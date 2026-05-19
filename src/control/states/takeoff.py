"""Phase 1: ascend to the configured altitude before any horizontal motion.

Holds the captured start XY and original yaw, ramps the height target up,
then waits for the drone to dwell inside the settle band for SETTLE_TIME_S
before handing off to search.
"""

from __future__ import annotations

import logging

from src.control.states.base import Context, State

logger = logging.getLogger(__name__)


class TakeoffState(State):
    SETTLE_TOLERANCE_M = 0.08
    SETTLE_TIME_S = 1.0
    TAKEOFF_SPEED_MPS = 0.2

    def __init__(self) -> None:
        self._settled_at: float | None = None

    def tick(self, ctx: Context) -> State | None:
        ctx.emit(
            ctx.start_x, ctx.start_y, ctx.takeoff_height_m,
            ctx.start_yaw_rad, self.TAKEOFF_SPEED_MPS,
        )
        if abs(ctx.pose.z - ctx.takeoff_height_m) <= self.SETTLE_TOLERANCE_M:
            if self._settled_at is None:
                self._settled_at = ctx.pose.timestamp
            elif ctx.pose.timestamp - self._settled_at >= self.SETTLE_TIME_S:
                logger.info("Takeoff complete; searching for first gate")
                from src.control.states.search import SearchState
                return SearchState()
        else:
            self._settled_at = None
        return None
