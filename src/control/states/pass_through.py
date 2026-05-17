"""Drive through the gate plane to a fixed pass-through target.

No detection consumption here on purpose: once we're committed, we trust
the previous measurement rather than letting a partial detection of the
gate frame as we fly into it perturb the target. After reaching the target
we bump `gates_done`, reset the tracker, and either loop back to search
for the next gate or hand off to return-home if we've cleared all of them.
"""

from __future__ import annotations

import logging

import numpy as np

from src.control.states.base import Context, State

logger = logging.getLogger(__name__)


class PassThroughState(State):
    REACHED_M = 0.20
    PASS_SPEED_MPS = 0.8

    def __init__(self, target_pos: np.ndarray, target_yaw_rad: float) -> None:
        self._pos = target_pos.copy()
        self._yaw = target_yaw_rad

    def tick(self, ctx: Context) -> State | None:
        ctx.emit(self._pos[0], self._pos[1], self._pos[2], self._yaw, self.PASS_SPEED_MPS)

        drone_pos = np.array([ctx.pose.x, ctx.pose.y, ctx.pose.z])
        if np.linalg.norm(self._pos - drone_pos) > self.REACHED_M:
            return None

        ctx.gates_done += 1
        logger.info("Gate %d/%d cleared", ctx.gates_done, ctx.n_gates)
        ctx.tracker.reset()

        if ctx.gates_done >= ctx.n_gates:
            from src.control.states.return_home import ReturnHomeState
            return ReturnHomeState()
        from src.control.states.search import SearchState
        return SearchState()
