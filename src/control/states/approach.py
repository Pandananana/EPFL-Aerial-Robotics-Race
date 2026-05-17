"""Fly to a stand-off point on the approach side of the gate.

Each tick refreshes the target from the latest filter estimate, picking the
side of the gate plane closer to where the drone already is (so the
approach doesn't flip mid-flight if the detector momentarily reorders
corners). Transitions to MEASURE once both position and yaw are within
tolerance.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.control.states.base import Context, State
from src.messages import Gate3D

logger = logging.getLogger(__name__)


def _yaw_err_rad(target_rad: float, current_deg: float) -> float:
    err = target_rad - math.radians(current_deg)
    return abs(((err + math.pi) % (2 * math.pi)) - math.pi)


class ApproachState(State):
    APPROACH_DISTANCE_M = 1
    REACHED_DIST_M = 0.15
    REACHED_YAW_RAD = math.radians(8.0)
    APPROACH_SPEED_MPS = 0.6

    def __init__(self, approach_pos: np.ndarray, target_yaw_rad: float) -> None:
        self._target_pos = approach_pos.copy()
        self._target_yaw_rad = target_yaw_rad

    def on_gate(self, ctx: Context, gate: Gate3D) -> None:
        ctx.tracker.update(gate, ctx.pose)

    def tick(self, ctx: Context) -> State | None:
        self._refresh_target(ctx)

        ctx.emit(
            self._target_pos[0], self._target_pos[1], self._target_pos[2],
            self._target_yaw_rad, self.APPROACH_SPEED_MPS,
        )

        drone_pos = np.array([ctx.pose.x, ctx.pose.y, ctx.pose.z])
        dist = float(np.linalg.norm(self._target_pos - drone_pos))
        yaw_err = _yaw_err_rad(self._target_yaw_rad, ctx.pose.yaw)
        if dist < self.REACHED_DIST_M and yaw_err < self.REACHED_YAW_RAD:
            logger.info("Approach point reached; measuring")
            from src.control.states.measure import MeasureState
            ctx.tracker.reset_filter_only()
            return MeasureState(self._target_pos.copy(), self._target_yaw_rad)
        return None

    def _refresh_target(self, ctx: Context) -> None:
        if not ctx.tracker.has_estimate:
            return
        normal = ctx.tracker.oriented_normal()
        if normal is None:
            return
        assert ctx.tracker.kalman is not None
        center = ctx.tracker.kalman.center()
        drone_pos = np.array([ctx.pose.x, ctx.pose.y, ctx.pose.z])
        side_a = center + normal * self.APPROACH_DISTANCE_M
        side_b = center - normal * self.APPROACH_DISTANCE_M
        chosen = normal if np.linalg.norm(side_a - drone_pos) <= np.linalg.norm(side_b - drone_pos) else -normal
        ctx.tracker.approach_normal = chosen
        self._target_pos = center + chosen * self.APPROACH_DISTANCE_M
        direction = center - drone_pos
        self._target_yaw_rad = math.atan2(direction[1], direction[0])
