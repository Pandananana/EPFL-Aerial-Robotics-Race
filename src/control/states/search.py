"""Yaw in place until the gate detector locks onto something.

Increments a yaw setpoint by YAW_STEP_RAD each time the drone catches up
to within YAW_ADVANCE_TOL of the current target — i.e., a continuous slow
rotation built out of small overlapping setpoints. Detections feed the
tracker; as soon as it has an estimate, we pick the closer side of the
gate plane as the approach point and transition.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.control.states.base import Context, State
from src.messages import Gate3D

logger = logging.getLogger(__name__)


def _angle_diff(a: float, b: float) -> float:
    return abs(((a - b + math.pi) % (2 * math.pi)) - math.pi)


class SearchState(State):
    YAW_STEP_RAD = 0.3                          # ~17 deg between target increments
    YAW_ADVANCE_TOL_RAD = math.radians(5.0)
    SEARCH_SPEED_MPS = 0.2
    APPROACH_DISTANCE_M = 0.7
    MIN_ESTIMATES_BEFORE_APPROACH = 3

    def __init__(self) -> None:
        self._yaw_target_rad: float | None = None
        self._z_m: float | None = None

    def on_gate(self, ctx: Context, gate: Gate3D) -> None:
        ctx.tracker.update(gate, ctx.pose)

    def tick(self, ctx: Context) -> State | None:
        if (
            ctx.tracker.has_estimate
            and ctx.tracker.estimate_count >= self.MIN_ESTIMATES_BEFORE_APPROACH
        ):
            transition = self._build_approach(ctx)
            if transition is not None:
                return transition

        if self._z_m is None:
            self._z_m = ctx.pose.z

        current_yaw = math.radians(ctx.pose.yaw)
        if self._yaw_target_rad is None:
            self._yaw_target_rad = current_yaw + self.YAW_STEP_RAD
        elif _angle_diff(self._yaw_target_rad, current_yaw) < self.YAW_ADVANCE_TOL_RAD:
            self._yaw_target_rad = current_yaw + self.YAW_STEP_RAD

        ctx.emit(
            ctx.pose.x, ctx.pose.y, self._z_m,
            self._yaw_target_rad, self.SEARCH_SPEED_MPS,
        )
        return None

    def _build_approach(self, ctx: Context) -> State | None:
        from src.control.states.approach import ApproachState

        assert ctx.tracker.kalman is not None
        normal = ctx.tracker.oriented_normal()
        if normal is None:
            # Filter is populated but the corners are degenerate; drop and keep yawing.
            ctx.tracker.reset()
            return None

        center = ctx.tracker.kalman.center()
        drone_pos = np.array([ctx.pose.x, ctx.pose.y, ctx.pose.z])
        side_a = center + normal * self.APPROACH_DISTANCE_M
        side_b = center - normal * self.APPROACH_DISTANCE_M
        if np.linalg.norm(side_a - drone_pos) < np.linalg.norm(side_b - drone_pos):
            approach_pos = side_a
            ctx.tracker.approach_normal = normal.copy()
        else:
            approach_pos = side_b
            ctx.tracker.approach_normal = -normal.copy()

        direction = center - drone_pos
        target_yaw = math.atan2(direction[1], direction[0])
        logger.info(
            "Gate %d/%d detected at %s; approaching from %s",
            ctx.gates_done + 1, ctx.n_gates, np.round(center, 2), np.round(approach_pos, 2),
        )
        return ApproachState(approach_pos=approach_pos, target_yaw_rad=target_yaw)
