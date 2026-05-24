"""Hover at the approach point so the Kalman filter shrinks, then aim past the gate.

Holds for HOLD_S seconds (long enough at the AI-deck's ~3 FPS to get a
handful of fresh detections after the filter reset) before computing the
pass-through target as a point PASS_OVERSHOOT_M past the gate centre along
the drone's heading vector.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.control.states.base import Context, State
from src.messages import Gate3D, GateEstimate

logger = logging.getLogger(__name__)


class MeasureState(State):
    HOLD_S = 2.0
    PASS_OVERSHOOT_M = 0.4
    MEASURE_SPEED_MPS = 0.3
    MIN_ESTIMATES_BEFORE_PASS = 5

    def __init__(self, hold_pos: np.ndarray, hold_yaw_rad: float) -> None:
        self._pos = hold_pos.copy()
        self._yaw = hold_yaw_rad
        self._start_t: float | None = None

    def on_gate(self, ctx: Context, gate: Gate3D) -> None:
        center = ctx.tracker.update(gate, ctx.pose)
        if center is not None:
            ctx.notify_measurement_accepted(ctx.gates_done + 1, center)
        # Keep `approach_normal` aligned with the filter's current normal so
        # the next pass-through computation can't pick the wrong side.
        n = ctx.tracker.oriented_normal()
        if n is not None:
            ctx.tracker.approach_normal = n

    def tick(self, ctx: Context) -> State | None:
        if self._start_t is None:
            self._start_t = ctx.pose.timestamp

        ctx.emit(self._pos[0], self._pos[1], self._pos[2], self._yaw, self.MEASURE_SPEED_MPS)
        min_estimates = (
            self.MIN_ESTIMATES_BEFORE_PASS
            if ctx.require_measurement_count else 1
        )

        if (
            ctx.pose.timestamp - self._start_t < self.HOLD_S
            or not ctx.tracker.has_estimate
            or ctx.tracker.estimate_count < min_estimates
        ):
            return None

        assert ctx.tracker.kalman is not None
        center = ctx.tracker.kalman.center()
        drone_pos = np.array([ctx.pose.x, ctx.pose.y, ctx.pose.z])
        direction = center - drone_pos
        d_norm = float(np.linalg.norm(direction))
        if d_norm < 1e-6:
            return None
        direction = direction / d_norm
        PRE_GATE_M = 0.20
        pass_target = center + direction * self.PASS_OVERSHOOT_M
        pre_gate = center - direction * PRE_GATE_M
        pass_yaw = math.atan2(direction[1], direction[0])

        rec = ctx.tracker.record_current_gate()
        if rec is not None:
            logger.info(
                "Recorded gate %d: center=%s normal=%s width=%.2fm height=%.2fm",
                len(ctx.tracker.recorded_gates),
                np.round(rec.center, 3), np.round(rec.normal, 3),
                rec.width_m, rec.height_m,
            )

        logger.info(
            "Pass-through pre-gate %s -> target %s yaw=%.1f deg",
            np.round(pre_gate, 2), np.round(pass_target, 2), math.degrees(pass_yaw),
        )
        self._emit_estimate(ctx)
        from src.control.states.pass_through import PassThroughState
        return PassThroughState(pass_target, pass_yaw, pre_gate_pos=pre_gate)

    def _emit_estimate(self, ctx: Context) -> None:
        if ctx.tracker.kalman is None:
            return
        corners = ctx.tracker.kalman.corners()
        center = ctx.tracker.kalman.center()
        normal = ctx.tracker.oriented_normal()
        theta = math.atan2(normal[1], normal[0]) if normal is not None else 0.0
        width = float(np.linalg.norm(corners[1] - corners[0]))
        height = float(np.linalg.norm(corners[3] - corners[0]))
        ctx.notify_gate_estimated(GateEstimate(
            gate_num=ctx.gates_done + 1,
            x=float(center[0]),
            y=float(center[1]),
            z=float(center[2]),
            theta_rad=theta,
            width_m=width,
            height_m=height,
        ))
