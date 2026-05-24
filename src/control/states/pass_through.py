"""Drive through the gate plane to a fixed pass-through target.

No detection consumption here on purpose: once we're committed, we trust
the previous measurement rather than letting a partial detection of the
gate frame as we fly into it perturb the target. After reaching the target
we bump `gates_done`, reset the tracker, and either loop back to search
for the next gate or hand off into the post-recon chain (save gates,
re-takeoff, race, return-home, land) if we've cleared all of them.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.control.states.base import Context, State

logger = logging.getLogger(__name__)


def _angle_diff(a: float, b: float) -> float:
    return abs(((a - b + math.pi) % (2 * math.pi)) - math.pi)


class PassThroughState(State):
    REACHED_M = 0.20
    PASS_SPEED_MPS = 0.1
    TURN_RIGHT_RAD = -math.pi / 4          # 45 deg clockwise (right) after pass-through
    YAW_TOL_RAD = math.radians(5.0)
    HOLD_S = 1.0
    HOLD_SPEED_MPS = 0.1

    def __init__(
        self,
        target_pos: np.ndarray,
        target_yaw_rad: float,
        pre_gate_pos: np.ndarray | None = None,
    ) -> None:
        self._pos = target_pos.copy()
        self._yaw = target_yaw_rad
        self._pre_gate_pos = pre_gate_pos.copy() if pre_gate_pos is not None else None
        self._pre_gate_reached = pre_gate_pos is None
        self._reached = False
        self._turn_yaw_rad: float | None = None
        self._hold_start_t: float | None = None

    def tick(self, ctx: Context) -> State | None:
        if not self._pre_gate_reached:
            assert self._pre_gate_pos is not None
            ctx.emit(self._pre_gate_pos[0], self._pre_gate_pos[1], self._pre_gate_pos[2], self._yaw, self.PASS_SPEED_MPS)
            drone_pos = np.array([ctx.pose.x, ctx.pose.y, ctx.pose.z])
            if np.linalg.norm(self._pre_gate_pos - drone_pos) > self.REACHED_M:
                return None
            logger.info("Pre-gate point reached; flying through")
            self._pre_gate_reached = True

        if not self._reached:
            ctx.emit(self._pos[0], self._pos[1], self._pos[2], self._yaw, self.PASS_SPEED_MPS)
            drone_pos = np.array([ctx.pose.x, ctx.pose.y, ctx.pose.z])
            if np.linalg.norm(self._pos - drone_pos) > self.REACHED_M:
                return None
            self._reached = True
            self._turn_yaw_rad = self._yaw + self.TURN_RIGHT_RAD

        assert self._turn_yaw_rad is not None
        ctx.emit(
            self._pos[0], self._pos[1], self._pos[2],
            self._turn_yaw_rad, self.HOLD_SPEED_MPS,
        )

        if self._hold_start_t is None:
            current_yaw_rad = math.radians(ctx.pose.yaw)
            if _angle_diff(self._turn_yaw_rad, current_yaw_rad) > self.YAW_TOL_RAD:
                return None
            self._hold_start_t = ctx.pose.timestamp

        if ctx.pose.timestamp - self._hold_start_t < self.HOLD_S:
            return None

        ctx.gates_done += 1
        logger.info("Gate %d/%d cleared", ctx.gates_done, ctx.n_gates)
        ctx.tracker.reset()

        if ctx.gates_done < ctx.n_gates:
            from src.control.states.search import SearchState
            return SearchState()

        # Recon lap complete. Build the post-recon chain:
        #   ReturnHome -> Land (intermediate, motors stay armed)
        #            -> SaveGates (writes CSV, holds briefly)
        #            -> Takeoff   (back to recon altitude)
        #            -> Race      (polynomial lap)
        #            -> ReturnHome -> Land (terminal, mission_done)
        from src.control.states.race import RaceState
        from src.control.states.return_home import ReturnHomeState
        from src.control.states.save_gates import SaveGatesState
        from src.control.states.takeoff import TakeoffState

        race_gates = list(ctx.tracker.recorded_gates)
        if not race_gates:
            logger.warning("Recon complete but no gates were recorded; landing terminally")
            return ReturnHomeState()

        race = RaceState(race_gates)
        save = SaveGatesState(
            gates=race_gates,
            save_path=ctx.gates_save_path,
            then=TakeoffState(then=race),
        )
        return ReturnHomeState(then_after_land=save)
