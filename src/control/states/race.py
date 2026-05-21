"""Race lap state: fly a polynomial trajectory through cached gate centres.

Builds a minimum-jerk PolyTrajectory from the drone's current pose, then
chains `num_laps` repetitions of the gate sequence followed by a return to
the first gate (so the timer naturally stops near the start). Each tick
samples the trajectory at the current elapsed time and emits a Waypoint —
the position controller's P-loop tracks it. Yaw is derived from the
trajectory's velocity direction so the drone faces where it's going.

When the trajectory completes, hand off to ReturnHome -> Land (terminal):
detection isn't running so we lean entirely on the cached gate data.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.control.gates_csv import RecordedGate
from src.control.states.base import Context, State
from src.control.trajectory import PolyTrajectory

logger = logging.getLogger(__name__)


class RaceState(State):
    RACE_SPEED_MPS = 3.0  # waypoint cap; the trajectory itself stays inside PolyTrajectory's limits
    FINISH_OVERSHOOT_M = 0.4  # extrapolate past gate 0 so the closing pass actually flies through

    def __init__(self, gates: list[RecordedGate], num_laps: int = 2) -> None:
        if not gates:
            raise ValueError("RaceState needs at least one gate")
        self._gates = gates
        self._num_laps = num_laps
        self._trajectory: PolyTrajectory | None = None
        self._t_start: float | None = None

    def _build_trajectory(self, ctx: Context) -> PolyTrajectory:
        # Start from the drone's current pose, loop through every gate
        # `num_laps` times, then close the circuit by flying through gate 0
        # (matching the aerial-robotics race convention where the timer stops
        # at the start/finish gate). The closing pass uses an overshoot point
        # past gate 0 along the entry direction so the trajectory actually
        # carries the drone through the gate plane instead of decelerating to
        # rest at its centre.
        start = np.array([ctx.pose.x, ctx.pose.y, ctx.pose.z], dtype=np.float64)
        waypoints: list[np.ndarray] = [start]
        for _ in range(self._num_laps):
            for g in self._gates:
                waypoints.append(g.center.copy())
        g0 = self._gates[0].center
        entry_dir = g0 - self._gates[-1].center
        entry_norm = float(np.linalg.norm(entry_dir))
        if entry_norm > 1e-6:
            waypoints.append(g0 + (entry_dir / entry_norm) * self.FINISH_OVERSHOOT_M)
        else:
            waypoints.append(g0.copy())
        traj = PolyTrajectory(waypoints)
        logger.info(
            "Race trajectory: %.2fm over %.2fs through %d waypoints "
            "(avg %.2f m/s, %d laps x %d gates)",
            traj.total_length, traj.total_time, len(waypoints),
            traj.total_length / traj.total_time if traj.total_time > 0 else 0.0,
            self._num_laps, len(self._gates),
        )
        return traj

    def tick(self, ctx: Context) -> State | None:
        if self._trajectory is None:
            self._trajectory = self._build_trajectory(ctx)
            self._t_start = ctx.pose.timestamp

        assert self._trajectory is not None and self._t_start is not None
        t = ctx.pose.timestamp - self._t_start

        if t >= self._trajectory.total_time:
            # Hold the last point until ReturnHome takes over.
            target = self._trajectory.position_at(self._trajectory.total_time)
            target_yaw = math.radians(ctx.pose.yaw)  # whatever we're at; ReturnHome will rebase
            ctx.emit(target[0], target[1], target[2], target_yaw, self.RACE_SPEED_MPS)
            logger.info("Race trajectory complete; returning home")
            from src.control.states.return_home import ReturnHomeState
            return ReturnHomeState()

        target = self._trajectory.position_at(t)
        target_yaw = self._trajectory.yaw_at(t)
        ctx.emit(target[0], target[1], target[2], target_yaw, self.RACE_SPEED_MPS)
        return None
