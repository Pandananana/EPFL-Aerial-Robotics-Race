"""Race lap state: fly a polynomial trajectory through cached gate centres.

Builds a minimum-jerk PolyTrajectory from the drone's current pose, then
chains `num_laps` repetitions of the gate sequence and finishes 1 m past
the last gate so the closing pass actually carries the drone through the
final gate plane. Each tick samples the trajectory at the current elapsed
time and emits a Waypoint — the position controller's P-loop tracks it.
Yaw is derived from the trajectory's velocity direction so the drone
faces where it's going.

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
from src.messages import Waypoint

logger = logging.getLogger(__name__)


class RaceState(State):
    RACE_SPEED_MPS = 3.0  # waypoint cap; the trajectory itself stays inside PolyTrajectory's limits
    FINISH_OVERSHOOT_M = 1.0  # extrapolate past the last gate so the closing pass actually flies through

    def __init__(self, gates: list[RecordedGate], num_laps: int = 2) -> None:
        if not gates:
            raise ValueError("RaceState needs at least one gate")
        self._gates = gates
        self._num_laps = num_laps
        self._trajectory: PolyTrajectory | None = None
        self._t_start: float | None = None

    def _build_trajectory(self, ctx: Context) -> PolyTrajectory:
        # Start from the drone's current pose, loop through every gate
        # `num_laps` times, then finish at a point past the last gate along
        # its entry direction so the trajectory actually carries the drone
        # through the final gate plane instead of decelerating to rest at
        # its centre. Between laps, route through a transition waypoint
        # 1.2 m directly above the takeoff spot.
        start = np.array([ctx.pose.x, ctx.pose.y, ctx.pose.z], dtype=np.float64)
        g_last = self._gates[-1].center
        transition = np.array([ctx.start_x, ctx.start_y, 1.2], dtype=np.float64)
        waypoints: list[np.ndarray] = [start]
        for lap in range(self._num_laps):
            if lap > 0:
                waypoints.append(transition.copy())
            for g in self._gates:
                waypoints.append(g.center.copy())
        prev_center = self._gates[-2].center if len(self._gates) >= 2 else start
        entry_dir = g_last - prev_center
        entry_norm = float(np.linalg.norm(entry_dir))
        if entry_norm > 1e-6:
            waypoints.append(g_last + (entry_dir / entry_norm) * self.FINISH_OVERSHOOT_M)
        else:
            waypoints.append(g_last.copy())
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
            if ctx.emit_race_trajectory is not None:
                ts = np.linspace(0.0, self._trajectory.total_time, 400)
                samples = np.array(
                    [self._trajectory.position_at(t) for t in ts],
                    dtype=np.float64,
                )
                ctx.emit_race_trajectory(samples)

        assert self._trajectory is not None and self._t_start is not None
        t = ctx.pose.timestamp - self._t_start

        if t >= self._trajectory.total_time:
            # Hold the last point until ReturnHome takes over. No feedforward
            # — the trajectory is over, we want the controller to settle.
            target = self._trajectory.position_at(self._trajectory.total_time)
            target_yaw = math.radians(ctx.pose.yaw)  # whatever we're at; ReturnHome will rebase
            ctx.emit(target[0], target[1], target[2], target_yaw, self.RACE_SPEED_MPS)
            logger.info("Race trajectory complete; returning home")
            from src.control.states.return_home import ReturnHomeState
            return ReturnHomeState()

        target = self._trajectory.position_at(t)
        v_ff = self._trajectory.velocity_at(t)
        target_yaw = self._trajectory.yaw_at(t)
        ctx.emit_waypoint(Waypoint(
            timestamp=ctx.pose.timestamp,
            x=float(target[0]), y=float(target[1]), z=float(target[2]),
            yaw=math.degrees(target_yaw),
            max_speed_mps=self.RACE_SPEED_MPS,
            vx_ff=float(v_ff[0]),
            vy_ff=float(v_ff[1]),
        ))
        return None
