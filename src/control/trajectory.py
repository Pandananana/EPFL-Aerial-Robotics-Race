"""Minimum-jerk polynomial trajectory through 3D waypoints, time-parameterized.

Ported from the aerial-robotics race controller. The trajectory is a chain of
5th-order polynomials, one per segment, stitched together with C4 continuity
(position/velocity/acceleration/jerk/snap match at interior waypoints). Segment
times are allocated proportional to segment length, and the total time is tuned
so the sampled trajectory just respects per-axis velocity / acceleration limits.

`position_at(t)`, `velocity_at(t)`, and `yaw_at(t)` are the runtime sample
points. The race state ticks at controller rate and emits a waypoint from each.
Feedforward velocity / acceleration aren't surfaced here because the
epfl-drone-race controller only takes a single Waypoint and runs a P-loop on
position — the smoothed reference does the work.
"""

from __future__ import annotations

import numpy as np


class PolyTrajectory:
    VEL_LIM_XY = 0.75
    VEL_LIM_Z = 0.5
    ACC_LIM_XY = 2.5
    ACC_LIM_Z = 2.0
    DISC_STEPS = 20

    def __init__(
        self,
        waypoints,
        v_init=None,
        a_init=None,
        v_final=None,
        a_final=None,
        *,
        vel_lim_xy: float | None = None,
        vel_lim_z: float | None = None,
        acc_lim_xy: float | None = None,
        acc_lim_z: float | None = None,
    ) -> None:
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.m = len(self.waypoints)
        if self.m < 2:
            raise ValueError("PolyTrajectory needs at least two waypoints")

        self.v_init = np.zeros(3) if v_init is None else np.asarray(v_init, dtype=float)
        self.a_init = np.zeros(3) if a_init is None else np.asarray(a_init, dtype=float)
        self.v_final = np.zeros(3) if v_final is None else np.asarray(v_final, dtype=float)
        self.a_final = np.zeros(3) if a_final is None else np.asarray(a_final, dtype=float)

        if vel_lim_xy is not None: self.VEL_LIM_XY = vel_lim_xy
        if vel_lim_z is not None: self.VEL_LIM_Z = vel_lim_z
        if acc_lim_xy is not None: self.ACC_LIM_XY = acc_lim_xy
        if acc_lim_z is not None: self.ACC_LIM_Z = acc_lim_z

        diffs = np.diff(self.waypoints, axis=0)
        self._seg_lengths = np.linalg.norm(diffs, axis=1)
        self.total_length = float(self._seg_lengths.sum())

        # Conservative initial guess — `_tune` scales it up to respect limits.
        t_f_guess = self.total_length / (0.5 * self.VEL_LIM_XY)
        self.total_time = self._tune(t_f_guess)
        self._solve(self.total_time)

    @staticmethod
    def _poly_matrix(t):
        return np.array([
            [t**5,    t**4,    t**3,    t**2, t, 1],
            [5*t**4,  4*t**3,  3*t**2,  2*t,  1, 0],
            [20*t**3, 12*t**2, 6*t,     2,    0, 0],
            [60*t**2, 24*t,    6,       0,    0, 0],
            [120*t,   24,      0,       0,    0, 0],
        ])

    def _seg_times(self, t_f):
        return t_f * self._seg_lengths / self._seg_lengths.sum()

    def _solve(self, t_f):
        seg_times = self._seg_times(t_f)
        self.times = np.concatenate([[0.0], np.cumsum(seg_times)])
        m = self.m
        n = 6 * (m - 1)
        self.coeffs = np.zeros((n, 3))
        A_0 = self._poly_matrix(0.0)

        for dim in range(3):
            A = np.zeros((n, n))
            b = np.zeros(n)
            pos = self.waypoints[:, dim]

            if m == 2:
                A_f = self._poly_matrix(seg_times[0])
                A[0, :6] = A_0[0]; b[0] = pos[0]
                A[1, :6] = A_f[0]; b[1] = pos[1]
                A[2, :6] = A_0[1]; b[2] = self.v_init[dim]
                A[3, :6] = A_f[1]; b[3] = self.v_final[dim]
                A[4, :6] = A_0[2]; b[4] = self.a_init[dim]
                A[5, :6] = A_f[2]; b[5] = self.a_final[dim]
            else:
                row = 0
                for i in range(m - 1):
                    A_f = self._poly_matrix(seg_times[i])
                    if i == 0:
                        A[row, :6] = A_0[0]; b[row] = pos[0]; row += 1
                        A[row, :6] = A_f[0]; b[row] = pos[1]; row += 1
                        A[row, :6] = A_0[1]; b[row] = self.v_init[dim]; row += 1
                        A[row, :6] = A_0[2]; b[row] = self.a_init[dim]; row += 1
                        A[row:row+4, :6] = A_f[1:]
                        A[row:row+4, 6:12] = -A_0[1:]
                        row += 4
                    elif i < m - 2:
                        A[row, i*6:(i+1)*6] = A_0[0]; b[row] = pos[i]; row += 1
                        A[row, i*6:(i+1)*6] = A_f[0]; b[row] = pos[i+1]; row += 1
                        A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                        A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                        row += 4
                    else:
                        A[row, i*6:(i+1)*6] = A_0[0]; b[row] = pos[i]; row += 1
                        A[row, i*6:(i+1)*6] = A_f[0]; b[row] = pos[i+1]; row += 1
                        A[row, i*6:(i+1)*6] = A_f[1]; b[row] = self.v_final[dim]; row += 1
                        A[row, i*6:(i+1)*6] = A_f[2]; b[row] = self.a_final[dim]; row += 1

            self.coeffs[:, dim] = np.linalg.solve(A, b)

    def _tune(self, t_f, max_iters=10, safety=1.05, tol=0.01):
        for _ in range(max_iters):
            self._solve(t_f)
            v_xy, v_z, a_xy, a_z = self._sample_limits()
            k = max(
                v_xy / self.VEL_LIM_XY,
                v_z / self.VEL_LIM_Z,
                np.sqrt(a_xy / self.ACC_LIM_XY),
                np.sqrt(a_z / self.ACC_LIM_Z),
            )
            new_t_f = t_f * k * safety
            if abs(new_t_f - t_f) / t_f < tol:
                return t_f
            t_f = new_t_f
        return t_f

    def _sample_limits(self):
        ts = np.linspace(0.0, self.times[-1], self.DISC_STEPS * self.m)
        v_xy_max = v_z_max = a_xy_max = a_z_max = 0.0
        for t in ts:
            seg, t_local = self._seg_index(t)
            M = self._poly_matrix(t_local)
            c = self.coeffs[seg*6:(seg+1)*6, :]
            v = M[1] @ c
            a = M[2] @ c
            v_xy_max = max(v_xy_max, float(np.hypot(v[0], v[1])))
            v_z_max = max(v_z_max, float(abs(v[2])))
            a_xy_max = max(a_xy_max, float(np.hypot(a[0], a[1])))
            a_z_max = max(a_z_max, float(abs(a[2])))
        return v_xy_max, v_z_max, a_xy_max, a_z_max

    def _seg_index(self, t):
        t_clamped = float(np.clip(t, 0.0, self.times[-1]))
        seg = int(min(max(np.searchsorted(self.times, t_clamped) - 1, 0), self.m - 2))
        return seg, t_clamped - self.times[seg]

    def position_at(self, t):
        seg, t_local = self._seg_index(t)
        return self._poly_matrix(t_local)[0] @ self.coeffs[seg*6:(seg+1)*6, :]

    def velocity_at(self, t):
        seg, t_local = self._seg_index(t)
        return self._poly_matrix(t_local)[1] @ self.coeffs[seg*6:(seg+1)*6, :]

    def yaw_at(self, t):
        """Heading from horizontal velocity direction; zero when stationary."""
        v = self.velocity_at(t)
        if np.hypot(v[0], v[1]) < 1e-6:
            return 0.0
        return float(np.arctan2(v[1], v[0]))
