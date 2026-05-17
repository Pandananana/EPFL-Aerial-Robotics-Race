"""Cascaded PID controller for the Webots Crazyflie.

Inputs match the real drone's `cflib.commander.send_hover_setpoint` surface:

    (vx, vy, yaw_rate, height)

where vx/vy are body-frame velocities in m/s, yaw_rate is in deg/s, and
height is the absolute z target in metres. The cascade is:

    height (m)     -> pos_z   -> v_z_target (m/s)  -> vel_z -> acc_z (m/s^2)
    vx,vy (body)   -> (rotate to inertial)         -> vel_xy -> acc_xy (m/s^2)
    yaw_rate (rad) -> rate_yaw directly

Inner attitude + body-rate loops convert (acc, yaw_rate) to motor PWM, same
as the firmware's velocity controller would.

Gains are borrowed from the EPFL aerial-robotics course's tuned controller
(controllers/main/exercises/ex1_pid_control.py) so they're known-good for
the Crazyflie model used by the Webots Crazyflie proto.

The PID helper class is a stripped-down copy of `simple_pid.PID` from the
same source — only the bits we need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


def _clamp(x: float, low: float | None, high: float | None) -> float:
    if high is not None and x > high:
        return high
    if low is not None and x < low:
        return low
    return x


class _PID:
    def __init__(self, kp: float, ki: float, kd: float,
                 out_lo: float | None = None, out_hi: float | None = None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_lo, self.out_hi = out_lo, out_hi
        self.setpoint = 0.0
        self._integral = 0.0
        self._last_measurement: float | None = None

    def set(self, setpoint: float) -> None:
        self.setpoint = setpoint

    def __call__(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement
        last = measurement if self._last_measurement is None else self._last_measurement
        d_meas = measurement - last
        p = self.kp * error
        self._integral = _clamp(self._integral + self.ki * error * dt, self.out_lo, self.out_hi)
        d = -self.kd * d_meas / dt if dt > 0 else 0.0
        self._last_measurement = measurement
        return _clamp(p + self._integral + d, self.out_lo, self.out_hi)


@dataclass
class _Sensors:
    """The subset of Webots sensor readings the controller needs."""
    x: float
    y: float
    z: float
    vx: float           # world frame
    vy: float
    vz: float
    roll: float         # radians
    pitch: float
    yaw: float
    rate_roll: float    # rad/s
    rate_pitch: float
    rate_yaw: float
    quat: tuple[float, float, float, float]  # x, y, z, w


class HoverController:
    """Convert a hover Setpoint + current Sensors to 4 motor PWM commands.

    Mirrors the cflib send_hover_setpoint surface: pure velocity tracking on
    xy (no position loop), altitude hold on z, direct yaw rate command.
    """

    G = 9.81
    MASS_KG = 0.0552  # Crazyflie 2.x

    # Per-loop output saturations (copied from aerial-robotics, known-good).
    L_VEL_XY = 3.0
    L_VEL_Z = 0.75
    L_ACC_RP = np.pi / 4   # tilt limit, rad
    L_RATE_RP = 2.0        # rad/s
    L_RATE_Y = 3.0         # rad/s

    def __init__(self) -> None:
        # Altitude: pos_z -> vel_z (gains from aerial-robotics).
        self.pid_pos_z = _PID(5.0, 0.0, 0.8, -self.L_VEL_Z, self.L_VEL_Z)
        self.pid_vel_z = _PID(7.0, 0.0, 2.0)
        # Body-frame velocity (we run two copies in the inertial frame, then
        # rotate the resulting acceleration into the body frame).
        self.pid_vel_x = _PID(0.5, 0.0, 0.015, -self.L_ACC_RP, self.L_ACC_RP)
        self.pid_vel_y = _PID(0.5, 0.0, 0.015, -self.L_ACC_RP, self.L_ACC_RP)
        # Attitude (roll, pitch) — tilt angle -> body rate.
        self.pid_att_x = _PID(10.0, 0.0, 0.2, -self.L_RATE_RP, self.L_RATE_RP)
        self.pid_att_y = _PID(10.0, 0.0, 0.2, -self.L_RATE_RP, self.L_RATE_RP)
        # Body rates -> torque command.
        self.pid_rate_roll = _PID(1.5, 0.0, 0.1)
        self.pid_rate_pitch = _PID(1.5, 0.0, 0.1)
        self.pid_rate_yaw = _PID(0.02, 0.0, 0.001)

    def step(self, *, vx_body: float, vy_body: float, yaw_rate_rad: float,
             height: float, dt: float, sensors: _Sensors) -> list[float]:
        # Altitude hold: z position -> z velocity target -> z acceleration.
        self.pid_pos_z.set(height)
        v_z_target = self.pid_pos_z(sensors.z, dt)
        self.pid_vel_z.set(v_z_target)
        acc_z = self.pid_vel_z(sensors.vz, dt)

        # XY velocity tracking. The hover setpoint is body-frame; rotate it
        # into the inertial frame so the velocity-PID setpoints decouple from
        # current yaw, then rotate the resulting acceleration back to body.
        c, s = np.cos(sensors.yaw), np.sin(sensors.yaw)
        vx_world = c * vx_body - s * vy_body
        vy_world = s * vx_body + c * vy_body
        self.pid_vel_x.set(vx_world)
        self.pid_vel_y.set(vy_world)
        acc_x_world = self.pid_vel_x(sensors.vx, dt)
        acc_y_world = self.pid_vel_y(sensors.vy, dt)
        rot = R.from_quat(list(sensors.quat))
        acc_body = rot.as_matrix().T @ np.array([acc_x_world, acc_y_world, 0.0])
        acc_x_body, acc_y_body = float(acc_body[0]), float(acc_body[1])

        # Attitude: lateral acceleration -> tilt angle setpoint.
        self.pid_att_x.set(np.clip(-acc_y_body, -self.L_ACC_RP, self.L_ACC_RP))
        self.pid_att_y.set(np.clip(acc_x_body, -self.L_ACC_RP, self.L_ACC_RP))
        rate_roll_sp = self.pid_att_x(sensors.roll, dt)
        rate_pitch_sp = self.pid_att_y(sensors.pitch, dt)

        # Yaw: direct rate command from the hover setpoint.
        rate_yaw_sp = float(np.clip(yaw_rate_rad, -self.L_RATE_Y, self.L_RATE_Y))

        # Rate loop -> body moments.
        self.pid_rate_roll.set(rate_roll_sp)
        self.pid_rate_pitch.set(rate_pitch_sp)
        self.pid_rate_yaw.set(rate_yaw_sp)
        roll_cmd = self.pid_rate_roll(sensors.rate_roll, dt)
        pitch_cmd = self.pid_rate_pitch(sensors.rate_pitch, dt)
        yaw_cmd = self.pid_rate_yaw(sensors.rate_yaw, dt)

        # Thrust mixing (k_* and motor formulas straight from aerial-robotics).
        k_thrust = 100
        k_rp = k_thrust * 0.7
        k_yaw = k_thrust * 10
        commanded = self.MASS_KG * np.array([acc_x_body, acc_y_body, acc_z + self.G])
        thrust = float(np.linalg.norm(commanded))
        m1 = k_thrust * thrust - k_rp * roll_cmd - k_rp * pitch_cmd + k_yaw * yaw_cmd
        m2 = k_thrust * thrust - k_rp * roll_cmd + k_rp * pitch_cmd - k_yaw * yaw_cmd
        m3 = k_thrust * thrust + k_rp * roll_cmd + k_rp * pitch_cmd + k_yaw * yaw_cmd
        m4 = k_thrust * thrust + k_rp * roll_cmd - k_rp * pitch_cmd - k_yaw * yaw_cmd
        return [float(np.clip(m, 0, 600)) for m in (m1, m2, m3, m4)]
