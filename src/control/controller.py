"""Maps the current Waypoint + DronePose to a hover Setpoint.

Setpoints follow the cflib `send_hover_setpoint` format:
  vx, vy   body-frame velocities (m/s)
  yaw_rate deg/s
  height   absolute z (m)

For lateral tracking we run a P-loop on world-frame position error plus a
world-frame velocity feedforward from the waypoint (zero for stationary
targets, the trajectory's tangent velocity during race laps — without the
FF a pure P-loop lags the reference and cuts curves to the inside). The
combined world-frame velocity is capped at the waypoint's `max_speed_mps`
and then rotated into the body frame via the current yaw. Yaw is closed
the same way: a P-loop on yaw error produces a yaw rate, capped at
MAX_YAW_RATE_DPS. Altitude tracking stays where it was — set `height`
from the waypoint and let the firmware (or the in-sim PID) close the z
loop.
"""

from __future__ import annotations

import math

from PyQt6 import QtCore

from src.bus import Latest
from src.messages import DronePose, Setpoint, Waypoint


class Controller(QtCore.QObject):
    setpoint_ready = QtCore.pyqtSignal(object)  # Setpoint

    XY_POS_GAIN_S_INV = 1.4         # world velocity (m/s) per metre of position error
    YAW_RATE_GAIN_PER_S = 3.0       # yaw rate (deg/s) per degree of yaw error
    MAX_YAW_RATE_DPS = 120.0

    def __init__(self, *, default_height_m: float, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._pose: Latest[DronePose] = Latest()
        self._default_height = default_height_m

    @QtCore.pyqtSlot(object)
    def on_pose(self, pose: DronePose) -> None:
        self._pose.set(pose)

    @QtCore.pyqtSlot(object)
    def on_waypoint(self, waypoint: Waypoint) -> None:
        pose = self._pose.get()
        if pose is None:
            # No pose yet — fall back to altitude hold so the firmware doesn't
            # see an undefined setpoint while the link warms up.
            self.setpoint_ready.emit(Setpoint(
                vx=0.0, vy=0.0, yaw_rate=0.0, height=waypoint.z,
            ))
            return

        # World-frame velocity: P-correction on position error plus the
        # waypoint's velocity feedforward, then capped at the waypoint speed.
        dx = waypoint.x - pose.x
        dy = waypoint.y - pose.y
        vx_world = self.XY_POS_GAIN_S_INV * dx + waypoint.vx_ff
        vy_world = self.XY_POS_GAIN_S_INV * dy + waypoint.vy_ff
        speed = math.hypot(vx_world, vy_world)
        if waypoint.max_speed_mps > 0.0 and speed > waypoint.max_speed_mps:
            scale = waypoint.max_speed_mps / speed
            vx_world *= scale
            vy_world *= scale

        # Rotate world velocity into the body frame using current yaw.
        yaw_rad = math.radians(pose.yaw)
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        vx_body = c * vx_world + s * vy_world
        vy_body = -s * vx_world + c * vy_world

        # Yaw error -> yaw rate (degrees throughout to match the hover surface).
        yaw_err_deg = ((waypoint.yaw - pose.yaw + 180.0) % 360.0) - 180.0
        yaw_rate = self.YAW_RATE_GAIN_PER_S * yaw_err_deg
        yaw_rate = max(-self.MAX_YAW_RATE_DPS, min(self.MAX_YAW_RATE_DPS, yaw_rate))

        self.setpoint_ready.emit(Setpoint(
            vx=float(vx_body),
            vy=float(vy_body),
            yaw_rate=float(yaw_rate),
            height=float(waypoint.z),
        ))
