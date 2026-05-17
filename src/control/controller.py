"""Maps the current Waypoint + DronePose to a Setpoint.

Setpoints follow the cflib send_hover_setpoint format:
  vx, vy   body-frame velocities (m/s)
  yaw_rate deg/s
  height   absolute z (m)

The Crazyflie's hover commander already does altitude tracking from the
`height` field and holds lateral position when vx=vy=0. That gives us a
free, safe takeoff for the cost of emitting one Setpoint per waypoint:
zero velocity, target height. World->body velocity control for lateral
waypoint tracking will be added when RECON_LAP / RACE_LAP need it.
"""

from __future__ import annotations

from PyQt6 import QtCore

from src.bus import Latest
from src.messages import DronePose, Setpoint, Waypoint


class Controller(QtCore.QObject):
    setpoint_ready = QtCore.pyqtSignal(object)  # Setpoint

    def __init__(self, *, default_height_m: float, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._pose: Latest[DronePose] = Latest()
        self._default_height = default_height_m

    @QtCore.pyqtSlot(object)
    def on_pose(self, pose: DronePose) -> None:
        self._pose.set(pose)

    @QtCore.pyqtSlot(object)
    def on_waypoint(self, waypoint: Waypoint) -> None:
        # Takeoff / hover: zero velocity, target height from the waypoint.
        # Lateral position is held by the firmware. Needs Implementation:
        # compute body-frame velocity from the world-frame position error
        # for the racing phases.
        self.setpoint_ready.emit(Setpoint(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            height=waypoint.z,
        ))
